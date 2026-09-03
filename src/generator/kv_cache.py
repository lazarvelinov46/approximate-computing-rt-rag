"""KV-cache precision (knob 3).

Builds the `generate()` kwargs for a quantized KV cache. Everything here was
established empirically in notebooks/09_ac_knob3_env_audit against
transformers 5.0.0; results/knob3_env_audit.json is the record.

Why this returns a FRESH dict on every call
-------------------------------------------
transformers/generation/utils.py does, for cache_implementation="quantized":

    cache_config = generation_config.cache_config or {}
    cache_config["config"] = self.config.get_text_config()
    backend = cache_config.pop("backend", "quanto")
    QuantizedCache(backend=backend, **cache_config)

The dict is taken BY REFERENCE and mutated: `backend` is popped out and the
model config written in. Hand the same dict to two generate() calls and the
second one silently falls back to quanto. So callers pass a serializable
SPEC (plain params) and the mutable config is rebuilt per batch.

Frozen constants and why
------------------------
RESIDUAL_LENGTH = 512
    Above max_new_tokens in both regimes (32 short / 256 explain), so the
    fp16 decode buffer never fills and never flushes. The prompt is quantized
    exactly once at prefill and every question gets identical treatment.
    The library default of 128 flushes for answers longer than 128 tokens and
    not for shorter ones (measured: 0 flushes at decode=92, 1 at decode=224),
    making the approximation depend on the outcome. residual_length=0 flushes
    on alternating decode steps, re-quantizing the whole store from already-
    lossy values each time — compounding drift, not precision scaling.

Q_GROUP_SIZE = 64
    Library default. Metadata is one fp16 scale + one fp16 zero per group, so
    effective bits = nbits + 32/group_size, confirmed exactly at nbits 8, 4
    and 2 across groups 16..256. Group size is a second precision axis
    trading the same currency as nbits; at matched effective bits, more bits
    with coarser groups beat fewer bits with finer groups (4-bit/G256 at 4.12
    bits gives RMSE 0.397, 2-bit/G16 at 4.00 bits gives 1.182).

AXIS: hqq 0, quanto -1
    Both are per-channel. The HF docs recommend the opposite (hqq 1, quanto
    0) but those pages document the pre-5.0 API and are stale; the installed
    docstring and defaults agree with the measurement. On tensors with
    outlier channels, per-channel beats per-token by ~1.8x at 4 and 8 bits.
    The two backends use mirrored conventions: hqq axis 0 pairs with quanto
    axis -1 (RMSE 0.332 vs 0.335 at 4-bit/G64).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# --- frozen study constants ---------------------------------------------
Q_GROUP_SIZE = 64
RESIDUAL_LENGTH = 512
DEFAULT_BACKEND = "hqq"

# --- library limits, read from transformers 5.0.0 cache_utils.py ---------
BACKEND_NBITS = {"quanto": (2, 4), "hqq": (1, 2, 3, 4, 8)}
BACKEND_AXES = {"quanto": (0, -1), "hqq": (0, 1)}
PER_CHANNEL_AXIS = {"quanto": -1, "hqq": 0}

# hqq nbits=3 raises inside the packer at axis 1, every group size.
BACKEND_AXIS_EXCEPTIONS = {("hqq", 3): (0,)}

# Verified to follow nbits + 32/G exactly at groups 16..256
# (audit cell 12 for 8/4/2; notebook 10 cell 3 for 1, with 8 as control).
_FORMULA_VERIFIED_NBITS = (8, 4, 2, 1)

# 3-bit uses 3bit_32 packing whose payload varies with the reshape, so it does
# NOT follow the formula. Measured, seed 42.
_EFF_BITS_MEASURED = {
    (3, 16): 6.00, (3, 32): 5.00, (3, 64): 4.00, (3, 128): 3.50, (3, 256): 3.38,
}

FP16_BITS = 16


def allowed_nbits(backend: str = DEFAULT_BACKEND):
    """Bit-widths this backend accepts, plus 16 meaning 'no quantization'."""
    _check_backend(backend)
    return (FP16_BITS,) + BACKEND_NBITS[backend]


def _check_backend(backend: str) -> None:
    if backend not in BACKEND_NBITS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of "
            f"{sorted(BACKEND_NBITS)}")


def make_cache(bits: int = FP16_BITS,
               backend: str = DEFAULT_BACKEND,
               q_group_size: int = Q_GROUP_SIZE,
               residual_length: int = RESIDUAL_LENGTH,
               axis: Optional[int] = None,
               allow_flush: bool = False) -> Dict[str, Any]:
    """-> FRESH generate() kwargs. Call once per generate() call.

    bits=16 returns {} so `model.generate(**enc, **make_cache(16))` is
    byte-identical to the Phase 1 code path — no cache object is constructed
    and the fp16 DynamicCache default applies.

    `axis=None` selects the per-channel setting for the backend. `allow_flush`
    is inert here; generate_batch checks it against max_new_tokens.
    """
    if bits == FP16_BITS:
        return {}

    _check_backend(backend)
    if bits not in BACKEND_NBITS[backend]:
        raise ValueError(
            f"backend {backend!r} accepts nbits {list(BACKEND_NBITS[backend])}, "
            f"got {bits}. hqq is the only backend spanning 8/4/2; quanto tops "
            f"out at 4 bits.")

    if axis is None:
        axis = PER_CHANNEL_AXIS[backend]
    permitted = BACKEND_AXIS_EXCEPTIONS.get((backend, bits),
                                            BACKEND_AXES[backend])
    if axis not in permitted:
        raise ValueError(
            f"axis {axis} invalid for {backend} at nbits {bits}; "
            f"permitted {list(permitted)}"
            + (" (nbits=3 is per-channel only in this build)"
               if (backend, bits) == ("hqq", 3) else ""))

    if q_group_size <= 0 or q_group_size % 2:
        raise ValueError(f"q_group_size must be a positive even int, "
                         f"got {q_group_size}")
    if residual_length < 0:
        raise ValueError(f"residual_length must be >= 0, got {residual_length}")

    # Fresh inner dict every call — see module docstring.
    return {
        "cache_implementation": "quantized",
        "cache_config": {
            "backend": backend,
            "nbits": bits,
            "axis_key": axis,
            "axis_value": axis,
            "q_group_size": q_group_size,
            "residual_length": residual_length,
        },
    }


def effective_bits(bits: int, q_group_size: int = Q_GROUP_SIZE) -> float:
    """Measured storage cost per element, including scale/zero metadata.

    This, not the nominal nbits, is the compression axis for the Pareto plot,
    so it refuses to extrapolate. nbits 3 already proved the formula is not
    universal: its packed payload runs 4.00, 4.00, 3.50, 3.25, 3.26 bits
    across groups 16..256 instead of a flat 3.
    """
    if bits == FP16_BITS:
        return float(FP16_BITS)
    if (bits, q_group_size) in _EFF_BITS_MEASURED:
        return _EFF_BITS_MEASURED[(bits, q_group_size)]
    if bits in _FORMULA_VERIFIED_NBITS:
        return bits + 32.0 / q_group_size
    raise NotImplementedError(
        f"effective bits for nbits={bits} at group {q_group_size} were never "
        f"measured. The formula nbits + 32/G is verified only for nbits "
        f"{list(_FORMULA_VERIFIED_NBITS)}; nbits=3 violates it. Measure with "
        f"notebooks/10_ac_knob3_kv_cache CELL 2 and add the result to "
        f"_EFF_BITS_MEASURED before using this setting.")


def channels_per_group(batch: int, seq_len: int, q_group_size: int,
                       num_kv_heads: int = 2, head_dim: int = 128) -> int:
    """How many distinct channels share one scale under per-channel grouping.

    1 is true per-channel. Higher values mean the stride slipped and groups
    mixed channels, which costs ~20% RMSE per doubling (measured: 1 -> 0.362,
    2 -> 0.436, 4 -> 0.508 at batch 16, 4-bit).
    """
    from math import gcd
    numel = batch * num_kv_heads * seq_len * head_dim
    if numel % q_group_size:
        raise ValueError("tensor is not divisible by q_group_size")
    return head_dim // gcd(numel // q_group_size, head_dim)


def label(bits: int = FP16_BITS, backend: str = DEFAULT_BACKEND,
          q_group_size: int = Q_GROUP_SIZE,
          residual_length: int = RESIDUAL_LENGTH,
          axis: Optional[int] = None, **_) -> str:
    """Stable setting name for CSV/JSON, e.g. 'kv_hqq_n4_g64_r512'."""
    if bits == FP16_BITS:
        return "kv_fp16"
    if axis is None:
        axis = PER_CHANNEL_AXIS[backend]
    name = f"kv_{backend}_n{bits}_g{q_group_size}_r{residual_length}"
    if axis != PER_CHANNEL_AXIS[backend]:
        name += f"_ax{axis}"
    return name


def describe(bits: int = FP16_BITS, **kw) -> Dict[str, Any]:
    """Serializable record of one setting, for the summary JSON."""
    backend = kw.get("backend", DEFAULT_BACKEND)
    q_group_size = kw.get("q_group_size", Q_GROUP_SIZE)
    residual_length = kw.get("residual_length", RESIDUAL_LENGTH)
    axis = kw.get("axis")
    if axis is None and bits != FP16_BITS:
        axis = PER_CHANNEL_AXIS[backend]
    eff = effective_bits(bits, q_group_size)
    return {
        "label": label(bits, backend, q_group_size, residual_length, axis),
        "bits": bits,
        "backend": None if bits == FP16_BITS else backend,
        "axis": axis,
        "q_group_size": None if bits == FP16_BITS else q_group_size,
        "residual_length": None if bits == FP16_BITS else residual_length,
        "effective_bits": eff,
        "compression_vs_fp16": round(FP16_BITS / eff, 3),
    }


def selftest() -> None:
    """Cheap invariants. No torch, no GPU."""
    assert make_cache(16) == {}
    assert make_cache(16, backend="quanto") == {}

    # The mutation guard: two consecutive calls must BOTH carry a backend.
    a = make_cache(4)
    a["cache_config"].pop("backend")            # simulate generate()
    b = make_cache(4)
    assert b["cache_config"]["backend"] == "hqq", "make_cache leaked state"
    assert a["cache_config"] is not b["cache_config"]

    assert make_cache(4)["cache_config"]["axis_key"] == 0
    assert make_cache(4, backend="quanto")["cache_config"]["axis_key"] == -1

    for bad in (lambda: make_cache(8, backend="quanto"),
                lambda: make_cache(3, axis=1),
                lambda: make_cache(4, q_group_size=0),
                lambda: make_cache(4, residual_length=-1),
                lambda: make_cache(4, backend="gptq")):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    assert effective_bits(4, 64) == 4.5
    assert effective_bits(2, 64) == 2.5
    assert effective_bits(3, 64) == 4.0
    assert effective_bits(16) == 16.0

    assert channels_per_group(16, 744, 64) == 1
    assert channels_per_group(16, 745, 64) == 2
    assert channels_per_group(16, 745, 128) == 4
    assert channels_per_group(16, 745, 32) == 1

    assert label(16) == "kv_fp16"
    assert label(4) == "kv_hqq_n4_g64_r512"
    print("kv_cache selftest OK")
