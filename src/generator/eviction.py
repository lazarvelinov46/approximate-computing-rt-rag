# src/generator/eviction.py — REPLACES the previous version
"""KV-cache eviction (knob 4).

Builds a `Cache` object for `past_key_values=`. Everything here was
established against the installed transformers 5.0.0 in
notebooks/21_ac_knob4_env_audit; results/knob4_env_audit.json is the record.

Why this is hand-written
------------------------
transformers 5.0.0 has NO eviction primitive. SinkCache was removed; the only
budgeted layers are DynamicSlidingWindowLayer / StaticSlidingWindowLayer,
which are recency-only with no attention sink. `cache_implementation` has no
registry for custom classes and `cache_config` reaches QuantizedCache alone,
so knob 3's path in kv_cache.py is not reusable. The supported entry point is
`past_key_values=<Cache instance>` — "escape route 1" in
GenerationMixin._prepare_cache_for_generation, which short-circuits and
raises if cache_implementation is also set.

Why keep_ratio and not an absolute budget
-----------------------------------------
configs/knobs.yaml commits keep_ratio [1.0, 0.75, 0.5, 0.25], and it is the
right axis. Regime-1 prompts run 344 to 1524 tokens, so a fixed budget would
evict nothing for short prompts and heavily for long ones — one setting would
then average two populations, which is exactly the outcome-dependent
approximation that residual_length=512 was chosen to avoid in knob 3. A ratio
treats every question identically; the footprint is reported as a mean.

The budget is fixed at PREFILL and held for the run. Recomputing it per decode
step would let it drift upward as the sequence grows, making the footprint a
moving target within a single setting.

Why batch_size must be 1
------------------------
MEASURED in notebook 21 cells 5-6. get_mask_sizes can only report
(kv_length, kv_offset) and sdpa_mask builds `arange(kv_length) + kv_offset`,
so the mask machinery can describe a CONTIGUOUS kv range and nothing else.
A keep-set that is not a contiguous suffix therefore cannot be described.
Under a uniform (all-ones) padding mask this costs nothing — sink+recent
passed all four rows of a length-uniform batch with eviction firing. With
padding_side="left" it fails, SILENTLY and with no exception, on exactly the
padded rows. Recency passed both because a suffix is contiguous by
construction. Separately, "keep first N" under left padding pins N PAD tokens
rather than the attention sink.

Notebook 23 then measured that eager attention produces NaN under left padding
(all -inf softmax row), so batch 1 is required for the attention family too.
At batch 1, eager and sdpa are bit-identical (0/32 prediction strings differ),
so all four families share one numerics regime at no cost.

Why the keys are re-rotated (SHIFT_ROPE)
---------------------------------------
Keys are stored AFTER rotary embedding, so the angle is baked in. Deleting the
middle deletes the tokens but not their phases, leaving the model a distance
ladder with a hole in it. StreamingLLM re-assigns positions within the cache;
skipping that is a DIFFERENT and weaker method, so calling it StreamingLLM
would be false.

Rotations compose — R(d) . R(p) = R(p+d) — so a stored key is moved to a new
position exactly by applying one further rotation. No pre-RoPE copy needed.
Verified at max |composed - direct| = 5.8e-07 in notebook 22.

The invariant maintained here: the newest key keeps its true position, and
every retained key is packed contiguously backwards from it. Recency satisfies
this already (all deltas zero, hence no rotation and no cost); sink+recent
moves only the sink block, by +1 per decode step; random moves whatever it has
to. With the invariant, `get_mask_sizes` returning (stored + q, cumulative -
stored) describes the stored phases EXACTLY rather than approximately.

attention_scaling is deliberately NOT applied in _rotate. The stored key is
already s.R(p).k_raw; applying R(d) unscaled yields s.R(p+d).k_raw, which is
correct. Applying it scaled would square s.

Rotation composition is exact only while inv_freq is FIXED for the run, so
rope_type must be one of default/linear/yarn/llama3. dynamic and longrope
recompute it mid-generation. Qwen2.5-3B is default.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

# --- frozen study constants ----------------------------------------------
# StreamingLLM's standard sink width. Xiao et al. report the sink effect is
# saturated by 4 tokens; a larger sink buys nothing and eats the budget.
DEFAULT_SINK = 4
DEFAULT_SEED = 42
SHIFT_ROPE = True

# rope types that build inv_freq once. dynamic/longrope recompute it from the
# sequence length mid-generation, which breaks rotation composition.
STATIC_ROPE = ("default", "linear", "yarn", "llama3")

# Contiguous keep-sets tolerate a padded batch; the others do not. MEASURED,
# notebook 21 cell 6.
POLICIES = ("none", "recency", "sink_recent", "random")
CONTIGUOUS = ("none", "recency")

NO_EVICTION = 1.0


def budget_for(prefill_len: int, keep_ratio: float, policy: str,
               sink: int = DEFAULT_SINK) -> int:
    """Tokens retained, fixed at prefill and held for the run. Pure python."""
    if keep_ratio >= NO_EVICTION:
        return prefill_len
    b = max(1, math.ceil(keep_ratio * prefill_len))
    floor = sink + 1 if policy == "sink_recent" else 1
    return min(prefill_len, max(b, floor))


def keep_indices(policy: str, seq_len: int, budget: int,
                 sink: int = DEFAULT_SINK,
                 rng: Optional[random.Random] = None) -> List[int]:
    """Which positions of a `seq_len` block survive. Pure python, no torch.

    Returns SORTED indices into the current block. Length is
    min(seq_len, budget) — every policy spends the same budget, which is what
    makes the position-vs-attention contrast a mechanism test rather than a
    budget test.
    """
    if seq_len <= budget:
        return list(range(seq_len))

    if policy == "recency":
        return list(range(seq_len - budget, seq_len))

    if policy == "sink_recent":
        recent = budget - sink
        return list(range(sink)) + list(range(seq_len - recent, seq_len))

    if policy == "random":
        # The newest token is always retained: it is the query's own key at
        # this step, and dropping it makes the step ill-formed rather than
        # approximate. The control is over the OTHER budget-1 slots.
        if rng is None:
            raise ValueError("policy 'random' needs a seeded rng")
        return sorted(rng.sample(range(seq_len - 1), budget - 1)) + [seq_len - 1]

    raise ValueError(f"unknown policy {policy!r}; expected {list(POLICIES)}")


def _rtag(keep_ratio: float) -> str:
    return f"r{int(round(keep_ratio * 100)):03d}"


def label(policy: str = "none", keep_ratio: float = NO_EVICTION,
          sink: int = DEFAULT_SINK, seed: int = DEFAULT_SEED,
          shift_rope: bool = SHIFT_ROPE, **_) -> str:
    """Stable setting name for CSV/JSON, e.g. 'evict_sink4_r050'."""
    if policy == "none" or keep_ratio >= NO_EVICTION:
        return "evict_none"
    if policy == "recency":
        name = f"evict_recency_{_rtag(keep_ratio)}"
    elif policy == "sink_recent":
        name = f"evict_sink{sink}_{_rtag(keep_ratio)}"
    elif policy == "random":
        name = f"evict_random_{_rtag(keep_ratio)}_s{seed}"
    else:
        raise ValueError(f"unknown policy {policy!r}")
    if not shift_rope and policy not in CONTIGUOUS:
        name += "_naive"
    return name


def describe(policy: str = "none", keep_ratio: float = NO_EVICTION,
             sink: int = DEFAULT_SINK, seed: int = DEFAULT_SEED,
             shift_rope: bool = SHIFT_ROPE,
             prompt_tokens: Optional[int] = None,
             kv_bytes_per_token: int = 36 * 1024, **_) -> Dict[str, Any]:
    """Serializable record of one setting, for the summary JSON."""
    evicting = policy != "none" and keep_ratio < NO_EVICTION
    rec = {
        "label": label(policy, keep_ratio, sink, seed, shift_rope),
        "policy": policy if evicting else "none",
        "keep_ratio": keep_ratio if evicting else NO_EVICTION,
        "sink": sink if (evicting and policy == "sink_recent") else None,
        "seed": seed if (evicting and policy == "random") else None,
        "shift_rope": shift_rope if (evicting and policy not in CONTIGUOUS) else None,
        "contiguous": policy in CONTIGUOUS,
        "max_batch_size": 1 if (evicting and policy not in CONTIGUOUS) else None,
    }
    if prompt_tokens is not None:
        b = budget_for(prompt_tokens, keep_ratio if evicting else NO_EVICTION,
                       policy, sink)
        rec["budget_at_prompt"] = b
        rec["decode_cache_bytes"] = b * kv_bytes_per_token
    return rec


def _build_layer(policy, keep_ratio, sink, seed, shift_rope, rope, layer_idx):
    import torch
    from transformers.cache_utils import DynamicLayer
    from transformers.models.qwen2.modeling_qwen2 import rotate_half

    class EvictLayer(DynamicLayer):
        """Store a subset, return the full states.

        This is DynamicSlidingWindowLayer's contract and it is not optional.
        Returning the truncated states would change the hidden states of every
        position in the prefill forward, which is a different computation, not
        an evicted cache. Eviction affects what the DECODE steps carry.
        """

        is_sliding = False

        def __init__(self):
            super().__init__()
            self.policy = policy
            self.keep_ratio = keep_ratio
            self.sink = sink
            self.shift_rope = shift_rope
            self._rope = rope
            self.budget = None          # fixed at first update, from prefill
            self.cumulative_length = 0
            self.evicted = 0
            self.rotations = 0
            self.phase = None           # position each stored key is rotated to
            self._rng = random.Random(seed * 1000 + layer_idx)

        def reset(self):
            super().reset()
            self.budget = None
            self.cumulative_length = 0
            self.evicted = 0
            self.rotations = 0
            self.phase = None
            self._rng = random.Random(seed * 1000 + layer_idx)

        def _rotate(self, keys, delta):
            inv_freq = self._rope.to(device=keys.device, dtype=torch.float32)
            freqs = delta.to(torch.float32)[:, None] * inv_freq[None, :]
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().to(keys.dtype)[None, None, :, :]
            sin = emb.sin().to(keys.dtype)[None, None, :, :]
            return keys * cos + rotate_half(keys) * sin

        def update(self, key_states, value_states, cache_kwargs=None):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
                self.phase = torch.empty(0, dtype=torch.long,
                                         device=key_states.device)

            pos = (cache_kwargs or {}).get("cache_position")
            if pos is None:
                raise RuntimeError(
                    "EvictLayer needs cache_position to track rotary phase; "
                    "cache_kwargs carried only "
                    f"{sorted((cache_kwargs or {}).keys())}")

            n_new = key_states.shape[-2]
            if self.budget is None:
                self.budget = budget_for(n_new, self.keep_ratio, self.policy,
                                         self.sink)
            self.cumulative_length += n_new

            full_k = torch.cat([self.keys, key_states], dim=-2)
            full_v = torch.cat([self.values, value_states], dim=-2)
            full_phase = torch.cat([self.phase, pos.to(self.phase.device)])
            seq_len = full_k.shape[-2]

            if seq_len <= self.budget:
                self.keys, self.values, self.phase = full_k, full_v, full_phase
                return full_k, full_v

            keep = torch.tensor(
                keep_indices(self.policy, seq_len, self.budget, self.sink,
                             self._rng),
                dtype=torch.long, device=full_k.device)
            self.evicted += seq_len - keep.numel()

            kept_k = full_k.index_select(-2, keep)
            kept_v = full_v.index_select(-2, keep)
            kept_phase = full_phase.index_select(0, keep.to(full_phase.device))

            # Pack contiguously backwards from the newest token, which keeps
            # its true position. Recency already satisfies this, so its delta
            # is identically zero and no rotation is applied.
            newest = int(full_phase[-1])
            n_keep = keep.numel()
            target = torch.arange(newest - n_keep + 1, newest + 1,
                                  dtype=torch.long, device=kept_phase.device)
            delta = target - kept_phase

            if self.shift_rope and bool((delta != 0).any()):
                kept_k = self._rotate(kept_k, delta.to(kept_k.device))
                self.rotations += 1
                self.phase = target
            else:
                self.phase = kept_phase

            self.keys, self.values = kept_k, kept_v
            return full_k, full_v

        def get_seq_length(self) -> int:
            return self.cumulative_length

        def get_max_cache_shape(self) -> int:
            return self.budget if self.budget is not None else -1

        def get_mask_sizes(self, cache_position):
            query_length = cache_position.shape[0]
            stored = self.keys.shape[-2] if self.is_initialized else 0
            return stored + query_length, max(self.cumulative_length - stored, 0)

        def crop(self, max_length: int) -> None:
            raise NotImplementedError(
                "crop() on an evicted cache would silently restore states that "
                "were dropped. Not used by greedy generate; refuse loudly.")

    return EvictLayer()


def make_cache(model=None, policy: str = "none",
               keep_ratio: float = NO_EVICTION, sink: int = DEFAULT_SINK,
               seed: int = DEFAULT_SEED, shift_rope: bool = SHIFT_ROPE,
               batch_size: int = 1, allow_naive: bool = False,
               allow_padded: bool = False):
    """-> a FRESH Cache instance for past_key_values=, or None.

    policy='none' or keep_ratio=1.0 returns None, so
    `generate(**enc, past_key_values=None)` is byte-identical to the Phase 1
    path: no cache object is constructed and the fp16 DynamicCache applies.
    keep_ratio 1.0 is knobs.yaml's precise baseline for this knob.

    A fresh object every call is mandatory for a stronger reason than knob 3's:
    a Cache carries the previous batch's KEYS, not just a config dict. Reuse
    raises IndexError (stale cumulative_length empties the input_ids slice).
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected {list(POLICIES)}")
    if not 0.0 < keep_ratio <= NO_EVICTION:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    if policy == "none" or keep_ratio >= NO_EVICTION:
        return None

    if model is None:
        raise ValueError("make_cache needs the model to read rope inv_freq")
    if policy == "sink_recent" and sink < 1:
        raise ValueError(f"sink_recent needs sink >= 1, got {sink}")

    if policy not in CONTIGUOUS and batch_size != 1 and not allow_padded:
        raise ValueError(
            f"policy {policy!r} builds a non-contiguous keep-set, which the "
            f"mask builder cannot describe under a padded batch. MEASURED in "
            f"notebook 21 cell 6: it fails silently on padded rows, with no "
            f"exception. batch_size must be 1 (got {batch_size}). Pass "
            f"allow_padded=True only for a deliberate length-uniform batch.")

    if not shift_rope and policy not in CONTIGUOUS and not allow_naive:
        raise ValueError(
            "shift_rope=False leaves retained keys carrying their original "
            "rotary phase, which is a weaker method than StreamingLLM and not "
            "the pre-registered headline. Pass allow_naive=True to run the "
            "ablation deliberately.")

    rp = getattr(model.config, "rope_parameters", None) or {}
    rope_type = rp.get("rope_type", "default")
    if rope_type not in STATIC_ROPE:
        raise ValueError(
            f"rope_type={rope_type!r} recomputes inv_freq mid-generation, so "
            f"R(d).R(p) != R(p+d) and the shift is not exact. Expected one of "
            f"{list(STATIC_ROPE)}.")

    from transformers.cache_utils import Cache

    inv_freq = model.model.rotary_emb.inv_freq.detach()
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads)
    if 2 * inv_freq.numel() != head_dim:
        raise ValueError(
            f"partial rotary: inv_freq covers {2 * inv_freq.numel()} of "
            f"{head_dim} head dims, and _rotate assumes the full width.")

    n_layers = model.config.num_hidden_layers
    return Cache(layers=[
        _build_layer(policy, keep_ratio, sink, seed, shift_rope, inv_freq, i)
        for i in range(n_layers)])


def selftest() -> None:
    """Cheap invariants. No torch, no GPU."""
    # budget is a fraction of the PREFILL length, floored above the sink
    assert budget_for(744, 1.0, "recency") == 744
    assert budget_for(744, 0.5, "recency") == 372
    assert budget_for(744, 0.25, "sink_recent") == 186
    assert budget_for(10, 0.25, "sink_recent", sink=4) == 5    # sink+1 floor
    assert budget_for(10, 0.25, "recency") == 3
    assert budget_for(344, 0.75, "recency") == 258             # shortest prompt
    assert budget_for(1524, 0.75, "recency") == 1143           # longest prompt

    assert keep_indices("recency", 10, 20) == list(range(10))
    assert keep_indices("recency", 12, 6) == [6, 7, 8, 9, 10, 11]
    assert keep_indices("sink_recent", 12, 6, sink=2) == [0, 1, 8, 9, 10, 11]

    r = keep_indices("random", 12, 6, rng=random.Random(0))
    assert len(r) == 6 and r == sorted(r) and r[-1] == 11 and len(set(r)) == 6

    # Every policy spends the same budget — the matched-budget invariant.
    for p, kw in (("recency", {}), ("sink_recent", {"sink": 2}),
                  ("random", {"rng": random.Random(1)})):
        assert len(keep_indices(p, 40, 8, **kw)) == 8, p

    assert keep_indices("random", 40, 8, rng=random.Random(7)) == \
           keep_indices("random", 40, 8, rng=random.Random(7))

    assert label() == "evict_none"
    assert label("recency", 1.0) == "evict_none"          # yaml's baseline
    assert label("recency", 0.5) == "evict_recency_r050"
    assert label("sink_recent", 0.75) == "evict_sink4_r075"
    assert label("random", 0.25) == "evict_random_r025_s42"
    assert label("sink_recent", 0.5, shift_rope=False) == "evict_sink4_r050_naive"
    # Shifting is meaningless for a contiguous policy, so it never tags one.
    assert label("recency", 0.5, shift_rope=False) == "evict_recency_r050"

    d = describe("recency", 0.5, prompt_tokens=744)
    assert d["budget_at_prompt"] == 372
    assert d["decode_cache_bytes"] == 372 * 36 * 1024
    assert describe("recency", 1.0)["policy"] == "none"
    assert describe("sink_recent", 0.5)["max_batch_size"] == 1
    assert describe("recency", 0.5)["max_batch_size"] is None

    for bad in (lambda: make_cache(policy="oops"),
                lambda: keep_indices("oops", 10, 5),
                lambda: keep_indices("random", 10, 5),
                lambda: make_cache(object(), "recency", keep_ratio=0.0),
                lambda: make_cache(object(), "recency", keep_ratio=1.5),
                lambda: make_cache(object(), "sink_recent", 0.5, sink=0),
                lambda: make_cache(object(), "sink_recent", 0.5, batch_size=16),
                lambda: make_cache(object(), "sink_recent", 0.5,
                                   shift_rope=False)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    assert make_cache(policy="none") is None
    assert make_cache(policy="recency", keep_ratio=1.0) is None
    print("eviction selftest OK")
