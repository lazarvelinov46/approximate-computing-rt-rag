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
GenerationMixin._prepare_cache_for_generation.

Why keep_ratio and not an absolute budget
-----------------------------------------
configs/knobs.yaml commits keep_ratio [1.0, 0.75, 0.5, 0.25]. Regime-1 prompts
run 344 to 1524 tokens, so a fixed budget would evict nothing for short prompts
and heavily for long ones — one setting averaging two populations, the
outcome-dependent approximation residual_length=512 was chosen to avoid in
knob 3. The budget is fixed at PREFILL and held for the run; recomputing per
step would let it drift upward as the sequence grows.

Why batch_size must be 1
------------------------
MEASURED, notebook 21 cells 5-6. get_mask_sizes reports (kv_length, kv_offset)
and sdpa_mask builds `arange(kv_length) + kv_offset`, so only a CONTIGUOUS kv
range can be described. Non-contiguous keep-sets pass a length-uniform batch
and fail SILENTLY on padded rows. Separately, "keep first N" under
padding_side=left pins N PAD tokens rather than the sink. Notebook 23 then
measured that eager attention NaNs under left padding (all -inf softmax row),
which the attention policy needs. At batch 1 eager and sdpa are bit-identical
(0/32 strings), so all families share one numerics regime at no cost.

Why the keys are re-rotated (SHIFT_ROPE)
---------------------------------------
Keys are stored AFTER rotary embedding. Deleting the middle deletes the tokens
but not their phases, leaving a distance ladder with a hole in it.
StreamingLLM re-assigns positions within the cache; skipping that is a weaker
method. Rotations compose — R(d).R(p) = R(p+d) — verified at 5.8e-07 in
notebook 22. Invariant: the newest key keeps its true position and every
retained key packs contiguously backwards from it, which also makes the
(kv_length, kv_offset) handed to the mask builder exactly true.
attention_scaling is NOT applied in _rotate: the stored key is already
s.R(p).k, so R(d) unscaled gives s.R(p+d).k. Composition is exact only while
inv_freq is fixed, so rope_type must be in STATIC_ROPE.

Eviction is incremental and path-dependent
------------------------------------------
At steady state each decode step selects from budget+1 candidates and drops
one; an evicted key never returns, even if its accumulated mass would later
rank it highly. This is H2O's greedy formulation rather than a global
re-selection, and it applies to every policy here.

Provenance tracking (`origin`)
------------------------------
`phase` is overwritten with the packed target arange, and `_last_keep` indexes
the CURRENT block, so neither answers "which prompt tokens survived". `origin`
carries each stored key's original absolute position through every
index_select. Purely diagnostic — it never touches the computation — but it is
what turns the close-out from "attention scored higher EM" into "attention
scored higher EM because it kept the gold span where recency dropped it".

The attention policy (H2O / SnapKV family)
------------------------------------------
Scores come from a forward hook on layers[i].self_attn, which receives the
module's (attn_output, attn_weights) tuple. attn_weights is real only under
attn_implementation="eager"; Qwen2DecoderLayer discards it, and update()
receives no query states, so no other route exists.

Pre-registered choices, none tuned:
  GQA REDUCTION: the 8 query heads sharing each KV head are reduced by MEAN.
  HEAD REDUCTION: a further MEAN over the KV heads. Forced, not chosen — the
    layer stores one [B, n_kv, S, D] tensor and index_select on the sequence
    dim applies across heads, so the keep-set is per-LAYER. Per-head keep-sets
    are implementable with gather (the phase invariant keeps them maskable
    since every head packs to the same ladder) and are recorded as NOT TAKEN.
  ACCUMULATION: MEAN over steps observed, not SUM. Sum rewards age — a key
    present for 30 steps would outscore one present for 3.
  CAUSALITY CORRECTION: at prefill, key j is visible to queries j..S-1, so a
    plain mean over the query axis rewards early keys for being early and
    would hand the policy the sink as an artifact. Each key's mass is divided
    by its visible-query count, which removes this exactly.
  PREFILL KEEPS EVERYTHING. The hook fires after the update() it would inform,
    so no scores exist at prefill. Eviction begins at the first decode step,
    which drops (prompt_len + 1 - budget) keys at once. Position policies
    evict at prefill; budgets match from step 1 onward and eviction cannot
    lower peak memory either way, so the contrast holds. The difference in
    first application is recorded, not hidden.
  THE SINK IS NOT SPECIAL-CASED. If attention rediscovers it, that is a result.

The hook sees attention over the FULL returned states, not the stored subset,
so update() records `_last_keep` and the hook maps full-length mass onto the
retained keys through it.
"""

from __future__ import annotations

import contextlib
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- frozen study constants ----------------------------------------------
# StreamingLLM's standard sink width; Xiao et al. report saturation by 4.
DEFAULT_SINK = 4
DEFAULT_SEED = 42
SHIFT_ROPE = True

# rope types that build inv_freq once. dynamic/longrope recompute it
# mid-generation, which breaks rotation composition.
STATIC_ROPE = ("default", "linear", "yarn", "llama3")

POLICIES = ("none", "recency", "sink_recent", "random", "attention")
CONTIGUOUS = ("none", "recency")
NEEDS_EAGER = ("attention",)

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
                 rng: Optional[random.Random] = None,
                 scores: Optional[Sequence[float]] = None) -> List[int]:
    """Which positions of a `seq_len` block survive. Pure python, no torch.

    SORTED indices into the current block, length min(seq_len, budget). Every
    policy spends the same budget, which is what makes the position-vs-attention
    contrast a mechanism test rather than a budget test.
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

    if policy == "attention":
        if scores is None or len(scores) != seq_len:
            raise ValueError(
                f"policy 'attention' needs {seq_len} scores, got "
                f"{None if scores is None else len(scores)}")
        # Newest is force-kept for the same reason as random, and because it
        # has never been scored (the hook has not fired for it yet).
        pool = sorted(range(seq_len - 1),
                      key=lambda j: (-scores[j], -j))   # ties -> more recent
        return sorted(pool[:budget - 1]) + [seq_len - 1]

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
    elif policy == "attention":
        name = f"evict_attn_{_rtag(keep_ratio)}"
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
        "needs_eager": policy in NEEDS_EAGER,
        "max_batch_size": 1 if (evicting and policy not in CONTIGUOUS) else None,
        "first_evict": "decode" if (evicting and policy == "attention")
                       else ("prefill" if evicting else None),
    }
    if evicting and policy == "attention":
        rec["gqa_reduce"] = "mean"
        rec["head_reduce"] = "mean"
        rec["accumulate"] = "mean_over_steps"
        rec["keep_set_scope"] = "layer"
    if prompt_tokens is not None:
        b = budget_for(prompt_tokens, keep_ratio if evicting else NO_EVICTION,
                       policy, sink)
        rec["budget_at_prompt"] = b
        rec["decode_cache_bytes"] = b * kv_bytes_per_token
    return rec


# --- provenance helpers ---------------------------------------------------

def kept_origins(cache) -> List[List[int]]:
    """-> per-layer sorted lists of ORIGINAL absolute positions still cached."""
    if cache is None:
        return []
    return [sorted(int(x) for x in lay.origin.tolist()) for lay in cache.layers]


def span_retention(cache, start: int, end: int) -> Dict[str, Any]:
    """What fraction of prompt tokens [start, end) survived, per layer.

    `start`/`end` are token offsets into the prompt, so the caller supplies the
    gold-passage span from prompt construction. Diagnostic only.
    """
    if cache is None:
        return {"span": (start, end), "n_span": end - start, "per_layer": [],
                "mean": 1.0, "min": 1.0, "max": 1.0}
    n = max(end - start, 1)
    per = [len([p for p in o if start <= p < end]) / n for o in kept_origins(cache)]
    return {"span": (start, end), "n_span": end - start, "per_layer": per,
            "mean": sum(per) / len(per), "min": min(per), "max": max(per)}


def _build_layer(policy, keep_ratio, sink, seed, shift_rope, rope, layer_idx):
    import torch
    from transformers.cache_utils import DynamicLayer
    from transformers.models.qwen2.modeling_qwen2 import rotate_half

    class EvictLayer(DynamicLayer):
        """Store a subset, return the full states.

        DynamicSlidingWindowLayer's contract, and not optional: returning the
        truncated states would change the hidden states of every prefill
        position, which is a different computation, not an evicted cache.
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
            self.n_updates = 0
            self.evicted = 0
            self.rotations = 0
            self.phase = None           # position each stored key is rotated to
            self.origin = None          # ORIGINAL position of each stored key
            self.scores = None          # running SUM of per-step mean mass
            self.n_obs = None           # steps each stored key was observed
            self._last_keep = None      # current-block indices kept, for the hook
            self._rng = random.Random(seed * 1000 + layer_idx)

        def reset(self):
            super().reset()
            self.budget = None
            self.cumulative_length = 0
            self.n_updates = 0
            self.evicted = 0
            self.rotations = 0
            self.phase = None
            self.origin = None
            self.scores = None
            self.n_obs = None
            self._last_keep = None
            self._rng = random.Random(seed * 1000 + layer_idx)

        def _rotate(self, keys, delta):
            inv_freq = self._rope.to(device=keys.device, dtype=torch.float32)
            freqs = delta.to(torch.float32)[:, None] * inv_freq[None, :]
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().to(keys.dtype)[None, None, :, :]
            sin = emb.sin().to(keys.dtype)[None, None, :, :]
            return keys * cos + rotate_half(keys) * sin

        def observe(self, mass) -> None:
            """Called by the scorer hook with per-key mass over the FULL block."""
            if self._last_keep is None:
                return
            self.scores = self.scores + mass.to(self.scores.device
                                                ).index_select(0, self._last_keep)
            self.n_obs = self.n_obs + 1

        def update(self, key_states, value_states, cache_kwargs=None):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
                dev = key_states.device
                self.phase = torch.empty(0, dtype=torch.long, device=dev)
                self.origin = torch.empty(0, dtype=torch.long, device=dev)
                self.scores = torch.zeros(0, dtype=torch.float32, device=dev)
                self.n_obs = torch.zeros(0, dtype=torch.float32, device=dev)

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
            self.n_updates += 1

            full_k = torch.cat([self.keys, key_states], dim=-2)
            full_v = torch.cat([self.values, value_states], dim=-2)
            p = pos.to(self.phase.device)
            full_phase = torch.cat([self.phase, p])
            full_origin = torch.cat([self.origin, p])
            z = torch.zeros(n_new, dtype=torch.float32, device=self.scores.device)
            full_scores = torch.cat([self.scores, z])
            full_nobs = torch.cat([self.n_obs, z])
            seq_len = full_k.shape[-2]

            # The attention policy has no scores until the hook has fired, and
            # the hook fires after the update it would inform. Prefill keeps
            # everything; eviction begins at the first decode step.
            defer = self.policy == "attention" and self.n_updates == 1

            if seq_len <= self.budget or defer:
                self.keys, self.values = full_k, full_v
                self.phase, self.origin = full_phase, full_origin
                self.scores, self.n_obs = full_scores, full_nobs
                self._last_keep = torch.arange(seq_len, device=full_k.device)
                return full_k, full_v

            sc = None
            if self.policy == "attention":
                sc = (full_scores / full_nobs.clamp(min=1.0)).tolist()
            keep = torch.tensor(
                keep_indices(self.policy, seq_len, self.budget, self.sink,
                             self._rng, sc),
                dtype=torch.long, device=full_k.device)
            self.evicted += seq_len - keep.numel()
            self._last_keep = keep

            kept_k = full_k.index_select(-2, keep)
            kept_v = full_v.index_select(-2, keep)
            k_meta = keep.to(full_phase.device)
            kept_phase = full_phase.index_select(0, k_meta)
            self.origin = full_origin.index_select(0, k_meta)
            self.scores = full_scores.index_select(0, keep.to(self.scores.device))
            self.n_obs = full_nobs.index_select(0, keep.to(self.n_obs.device))

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


@contextlib.contextmanager
def attention_scoring(model, cache, n_kv_heads: Optional[int] = None):
    """Hook layers[i].self_attn so each EvictLayer sees its attention mass.

    A no-op for non-attention policies and for cache=None, so callers can wrap
    every generate() call unconditionally.
    """
    import torch

    if cache is None or getattr(cache.layers[0], "policy", None) != "attention":
        yield
        return

    if model.config._attn_implementation != "eager":
        raise RuntimeError(
            f"attention scoring needs attn_implementation='eager'; the model "
            f"resolved to {model.config._attn_implementation!r}. Under sdpa "
            f"attn_weights is None — there is nothing to hook.")

    n_kv = n_kv_heads or model.config.num_key_value_heads

    def make_hook(layer):
        def hook(module, args, out):
            w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            if w is None:
                raise RuntimeError(
                    "self_attn returned no attention weights. Under sdpa this "
                    "is expected; eager is required for the attention policy.")
            if w.shape[0] != 1:
                raise RuntimeError(
                    f"attention scoring assumes batch 1, got {w.shape[0]}. "
                    "Padded rows would misalign the mass with the keep-set.")
            w = w[0].float()                                # [H, q, kv]
            H, q, kv = w.shape
            w = w.view(n_kv, H // n_kv, q, kv).mean(dim=1)  # GQA mean
            w = w.mean(dim=0)                               # head mean -> [q, kv]
            total = w.sum(dim=0)                            # [kv]
            # Key j is visible to queries j..S-1 at prefill (q == kv) and to
            # the single query at decode. Dividing by the visible count stops
            # early keys scoring high purely for being early.
            if q == kv:
                seen = torch.arange(kv, 0, -1, dtype=total.dtype,
                                    device=total.device)
            else:
                seen = torch.full((kv,), float(q), dtype=total.dtype,
                                  device=total.device)
            layer.observe(total / seen)
        return hook

    handles = [blk.self_attn.register_forward_hook(make_hook(lay))
               for blk, lay in zip(model.model.layers, cache.layers)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def make_cache(model=None, policy: str = "none",
               keep_ratio: float = NO_EVICTION, sink: int = DEFAULT_SINK,
               seed: int = DEFAULT_SEED, shift_rope: bool = SHIFT_ROPE,
               batch_size: int = 1, allow_naive: bool = False,
               allow_padded: bool = False):
    """-> a FRESH Cache instance for past_key_values=, or None.

    policy='none' or keep_ratio=1.0 returns None, so
    `generate(**enc, past_key_values=None)` is byte-identical to the Phase 1
    path. keep_ratio 1.0 is knobs.yaml's precise baseline for this knob.

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

    if policy in NEEDS_EAGER and model.config._attn_implementation != "eager":
        raise RuntimeError(
            f"policy {policy!r} reads attention weights, which are None under "
            f"{model.config._attn_implementation!r}. Call "
            f"model.set_attn_implementation('eager') first. NOTE: eager NaNs "
            f"under left padding (notebook 23), so batch must stay at 1.")

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
    assert budget_for(744, 1.0, "recency") == 744
    assert budget_for(744, 0.5, "recency") == 372
    assert budget_for(744, 0.25, "sink_recent") == 186
    assert budget_for(10, 0.25, "sink_recent", sink=4) == 5    # sink+1 floor
    assert budget_for(344, 0.75, "recency") == 258             # shortest prompt
    assert budget_for(1524, 0.75, "recency") == 1143           # longest prompt

    assert keep_indices("recency", 12, 6) == [6, 7, 8, 9, 10, 11]
    assert keep_indices("sink_recent", 12, 6, sink=2) == [0, 1, 8, 9, 10, 11]
    r = keep_indices("random", 12, 6, rng=random.Random(0))
    assert len(r) == 6 and r == sorted(r) and r[-1] == 11

    sc = [0.9, 0.1, 0.8, 0.2, 0.7, 0.0]
    assert keep_indices("attention", 6, 3, scores=sc) == [0, 2, 5]
    assert keep_indices("attention", 6, 2, scores=sc) == [0, 5]
    assert keep_indices("attention", 5, 3, scores=[1.0, 1.0, 1.0, 0.0, 0.0]) \
           == [1, 2, 4]

    # Every policy spends the same budget — the matched-budget invariant.
    for p, kw in (("recency", {}), ("sink_recent", {"sink": 2}),
                  ("random", {"rng": random.Random(1)}),
                  ("attention", {"scores": [i / 40 for i in range(40)]})):
        assert len(keep_indices(p, 40, 8, **kw)) == 8, p

    assert label() == "evict_none"
    assert label("recency", 1.0) == "evict_none"
    assert label("recency", 0.5) == "evict_recency_r050"
    assert label("sink_recent", 0.75) == "evict_sink4_r075"
    assert label("random", 0.25) == "evict_random_r025_s42"
    assert label("attention", 0.5) == "evict_attn_r050"
    assert label("attention", 0.5, shift_rope=False) == "evict_attn_r050_naive"
    assert label("recency", 0.5, shift_rope=False) == "evict_recency_r050"

    d = describe("attention", 0.5, prompt_tokens=744)
    assert d["budget_at_prompt"] == 372 and d["needs_eager"] is True
    assert d["first_evict"] == "decode" and d["keep_set_scope"] == "layer"
    assert describe("recency", 0.5)["first_evict"] == "prefill"
    assert describe("recency", 1.0)["policy"] == "none"
    assert describe("sink_recent", 0.5)["max_batch_size"] == 1

    # provenance helpers are inert on the baseline path
    assert kept_origins(None) == []
    assert span_retention(None, 0, 10)["mean"] == 1.0

    for bad in (lambda: make_cache(policy="oops"),
                lambda: keep_indices("oops", 10, 5),
                lambda: keep_indices("random", 10, 5),
                lambda: keep_indices("attention", 10, 5),
                lambda: keep_indices("attention", 10, 5, scores=[0.0] * 3),
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