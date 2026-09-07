# src/generator/eviction.py — NEW FILE
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

Why batch_size must be 1
------------------------
MEASURED in notebook 21 cell 5/6. get_mask_sizes can only report
(kv_length, kv_offset) and sdpa_mask builds `arange(kv_length) + kv_offset`,
so the mask machinery can describe a CONTIGUOUS kv range and nothing else.
A keep-set that is not a contiguous suffix therefore cannot be described.
Under a uniform (all-ones) padding mask this costs nothing — sink+recent
passed all four rows of a length-uniform batch with eviction firing. With
padding_side="left" it fails, SILENTLY and with no exception, on exactly the
padded rows. Recency passed both because a suffix is contiguous by
construction. So: recency runs at any batch size, everything else runs at
batch 1. The matched control is a batch-1 baseline vs the frozen batch-16
baseline (knob 5 measured batch 8 vs 16 at 0 discordant EM pairs / 1000).

Why the keys are re-rotated (SHIFT_ROPE)
---------------------------------------
Keys are stored AFTER rotary embedding, so the angle is baked in. Deleting
the middle deletes the tokens but not their phases, leaving the model a
distance ladder with a hole in it: two keys 12 and 11 steps back, then a
cliff, then four keys 4..1 back. Training never contained such a ladder.
StreamingLLM re-assigns positions within the cache; skipping that is a
DIFFERENT and weaker method, so calling it StreamingLLM would be false.

Rotations compose — R(d) . R(p) = R(p+d) — so a stored key is moved to a new
position exactly by applying one further rotation. No pre-RoPE copy needed.
The invariant maintained here: the newest key keeps its true position, and
every retained key is packed contiguously backwards from it. Recency
satisfies this already (all deltas zero, hence no rotation and no cost);
sink+recent moves only the sink block, by +1 per decode step; random moves
whatever it has to.

The invariant has a second payoff. With it, `get_mask_sizes` returning
(stored + q, cumulative - stored) describes the stored phases EXACTLY rather
than approximately, which is what the audit found broken for a naive
non-contiguous keep-set.

attention_scaling is deliberately NOT applied in _rotate. The stored key is
already s.R(p).k_raw; applying R(d) unscaled yields s.R(p+d).k_raw, which is
correct. Applying it scaled would square s. For Qwen2.5-3B (default rope,
rope_scaling None) s is 1.0 anyway; describe() records it either way.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

# --- frozen study constants ----------------------------------------------
# StreamingLLM's standard sink width. Xiao et al. report the sink effect is
# saturated by 4 tokens; larger sinks buy nothing and eat the budget.
DEFAULT_SINK = 4
DEFAULT_SEED = 42
SHIFT_ROPE = True

# Contiguous keep-sets tolerate a padded batch; the others do not. MEASURED,
# notebook 21 cell 6.
POLICIES = ("none", "recency", "sink_recent", "random")
CONTIGUOUS = ("none", "recency")


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
        pool = range(seq_len - 1)
        return sorted(rng.sample(list(pool), budget - 1)) + [seq_len - 1]

    raise ValueError(f"unknown policy {policy!r}; expected {list(POLICIES)}")


def label(policy: str = "none", budget: int = 0, sink: int = DEFAULT_SINK,
          seed: int = DEFAULT_SEED, shift_rope: bool = SHIFT_ROPE, **_) -> str:
    """Stable setting name for CSV/JSON, e.g. 'evict_sink4_recent252_b256'."""
    if policy == "none":
        return "evict_none"
    if policy == "recency":
        name = f"evict_recency_b{budget}"
    elif policy == "sink_recent":
        name = f"evict_sink{sink}_recent{budget - sink}_b{budget}"
    elif policy == "random":
        name = f"evict_random_b{budget}_s{seed}"
    else:
        raise ValueError(f"unknown policy {policy!r}")
    if not shift_rope and policy not in CONTIGUOUS:
        name += "_naive"
    return name


def describe(policy: str = "none", budget: int = 0, sink: int = DEFAULT_SINK,
             seed: int = DEFAULT_SEED, shift_rope: bool = SHIFT_ROPE,
             prompt_tokens: Optional[int] = None,
             kv_bytes_per_token: int = 36 * 1024, **_) -> Dict[str, Any]:
    """Serializable record of one setting, for the summary JSON."""
    rec = {
        "label": label(policy, budget, sink, seed, shift_rope),
        "policy": policy,
        "budget": None if policy == "none" else budget,
        "sink": sink if policy == "sink_recent" else None,
        "seed": seed if policy == "random" else None,
        "shift_rope": None if policy in CONTIGUOUS else shift_rope,
        "contiguous": policy in CONTIGUOUS,
        "max_batch_size": None if policy in CONTIGUOUS else 1,
    }
    if prompt_tokens is not None and policy != "none":
        kept = min(prompt_tokens, budget)
        rec["kept_frac_at_prompt"] = round(kept / prompt_tokens, 4)
        rec["decode_cache_bytes"] = kept * kv_bytes_per_token
    return rec


def _build_layer(policy, budget, sink, seed, shift_rope, rope, layer_idx):
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
            self.budget = budget
            self.sink = sink
            self.shift_rope = shift_rope
            self._rope = rope
            self.cumulative_length = 0
            self.evicted = 0
            self.rotations = 0
            self.phase = None          # absolute position each stored key is rotated to
            self._rng = random.Random(seed * 1000 + layer_idx)

        def reset(self):
            super().reset()
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
            return self.budget

        def get_mask_sizes(self, cache_position):
            query_length = cache_position.shape[0]
            stored = self.keys.shape[-2] if self.is_initialized else 0
            return stored + query_length, max(self.cumulative_length - stored, 0)

        def crop(self, max_length: int) -> None:
            raise NotImplementedError(
                "crop() on an evicted cache would silently restore states that "
                "were dropped. Not used by greedy generate; refuse loudly.")

    return EvictLayer()


def make_cache(model=None, policy: str = "none", budget: int = 0,
               sink: int = DEFAULT_SINK, seed: int = DEFAULT_SEED,
               shift_rope: bool = SHIFT_ROPE, batch_size: int = 1,
               allow_naive: bool = False, allow_padded: bool = False):
    """-> a FRESH Cache instance for past_key_values=, or None.

    policy='none' returns None so `generate(**enc, past_key_values=None)` is
    byte-identical to the Phase 1 path: no cache object is constructed and the
    fp16 DynamicCache default applies.

    A fresh object every call is mandatory for a different reason than knob
    3's: a Cache carries the previous batch's KEYS, not just a config dict.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected {list(POLICIES)}")
    if policy == "none":
        return None

    if model is None:
        raise ValueError("make_cache needs the model to read rope inv_freq")
    if budget < 2:
        raise ValueError(f"budget must be >= 2, got {budget}")
    if policy == "sink_recent":
        if sink < 1:
            raise ValueError(f"sink_recent needs sink >= 1, got {sink}")
        if sink >= budget:
            raise ValueError(
                f"sink={sink} >= budget={budget} leaves no recent window")

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

    from transformers.cache_utils import Cache

    rope = model.model.rotary_emb.inv_freq.detach()
    n_layers = model.config.num_hidden_layers
    return Cache(layers=[
        _build_layer(policy, budget, sink, seed, shift_rope, rope, i)
        for i in range(n_layers)])


def selftest() -> None:
    """Cheap invariants. No torch, no GPU."""
    assert keep_indices("recency", 10, 20) == list(range(10))
    assert keep_indices("sink_recent", 10, 20) == list(range(10))

    assert keep_indices("recency", 12, 6) == [6, 7, 8, 9, 10, 11]
    assert keep_indices("sink_recent", 12, 6, sink=2) == [0, 1, 8, 9, 10, 11]

    r = keep_indices("random", 12, 6, rng=random.Random(0))
    assert len(r) == 6 and r == sorted(r) and r[-1] == 11 and len(set(r)) == 6

    # Every policy spends the same budget — the matched-budget invariant.
    for p, kw in (("recency", {}), ("sink_recent", {"sink": 2}),
                  ("random", {"rng": random.Random(1)})):
        assert len(keep_indices(p, 40, 8, **kw)) == 8, p

    # Seeded random is reproducible and layer-dependent draws differ.
    assert keep_indices("random", 40, 8, rng=random.Random(7)) == \
           keep_indices("random", 40, 8, rng=random.Random(7))

    assert label() == "evict_none"
    assert label("recency", 256) == "evict_recency_b256"
    assert label("sink_recent", 256) == "evict_sink4_recent252_b256"
    assert label("random", 256) == "evict_random_b256_s42"
    assert label("sink_recent", 256, shift_rope=False) == \
           "evict_sink4_recent252_b256_naive"
    # Shifting is meaningless for a contiguous policy, so it never tags one.
    assert label("recency", 256, shift_rope=False) == "evict_recency_b256"

    assert describe("recency", 256, prompt_tokens=744)["kept_frac_at_prompt"] \
           == 0.3441
    assert describe()["budget"] is None
    assert describe("sink_recent", 256)["max_batch_size"] == 1
    assert describe("recency", 256)["max_batch_size"] is None

    for bad in (lambda: make_cache(policy="oops"),
                lambda: keep_indices("oops", 10, 5),
                lambda: keep_indices("random", 10, 5),
                lambda: make_cache(object(), "recency", budget=1),
                lambda: make_cache(object(), "sink_recent", 256, sink=0),
                lambda: make_cache(object(), "sink_recent", 8, sink=8),
                lambda: make_cache(object(), "sink_recent", 256, batch_size=16),
                lambda: make_cache(object(), "sink_recent", 256,
                                   shift_rope=False)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    assert make_cache(policy="none") is None
    print("eviction selftest OK")
