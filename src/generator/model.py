"""Qwen2.5 load, prompt construction, batched greedy generation. Phase 1.

Precise baseline for KNOBS 3 & 4: fp16 weights, full fp16 KV cache (the
default DynamicCache), no eviction. Phase 2 swaps only the cache object.

Experimental-setup invariants — must not change between sweeps:
  * passages presented in descending retrieval-rank order, numbered [1]..[k]
  * this exact system prompt (EM is literal; the answer format is part of the
    measurement, not a cosmetic choice)
  * greedy decoding, max_new_tokens fixed
  * left padding (mandatory for correct batched decode)
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Optional

import torch
import re

SYSTEM = (
    "You answer questions using only the numbered context passages.\n"
    "Output only the answer itself, copied as a minimal span from the passages. "
    "Never explain, never restate the question, never write a full sentence.\n"
    "\n"
    "Example question: Which city is home to the Eiffel Tower?\n"
    "Example answer: Paris"
)

# --- Regime B: explain-then-answer ---------------------------------------
# Long decode (~90 tokens vs ~3) so KV eviction has decode-time context to act
# on and decode throughput is measurable in Phase 3. Selected on a 200-question
# dev sample disjoint from the frozen subset: EM 0.425 vs 0.435 for a shorter
# variant (inside 1 SE), 100% parse, 0% truncation, median decode 90 vs 53.
SYSTEM_EXPLAIN = (
    "You answer questions using only the numbered context passages.\n"
    "Work through EVERY passage in order. For each one, write a sentence "
    "stating what it says and whether it helps answer the question.\n"
    "Then write one or two sentences connecting the useful passages into a "
    "chain of reasoning.\n"
    "Then write a final line in exactly this form: Answer: <answer>\n"
    "The answer itself must be a minimal span copied from the passages — "
    "never a full sentence."
)

_ANSWER_RE = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)

# --- model-portability constants (cross-model arms) -----------------------
# Llama 3.x templates write "Today Date: <strftime_now()>" into the system
# block when no date_string is passed, so the prompt would change every day.
# Pinned to the template's own fallback. Templates that never read it
# (Qwen2.5) ignore the kwarg — byte-identity verified on every frozen Qwen
# prompt in notebook 33 cell 2.
TEMPLATE_KWARGS = {"date_string": "26 Jul 2024"}

# The chat template already emits any BOS the model needs (Llama's opens with
# <|begin_of_text|>); add_special_tokens=True would prepend a SECOND one. Qwen2
# adds no special tokens either way — id-identity verified in notebook 33.
ENCODE_KWARGS = {"add_special_tokens": False}


def n_tokens(tok, text: str) -> int:
    """Token count exactly as generate_batch tokenizes. Single source for the
    prompt_tokens and decode_tokens columns."""
    return len(tok.encode(text, **ENCODE_KWARGS))


def kv_geometry(config) -> dict:
    """KV shape and fp16 bytes per cached token (K and V, all layers).

    36 KiB for Qwen2.5-3B (36 layers x 2 KV heads), 112 KiB for Llama-3.2-3B
    (28 x 8). eviction.describe() defaults to the Qwen figure — pass this.
    """
    n_kv = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads)
    n_layers = config.num_hidden_layers
    return {
        "n_layers": n_layers,
        "n_kv_heads": n_kv,
        "q_per_kv": config.num_attention_heads // n_kv,
        "head_dim": head_dim,
        "kv_bytes_per_token": 2 * n_layers * n_kv * head_dim * 2,
    }

# --- per-generator short-mode system prompt --------------------------------
# Under SYSTEM, Llama-3.2-3B answered 27% of dev-200 with a bare passage number
# ("[1]") and 11% citation-prefixed, flattening its evidence sensitivity
# (em_complete - em_incomplete 0.050 vs Qwen's 0.312). One added clause removes
# both (0.000 / 0.000) and restores it (0.246); dev EM 0.265 -> 0.435.
# Notebook 33 cell 15. Qwen models keep the frozen SYSTEM unchanged.
SYSTEM_NO_CITE = SYSTEM.replace(
    "never write a full sentence.",
    "never write a full sentence, and never answer with a passage number such as [1].")
assert SYSTEM_NO_CITE != SYSTEM

SYSTEM_BY_GENERATOR = {
    "meta-llama/Llama-3.2-3B-Instruct": SYSTEM_NO_CITE,
}


def system_for(generator: str, mode: str) -> str:
    """System prompt for this generator and regime. Explain is shared."""
    if mode == "explain":
        return SYSTEM_EXPLAIN
    return SYSTEM_BY_GENERATOR.get(generator, SYSTEM)


def parse_answer(text: str) -> Tuple[str, bool]:
    """Extract the marked answer span -> (answer, parsed_ok).

    Uses the LAST marker match: the reasoning may mention the word "answer"
    in passing, and the final line is the committed one.

    On failure, falls back to the last non-empty line rather than returning
    empty — a formatting slip should not become an automatic zero the way
    the old yes/no misfire did. `parsed_ok` is False so the fallback rate
    is measurable rather than silent.
    """
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip(), True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[-1] if lines else ""), False


def load_tokenizer(name: str):
    """The tokenizer exactly as load_generator configures it. No weights, CPU-safe.

    pad = eos only when the model ships no pad token (Llama 3.2 ships none, so
    pad becomes <|eot_id|>). Harmless under left padding: the attention mask,
    not the pad id, excludes the pad prefix, and finished rows are filled with
    a special token that batch_decode strips.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

# --- multi-device placement (Qwen2.5-7B arm) -------------------------------
# fp16 Qwen2.5-7B is ~14.2 GiB of weights against ~14.56 GiB usable on one T4,
# so that arm is pipeline-split across both cards. The map is explicit and
# pinned, never "auto": "auto" sizes the split from free memory at load time,
# so it could differ between sessions. Every module owning a parameter or
# buffer is named, because transformers does NOT raise for an unnamed one — it
# falls back to device_map.get("", "cpu") and the module lands on CPU silently.
def split_device_map(name: str, first_layer_on_gpu1: int) -> Dict[str, int]:
    """Two-card split: embed_tokens, rotary_emb and decoder layers
    [0, first_layer_on_gpu1) on GPU 0; the remaining layers, the final norm
    and lm_head on GPU 1. Reads config.json only."""
    from transformers import AutoConfig

    n = AutoConfig.from_pretrained(name).num_hidden_layers
    if not 0 < first_layer_on_gpu1 < n:
        raise ValueError(
            f"first_layer_on_gpu1 must be in (0, {n}), got {first_layer_on_gpu1}")
    dm = {"model.embed_tokens": 0, "model.rotary_emb": 0}
    dm.update({f"model.layers.{i}": int(i >= first_layer_on_gpu1)
               for i in range(n)})
    dm.update({"model.norm": 1, "lm_head": 1})
    return dm


def device_map_for(cfg) -> Optional[Dict[str, int]]:
    """The pinned split a model overlay names under `placement`, else None
    (None = the single-device default every reference arm uses)."""
    pl = cfg.get("placement")
    if not pl:
        return None
    return split_device_map(cfg["models"]["generator"], pl["first_layer_on_gpu1"])


def device_for(tensor_name: str, device_map: Dict[str, int]):
    """Device a parameter/buffer name resolves to, by the longest matching
    module prefix (the resolution transformers applies). None = unmapped."""
    for k in sorted(device_map, key=lambda k: (k.count("."), len(k)), reverse=True):
        if k == "" or tensor_name == k or tensor_name.startswith(k + "."):
            return device_map[k]
    return None


def check_placement(model, device_map: Dict[str, int]) -> Dict[str, float]:
    """Raise unless every parameter and buffer sits where device_map put it.

    -> GiB of parameters + buffers per device, keyed "cuda:<index>".
    """
    gib, bad = {}, []
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        want = device_for(name, device_map)
        got = t.device.index if t.device.type == "cuda" else t.device.type
        if want is None or got != want:
            bad.append(f"{name}: on {t.device}, device_map says {want}")
        dev = f"cuda:{got}" if isinstance(got, int) else str(got)
        gib[dev] = gib.get(dev, 0.0) + t.numel() * t.element_size() / 2**30
    if bad:
        raise RuntimeError(f"{len(bad)} tensors off their mapped device, e.g. "
                           + "; ".join(bad[:5]))
    return {k: round(v, 3) for k, v in sorted(gib.items())}


def load_generator(name: str, dtype=torch.float16, device: int = 0,
                   device_map: Optional[Dict[str, int]] = None,
                   weights_dir: Optional[str] = None):
    """-> (model, tokenizer).

    Default: device_map={"": device} pins every layer to one card. "auto" would
    shard across the T4 pair and make peak-memory numbers meaningless. Every
    reference arm runs this path, and it is unchanged.

    `device_map`: a pinned multi-card map from split_device_map(), for a model
    that cannot fit on one card. Placement is verified after loading. Peak
    memory from such a run is per card and NOT comparable to the single-GPU
    arms (src/metrics/memory.py).

    `weights_dir`: load weights from a local copy (Kaggle Models / Dataset
    mount) instead of the Hub. name_or_path is reset to `name`, because
    results_dir, check_out_path and system_for key on it, and a mount path
    would fail those lookups.
    """
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(name)
    model = AutoModelForCausalLM.from_pretrained(
        weights_dir or name, dtype=dtype,
        device_map=device_map if device_map is not None else {"": device},
    ).eval()
    if weights_dir:
        model.config.name_or_path = name
    if device_map is not None:
        check_placement(model, device_map)
    return model, tok


def build_context(paragraphs: Sequence[str]) -> str:
    """Numbered passage block, given in retrieval-rank order."""
    return "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(paragraphs))


def build_prompt(tok, question: str, paragraphs: Sequence[str],
                 system: Optional[str] = None) -> str:
    """Chat-templated prompt string, ready to tokenize.

    `system` overrides the frozen SYSTEM constant. It exists ONLY for prompt
    A/B testing on a dev sample — every measurement run uses the default.
    """
    user = f"Context:\n{build_context(paragraphs)}\n\nQuestion: {question}"
    return tok.apply_chat_template(
        [{"role": "system", "content": system or SYSTEM},
         {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        **TEMPLATE_KWARGS,
    )


@torch.inference_mode()
def generate_batch(model, tok, prompts: Sequence[str],
                   max_new_tokens: int = 32,
                   kv: Optional[dict] = None,
                   evict: Optional[dict] = None) -> List[str]:
    """Greedy-decode a batch of prompts -> list of answer strings.

    With padding_side="left" every sequence in the batch shares one padded
    input length, so slicing at that length strips the prompt from all rows
    at once. With right padding this slice would be wrong for every row that
    got padded, and the bug is silent: you get plausible-looking garbage.

    `kv` is a knob-3 spec: plain serializable params, e.g. {"bits": 4} or
    {"bits": 2, "backend": "quanto"}. None or {"bits": 16} leaves this
    byte-identical to the Phase 1 path. The mutable cache_config is rebuilt
    HERE, per batch, because generate() pops "backend" out of whatever dict
    it is handed — a reused dict silently reverts to the quanto backend after
    the first batch.

    `evict` is a knob-4 spec, e.g. {"policy": "recency", "keep_ratio": 0.5}.
    None, policy "none", or keep_ratio 1.0 leaves past_key_values=None, which
    _prepare_cache_for_generation treats as "no user cache" (the escape-route
    guard tests `is not None`), so the Phase 1 path is untouched. The Cache is
    rebuilt HERE, per batch, for a stronger reason than the kv dict: a Cache
    carries the previous batch's KEYS, and reuse raises IndexError.

    The two specs are mutually exclusive until a knob3 x knob4 joint is
    explicitly scoped: a QuantizedCache and an EvictLayer cannot both be the
    cache, and combining them needs QuantizedLayer subclassing, not two kwargs.
    """
    from src.generator import kv_cache as KV
    from src.generator import eviction as EV

    spec = dict(kv or {})
    allow_flush = spec.pop("allow_flush", False)
    residual_length = spec.get("residual_length", KV.RESIDUAL_LENGTH)
    if spec.get("bits", KV.FP16_BITS) != KV.FP16_BITS \
            and residual_length <= max_new_tokens and not allow_flush:
        raise ValueError(
            f"residual_length={residual_length} <= max_new_tokens="
            f"{max_new_tokens}: the fp16 buffer would flush mid-generation, "
            f"re-quantizing the whole store from already-lossy values. The "
            f"pre-registered headline setting is residual_length=512. Pass "
            f"allow_flush=True to run this deliberately.")

    cache_kwargs = KV.make_cache(**spec)

    ev = dict(evict or {})
    evicting = (ev.get("policy", "none") != "none"
                and ev.get("keep_ratio", EV.NO_EVICTION) < EV.NO_EVICTION)
    if evicting and cache_kwargs:
        raise ValueError(
            f"kv={kv} and evict={evict} together is a knob3 x knob4 joint, "
            f"which is NOT scoped: a QuantizedCache and an EvictLayer cannot "
            f"both be the cache object. Combining them requires subclassing "
            f"QuantizedLayer. Run one knob at a time.")

    # Built from len(prompts) rather than the padded tensor so the batch guard
    # fires before tokenization — that keeps the guard testable with model=None.
    cache = EV.make_cache(model=model, batch_size=len(prompts),
                          **ev) if evicting else None

    enc = tok(list(prompts), return_tensors="pt", padding=True,
              **ENCODE_KWARGS).to(model.device)
    with EV.attention_scoring(model, cache):
        out = model.generate(
            **enc,
            **cache_kwargs,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    gen = out[:, enc["input_ids"].shape[-1]:]
    return [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]
