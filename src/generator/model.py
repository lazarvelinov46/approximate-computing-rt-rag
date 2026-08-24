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

from typing import List, Sequence, Tuple, Optional

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
# Long decode so KV-cache eviction (knob 4) has accumulated context to act on.
# The final line is machine-parsed, so EM/F1 stay identical to regime A.
SYSTEM_EXPLAIN = (
    "You answer questions using only the numbered context passages.\n"
    "First explain your reasoning in two or three sentences, naming the "
    "passages you used by their number.\n"
    "Then write a final line in exactly this form: Answer: <answer>\n"
    "The answer itself must be a minimal span copied from the passages — "
    "never a full sentence.\n"
    "\n"
    "Example question: Which city is home to the Eiffel Tower?\n"
    "Example response:\n"
    "Passage [2] describes the Eiffel Tower as a wrought-iron tower on the "
    "Champ de Mars. Passage [2] places the Champ de Mars in Paris, so the "
    "tower stands in Paris.\n"
    "Answer: Paris"
)

_ANSWER_RE = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)


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


def load_generator(name: str, dtype=torch.float16, device: int = 0):
    """-> (model, tokenizer), pinned to a single GPU.

    device_map={"": device} pins every layer to one card. "auto" would shard
    across the T4 pair and make peak-memory numbers meaningless.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, device_map={"": device},
    ).eval()
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
    )


@torch.inference_mode()
def generate_batch(model, tok, prompts: Sequence[str],
                   max_new_tokens: int = 32) -> List[str]:
    """Greedy-decode a batch of prompts -> list of answer strings.

    With padding_side="left" every sequence in the batch shares one padded
    input length, so slicing at that length strips the prompt from all rows
    at once. With right padding this slice would be wrong for every row that
    got padded, and the bug is silent: you get plausible-looking garbage.
    """
    enc = tok(list(prompts), return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[:, enc["input_ids"].shape[-1]:]
    return [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]