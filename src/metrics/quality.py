"""Quality metrics — hardware-independent, safe on shared free GPUs. Phase 1.

Answer metrics follow the official HotpotQA evaluation script (itself
inherited from SQuAD): normalise both sides identically, then compare.
The normaliser IS the metric — any deviation makes these numbers
incomparable to published HotpotQA results.

Retrieval metric is paragraph-level recall@k against the supporting-fact
paragraphs. Scores knobs 1, 2 and 5.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    """Official HotpotQA/SQuAD normalisation: lowercase, drop punctuation,
    drop articles, collapse whitespace. Order matters — articles are removed
    after punctuation so "the," becomes "" rather than "the,"."""
    s = str(s).lower()
    s = s.translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return _WS.sub(" ", s).strip()


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred: str, gold: str) -> float:
    """Token-level F1 over multisets.

    Counter intersection rather than set intersection: a prediction that
    repeats a token earns credit only as many times as gold contains it.

    Yes/no and other single-token golds degenerate to 0 or 1, so they carry
    no partial credit — worth remembering when reading aggregate F1.
    """
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()

    # Empty-string edge case: two empties agree, one empty does not.
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)

    common = sum((Counter(p_toks) & Counter(g_toks)).values())
    if common == 0:
        return 0.0
    precision = common / len(p_toks)
    recall = common / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved_ids: Sequence[int], gold_ids: Sequence[int],
                k: int) -> float:
    """Fraction of gold paragraphs appearing in the top-k retrieved ids."""
    gold = set(gold_ids)
    if not gold:
        raise ValueError("recall@k is undefined with no gold ids")
    return len(gold & set(retrieved_ids[:k])) / len(gold)


def _parse_ids(v: Any) -> List[int]:
    """CSV round-trips id lists as space-separated strings."""
    if isinstance(v, str):
        return [int(x) for x in v.split()] if v.strip() else []
    return [int(x) for x in v]


def score_rows(rows: Sequence[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    """Attach em / f1 / recall / retrieval-completeness to each row."""
    out = []
    for r in rows:
        gold_ids = _parse_ids(r["gold_ids"])
        ret_ids = _parse_ids(r["retrieved_ids"])
        rec = recall_at_k(ret_ids, gold_ids, k)
        out.append({
            **r,
            "em": exact_match(r["prediction"], r["gold_answer"]),
            "f1": f1(r["prediction"], r["gold_answer"]),
            "recall": rec,
            "complete": float(rec == 1.0),   # all gold evidence was supplied
        })
    return out


def aggregate(rows: Sequence[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    """Headline numbers plus the breakdowns that make them interpretable."""
    import statistics as st

    scored = score_rows(rows, k)

    def mean(xs, key):
        xs = [x[key] for x in xs]
        return round(st.mean(xs), 4) if xs else float("nan")

    agg: Dict[str, Any] = {
        "n": len(scored),
        "em": mean(scored, "em"),
        "f1": mean(scored, "f1"),
        f"recall@{k}": mean(scored, "recall"),
        "complete_frac": mean(scored, "complete"),
    }

    # EM conditioned on whether the model actually had the evidence. The gap
    # separates generator failure from retriever failure.
    for label, subset in (("complete", [r for r in scored if r["complete"] == 1.0]),
                          ("incomplete", [r for r in scored if r["complete"] < 1.0])):
        agg[f"em_{label}"] = mean(subset, "em")
        agg[f"n_{label}"] = len(subset)

    for qtype in sorted({r["qtype"] for r in scored if r.get("qtype")}):
        sub = [r for r in scored if r["qtype"] == qtype]
        agg[f"em_{qtype}"] = mean(sub, "em")
        agg[f"n_{qtype}"] = len(sub)

    yn = [r for r in scored if normalize_answer(r["gold_answer"]) in {"yes", "no"}]
    agg["em_yesno"] = mean(yn, "em")
    agg["n_yesno"] = len(yn)
    return agg