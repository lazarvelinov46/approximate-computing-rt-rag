"""HotpotQA (distractor) loader for the precise baseline. Phase 1.

Index regime 1: every question carries its own corpus of ~10 paragraphs
(2 gold + 8 distractors). Gold retrieval targets are the paragraph indices
whose titles appear in `supporting_facts` -> these are the ground truth for
recall@k. `answer` is the ground truth for EM / token-F1.

Subset selection is deterministic and library-version independent: we shuffle
the *index list* with random.Random(seed), not datasets.shuffle(), so the same
500 questions come back on any machine and any `datasets` release.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from random import Random
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATASET_NAME = "hotpotqa/hotpot_qa"   # parquet-converted; no trust_remote_code needed
_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", str(text)).strip()


@dataclass
class Example:
    """One HotpotQA question plus its private 10-paragraph corpus."""
    qid: str
    question: str
    answer: str
    level: str          # easy | medium | hard
    qtype: str          # bridge | comparison
    titles: List[str]
    paragraphs: List[str]     # "Title. sent1 sent2 ..."  (index i <-> titles[i])
    gold_ids: List[int]       # indices into paragraphs; ground truth for recall@k

    @property
    def n_paragraphs(self) -> int:
        return len(self.paragraphs)

    @property
    def gold_titles(self) -> List[str]:
        return [self.titles[i] for i in self.gold_ids]


# --- schema normalisation -------------------------------------------------
# `datasets` may hand back a Sequence-of-struct as dict-of-lists (classic) or
# list-of-dicts (newer releases). Accept both so a library bump can't break us.

def _pairs(field: Any, key_a: str, key_b: str) -> List[Tuple[Any, Any]]:
    if isinstance(field, dict):
        return list(zip(field[key_a], field[key_b]))
    return [(row[key_a], row[key_b]) for row in field]


def _to_example(row: Dict[str, Any]) -> Optional[Example]:
    titles, paragraphs = [], []
    for title, sentences in _pairs(row["context"], "title", "sentences"):
        t = _clean(title)
        # HotpotQA sentences already carry their own leading spaces -> "".join
        body = _clean("".join(sentences))
        titles.append(t)
        paragraphs.append(_clean(f"{t}. {body}"))

    sup_titles = {_clean(t) for t, _ in _pairs(row["supporting_facts"], "title", "sent_id")}
    gold_ids = [i for i, t in enumerate(titles) if t in sup_titles]

    # A handful of rows have malformed context / unmatched supporting titles.
    if len(paragraphs) < 2 or not gold_ids:
        return None

    return Example(
        qid=row["id"],
        question=_clean(row["question"]),
        answer=_clean(row["answer"]),
        level=row.get("level", ""),
        qtype=row.get("type", ""),
        titles=titles,
        paragraphs=paragraphs,
        gold_ids=gold_ids,
    )


# --- public API -----------------------------------------------------------

def load_hotpotqa(
    subset_size: Optional[int] = 500,
    seed: int = 42,
    split: str = "validation",
    config: str = "distractor",
    dataset_name: str = DATASET_NAME,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> List[Example]:
    """Return `subset_size` Examples, deterministically sampled."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, config, split=split, cache_dir=cache_dir)

    order = list(range(len(ds)))
    Random(seed).shuffle(order)

    out: List[Example] = []
    skipped = 0
    for i in order:
        ex = _to_example(ds[i])
        if ex is None:
            skipped += 1
            continue
        out.append(ex)
        if subset_size is not None and len(out) >= subset_size:
            break

    if subset_size is not None and len(out) < subset_size:
        raise RuntimeError(f"only {len(out)}/{subset_size} usable examples in {split}")
    if verbose:
        print(f"loaded {len(out)} examples from {dataset_name}/{config}:{split} "
              f"(seed {seed}, {skipped} malformed rows skipped)")
    return out


def load_from_config(cfg: Dict[str, Any], **overrides) -> List[Example]:
    """Wire configs/default.yaml -> load_hotpotqa."""
    d = cfg["dataset"]
    kwargs = dict(
        dataset_name=d.get("name", DATASET_NAME),
        config=d.get("config", "distractor"),
        split=d.get("split", "validation"),
        subset_size=d.get("subset_size", 500),
        seed=cfg.get("seed", 42),
    )
    kwargs.update(overrides)
    return load_hotpotqa(**kwargs)


def describe(examples: Sequence[Example]) -> Dict[str, Any]:
    """Sanity stats. Printed in 01_baseline before anything expensive runs."""
    import statistics as st

    n_par = [e.n_paragraphs for e in examples]
    n_gold = [len(e.gold_ids) for e in examples]
    chars = [len(p) for e in examples for p in e.paragraphs]
    yes_no = sum(e.answer.lower() in {"yes", "no"} for e in examples)

    stats = {
        "n_examples": len(examples),
        "paragraphs_per_q_min": min(n_par),
        "paragraphs_per_q_max": max(n_par),
        "gold_per_q_counts": {k: n_gold.count(k) for k in sorted(set(n_gold))},
        "paragraph_chars_mean": round(st.mean(chars), 1),
        "paragraph_chars_p99": sorted(chars)[int(0.99 * len(chars)) - 1],
        "yes_no_answers": yes_no,
        "levels": {k: sum(e.level == k for e in examples) for k in sorted({e.level for e in examples})},
        "types": {k: sum(e.qtype == k for e in examples) for k in sorted({e.qtype for e in examples})},
    }
    for k, v in stats.items():
        print(f"{k:24s}: {v}")
    return stats


if __name__ == "__main__":
    exs = load_hotpotqa()
    describe(exs)
    e = exs[0]
    print(f"\nqid={e.qid}\nQ: {e.question}\nA: {e.answer}\ngold_ids={e.gold_ids} "
          f"titles={e.gold_titles}\n\n{e.paragraphs[e.gold_ids[0]][:300]}...")