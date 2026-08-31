"""Pooled corpus construction for INDEX REGIME 2. Phase 2.

Regime 1 gives each question its own ~10-paragraph corpus, which is degenerate
for approximate search: an ANN index over 10 vectors returns the same 10 as an
exact one. Regime 2 pools paragraphs into one shared index so that knobs 1
(search effort) and 2 (embedding precision) have somewhere to act.

Corpus tiers (measured in notebooks/04_ac_corpus_audit):
  A1 = validation split          ->  66,581 passages
  A2 = validation + train        -> 507,494 passages  (7.62x A1)

Experimental-setup invariants — changing any of these invalidates prior runs:
  * built from the RAW split, never through load_hotpotqa(): that function
    applies min_paragraphs filtering and seed-42 sampling, which define the
    frozen 1000 QUESTIONS. Corpus and question set are independent choices,
    and routing one through the other would risk disturbing the frozen subset.
  * dedup key is the normalised TITLE. Verified vacuous on A1 (66,581 titles,
    66,581 distinct texts, zero titles with multiple texts) but retained
    because A2 may not have that property.
  * splits are supplied in PRIORITY order and the first writer wins, so
    build_corpus(("validation", "train")) keeps validation text for the
    41,108 titles the two splits share. This is what stops A2's gold context
    from drifting away from regime 1's.
  * corpus order is SORTED BY TITLE, not dataset iteration order. HNSW graph
    construction depends on insertion order, so pinning the order is what
    makes an index rebuild reproducible.
  * text normalisation is imported from the loader, not reimplemented.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Sequence, Tuple

# Private imports are deliberate. The pooled paragraph text MUST be byte
# identical to what _to_example() produced in Phase 1 — a divergent copy of
# the normaliser is the single most damaging silent failure available here.
# If these are ever renamed, this import should break loudly.
from src.data.loader import DATASET_NAME, _clean, _pairs

BUILDER_VERSION = 1


@dataclass
class Corpus:
    """An ordered, deduplicated passage collection plus its provenance."""
    titles: List[str]              # index i <-> texts[i]
    texts: List[str]               # "Title. sent1 sent2 ..." — prompt-ready
    title_to_id: Dict[str, int]
    manifest: Dict[str, Any]

    def __len__(self) -> int:
        return len(self.texts)


def _iter_pairs(split: str) -> Iterator[Tuple[str, str]]:
    """(title, paragraph) for every context in every row of a raw split.

    Mirrors _to_example()'s construction exactly, including the double _clean
    (once on the joined sentences, once on the title-prefixed result).
    """
    from datasets import load_dataset

    raw = load_dataset(DATASET_NAME, "distractor", split=split)
    for row in raw:
        for title, sentences in _pairs(row["context"], "title", "sentences"):
            t = _clean(title)
            body = _clean("".join(sentences))
            yield t, _clean(f"{t}. {body}")


def corpus_hash(titles: Sequence[str], texts: Sequence[str]) -> str:
    """SHA-256 over the ORDERED (position, title, text) triples.

    Position is included so that a permutation which preserves contents still
    changes the hash — order is part of the artifact, because the ANN graph
    depends on it.
    """
    h = hashlib.sha256()
    for i, (t, x) in enumerate(zip(titles, texts)):
        h.update(f"{i}\t{t}\t{x}\n".encode("utf-8"))
    return h.hexdigest()


def build_corpus(splits: Sequence[str] = ("validation",),
                 verbose: bool = True) -> Corpus:
    """Pool + dedup paragraphs across `splits`, earlier splits winning ties."""
    seen: Dict[str, str] = {}
    n_raw: Dict[str, int] = {}
    collisions: List[Tuple[str, str]] = []   # (split, title) with differing text

    for split in splits:
        count = 0
        for title, text in _iter_pairs(split):
            count += 1
            if title not in seen:
                seen[title] = text
            elif seen[title] != text:
                collisions.append((split, title))
        n_raw[split] = count
        if verbose:
            print(f"  {split:<12} {count:>7} raw pairs -> {len(seen):>7} unique titles")

    titles = sorted(seen)                     # deterministic, content-derived
    texts = [seen[t] for t in titles]

    manifest = {
        "builder_version": BUILDER_VERSION,
        "dataset": DATASET_NAME,
        "config": "distractor",
        "splits": list(splits),
        "dedup_key": "normalised_title",
        "tie_break": "first split in `splits` wins",
        "order": "sorted by normalised title",
        "n_raw_pairs": n_raw,
        "n_passages": len(texts),
        "n_title_collisions": len(collisions),
        "collision_examples": [t for _, t in collisions[:10]],
        "sha256": corpus_hash(titles, texts),
    }

    if verbose:
        print(f"\ncorpus: {len(texts)} passages   sha256 {manifest['sha256'][:16]}...")
        if collisions:
            print(f"WARNING: {len(collisions)} title collisions with differing "
                  f"text — tie-break applied, see manifest")

    return Corpus(titles=titles, texts=texts,
                  title_to_id={t: i for i, t in enumerate(titles)},
                  manifest=manifest)


def resolve_gold(corpus: Corpus, examples) -> List[List[int]]:
    """Regime-2 gold ids: Example.gold_titles -> global corpus ids.

    In regime 1 gold ids are POSITIONS in a per-question paragraph list; here
    they are positions in the shared corpus. Same metric code (recall_at_k is
    a set intersection), completely different id space — which is why
    regime-2 results must go to their own CSV files.
    """
    out = []
    for e in examples:
        ids = []
        for t in e.gold_titles:
            if t not in corpus.title_to_id:
                raise KeyError(f"gold title absent from corpus: {t!r} (qid={e.qid})")
            ids.append(corpus.title_to_id[t])
        out.append(sorted(ids))
    return out


def audit_against(corpus: Corpus, examples) -> Dict[str, int]:
    """Confirm the corpus can stand in for regime 1's per-question paragraphs.

    `text_drift` must be 0: the exact string Phase 1 fed to the generator has
    to be present under its own title. Non-zero means normalisation diverged
    and nothing downstream of this corpus is comparable to the baselines.
    """
    missing = sum(t not in corpus.title_to_id for e in examples for t in e.gold_titles)
    drift = sum(corpus.texts[corpus.title_to_id[e.titles[gi]]] != e.paragraphs[gi]
                for e in examples for gi in e.gold_ids
                if e.titles[gi] in corpus.title_to_id)
    return {
        "n_examples": len(examples),
        "gold_slots": sum(len(e.gold_titles) for e in examples),
        "gold_titles_missing": missing,
        "gold_text_drift": drift,
    }


def save_manifest(corpus: Corpus, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(corpus.manifest, fh, indent=2)


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)
