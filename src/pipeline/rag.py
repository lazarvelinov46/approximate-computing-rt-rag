"""End-to-end RAG orchestration: embed -> retrieve -> prompt -> generate.
Home of KNOB 5 (top-k). Phase 1.

Precise baseline: exact flat index, fp32 embeddings, fp16 KV cache, top_k=5.

Experimental-setup invariants:
  * examples are processed in loader order — never sorted, never shuffled
  * batches are fixed chunks of that order, so batch composition is identical
    across every knob setting (batched greedy decode is not bitwise
    reproducible across different batch compositions, so a change in
    composition would enter the results as if it were a knob effect)
  * embeddings are recomputed on every run; there is no cache to go stale
    when knob 2 changes the vector representation
"""
from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, List, Optional, Sequence

FIELDS = [
    "setting", "qid", "level", "qtype", "question", "gold_answer",
    "gold_ids", "retrieved_ids", "prediction", "prompt_tokens", "n_paragraphs",
]


def _chunks(seq: Sequence[Any], n: int) -> List[Sequence[Any]]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def _completed_qids(path: Optional[str], setting: str) -> set:
    """qids already written for this setting, so a killed session can resume."""
    if not path or not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {r["qid"] for r in csv.DictReader(fh) if r["setting"] == setting}


def _writer(path: str):
    """Append-mode writer; emits the header only for a fresh file."""
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if fresh:
        w.writeheader()
    return fh, w


def retrieve_all(embedder, examples, top_k: int, progress: bool = False):
    """-> (retrieved_ids per question, in descending relevance order).

    Regime 1: one exact index per question over its own ~10 paragraphs.
    """
    from src.retriever import embed as E, index as I

    corpus_vecs = E.embed_grouped(embedder, [e.paragraphs for e in examples],
                                  progress=progress)
    qvecs = E.embed_queries(embedder, [e.question for e in examples])

    out = []
    for cv, qv in zip(corpus_vecs, qvecs):
        idx = I.build_flat_index(cv)
        ids, _ = I.search(idx, qv[None, :], k=top_k)
        out.append([int(i) for i in ids[0]])
    return out


def run_pipeline(
    examples,
    cfg: Dict[str, Any],
    *,
    embedder=None,
    generator=None,
    gen_tok=None,
    top_k: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    setting: str = "baseline",
    out_csv: Optional[str] = None,
    resume: bool = True,
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """Run the full pipeline over `examples`. Returns rows produced THIS call.

    When resuming, previously-written rows are not returned — the CSV is the
    source of truth for a completed run.

    `setting` tags every row so Phase 2 can append many knob settings to one
    file and still tell them apart.
    """
    from src.retriever import embed as E
    from src.generator import model as G

    top_k = top_k if top_k is not None else cfg["retrieval"]["top_k"]
    batch_size = batch_size or cfg["generation"]["batch_size"]
    max_new_tokens = max_new_tokens or cfg["generation"]["max_new_tokens"]

    if embedder is None:
        embedder = E.load_embedder(cfg["models"]["embedder"])
    if generator is None or gen_tok is None:
        generator, gen_tok = G.load_generator(cfg["models"]["generator"])

    t0 = time.time()
    retrieved = retrieve_all(embedder, examples, top_k, progress=progress)
    prompts = [
        G.build_prompt(gen_tok, e.question, [e.paragraphs[i] for i in r])
        for e, r in zip(examples, retrieved)
    ]
    plens = [len(gen_tok.encode(p)) for p in prompts]
    if progress:
        print(f"retrieved + prompted {len(examples)} questions "
              f"in {time.time() - t0:.1f}s")

    # Chunk the FULL order first, then skip complete batches. Filtering
    # examples before chunking would re-partition the survivors and change
    # batch composition, which is not reproducible across runs.
    batches = _chunks(list(range(len(examples))), batch_size)
    done = _completed_qids(out_csv, setting) if resume else set()

    fh = w = None
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        fh, w = _writer(out_csv)

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    try:
        for bi, idxs in enumerate(batches):
            if done and all(examples[i].qid in done for i in idxs):
                continue
            preds = G.generate_batch(generator, gen_tok,
                                     [prompts[i] for i in idxs], max_new_tokens)
            for i, pred in zip(idxs, preds):
                e = examples[i]
                row = {
                    "setting": setting,
                    "qid": e.qid,
                    "level": e.level,
                    "qtype": e.qtype,
                    "question": e.question,
                    "gold_answer": e.answer,
                    "gold_ids": " ".join(map(str, e.gold_ids)),
                    "retrieved_ids": " ".join(map(str, retrieved[i])),
                    "prediction": pred,
                    "prompt_tokens": plens[i],
                    "n_paragraphs": e.n_paragraphs,
                }
                rows.append(row)
                if w:
                    w.writerow(row)
            if fh:
                fh.flush()   # survive a hard session kill, not just an exception
            if progress and (bi + 1) % 5 == 0:
                print(f"  batch {bi + 1}/{len(batches)}  "
                      f"({time.time() - t0:.0f}s elapsed)")
    finally:
        if fh:
            fh.close()

    if progress:
        print(f"generated {len(rows)} answers in {time.time() - t0:.1f}s")
    return rows


def load_results(path: str, setting: Optional[str] = None):
    """Read a results CSV back as a DataFrame."""
    import pandas as pd

    df = pd.read_csv(path)
    return df[df["setting"] == setting] if setting else df