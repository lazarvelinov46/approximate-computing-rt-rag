"""End-to-end RAG over the POOLED corpus (index regime 2). Phase 2.

Deliberately separate from src/pipeline/rag.py rather than a branch inside it.
The Phase-1 functions produced the frozen baselines and are not modified here:
a bug in shared branching logic would silently invalidate results that cost
37 GPU-minutes to regenerate and anchor the whole study. The duplication below
is the price of that guarantee.

Differences from regime 1, all of them load-bearing:
  * ONE shared index over 66,581 passages, not one index per question
  * retrieved ids are GLOBAL corpus positions, not positions in e.paragraphs
  * gold ids come from corpus.resolve_gold(), i.e. title lookup
  * prompts are built from corpus.texts[i]
  * results go to their OWN csv files (FIELDS_POOLED differs from FIELDS, and
    mixing the two id spaces in one column would be silent corruption)

Invariants carried over unchanged from regime 1:
  * examples processed in loader order; batches are fixed chunks of that order
  * embeddings recomputed every run — no cache to go stale when knob 2 lands
  * greedy decode, fp16, padding_side="left"
"""
from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from src.pipeline.rag import _chunks, _mode_settings

FIELDS_POOLED = [
    "setting", "mode", "qid", "level", "qtype", "question", "gold_answer",
    "gold_ids", "retrieved_ids", "prediction", "raw_generation", "parsed_ok",
    "prompt_tokens", "decode_tokens", "n_corpus", "corpus_sha",
]


def _completed_qids(path: Optional[str], setting: str) -> set:
    if not path or not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {r["qid"] for r in csv.DictReader(fh) if r["setting"] == setting}


def _writer(path: str):
    """Append-mode writer for FIELDS_POOLED. Mirrors rag._writer's header check.

    A header mismatch means either the schema changed or this is a regime-1
    file. Both are reasons to fail rather than append.
    """
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    if not fresh:
        with open(path, newline="") as fh:
            existing = next(csv.reader(fh), [])
        if existing != FIELDS_POOLED:
            raise RuntimeError(
                f"{path} has an incompatible header — regime-1 file or schema "
                f"change. Write to a new file rather than appending.")
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS_POOLED)
    if fresh:
        w.writeheader()
    return fh, w


def embed_corpus(embedder, corpus, batch_size: int = 64, progress: bool = True):
    """(N, dim) fp32 passage matrix over the pooled corpus, in corpus order.

    Row i corresponds to corpus.texts[i], so FAISS ids ARE corpus ids. That
    identity is what makes gold resolution by title valid; do not sort,
    filter, or reorder the matrix anywhere downstream.
    """
    from src.retriever import embed as E

    t0 = time.time()
    vecs = E.embed_passages(embedder, corpus.texts, batch_size=batch_size,
                            progress=progress)
    if progress:
        print(f"embedded {len(corpus)} passages in {time.time() - t0:.1f}s "
              f"-> {vecs.shape} {vecs.dtype} ({vecs.nbytes / 2**20:.0f} MiB)")
    assert vecs.shape[0] == len(corpus), "vector count != corpus size"
    return vecs


def retrieve_all_pooled(embedder, examples, index, top_k: int,
                        progress: bool = False) -> List[List[int]]:
    """-> global corpus ids per question, descending relevance.

    `index` is supplied by the caller, not built here: KNOB 1 swaps the index
    constructor (flat -> HNSW/IVF) while everything else stays fixed, and
    rebuilding inside this function would re-embed the corpus per setting.
    """
    from src.retriever import embed as E, index as I

    qvecs = E.embed_queries(embedder, [e.question for e in examples])
    t0 = time.time()
    ids, _ = I.search(index, qvecs, k=top_k)
    if progress:
        print(f"searched {len(examples)} queries over {index.ntotal} passages "
              f"in {time.time() - t0:.2f}s")
    return [[int(i) for i in row] for row in ids]


def run_pipeline_pooled(
    examples,
    cfg: Dict[str, Any],
    corpus,
    index,
    *,
    embedder=None,
    generator=None,
    gen_tok=None,
    gold_global: Optional[Sequence[Sequence[int]]] = None,
    mode: str = "short",
    top_k: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    setting: str = "regime2_baseline",
    out_csv: Optional[str] = None,
    resume: bool = True,
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """Regime-2 counterpart of rag.run_pipeline. Returns rows written THIS call."""
    from src.data import corpus as C
    from src.retriever import embed as E
    from src.generator import model as G

    top_k = top_k if top_k is not None else cfg["retrieval"]["top_k"]
    batch_size = batch_size or cfg["generation"]["batch_size"]
    system, max_new_tokens = _mode_settings(mode, cfg, max_new_tokens)
    sha = corpus.manifest["sha256"]

    if top_k != cfg["retrieval"]["top_k"] and setting in (
        "regime2_a1_baseline", "regime2_a1_baseline_explain"):
    raise ValueError(
        f"setting={setting!r} is a frozen regime-2 tag but top_k={top_k} "
        f"differs from the baseline {cfg['retrieval']['top_k']}. Tag knob-5 "
        f"runs with their own setting name (e.g. 'r2_topk_{top_k:02d}').")

    if embedder is None:
        embedder = E.load_embedder(cfg["models"]["embedder"])
    if generator is None or gen_tok is None:
        generator, gen_tok = G.load_generator(cfg["models"]["generator"])
    if gold_global is None:
        gold_global = C.resolve_gold(corpus, examples)

    t0 = time.time()
    retrieved = retrieve_all_pooled(embedder, examples, index, top_k,
                                    progress=progress)
    prompts = [
        G.build_prompt(gen_tok, e.question, [corpus.texts[i] for i in r],
                       system=system)
        for e, r in zip(examples, retrieved)
    ]
    plens = [len(gen_tok.encode(p)) for p in prompts]
    if progress:
        print(f"retrieved + prompted {len(examples)} questions in "
              f"{time.time() - t0:.1f}s  [mode={mode}, regime 2, "
              f"n_corpus={len(corpus)}, max_new_tokens={max_new_tokens}]")

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
            raw = G.generate_batch(generator, gen_tok,
                                   [prompts[i] for i in idxs], max_new_tokens)
            if mode == "explain":
                parsed = [G.parse_answer(t) for t in raw]
            else:
                parsed = [(t, True) for t in raw]

            for i, text, (pred, ok) in zip(idxs, raw, parsed):
                e = examples[i]
                row = {
                    "setting": setting,
                    "mode": mode,
                    "qid": e.qid,
                    "level": e.level,
                    "qtype": e.qtype,
                    "question": e.question,
                    "gold_answer": e.answer,
                    "gold_ids": " ".join(map(str, gold_global[i])),
                    "retrieved_ids": " ".join(map(str, retrieved[i])),
                    "prediction": pred,
                    "raw_generation": text if mode == "explain" else "",
                    "parsed_ok": int(ok),
                    "prompt_tokens": plens[i],
                    "decode_tokens": len(gen_tok.encode(text)),
                    "n_corpus": len(corpus),
                    "corpus_sha": sha,
                }
                rows.append(row)
                if w:
                    w.writerow(row)
            if fh:
                fh.flush()
            if progress and (bi + 1) % 5 == 0:
                print(f"  batch {bi + 1}/{len(batches)}  "
                      f"({time.time() - t0:.0f}s elapsed)")
    finally:
        if fh:
            fh.close()

    if progress:
        print(f"generated {len(rows)} answers in {time.time() - t0:.1f}s")
    return rows


def ann_recall(approx_csv: str, exact_csv: str, approx_setting: str,
               exact_setting: str, k: Optional[int] = None) -> float:
    """Mean |approx top-k ∩ exact top-k| / k, joined on qid.

    This is NOT recall@k. recall@k measures agreement with GOLD; ann_recall
    measures agreement with EXACT SEARCH. An index can have perfect ann_recall
    and poor recall@k (the retriever is faithful but the embedder is weak), or
    the reverse. Conflating them would misattribute embedder error to the knob.
    """
    import pandas as pd

    a = pd.read_csv(approx_csv, keep_default_na=False, dtype=str)
    x = pd.read_csv(exact_csv, keep_default_na=False, dtype=str)
    a = a[a["setting"] == approx_setting].set_index("qid")["retrieved_ids"]
    x = x[x["setting"] == exact_setting].set_index("qid")["retrieved_ids"]
    common = a.index.intersection(x.index)
    if len(common) == 0:
        raise ValueError("no shared qids between the two settings")

    scores = []
    for q in common:
        ai = a.loc[q].split()
        xi = x.loc[q].split()
        kk = k or len(xi)
        scores.append(len(set(ai[:kk]) & set(xi[:kk])) / kk)
    return sum(scores) / len(scores)


def load_results_pooled(path: str, setting: Optional[str] = None,
                        mode: Optional[str] = None):
    """rag.load_results for the pooled schema (numeric coercion on n_corpus)."""
    import pandas as pd

    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    for col in ("prompt_tokens", "decode_tokens", "parsed_ok", "n_corpus"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if setting is not None:
        df = df[df["setting"] == setting]
    if mode is not None and "mode" in df.columns:
        df = df[df["mode"] == mode]
    return df
