"""End-to-end RAG orchestration: embed -> retrieve -> prompt -> generate.
Home of KNOB 5 (top-k). Phase 1.

Precise baseline: exact flat index, fp32 embeddings, fp16 KV cache, top_k=5.

Two generation REGIMES, held fixed within any sweep and swept independently:
  * "short"   — minimal answer span, ~3-5 decode tokens
  * "explain" — reasoning then a marked answer line, ~90 decode tokens.
                Gives KNOB 4 (KV eviction) decode-time context to act on and
                makes decode throughput measurable in Phase 3.

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
    "setting", "mode", "qid", "level", "qtype", "question", "gold_answer",
    "gold_ids", "retrieved_ids", "prediction", "raw_generation", "parsed_ok",
    "prompt_tokens", "decode_tokens", "n_paragraphs",
]

MODES = ("short", "explain")


def _chunks(seq: Sequence[Any], n: int) -> List[Sequence[Any]]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def _completed_qids(path: Optional[str], setting: str) -> set:
    """qids already written for this setting, so a killed session can resume."""
    if not path or not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {r["qid"] for r in csv.DictReader(fh) if r["setting"] == setting}


def _writer(path: str):
    """Append-mode writer; header only for a fresh file.

    A pre-existing file with a different header means the schema changed since
    it was written. Appending would misalign every column, so fail loudly.
    """
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    if not fresh:
        with open(path, newline="") as fh:
            existing = next(csv.reader(fh), [])
        if existing != FIELDS:
            raise RuntimeError(
                f"{path} has an incompatible header (schema changed). "
                f"Write to a new file rather than appending.")
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if fresh:
        w.writeheader()
    return fh, w


def retrieve_all(embedder, examples, top_k: int, progress: bool = False):
    """-> retrieved_ids per question, in descending relevance order.

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


def _mode_settings(mode: str, cfg: Dict[str, Any], max_new_tokens: Optional[int]):
    """-> (system prompt, max_new_tokens) for the requested regime."""
    from src.generator import model as G

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "short":
        return G.SYSTEM, max_new_tokens or cfg["generation"]["max_new_tokens"]
    return (G.SYSTEM_EXPLAIN,
            max_new_tokens or cfg["generation"].get("max_new_tokens_explain", 256))


def run_pipeline(
    examples,
    cfg: Dict[str, Any],
    *,
    embedder=None,
    generator=None,
    gen_tok=None,
    mode: str = "short",
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
    file and still tell them apart. `mode` selects the generation regime.
    """
    from src.retriever import embed as E
    from src.generator import model as G

    top_k = top_k if top_k is not None else cfg["retrieval"]["top_k"]
    batch_size = batch_size or cfg["generation"]["batch_size"]
    system, max_new_tokens = _mode_settings(mode, cfg, max_new_tokens)

    if embedder is None:
        embedder = E.load_embedder(cfg["models"]["embedder"])
    if generator is None or gen_tok is None:
        generator, gen_tok = G.load_generator(cfg["models"]["generator"])

    t0 = time.time()
    retrieved = retrieve_all(embedder, examples, top_k, progress=progress)
    prompts = [
        G.build_prompt(gen_tok, e.question, [e.paragraphs[i] for i in r],
                       system=system)
        for e, r in zip(examples, retrieved)
    ]
    plens = [len(gen_tok.encode(p)) for p in prompts]
    if progress:
        print(f"retrieved + prompted {len(examples)} questions "
              f"in {time.time() - t0:.1f}s  [mode={mode}, "
              f"max_new_tokens={max_new_tokens}]")

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
            raw = G.generate_batch(generator, gen_tok,
                                   [prompts[i] for i in idxs], max_new_tokens)

            # In explain mode the prediction is an EXTRACTION from a longer
            # text. Keep the raw generation so a parse failure stays
            # distinguishable from a wrong answer after the run.
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
                    "gold_ids": " ".join(map(str, e.gold_ids)),
                    "retrieved_ids": " ".join(map(str, retrieved[i])),
                    "prediction": pred,
                    "raw_generation": text if mode == "explain" else "",
                    "parsed_ok": int(ok),
                    "prompt_tokens": plens[i],
                    "decode_tokens": len(gen_tok.encode(text)),
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


def load_results(path: str, setting: Optional[str] = None,
                 mode: Optional[str] = None):
    """Read a results CSV back as a DataFrame, optionally filtered."""
    import pandas as pd

    df = pd.read_csv(path)
    if setting is not None:
        df = df[df["setting"] == setting]
    if mode is not None and "mode" in df.columns:
        df = df[df["mode"] == mode]
    return df
