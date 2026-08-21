"""FAISS indexes. KNOB 1 = search effort, KNOB 2 = embedding precision.

Precise baseline: IndexFlatIP — exhaustive, exact inner-product search over
fp32 vectors. Every vector is compared against the query, so there is no
recall loss to speak of; this is the reference every approximate index in
Phase 2 is measured against.

Phase 1 uses index regime 1: one small index per question (~10 vectors).
Building a FAISS index over 10 vectors is objectively wasteful compared to a
matmul, and we do it anyway so that the Phase 2 code path is identical except
for the index constructor. The knob must be the only thing that varies.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _as_faiss(x: np.ndarray) -> np.ndarray:
    """FAISS requires 2-D, fp32, C-contiguous. Fail loudly rather than segfault."""
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {a.shape}")
    return np.ascontiguousarray(a.astype("float32", copy=False))


def build_flat_index(vectors: np.ndarray):
    """Exact inner-product index over L2-normalised vectors (== cosine)."""
    import faiss

    v = _as_faiss(vectors)
    index = faiss.IndexFlatIP(v.shape[1])
    index.add(v)
    return index


def search(index, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """-> (ids, scores), each (n_queries, min(k, index.ntotal)).

    FAISS returns (distances, labels); we swap to (ids, scores) because ids are
    what the pipeline consumes. k is clamped to ntotal: asking for more
    neighbours than exist makes FAISS pad with -1, which would silently enter
    recall@k as a valid-looking id.
    """
    q = _as_faiss(queries)
    k_eff = min(k, index.ntotal)
    scores, ids = index.search(q, k_eff)
    return ids, scores


def build_hnsw_index(vectors, ef):          # KNOB 1
    raise NotImplementedError("Phase 2")


def build_ivfpq_index(vectors, nprobe, m):  # KNOBS 1 & 2
    raise NotImplementedError("Phase 2")