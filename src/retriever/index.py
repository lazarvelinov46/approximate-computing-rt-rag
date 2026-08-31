"""FAISS indexes. KNOB 1 = search effort, KNOB 2 = embedding precision.

Precise baseline: IndexFlatIP — exhaustive, exact inner-product search over
fp32 vectors. Every vector is compared against the query, so there is no
recall loss to speak of; this is the reference every approximate index in
Phase 2 is measured against.

Phase 1 uses index regime 1: one small index per question (~10 vectors).
Building a FAISS index over 10 vectors is objectively wasteful compared to a
matmul, and we do it anyway so that the Phase 2 code path is identical except
for the index constructor. The knob must be the only thing that varies.

--- Phase 2: keeping the two knobs unconfounded -------------------------------

FAISS lets you fuse skipping and precision loss into one structure (IVF-PQ).
That is convenient and, for a characterisation study, wrong: a curve produced
by varying `m` inside an IVF index is a curve of BOTH knobs at once. So the
constructors below separate them by error source:

  build_hnsw_index    KNOB 1 only  — full fp32 vectors, error = comparisons skipped
  build_ivfflat_index KNOB 1 only  — full fp32 vectors, error = cells not probed
  build_pq_index      KNOB 2 only  — exhaustive scan, error = quantisation only
  build_ivfpq_index   PHASE 4      — the composition of both; not a single-knob index

--- Build-time vs query-time --------------------------------------------------

`efSearch` and `nprobe` are QUERY-TIME fields: one build, then a dense sweep
over settings at ~0.3 s per 1000 queries. `M`, `efConstruction` and `nlist`
are BUILD-TIME; sweeping them is a different experiment (build cost vs
quality) and would confound the search-effort curve. They are frozen
constants, recorded in the index manifest.

--- Determinism ---------------------------------------------------------------

Unlike IndexFlatIP, these are not deterministic by construction:
  * IVF trains k-means, which is seeded and stochastic.
  * HNSW assigns node levels by RNG and inserts in parallel by default.
Both are built single-threaded here, and both consume vectors in corpus order
(pinned by the title sort in src/data/corpus.py). Determinism is ASSERTED
empirically in the sweep notebook rather than assumed from the seed argument.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# Build-time constants. Changing any of these invalidates every knob-1 result
# produced against the previous value.
HNSW_M = 32                 # links per node; 16-64 is the usual band
HNSW_EF_CONSTRUCTION = 200  # candidate width during insertion
IVF_NLIST = 1024            # ~4*sqrt(66581)=1032; 65 training pts/centroid
INDEX_SEED = 42


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


# ---------------------------------------------------------------- KNOB 1 -----

def build_hnsw_index(vectors: np.ndarray,
                     M: int = HNSW_M,
                     ef_construction: int = HNSW_EF_CONSTRUCTION,
                     single_thread: bool = True):
    """Navigable small-world graph over FULL fp32 vectors. Knob 1 only.

    Vectors are stored uncompressed, so the only error source is graph
    traversal that stops early — i.e. pure computation skipping. Set the
    query-time effort afterwards with set_ef().

    METRIC_INNER_PRODUCT matches the flat baseline exactly. On L2-normalised
    vectors L2 and IP induce identical rankings, but relying on that silently
    would be one more thing to get wrong.
    """
    import faiss

    v = _as_faiss(vectors)
    if single_thread:
        faiss.omp_set_num_threads(1)   # parallel insertion perturbs the graph
    index = faiss.IndexHNSWFlat(v.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(v)                        # insertion order == corpus order
    return index


def build_ivfflat_index(vectors: np.ndarray,
                        nlist: int = IVF_NLIST,
                        seed: int = INDEX_SEED,
                        single_thread: bool = True):
    """Coarse-quantised inverted file over FULL fp32 vectors. Knob 1 only.

    k-means partitions the space into `nlist` cells; search probes the nearest
    `nprobe` of them exhaustively. Vectors inside a cell are uncompressed, so
    again the only error is skipping. Set effort with set_nprobe().

    FAISS warns below ~39 training points per centroid. At N=66,581 and
    nlist=1024 that ratio is 65.
    """
    import faiss

    v = _as_faiss(vectors)
    if v.shape[0] < 39 * nlist:
        raise ValueError(
            f"nlist={nlist} needs ~{39 * nlist} training vectors, have {v.shape[0]}")
    if single_thread:
        faiss.omp_set_num_threads(1)

    quantizer = faiss.IndexFlatIP(v.shape[1])
    index = faiss.IndexIVFFlat(quantizer, v.shape[1], nlist,
                               faiss.METRIC_INNER_PRODUCT)
    index.cp.seed = seed                # k-means init is stochastic
    index.train(v)
    index.add(v)
    return index


def set_ef(index, ef: int):
    """Query-time HNSW effort. No rebuild — this is what makes the sweep free."""
    index.hnsw.efSearch = int(ef)
    return index


def set_nprobe(index, nprobe: int):
    """Query-time IVF effort. Clamped to nlist; above that it is exhaustive."""
    index.nprobe = int(min(nprobe, index.nlist))
    return index


# ---------------------------------------------------------------- KNOB 2 -----

def build_pq_index(vectors: np.ndarray, m: int, nbits: int = 8,
                   seed: int = INDEX_SEED, single_thread: bool = True):
    """Product-quantised index with EXHAUSTIVE scan. Knob 2 only.

    Every vector is still compared against the query — nothing is skipped —
    but each is stored as `m` sub-codes of `nbits` instead of `d` floats.
    The only error source is quantisation, which is what isolates knob 2.

    Compression: d*4 bytes -> m*nbits/8 bytes per vector. At d=384, m=48,
    nbits=8 that is 1536 -> 48 bytes, 32x.
    """
    import faiss

    v = _as_faiss(vectors)
    d = v.shape[1]
    if d % m:
        raise ValueError(f"m={m} must divide the embedding dim {d}")
    if single_thread:
        faiss.omp_set_num_threads(1)

    index = faiss.IndexPQ(d, m, nbits, faiss.METRIC_INNER_PRODUCT)
    index.cp.seed = seed
    index.train(v)
    index.add(v)
    return index


# --------------------------------------------------------------- PHASE 4 -----

def build_ivfpq_index(vectors: np.ndarray, m: int, nbits: int = 8,
                      nlist: int = IVF_NLIST, seed: int = INDEX_SEED,
                      single_thread: bool = True):
    """Skipping AND precision loss in one structure. NOT a single-knob index.

    Reserved for the Phase 4 joint sweep, where the question is precisely
    whether the two error sources compose as the single-knob curves predict.
    Using this for either knob alone would confound them.
    """
    import faiss

    v = _as_faiss(vectors)
    d = v.shape[1]
    if d % m:
        raise ValueError(f"m={m} must divide the embedding dim {d}")
    if single_thread:
        faiss.omp_set_num_threads(1)

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits,
                             faiss.METRIC_INNER_PRODUCT)
    index.cp.seed = seed
    index.train(v)
    index.add(v)
    return index


# ------------------------------------------------------------ work metric ----

def reset_search_stats() -> None:
    """Zero FAISS's global distance-computation counters before a timed search."""
    import faiss

    for name in ("hnsw_stats", "indexIVF_stats"):
        s = getattr(faiss.cvar, name, None)
        if s is not None:
            s.reset()


def read_search_stats(n_queries: int) -> Dict[str, Any]:
    """Distance computations per query — a HARDWARE-INDEPENDENT effort metric.

    Phase 2 measures quality only; wall-clock belongs to Phase 3 on the 4090.
    But `ndis` is a machine-independent count of work done, so it can go on the
    quality-phase x-axis without importing any timing assumptions.
    """
    import faiss

    out: Dict[str, Any] = {}
    for name, label in (("hnsw_stats", "hnsw"), ("indexIVF_stats", "ivf")):
        s = getattr(faiss.cvar, name, None)
        if s is None:
            continue
        ndis = getattr(s, "ndis", 0)
        if ndis:
            out[f"{label}_ndis_per_query"] = ndis / n_queries
            nhops = getattr(s, "nhops", 0)
            if nhops:
                out[f"{label}_nhops_per_query"] = nhops / n_queries
    return out


def index_manifest(index, kind: str, **params) -> Dict[str, Any]:
    """Provenance for a built index, to be stored alongside sweep results."""
    return {
        "kind": kind,
        "ntotal": int(index.ntotal),
        "d": int(index.d),
        "metric_inner_product": bool(index.metric_type == 0),
        **params,
    }
