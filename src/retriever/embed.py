"""Embedding model wrapper (bge-small via sentence-transformers). Phase 1.

Precise baseline for KNOB 2 (embedding precision): full-precision fp32 dense
vectors, L2-normalised so that inner product == cosine similarity, which is
the metric BGE was trained under.

Experimental-setup invariants — these must not change between sweeps:
  * the instruction prefix goes on the QUERY side only (bge is asymmetric)
  * vectors are always L2-normalised
  * vectors are always returned fp32 and C-contiguous (what FAISS requires)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_MODEL_CACHE: Dict[Tuple[str, str], object] = {}


def load_embedder(name: str = "BAAI/bge-small-en-v1.5", device: Optional[str] = None):
    """Load (and cache) the encoder. Cached so repeated sweep calls don't reload."""
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    key = (name, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = SentenceTransformer(name, device=device)
    return _MODEL_CACHE[key]


def _encode(model, texts: Sequence[str], batch_size: int, progress: bool) -> np.ndarray:
    vecs = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,   # inner product == cosine
        convert_to_numpy=True,
        show_progress_bar=progress,
    )
    # FAISS will reject anything that is not fp32 and C-contiguous.
    return np.ascontiguousarray(vecs.astype("float32", copy=False))


def embed_passages(model, texts: Sequence[str], batch_size: int = 64,
                   progress: bool = False) -> np.ndarray:
    """(N, dim) fp32 passage matrix. No prefix — passages are encoded bare."""
    return _encode(model, texts, batch_size, progress)


def embed_queries(model, questions: Sequence[str], batch_size: int = 64,
                  progress: bool = False) -> np.ndarray:
    """(M, dim) fp32 query matrix, with the bge instruction prefix applied."""
    return _encode(model, [QUERY_PREFIX + q for q in questions], batch_size, progress)


def embed_query(model, question: str) -> np.ndarray:
    """Single query -> (1, dim). Convenience wrapper for the pipeline."""
    return embed_queries(model, [question])


def embed_grouped(model, groups: Sequence[Sequence[str]], batch_size: int = 64,
                  progress: bool = False) -> List[np.ndarray]:
    """Embed per-question paragraph groups in ONE batched pass.

    Regime 1 gives 500 groups of ~10 paragraphs. Encoding each group separately
    would leave the GPU idle between 10-item batches; we flatten, encode once,
    then slice back so group i keeps paragraph order (index j in the returned
    matrix == paragraphs[j] of that question). That alignment is what makes
    gold_ids valid as retrieval ground truth.
    """
    flat, bounds, cursor = [], [], 0
    for g in groups:
        flat.extend(g)
        bounds.append((cursor, cursor + len(g)))
        cursor += len(g)

    matrix = _encode(model, flat, batch_size, progress)
    return [np.ascontiguousarray(matrix[a:b]) for a, b in bounds]


def token_lengths(model, texts: Sequence[str]) -> np.ndarray:
    """Untruncated token length per text — used to audit encoder truncation."""
    tok = model.tokenizer
    return np.array([len(tok.encode(t, truncation=False)) for t in texts])