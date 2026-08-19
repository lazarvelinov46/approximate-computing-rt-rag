"""FAISS indexes.  KNOB 1 = search effort, KNOB 2 = embedding precision.

Precise baseline: IndexFlatIP (exact nearest neighbour, FP32 vectors).
Approximate:      IndexHNSWFlat(ef=...) and IndexIVFPQ(nprobe=..., m=...).

TODO(Phase 1): build_flat_index(vectors)             # exact baseline
TODO(Phase 2): build_hnsw_index(vectors, ef)         # knob 1
TODO(Phase 2): build_ivfpq_index(vectors, nprobe, m) # knobs 1 & 2
TODO:          search(index, qvec, k) -> ids, scores
"""
def build_flat_index(vectors):
    raise NotImplementedError("Phase 1")
