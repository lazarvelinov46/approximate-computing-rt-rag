"""KNOB 4 — KV-cache eviction / token dropping (AC: computation skipping).

Precise baseline: keep_ratio = 1.0 (full cache).
Approximate:      SnapKV / H2O style importance-based eviction.

TODO(Phase 2/advanced): apply_eviction(model, keep_ratio)
"""
def apply_eviction(model, keep_ratio=1.0):
    raise NotImplementedError("Phase 2")
