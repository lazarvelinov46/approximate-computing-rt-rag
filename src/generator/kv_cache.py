"""KNOB 3 — KV-cache precision (AC: precision scaling).

Precise baseline: FP16 cache (default DynamicCache).
Approximate:      transformers QuantizedCache (optimum-quanto backend) at 8/4/2 bit.

TODO(Phase 2): make_cache(bits) -> cache config for model.generate(...)
"""
def make_cache(bits=16):
    raise NotImplementedError("Phase 2")
