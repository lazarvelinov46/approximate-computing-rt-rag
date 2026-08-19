"""Efficiency metrics — only meaningful on a dedicated GPU (Phase 3).

TODO(Phase 3): time_to_first_token, tokens_per_sec, peak_memory (torch.cuda),
               kv_memory, energy_per_query (pynvml power sampling).
"""
def peak_memory_mb():
    raise NotImplementedError("Phase 3")
