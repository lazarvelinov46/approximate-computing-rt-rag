"""Quality metrics — safe to run on shared free GPUs.

TODO(Phase 1): exact_match(pred, gold), f1(pred, gold)
TODO(Phase 2): recall_at_k(retrieved_ids, gold_ids, k), perplexity(model, text)
"""
def exact_match(pred, gold):
    raise NotImplementedError("Phase 1")
