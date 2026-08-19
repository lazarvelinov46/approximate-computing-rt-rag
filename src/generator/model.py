"""LLM load + generation wrapper (transformers).

TODO(Phase 1): load_model(name, weight_dtype) -> (model, tokenizer)
TODO(Phase 1): generate(model, tok, prompt, **gen_kwargs) -> text, timings
"""
def load_model(name, weight_dtype="float16"):
    raise NotImplementedError("Phase 1")
