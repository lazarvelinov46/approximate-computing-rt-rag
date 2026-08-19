# Approximate Computing for Real-Time RAG

A precise-vs-approximate evaluation of an end-to-end LLM + retrieval serving stack.
See `../AC-RAG-proposal.md` for the one-page proposal this repo implements.

## Goal
Take a working RAG pipeline, apply software approximate-computing (AC) techniques through
independently tunable knobs, and characterize the accuracy-vs-efficiency trade-off under a
real-time budget — per-knob curves plus a joint trade-off surface.

## The four knobs (+ optional 5th)

| # | Knob | Subsystem | Precise baseline | Approximate variant | AC technique | Module |
|---|------|-----------|------------------|---------------------|--------------|--------|
| 1 | Search effort | Retriever | Exact (FAISS Flat) | HNSW `ef` / IVF `nprobe` | Approximate search | `src/retriever/index.py` |
| 2 | Embedding precision | Retriever | FP32 vectors | Product quantization (IVF-PQ) | Precision scaling | `src/retriever/index.py` |
| 3 | KV-cache precision | Generator | FP16 cache | INT8 / INT4 / INT2 KV | Precision scaling | `src/generator/kv_cache.py` |
| 4 | KV-cache eviction | Generator | Full cache | Token dropping (H2O / SnapKV) | Computation skipping | `src/generator/eviction.py` |
| 5 | Top-k (optional) | Retriever->prompt | Full top-k | Reduced k | Data sampling | `src/pipeline/rag.py` |

## Environment
- **Compute now:** Kaggle Notebooks (free, ~30 GPU-hrs/week, P100 16 GB or 2xT4).
- **Compute later:** RTX 4090 (RunPod ~$0.34/hr, or local) for final latency/energy runs only.
- **Python** 3.11 + PyTorch. **VCS:** git + GitHub (free).

## Software stack (all free / open-source)

| Component | Choice | License | On Kaggle |
|-----------|--------|---------|-----------|
| Runtime | Python 3.11 + PyTorch | — | preinstalled |
| LLM runtime | HuggingFace `transformers` + `accelerate` | Apache-2.0 | preinstalled |
| Main model | `Qwen/Qwen2.5-3B-Instruct` | Apache-2.0 (ungated) | free HF token |
| Small model | `Qwen/Qwen2.5-1.5B-Instruct` | Apache-2.0 | free HF token |
| Embeddings | `BAAI/bge-small-en-v1.5` (`sentence-transformers`) | MIT | pip |
| Vector search | FAISS (`faiss-cpu`) | MIT | pip |
| Weight quant | `bitsandbytes` | MIT | pip |
| KV-cache quant | `transformers` quantized cache (`optimum-quanto`) | Apache/MIT | pip |
| KV eviction | SnapKV / H2O (custom port) | research code | Phase 2 |
| Datasets | HuggingFace `datasets` | Apache-2.0 | preinstalled |
| Metrics / plots | `pandas`, `matplotlib`, `pynvml` | BSD/MIT | preinstalled |
| Config | `pyyaml` | MIT | preinstalled |
| Tracking (optional) | CSV/JSON (default) or W&B free tier | — | optional |

## Project structure
```
ac-rag-stack/
├── configs/            # default.yaml (models, data) + knobs.yaml (sweep ranges)
├── src/
│   ├── data/           # dataset + corpus loading
│   ├── retriever/      # embeddings + FAISS  (knobs 1 & 2)
│   ├── generator/      # LLM + KV precision + eviction  (knobs 3 & 4)
│   ├── pipeline/       # end-to-end RAG orchestration  (knob 5)
│   ├── knobs/          # precise<->approx knob abstraction
│   ├── metrics/        # quality (EM/F1, recall@k) + efficiency (latency/mem/energy)
│   └── experiments/    # sweep runner
├── scripts/            # build_index.py, run_sweep.py
├── notebooks/          # smoke test + baseline
├── results/            # (gitignored) metrics, logs
└── figures/            # generated plots
```

## Roadmap
- **Phase 0** — scaffold + smoke test (env, model load, tiny retrieval). <- we are here
- **Phase 1** — precise baseline end-to-end (Flat index, FP16 cache, full context).
- **Phase 2** — single-knob sweeps for *quality* metrics (hardware-independent; free Kaggle).
- **Phase 3** — dedicated-GPU timing + energy runs (RTX 4090).
- **Phase 4** — joint sweeps, Pareto frontier, plots, write-up.

## Open decisions (confirm before Phase 1)
1. Benchmark: HotpotQA (bundled gold+distractor contexts — convenient) vs NQ-open (needs a Wikipedia corpus).
2. Default model size on Kaggle: Qwen2.5-3B (recommended) vs 1.5B (safest) — 7B reserved for the 4090.
3. Tracking: CSV/JSON + matplotlib (recommended, zero-setup) vs add Weights & Biases.
4. Workflow: GitHub repo cloned into a Kaggle notebook (recommended) vs Kaggle-only.
