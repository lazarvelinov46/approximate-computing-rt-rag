# Approximate Computing for Real-Time RAG

A precise-vs-approximate evaluation of an end-to-end LLM + retrieval serving stack.

Five approximation knobs are swept individually against a frozen exact baseline on
the same 1,000 HotpotQA questions, with every comparison decided by a paired
statistical test rather than by a gap between averages. Quality characterisation is
complete; efficiency measurement is future work.

## The five knobs

| # | Knob | Subsystem | Precise baseline | Approximate variant | AC technique | Module |
|---|------|-----------|------------------|---------------------|--------------|--------|
| 1 | Search effort | Retriever | Exact (FAISS Flat) | HNSW `ef` / IVF `nprobe` | Computation skipping | `src/retriever/index.py` |
| 2 | Embedding precision | Retriever | FP32 vectors | Product / scalar quantization (flat, exhaustive scan) | Precision scaling | `src/retriever/index.py` |
| 3 | KV-cache precision | Generator | FP16 cache | hqq 8/4/3/2/1-bit, group size 32 or 16 | Precision scaling | `src/generator/kv_cache.py` |
| 4 | KV-cache eviction | Generator | Full cache | Recency / sink+recent / attention (H2O, SnapKV) / uniform random control | Pruning | `src/generator/eviction.py` |
| 5 | Top-k | Retriever→prompt | k = 5 | k ∈ {2, 3, 7, 10, 20} | Data sampling | `src/pipeline/rag.py` |

IVF-PQ is deliberately absent from the single-knob rows: it fuses skipping (knob 1)
with precision loss (knob 2), so a curve produced by varying `m` inside an IVF index
is a curve of both knobs at once. It is reserved for a joint sweep, where the
question is whether the two error sources compose as the single-knob curves predict.

## Two retrieval regimes

- **Regime 1 — per-question corpora.** Each question sees only its own ~10 paragraphs
  (2 gold, 8 distractors). Retrieval difficulty is constant, so quality changes are
  attributable to the generator. Knobs 3 and 4 run here.
- **Regime 2 — pooled corpus.** All validation-split paragraphs pooled into one index:
  73,700 raw title/paragraph pairs deduplicated by normalised title to **66,581
  passages**, sorted for reproducible index construction and pinned by SHA-256.
  Approximate search is meaningless at N≈10, so knobs 1 and 2 run here.

Knob 5 runs in both.

## Baselines

Exact search, fp32 vectors, fp16 KV cache, no eviction, k = 5, n = 1000, seed 42,
greedy decoding.

| | Regime 1 | Regime 2 |
|---|---|---|
| recall@5 | 0.916 | 0.768 |
| Both gold retrieved | 0.838 | 0.573 |
| EM / F1 (short) | 0.391 / 0.507 | 0.306 / 0.404 |
| EM (explain) | 0.424 | 0.319 |

## Acceptance criterion

Every comparison is an exact McNemar test on per-question exact match, reported with
the **discordant count** — the number of questions on which two settings disagree,
which is the test's effective sample size.

- p ≤ 0.05 → real difference
- p > 0.05 with ≥ 30 discordant → evidence of no difference; reported as a claim
- < 10 discordant → uninformative
- 0 discordant with identical marginals → strongest possible null

"Free" in this repo means a powered null under these rules, not a small gap.

## Headline results

- **Embedding compression to 24× is free** (94 discordant, p = 1.000) even though the
  exact top-5 is returned on only 3.6% of queries. Ranking is destroyed; enough gold
  survives that the generator cannot tell.
- **8-bit KV quantization is bit-identical to fp16** — same answer on all 1,000
  questions at 9.00 effective bits. Quality holds to 5 effective bits, then collapses
  between 4.00 and 3.00.
- **KV eviction has no free operating point**, and delivered no memory saving: peak
  stayed at 6.11–6.15 GiB against 5.87 GiB of weights, because eviction acts after
  prefill and prefill is the peak.
- **Evidence delivered to the generator is a sufficient statistic for quality.**
  Replicated across two index families, two quantization schemes, and two distractor
  sources.
- **k = 5 is not optimal.** k = 10 is a significant improvement in both regimes;
  k = 2 is null in both at less than half the prompt length.

Negative and falsified results are kept deliberately — see `environment.md`.

## Environment

- **Compute:** Kaggle Notebooks, single NVIDIA Tesla T4 (14.56 GiB, sm_75). Two cards
  are allocated; one is used (`cuda:0`) so results don't depend on work distribution.
- **Runtime:** Python 3.12, PyTorch 2.10 (CUDA 12.8), transformers 5.0.0.
- **Later:** RTX 4090 for latency/energy runs.

Three hardware constraints removed comparisons from the design rather than merely
slowing them: optimum-quanto is unavailable on sm_75 (hqq used instead, so the
backend contrast does not exist); k = 20 in regime 2 OOMs at the frozen batch size
and ran at batch 8 with a matched control; and non-contiguous eviction produces NaNs
under left padding, forcing batch size 1 for that sweep.

## Software stack

| Component | Choice | License |
|---|---|---|
| LLM runtime | HuggingFace `transformers` + `accelerate` | Apache-2.0 |
| Main model | `Qwen/Qwen2.5-3B-Instruct` | Apache-2.0 |
| Embeddings | `BAAI/bge-small-en-v1.5` (`sentence-transformers`) | MIT |
| Vector search | FAISS (`faiss-cpu`) | MIT |
| KV-cache quant | `hqq` | Apache-2.0 |
| KV eviction | H2O / SnapKV-style (custom implementation) | — |
| Datasets | HuggingFace `datasets` | Apache-2.0 |
| Metrics / plots | `pandas`, `matplotlib` | BSD/MIT |

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
│   ├── metrics/        # quality (EM/F1, recall@k) + efficiency (stub)
│   └── experiments/    # sweep runner
├── scripts/            # plot_phase2.py, report_tables.py
├── notebooks/          # 00-32: smoke test, baselines, per-knob sweeps and closeouts
├── results/            # per-question CSVs, per-knob summaries, figures
└── figures/            # generated plots
```


`scripts/build_index.py` and `scripts/run_sweep.py` are unimplemented stubs — sweeps
are driven from the notebooks, which clone this repo into the Kaggle session so every
run uses versioned code rather than logic pasted into a cell.

## Reproducing the figures and tables

Both are local, CPU-only, and read `results/*_summary.{json,csv}`:

```bash
python scripts/plot_phase2.py --self-test   # verdict logic
python scripts/plot_phase2.py --check       # input inventory + schemas
python scripts/plot_phase2.py               # writes results/figures/
python scripts/report_tables.py             # markdown result tables
```

## Status

- **Complete** — baselines (both regimes, both answer formats) and all five knob
  sweeps, with figures and result tables.
- **Next** — efficiency measurement on dedicated hardware; joint two-knob sweeps and
  a Pareto frontier (KV precision × top-k first, then search effort × eviction);
  Qwen2.5-7B (GQA-4) as a within-family contrast for the KV knobs.
- **Queued** — quantization-error proxy recomputed on captured KV tensors; audit of
  the larger corpus tier before use.

## Settled decisions

- Benchmark: HotpotQA distractor, validation split.
- Large-index corpus: pooled HotpotQA paragraphs (66,581), not NQ-open.
- Answer scoring: EM and F1 only. An LLM judge was considered and rejected — F1
  already supplies the partial credit that motivated it.
- Tracking: CSV/JSON + matplotlib. Workflow: GitHub cloned into Kaggle.

## Open

- **Study scoping** — characterisation of approximate computing in short-context RAG,
  versus reproducing long-context KV findings. Prompt tokens are 88–99% of the KV
  cache here and prefill activations exceed KV by ~7:1, so the long-context regime is
  not reachable without changing model or corpus.