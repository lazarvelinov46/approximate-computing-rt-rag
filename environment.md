# Environment & setup

## A. Kaggle (primary, free)
1. Create a free Kaggle account; verify phone to unlock GPU.
2. New Notebook -> Settings -> Accelerator -> GPU T4 x2 (or P100).
3. Add your HuggingFace token as a Kaggle Secret named HF_TOKEN (free; needed to download models).
4. First cell:
   ```
   !git clone https://github.com/<you>/ac-rag-stack.git
   %cd ac-rag-stack
   !pip install -q sentence-transformers faiss-cpu bitsandbytes optimum-quanto pynvml
   ```
5. Quality sweeps (Phase 2) run fine here. Do NOT trust latency/energy numbers from Kaggle — shared, throttled.

## B. Local / RTX 4090 (later, for Phase 3 timing + energy)
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Run timing/energy sweeps with nothing else on the GPU.

## Notes on "is it free / accessible"
- Every library above is open-source (MIT / Apache / BSD) and pip-installable.
- Models are Apache-2.0 and ungated (Qwen2.5) — no license-acceptance step, unlike Llama.
- A free HuggingFace token is only needed to pull model weights.

## Measured constants — Kaggle T4 (14.56 GiB), Qwen2.5-3B fp16

Phase 0 (1024-token stub, all-ones mask, full 32-step decode):
- weights resident 5.76–5.88 GiB (varies slightly by session)
- batch 32 peak 12.97–13.09 GiB; OOM at 48

Phase 1, prompts (500 real HotpotQA top-k=5 prompts, batched left-padded):
- prompt tokens: mean 744, p50 728, p90 985, p99 1174, max 1524
  (measured under the final V1 system prompt)
- batches pad to the batch maximum, so the binding constraint is max, not p99
- batch 40 peak 12.65 GiB OK; batch 48 OOM (ladder run under the slightly
  longer V0 prompt, max 1539 — current prompts are shorter, so the measured
  ceiling is conservative)
- per-sequence cost is LOWER than the stub despite longer prompts: real
  short-regime answers hit EOS after ~3-5 tokens, so decode stops early and
  the KV cache never reaches max_new_tokens extent

generation.batch_size = 16 (frozen for all sweeps).
Batches must be fixed chunks of the example order — never length-sorted, since
knob changes would alter batch composition and contaminate the measured delta.

## Generation regimes (Phase 1)

Two regimes over the same frozen 500 questions and the same retrieval. Held
fixed within a sweep; swept independently.

| | short | explain |
|---|---|---|
| EM | 0.390 | 0.422 |
| F1 | 0.486 | 0.539 |
| recall@5 | 0.916 | 0.916 |
| EM, complete evidence (n=418) | 0.438 | 0.474 |
| EM, incomplete evidence (n=82) | 0.146 | 0.159 |
| decode tokens p50 | ~3 | 92 |
| decode tokens p99 | — | 213 |
| max_new_tokens | 32 | 256 |
| cap / parse failures | — | 0.8% |

McNemar exact on the paired EM outcomes: 68 explain-only correct vs 52
short-only, p = 0.171. The regimes are NOT significantly different on
quality; they differ in decode profile. They disagree on 120 of 500
questions while landing at comparable accuracy.

Explain-regime failures: 4/500 rows hit the 256-token cap, and those are
exactly the 4 rows that fail to parse — a single failure mode. They are soft
repetition loops (the model enumerates numbered sentences past the 5 passages
it was given), not long legitimate chains, so a higher cap does not recover
them (confirmed: same 1/32 truncation at caps of 256 and 512).

## Prompt selection

Both system prompts were selected on a 200-question dev sample drawn with
seed 7 and filtered to be disjoint from the frozen evaluation subset, with
criteria fixed before the numbers were seen.

Short regime — the yes/no clause was removed:
| variant | EM | misfire | em_yesno |
|---|---|---|---|
| V0, with yes/no clause | 0.385 | 12.0% | 0.800 |
| V1, clause removed | 0.430 | 0.0% | 0.800 |
| V2, clause replaced by exemplar | 0.430 | 0.0% | 0.800 |

V1 chosen: tied with V2 on quality, fewer tokens. Accuracy on real yes/no
questions was unchanged, so the clause did no useful work — it only created
misclassification opportunities. On the frozen 500 the misfire rate fell from
14.6% to 0.2% and EM rose 0.364 -> 0.390.

Explain regime — four variants, two rejected with diagnosed mechanisms:
| variant | EM | parse | decode p50 |
|---|---|---|---|
| B, two-to-three sentence explanation | 0.435 | 99.5% | 53 |
| C, per-passage walkthrough (chosen) | 0.425 | 100.0% | 90 |
| D, verbatim quoting | 0.140 | 82.5% | 143 |
| E, fixed five-line template | 0.075 | 28.0% | 54 |

C chosen: matches B on quality within one standard error and gives 70% more
decode steps. D fails because verbatim-copy instructions drive greedy
decoding into repetition loops. E fails because its numbered template
suppresses the answer marker and induces early commitment to a wrong
decomposition.

Note on sample size: an initial 32-question pilot showed B 0.500 vs C 0.375
and nearly caused C to be abandoned. At n=200 the gap vanished. Prompt
variants are compared on the dev sample only.