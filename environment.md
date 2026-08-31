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

Two regimes over the same frozen 1000 questions and the same retrieval.
Held fixed within a sweep; swept independently.

| | short | explain |
|---|---|---|
| EM | 0.391 | 0.424 |
| F1 | 0.507 | 0.559 |
| recall@5 | 0.916 | 0.916 |
| EM, complete evidence (n=838) | 0.442 | 0.485 |
| EM, incomplete evidence (n=162) | 0.130 | 0.111 |
| EM, bridge (n=808) | 0.359 | 0.412 |
| EM, comparison (n=192) | 0.526 | 0.474 |
| EM, yes/no (n=65) | 0.646 | 0.646 |
| abstention rate | 0.010 | 0.008 |
| decode tokens p50 | ~3 | 92 |
| decode tokens p99 | — | 224 |
| max_new_tokens | 32 | 256 |
| cap / parse failures | — | 0.6% |

McNemar exact on the paired EM outcomes at n=1000: 120 explain-only correct
vs 87 short-only, 207 discordant, p = 0.0259. The explain regime is
significantly better on quality as well as longer in decode profile.
(At n=500 the same test gave 68 vs 52, p = 0.171 — not significant. The
effect size was nearly identical, 56.7% vs 58.0% of discordant pairs; only
the power changed. A non-significant result at n=500 was absence of
evidence, not evidence of absence.)

Reasoning helps where chaining is required and costs a little where it is
not: bridge EM rises 0.359 -> 0.412 while comparison EM falls 0.526 -> 0.474.
Both subgroups are large enough at n=1000 for this to be a real pattern.

Explain-regime failures: 6/1000 rows hit the 256-token cap, and those are
exactly the 6 rows that fail to parse — a single failure mode. They are soft
repetition loops (the model enumerates numbered sentences past the passages
it was given), not long legitimate chains, so a higher cap does not recover
them (confirmed: same behaviour at caps of 256 and 512).

Abstention: the model occasionally answers the literal string "None".
10/1000 in the short regime, 8/1000 in explain. Tracked as `abstain_rate`
because it is a distinct degradation channel — a knob that damages fact
location may push the model toward abstaining rather than answering wrongly,
and aggregate EM cannot tell those apart.

## Subset size

The evaluation subset was raised from 500 to 1000 questions before Phase 2.
Because the loader shuffles an index list with a fixed seed and takes the
first N valid examples, the 1000-question set is a strict superset of the
500-question set: the original questions keep their positions, therefore
their batch partitions, and their predictions are bitwise identical across
both runs (verified: 0 changed predictions on the shared 500, both regimes).

The 500-question artefacts are retained as a subset-size robustness check.
1008 rows were drawn to yield 1000; 8 were excluded as short.
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

## Index regime 2 — pooled A1 corpus

Corpus: HotpotQA distractor validation split, pooled and deduplicated by
normalised title. 73,700 raw (title, paragraph) pairs -> 66,581 passages.
sha256 1c8d4442c84f145a... (manifest: results/corpus_a1_manifest.json).
Title->text bijection verified: zero titles with multiple texts, zero gold
titles missing, zero text drift against the frozen 1000.

Embedding: 66,581 passages in 199.6s (334/s), 98 MiB fp32. Recomputed per
session, never cached — knob 2 changes the vector representation.

Exact flat search: 1000 queries in 0.28s = 0.28 ms/query, against ~429 ms
per question for generation. Retrieval is 0.065% of end-to-end latency, so
knob 1 has almost no latency to recover at this scale; its efficiency story
is knob 2's memory footprint, not search time.

Prompt tokens (regime 2, n=1000): mean 761, p50 749, p90 981, p99 1178,
max 1325 — shorter worst case than regime 1 (1524) despite the far larger
search space, likely because bge truncates at 512 tokens and long passages
therefore rank lower. batch_size=16 retains more headroom than in Phase 1.

### Precise baseline (exact search, fp32, fp16 KV, top-k 5)

| | short | explain | [regime 1 short] |
|---|---|---|---|
| EM | 0.306 | 0.319 | 0.391 |
| F1 | 0.4037 | 0.4385 | 0.5067 |
| recall@5 | 0.768 | 0.768 | 0.916 |
| complete_frac | 0.573 | 0.573 | 0.838 |
| em_complete | 0.4503 | 0.4695 | 0.4415 |
| em_incomplete | 0.1124 | 0.1171 | 0.1296 |
| em_bridge | 0.2537 | 0.2884 | 0.3589 |
| em_comparison | 0.526 | 0.4479 | 0.526 |
| abstain_rate | 0.014 | 0.015 | 0.010 |

The EM drop is entirely an evidence-completeness effect. Projecting regime-1
conditional EMs onto regime-2 complete_frac predicted 0.308 / 0.325 before
the runs; observed 0.306 / 0.319, both within one SE. Conditional EMs are
unchanged across regimes, so the generator's transfer function is stable and
regime-2 EM movement during a sweep is attributable to retrieval.

### Bridge vs comparison (recall@5, exact search)

| qtype | n | regime 1 | regime 2 | Δ | both-gold r1 | both-gold r2 |
|---|---|---|---|---|---|---|
| bridge | 808 | 0.8967 | 0.7141 | -0.1825 | 0.801 | 0.474 |
| comparison | 192 | 0.9974 | 0.9948 | -0.0026 | 0.995 | 0.990 |

Comparison questions name both entities and are directly addressable by a
single dense query; they lose one question out of 192 across a 6,658x
expansion of the search space. Bridge questions reach the second hop only
through the first and absorb the entire degradation.

### Explain regime attenuates in regime 2

McNemar exact, paired EM, n=1000: 84 explain-only correct vs 71 short-only,
155 discordant, share 0.542, p = 0.335 (regime 1: 120/87, 207 discordant,
share 0.580, p = 0.0259).

Both the discordant count and the effect size fell, which is attenuation
rather than lost power (contrast the n=500 -> n=1000 expansion, where the
share held at ~0.58 and only power changed). Explain's benefit is
concentrated on bridge chaining, and 52.6% of bridge questions no longer
retrieve both passages — reasoning cannot chain evidence that was never
retrieved.

Consequence: knobs 1 and 2 sweep in SHORT mode only, no explain arm.
Knobs 3 and 4 retain explain arms; they run in regime 1, where the effect
holds, and knob 4 needs decode-time context to act on.
