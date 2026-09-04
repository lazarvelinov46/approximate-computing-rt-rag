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


## Knob 1 — retrieval search effort

Regime 2, A1 corpus, short mode only. Build constants frozen: HNSW M=32,
efConstruction=200; IVF nlist=1024; seed 42; inner-product metric;
single-threaded build. Rebuild determinism verified — two independent builds
of each index retrieve identically, so indexes are reproducible from their
parameters rather than needing to be persisted.

Method: `efSearch` and `nprobe` are query-time fields, so a dense sweep of 31
settings costs no GPU time and no generation. ANN recall, recall@5 and
complete_frac are deterministic given a fixed index and question set; only EM
carries sampling noise. The dense sweep therefore gets 31 points and EM gets
8, each metric sampled at the resolution it supports. End-to-end settings were
chosen before any EM was seen, by a pre-registered rule targeting ANN recall
1.00 / 0.95 / 0.90 / 0.80 / 0.60 / 0.40.

Full curves: results/knob1_dense_sweep.csv, results/knob1_summary.csv,
results/knob1_summary.json, results/knob1_curves.png.

### HNSW dominates IVF at equal work

| distance comps/query | HNSW ANN recall | IVF ANN recall |
|---|---|---|
| ~500 | 0.825 (ef=8) | 0.708 (nprobe=6) |
| ~1000 | 0.937 (ef=24) | 0.803 (nprobe=12) |
| ~2000 | 0.981 (ef=64) | 0.891 (nprobe=32) |
| ~3500 | 0.992 (ef=128) | 0.920 (nprobe=48) |

The margin widens with budget. IVF needs the full 66,581 comparisons to reach
ANN recall 1.0, so it has no efficiency advantage at the top of its range.

### Structure-independence at matched evidence

At equal complete_frac, HNSW and IVF give indistinguishable EM despite
different error geometry (graph-traversal truncation vs cell-boundary misses),
and despite differing in ANN recall by up to 0.036:

| pair | complete_frac | McNemar |
|---|---|---|
| ef=32 vs nprobe=48 | 0.527 / 0.527 | b=20 c=13, p=0.296 |
| ef=5 vs nprobe=6 | 0.365 / 0.368 | b=61 c=56, p=0.712 |

Both nulls are adequately powered (33 and 117 discordant pairs). Evidence
completeness is a sufficient statistic for downstream quality; *how* the
retriever lost the evidence does not propagate.

Replicated in knob 2: PQ vs SQ at matched footprint is also null (p=0.552,
45 discordant). Two knobs, four structures, two adequately powered nulls.
Downstream quality is a function of evidence delivered, not of the mechanism
that degraded it.

### Approximation is cheap but not free

| setting | ANN recall | ndis/query | EM | vs exact |
|---|---|---|---|---|
| exact | 1.0000 | 66,581 | 0.306 | — |
| hnsw ef=32 | 0.9562 | 1,228 | 0.297 | p=0.0225 (b=2, c=11) |
| hnsw ef=16 | 0.9028 | 766 | 0.292 | — |
| hnsw ef=8 | 0.8254 | 512 | 0.287 | — |
| hnsw ef=5 | 0.7430 | 406 | 0.275 | p=0.0008 (b=25, c=56) |
| ivf nprobe=1 | 0.3644 | 92 | 0.206 | p<0.0001 (b=39, c=139) |

HNSW at ef=32 does 54x less work for 0.009 EM. That gap is well inside one SE
(0.015) on an unpaired comparison and was initially called noise — wrongly.
Only 13 of 1000 questions changed state, but 11 of 13 went the wrong way, and
McNemar gives p=0.0225. **The quality loss from search-effort approximation is
small but statistically reliable from ef=32 downward.**

Methodological consequence: acceptance criteria for the remaining knobs use
the paired test, not the aggregate gap. Settings share questions, model and
decoding, so most predictions are identical across settings and the shared
variance buries small effects in the aggregate.

### The projection model breaks under degradation

EM projected from complete_frac and the baseline conditionals
(em = cf x 0.4503 + (1-cf) x 0.1124) was accurate to within one SE on the
regime-2 baseline, but underpredicts every degraded setting. Residuals: +0.007
at ANN recall 0.956, rising monotonically to +0.054 at 0.364. All seven
positive (sign test p ~ 0.008).

Mechanism — the conditional is not constant:

| setting | em_incomplete | abstain_rate |
|---|---|---|
| exact | 0.1124 | 0.014 |
| hnsw ef=32 | 0.1163 | 0.025 |
| hnsw ef=16 | 0.1290 | 0.034 |
| hnsw ef=8 | 0.1562 | 0.044 |
| hnsw ef=5 | 0.1559 | 0.052 |
| ivf nprobe=1 | 0.1753 | 0.108 |

Under exact search, incomplete retrieval still supplies five topically
plausible passages and the generator confidently answers from the wrong hop.
Under aggressive approximation it gets obvious junk, recognises it, and either
abstains or falls back on parametric memory — which for HotpotQA's famous
entities is sometimes correct. Abstention rises 7.7x and directly costs EM
(abstentions score zero), so the parametric-fallback effect more than
compensates for it.

This confirms the abstention prediction recorded in the Phase 1 notes: a knob
that damages fact location does push the model toward abstaining rather than
answering wrongly, and aggregate EM cannot separate those channels.

em_complete also drifts up (0.450 -> 0.482) as a selection effect: the
questions still retrieving complete evidence under a degraded index are the
easy ones. The exception is nprobe=1 (0.4397), where the complete bucket holds
only ~116 questions and is both noisy and oddly selected.

Use the projection model for baselines, not for degraded-retrieval regimes.

### Comparison questions are not immune here

Comparison EM held at 0.526 through nprobe=48, then fell to 0.443 (ef=5) and
0.380 (nprobe=1). They were immune to the regime-1 -> regime-2 transition
because both entities are named and directly addressable; they are not immune
to search-effort approximation, which fails to find passages that exist and
are findable.

### Efficiency framing

Exact search was already 0.28 ms/query against ~429 ms of generation — 0.065%
of end-to-end latency. Knob 1's payoff is therefore measured in distance
computations, not seconds. Together with the KV finding (activations exceed KV
~5.6:1), two of the five knobs act on parts of the system that do not dominate
the budget. This is the characterisation the study set out to produce, not a
disappointment, and it argues that short-context RAG's approximation headroom
lives in prefill and decode.

## Knob 2 — embedding precision

Regime 2, A1 corpus, short mode. nbits fixed at 8 for the headline curve.
Isolation: IndexPQ and IndexScalarQuantizer scan EXHAUSTIVELY, so the only
error source is quantisation — nothing is skipped. IVF-PQ fuses knobs 1 and 2
and is reserved for the Phase 4 joint sweep. Both index types verified to
rebuild identically; PQ's k-means seed lives at `index.pq.cp.seed`, not
`index.cp.seed` (IndexPQ is not an IVF index).

Selection anchored on complete_frac rather than ANN recall: complete_frac is
what predicts EM, and it barely moves until ANN recall falls below ~0.8, so
anchoring on ANN recall would have spent most runs where nothing happens.

Full curves: results/knob2_dense_sweep.csv, results/knob2_summary.csv,
results/knob2_summary.json, results/knob2_curves.png.

### Compression is free to 24x, breaks between 32x and 48x

| setting | compression | corpus MiB | ann_recall | complete_frac | EM | McNemar vs exact |
|---|---|---|---|---|---|---|
| exact | 1x | 97.5 | 1.0000 | 0.573 | 0.306 | — |
| sq 8-bit | 4x | 24.4 | 0.9926 | 0.573 | 0.308 | 3/1, d=4, p=0.625 * |
| pq m=192 | 8x | 12.2 | 0.9244 | 0.572 | 0.306 | 17/17, d=34, p=1.000 |
| sq 4-bit | 8x | 12.2 | 0.8964 | 0.568 | 0.301 | — |
| pq m=64 | 24x | 4.1 | 0.7146 | 0.527 | 0.306 | 47/47, d=94, p=1.000 |
| pq m=48 | 32x | 3.0 | 0.6564 | 0.498 | 0.286 | 39/59, d=98, p=0.054 |
| pq m=32 | 48x | 2.0 | 0.5420 | 0.423 | 0.275 | 42/73, d=115, p=0.0049 |
| pq m=16 | 96x | 1.0 | 0.3246 | 0.201 | 0.201 | 40/145, d=185, p<0.0001 |

* only 4 of 1000 answers changed at all; p is uninformative but the
  behavioural identity is near-total.

The m=64 null is the strongest result here: 94 discordant pairs — ample power
— splitting exactly 47/47 at 24x compression, despite complete_frac falling
0.046 below exact. Both well-powered nulls (17/17 and 47/47) are perfectly
symmetric, which fits the mechanism: quantisation noise reshuffles WHICH
questions receive complete evidence roughly at random, so as many questions
gain a gold passage as lose one. Approximation here behaves more like a
permutation of which questions succeed than a systematic degradation.

Operating point: free to 24x, boundary at 32x (p=0.054 — suggestive, not
established), significant loss from 48x.

### Quantisation reorders without losing documents

At pq m=192: ANN recall 0.9244 but exact_match_topk 0.396. The index disagrees
with exact search's full ordered top-5 on 60% of queries while recall@5 moves
0.768 -> 0.766. Quantisation error is large relative to the tiny gaps between
the top few candidates (all similar to the query) and small relative to the
gaps separating relevant from irrelevant.

This is the opposite corner from knob 1, whose errors DROPPED documents:
at ef=64, ANN recall 0.981 with list identity 0.926. Reporting both metrics is
what makes the distinction visible; either alone would mislead.

### The generator is untouched

em_complete across 1x to 48x compression: 0.4503, 0.4520, 0.4476, 0.4507,
0.4516, 0.4518, 0.4539 — never moving more than 0.004. Quantising the
embeddings 48-fold does not damage generation at all; conditioned on receiving
both gold passages the model answers exactly as well. All damage is in evidence
delivery. (At 96x, em_complete jumps to 0.5075 — the selection effect seen in
knob 1: at complete_frac 0.201 the surviving questions are the easy ones.)

Abstention rises 0.014 -> 0.083, more gently than knob 1's 0.014 -> 0.108 at
comparable complete_frac. PQ returns PLAUSIBLE wrong passages — nearest
neighbours in a lossy space are still semantically near — whereas aggressive
ANN search returns passages it stumbled into. Plausible-but-wrong context does
not trigger the "this is junk" response, which is why knob 2's projection
residuals are smaller than knob 1's.

### Byte count is not a sufficient statistic

Matched-footprint probe at nbits=4 vs 8, three pairs, all favouring
fewer-and-finer:

| bytes | more-and-coarser | fewer-and-finer | Δ ann_recall |
|---|---|---|---|
| 96 | m=192, nbits=4: 0.7876 | m=96, nbits=8: 0.8042 | +0.017 |
| 48 | m=96, nbits=4: 0.6266 | m=48, nbits=8: 0.6564 | +0.030 |
| 24 | m=48, nbits=4: 0.4262 | m=24, nbits=8: 0.4610 | +0.035 |

256 centroids per subspace is worth more than doubling the number of
subspaces. nbits=4 does train 8-15x faster (16 centroids vs 256), so the
8-bit advantage costs build time, not memory. Fixing nbits=8 was correct.

### PQ vs SQ at matched footprint: null

At 192 bytes, PQ's learned codebooks beat SQ's uniform grid by 0.028 ANN
recall (0.9244 vs 0.8964) — bge-small's vectors are not uniformly distributed.
That advantage does not reach EM: McNemar b=20, c=25, 45 discordant, p=0.552.
Adequately powered null.

### Scaling argument

At A1 scale, 97.5 MiB -> 4.1 MiB is a curiosity. The ratio is the point: the
same 24x on the 21M-passage DPR corpus rejected in P2-0 is 32 GB -> 1.3 GB,
which is the difference between "needs a server" and "fits in RAM". Knob 2 is
the knob whose result generalises beyond this corpus scale, and unlike knob 1
its efficiency axis (bytes) is a resource that actually binds.

ndis_per_query is the WRONG x-axis for this knob — every setting scans
exhaustively, so it reads 66,581 throughout. Note also that PQ search does not
decompress: FAISS precomputes query-to-centroid distances per subspace and
scores each passage with m table lookups instead of d multiply-adds, so PQ
makes each comparison cheaper as well as smaller. That second benefit is not
captured by any Phase 2 metric and belongs to Phase 3.


## Knob 3 — KV cache precision (regime 1, n=1000)

**Setup.** hqq backend, per-channel (axis 0), q_group_size 32,
residual_length 512, transformers 5.0.0. Retrieval untouched:
complete_frac and recall@5 constant across every setting.

**Frontier (short / explain, relative EM cost):**
- 8-bit @ G32 — 9.00 eff bits, 1.78x — 0.0% / not run. Zero discordant
  pairs against fp16: bit-for-bit invisible.
- 4-bit @ G32 — 5.00 eff bits, 3.20x — -1.5% / -1.9%. Explain McNemar
  b=67 c=75, 142 discordant, p=0.557: an ADEQUATELY POWERED null, the
  strongest equivalence claim in the study.
- 2-bit @ G16 — 4.00 eff bits, 4.00x — -5.4% / -21.2%.
- 2-bit @ G32 — 3.00 eff bits, 5.33x — collapse. Explain: 95%
  unparseable, 42% hit the token cap, 0% empty.

**Group size is a second precision axis, not a tuning detail.** At
identical 5.00 effective bits, 4-bit beats 3-bit 0.385 to 0.339. At
identical nbits=2, group 16 gives EM 0.370 against group 32's 0.046 —
one extra effective bit, eightfold difference. The nbits-only 8/4/2
ladder has no resolution between EM 0.385 and 0.046; adding group size
fills it.

**First knob to move em_complete.** 0.4415 -> 0.0000 with complete_frac
pinned, versus knob 2's maximum movement of 0.004 across 48x
compression. All damage is generator-side by construction.

**Abstention runs OPPOSITE to knob 1.** Retrieval damage raised
abstention 0.014 -> 0.108; cache damage lowers it 0.010 -> 0.000. A
model can see missing evidence and decline. It cannot see a corrupted
cache, so it fills the token budget with confident noise.

**The explain advantage inverts.** +0.033 at fp16, +0.031 at 4-bit,
-0.036 at 2-bit/G16, -0.046 at 2-bit/G32. Chain-of-thought helps with an
intact cache and hurts with a degraded one. Difference-in-differences,
so the baseline mode gap is controlled by construction.

**Damage concentrates on a fragile subset, in short mode.** Pairwise
overlap 3-18x above independent expectation. The 72 fragile questions
have zero yes/no answers (vs 13.2%, p=0.0002), longer gold answers
(2.57 vs 1.82 words, p<0.00001) and more bridge type (p=0.025) —
consistent with exact match punishing long answers. In explain mode
concentration weakens to 1.57x and none of the correlates hold
(yes/no p=1.0, length p=0.10). Cross-mode transfer 22.2% vs 12.8%,
p=0.088: suggestive, underpowered, reported as inconclusive.

**Free CPU proxy validated.** Reconstruction RMSE on quantized tensors
predicts EM at Spearman -0.919 (n=7), so dense sweeps can run on CPU and
GPU runs spent sparsely. One rank inversion (3-bit/G32 vs 2-bit/G16)
remains open; the RMSE surface used planted outliers, and recomputing on
results/knob3_real_kv_sample.pt would test whether that is an artifact.

**Overhead.** Short 1.25x / 1.38x / 2.00x at 8/4/2 bits; explain 1.32x
at 4-bit. residual_length=0 costs 8.00x in explain and was not run.

**Not established.** Backend contrast is CPU-proxy only: optimum-quanto
0.2.7 bundles Marlin kernels requiring sm_80 and cannot build on the T4.
hqq vs quanto agree within 4% on reconstruction error at matched nbits
and matched footprint, which is weaker than the EM-level replications in
knobs 1 and 2.


In explain mode the bridge correlate REVERSES: fragile questions are 68.0%
bridge vs 80.8% robust (p=0.0195), against 84.7% vs 71.8% in short mode. A
sign flip on a significant effect, not merely a null — comparison questions
survive span extraction from a perturbed cache but not a 90-step reasoning
chain over one.
