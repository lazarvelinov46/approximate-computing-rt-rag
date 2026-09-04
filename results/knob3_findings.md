
## Knob 3 — KV cache precision (regime 1, n=1000)

**Setup.** hqq backend, per-channel (axis 0), q_group_size 32,
residual_length 512, transformers 5.0.0. Retrieval untouched:
complete_frac and recall@5 constant across every setting.

**Frontier (short / explain, relative EM cost):**
- 8-bit @ G32 — 8.50 eff bits, 1.78x — 0.0% / not run. Zero discordant
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
