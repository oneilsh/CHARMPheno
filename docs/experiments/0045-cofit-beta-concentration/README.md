# 0045 — Co-fit-beta concentration recovery (CR-4)

**Date:** 2026-07-09
**Topic:** stm | lda | generation | concentration | beta-learning | diagnostics
**Status:** Complete (local numpy plant-and-recover; verdict: hypothesis NOT
confirmed — see insight 0042)

## What this tests

The frozen-beta diagnostic (exp 0038, insight 0038) planted documents at a known
per-document concentration over a known shared-vocab beta and recovered the
concentration with beta FROZEN at truth. It showed held-out predictive-LL
recovers the true concentration and that LDA's alpha-optimization does NOT read
hot — so the real-data STM-vs-LDA peakiness gap (STM median top_mass 0.269 vs
LDA 0.513, exps 0033/0034) is NOT an alpha-inference artifact. 0038 explicitly
left one test un-run ("What this does NOT claim"): a synthetic run where LDA and
STM each CO-FIT beta rather than freezing it at truth.

That is this experiment. The hypothesis (0038's proposed mechanism): when beta
is learned, LDA's Dirichlet document-sparsity pressure carves a SHARPER, more
document-specific beta, raising per-document top_mass (peakier patients), while
STM's logistic-normal stays more blended — reproducing the real gap via
beta-co-adaptation.

## Method

Additive helpers in `spark_vi/eval/topic/concentration_recovery.py` co-fit beta
by full-batch variational EM, reusing the SAME per-doc E-steps the frozen path
and the production models use (`_cavi_doc_inference` for LDA,
`_stm_doc_inference` for STM):

- `stm_cofit_beta` — learns beta under a FIXED logistic-normal prior N(0, c·I);
  Sigma is held at c·I so the swept knob c is a pure concentration prior.
- `lda_cofit_beta` — learns beta at Dirichlet alpha; optional joint
  empirical-Bayes alpha-optimization (Blei 2003 A.4.2).
- `beta_recovery_error` — permutation-invariant topic match (Hungarian /
  linear-sum-assignment, Kuhn 1955; scipy Jonker-Volgenant), reporting matched
  per-topic L1 and 1−cosine.
- `beta_sharpness` — mean top-k mass (higher = sharper) and mean inverse-Simpson
  effective vocabulary (Hill 1973 / Jost 2006; lower = sharper).
- `sweep_heldout_cofit` — for each knob, learn beta on TRAIN, score
  document-completion held-out-LL on a disjoint TEST corpus (Wallach et al.
  2009). The argmax is the held-out-LL-CALIBRATED knob — the 0038 gold standard,
  now applied with beta co-fit.

Per (mechanism × level) cell we plant a TRAIN and a disjoint TEST corpus over
one shared beta_true, then report, at the held-out-LL-calibrated knob for each
model: theta top_mass, beta-recovery error, beta-sharpness — alongside the
FROZEN-beta baseline (same docs, beta frozen at truth) so the delta from
co-fitting beta is explicit. Two regimes match 0038: clean (K=8, V=400,
doc_len=60) and real (K=60, V=5000, doc_len=44). A large-D control
(`real_bigD`, 5× the training corpus, two representative cells) isolates whether
any effect is small-sample beta-identifiability or fundamental. Deterministic
seeds throughout. Driver: `scripts/cofit_beta_concentration_experiment.py`.

## Results

Numbers: `results.md` (clean), `results-real-regime.md` (real),
`results-real_bigD-regime-subset.md` (large-D control), plus the `.json`
siblings. Headline (mean over cells; theta top_mass at the held-out-LL-
calibrated knob):

| regime | planted | FROZEN STM | FROZEN LDA | COFIT STM | COFIT LDA (held-out) | COFIT LDA (α-opt) |
|---|---|---|---|---|---|---|
| clean (K=8) | 0.480 | 0.471 | 0.458 | 0.408 | 0.408 | 0.421 |
| real (K=60, D=600) | 0.201 | 0.183 | 0.166 | 0.063 | 0.032 | 0.036 |
| real_bigD (K=60, D=3000) | 0.288 | 0.270 | 0.227 | 0.154 | 0.127 | 0.030 |

beta-sharpness (mean top-k mass; true beta 0.211–0.322 depending on regime) and
beta-recovery (matched 1−cosine):

| regime | STM sharp | LDA sharp | STM eff_vocab | LDA eff_vocab | STM βcos | LDA βcos |
|---|---|---|---|---|---|---|
| real (D=600) | 0.198 | 0.213 | 103 | 86 | 0.86 | 0.90 |
| real_bigD (D=3000) | 0.120 | 0.158 | 408 | 227 | 0.56 | 0.47 |

### Reading

1. **Co-fitting beta does NOT reproduce the real-data peakiness gap.** At the
   held-out-LL-calibrated knob, co-fit LDA theta is LESS peaky than co-fit STM
   in BOTH real regimes (0.032 < 0.063 at D=600; 0.127 < 0.154 at D=3000) — the
   OPPOSITE sign to the real data (LDA 0.513 > STM 0.269). The clean regime is a
   null: both land at 0.408.

2. **The LDA-beta-sharpening half of the hypothesis is real, but small, and
   causally inert.** LDA's co-fit beta IS marginally sharper than STM's
   (top-k mass 0.213 vs 0.198 at D=600; 0.158 vs 0.120 at D=3000; eff_vocab
   consistently lower for LDA). The effect GROWS with data. But it does NOT
   translate into peakier theta — STM's logistic-normal theta stays peakier at
   the calibrated knob. Sharper beta → peakier patients is FALSE here.

3. **Co-fitting beta COLLAPSES concentration; more data partly cures it.** With
   beta frozen at truth, both families recover the planted concentration well
   (real: 0.183/0.166 vs planted 0.201). Learning beta on the same docs drops
   recovered top_mass to 0.03–0.06 (D=600). This is a beta-identifiability
   failure: at D=600 the held-out-LL argmax sits at the DIFFUSE grid BOUNDARY
   (STM c=1, LDA α=3.0 in 7/8 cells) — no interior concentration optimum,
   unlike 0038's frozen-beta interior argmax. At D=3000 the argmax moves back
   INTERIOR (STM c=5, LDA α=0.3) and recovered top_mass climbs to 0.13–0.16, so
   the collapse is largely small-sample. Even cured, LDA stays cooler than STM.

4. **The real-data-matching LDA config (co-fit beta + α-opt) collapses hardest**
   — top_mass 0.030 at large D, the opposite of real LDA's 0.513.

## Verdict

The hypothesis is NOT confirmed. Co-fitting beta does not reproduce the
real-data STM-vs-LDA peakiness gap and does not causally link a sharper LDA beta
to peakier patients; if anything co-fit LDA under-concentrates most. Combined
with 0038 (alpha-inference is not the cause), NEITHER synthetic mechanism —
alpha-inference nor beta-co-adaptation — reproduces the real gap. The gap is
therefore a property of the real corpus / scale calibration this synthetic model
does not capture, reinforcing the decision-critical step: pin concentration via
held-out-LL on the REAL corpus rather than attribute the gap to a modeling
artifact. See insight 0042 for the full verdict.
