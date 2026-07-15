# 0038 — Held-out predictive likelihood recovers the true per-document concentration (validated on synthetic, both prior families, both regimes); and LDA α-optimization does NOT read hot — so the real STM-vs-LDA concentration gap is not an α-inference artifact

**Date:** 2026-07-05
**Topic:** stm | lda | generation | concentration | diagnostics | calibration
**Status:** Confirmed (synthetic plant-and-recover, `spark_vi/eval/topic/concentration_recovery.py`, `scripts/concentration_recovery_experiment.py`; results in `docs/experiments/0038-concentration-recovery/`)

Insight [0037](0037-in-band-pooled-scale-converges-below-export-scale-faithful-scale-not-fit-recoverable.md)
established that every fit-anchored generative scale under-concentrates and that the true
per-document concentration is unknown — a bracket [0.27, 0.51] in top_mass with no gold
standard to pin it. It proposed held-out within-document token prediction as the gold
standard. This insight VALIDATES that gold standard on synthetic data where the true
concentration is known, and in doing so corrects the "LDA α-optimization reads hot" story.

## The experiment

A controlled, fully local (numpy) plant-and-recover diagnostic. Plant documents at a KNOWN
per-document concentration over a SHARED-TERM topic matrix (`make_shared_beta` — topics share
a common vocabulary pool, so inference must disambiguate, unlike a disjoint-vocab toy). Two
planting mechanisms, so neither prior family is privileged: logistic-normal (η ~ N(0, level·I),
θ = softmax) and Dirichlet (θ ~ Dir(level·1)), four concentration levels each. The GROUND TRUTH
is the planted θ's measured concentration (top_mass = max_k θ_k; effective #topics =
inverse-Simpson / Hill order 2, Hill 1973 / Jost 2006), not the knob. Recover with STM (MAP
E-step under N(0, c·I), sweep c) and LDA (CAVI at Dirichlet α, sweep α), β FROZEN to the true
matrix so the test isolates concentration INFERENCE from topic learning. Held-out gold
standard: mask a random 30% of each document's tokens, infer θ̂ from the visible tokens, score
predictive log-likelihood of the held-out tokens, sweep the concentration knob; the argmax is
the estimated concentration. Run at a clean regime (K=8, V=400, doc_len=60) and the REAL regime
(K=60, V=5000, doc_len=44, matching the production corpus).

## Finding 1 — held-out predictive likelihood recovers the true concentration, both families, both regimes

In all 8 (mechanism × level) cells, at BOTH regimes, the held-out-LL argmax knob recovers a θ̂
whose median top_mass matches the planted median within a small tolerance (worst-case absolute
error 0.068 at the real regime, 0.048 at the clean regime; tolerance 0.08). The diffuse unit
scale never wins on a peaky corpus, and the argmax is interior. So held-out within-document
prediction is a trustworthy, PRIOR-FAMILY-AGNOSTIC, ground-truth-free calibration for
per-document concentration — and critically it holds at the real regime (K=60, short 44-token
documents, sparse 5000-term vocabulary), exactly where it would be applied to real data. This
licenses using real-data held-out-LL to pin the insight-0037 bracket.

## Finding 2 — STM recovers concentration more faithfully than LDA at fixed β

At the real regime the logistic-normal MAP recovers the planted concentration with lower mean
absolute error than Dirichlet CAVI (STM 0.019 vs LDA 0.033); the gap is wider at the real
regime than the clean one. The expected "matched prior" effect (each family recovering its own
generative story best) did NOT hold — STM was the better recoverer on BOTH logistic-normal- and
Dirichlet-planted data. So the logistic-normal is not the weaker concentration estimator; at
the right scale it is the better one.

## Finding 3 — LDA α-optimization does NOT read hot; it reads COOL — the "reads hot" story is wrong here

The hypothesis (from mean-field-VB under-dispersion, and stated in exp 0034 / the 2026-07-04
review) was that LDA α-optimization over-concentrates on short documents, explaining why real
LDA looked ~2× peakier than STM. The synthetic data refutes this as the cause: with β frozen at
truth, LDA's own α-optimization recovers a concentration AT or BELOW the planted value — mean
top_mass Δ = −0.014 at the real regime (several cells badly UNDER-concentrated, e.g.
logistic-normal level 3: α-opt gives top_mass 0.048 vs planted 0.211). Moreover, LDA's α-opt
picks a DIFFERENT (more diffuse) α than the held-out-LL optimum — its internal objective is not
the held-out-predictive one — which is a further argument for held-out-LL as the calibration
criterion over "just optimize α".

## Consequence — the real STM-vs-LDA gap is β-learning + fit-scale, not α-inference

On real data (exps 0033/0034) STM at its fit scale τ² = 2.36 gave median top_mass 0.269 and
LDA α-optimized gave 0.513. Findings 1–3 reattribute that gap: it is NOT an α-inference
artifact (Finding 3), so it is driven by (a) STM's fit scale being too low (insight 0037) and
(b) LDA co-fitting a sharper, more document-specific β — the synthetic FROZE β at truth, so it
cannot reproduce a β-learning effect; real LDA learns its own topics and can carve peakier ones.
The true per-document concentration is therefore still open, but now pinnable: run the
validated held-out-LL sweep on the REAL corpus (sweep STM's Σ scale on real documents; the
argmax is the faithful generative scale). That is the decision-critical next step for the
generative bundle, replacing every fit-anchored guess.

## What this does NOT claim

- It does not claim LDA is worse in general — only that at fixed β it is a slightly weaker, and
  cooler-reading, concentration estimator in these regimes.
- It does not measure the real documents' true concentration — the synthetic β and planting are
  a MODEL of the problem; the definitive number needs real-data held-out-LL (and, to test the
  β-co-adaptation hypothesis directly, a synthetic run where LDA and STM each CO-FIT β rather
  than freezing it at truth).
- It does not reopen a fit-time free variance (still falsified, exps 0022/0024/0032).

**Related:** insight 0037 (the bracket + the held-out-LL proposal this validates), exps 0033
(STM A1) / 0034 (LDA), the diagnostic + update reports under `docs/superpowers/specs/2026-07-04-*`
and `2026-07-05-*`. Code: `spark_vi/eval/topic/concentration_recovery.py`,
`scripts/concentration_recovery_experiment.py`; results
`docs/experiments/0038-concentration-recovery/results{,-real-regime}.{json,md}`.
Metric: Hill 1973 / Jost 2006. Mean-field VB under-dispersion (the refuted-here hypothesis):
Teh, Newman & Welling 2007; Asuncion et al. 2009.
