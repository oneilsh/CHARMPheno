# 0039 — Revisiting 0028: stabilized, gated STM recovers rare/minority sub-phenotypes; the "prior family, not stabilization" verdict was confounded

**Date:** 2026-07-08
**Topic:** stm | lda | gating | priors | rare-phenotypes | phenotyping | re-examination
**Status:** Observed (re-examination of [0028](0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md))

Insight [0028](0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md)
(2026-06-24) concluded that the document–topic **prior family** — Dirichlet (LDA)
vs logistic-normal (STM/CTM) — *intrinsically* governs rare-phenotype recovery,
that logistic-normal "washes them into the anchors regardless of gating," and that
the fix is to build gated LDA (PLDA). Later work shows that conclusion was
**confounded by missing stabilizers and fit-scale**, and is superseded in part.

## What the later runs show

1. **The collapse was a missing-stabilizer artifact, not the prior family.**
   [0029](0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md) diagnosed the
   STM σ-init collapse / Σ-blowup as a *missing-stabilizer* artifact (spectral init
   + K−1 reference-topic parameterization + Σ shrinkage), "**not a property of
   STM**." 0028's STM arm (exps 0005/0007) ran without any of that stack.
   [0030](0030-spectral-init-closes-stm-sigma-blowup-on-real-data.md): spectral
   init "resolves all K topics at the default σ_init=1" on the real cancer data.

2. **Same cohort, stabilized STM recovers what 0028 said it could not.**
   [0032](0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md)
   re-ran the **same `cancer_or_dementia` cohort** as stabilized gated full-Σ STM
   and split the dementia block into genuine sub-phenotypes (Alzheimer's/amnestic
   topic 41, vascular dementia topic 49) — the ones 0028 recorded as ~0.003 peak-β
   clones. It states it "delivers the rare-phenotype recovery that insight 0028
   found the combined logistic-normal STM could not."

3. **Extended to a far more extreme minority ratio.**
   [0035](0035-rare-disease-gated-foreground-recovers-eds-subphenotypes-on-full-population.md)
   recovered EDS sub-phenotypes with a **0.5% foreground (956 docs)** under the
   block-wise unit-diagonal STM ([ADR 0034](../decisions/0034-stm-blockwise-unit-diagonal-correlation-sigma.md)):
   no collapse, Σ bounded. This is far past 0028's 19% minority arm.

## Patient-concentration reframe

[0037](0037-in-band-pooled-scale-converges-below-export-scale-faithful-scale-not-fit-recoverable.md)
and [0038](0038-heldout-ll-recovers-true-concentration-and-lda-alpha-opt-is-not-hot.md)
recast the STM-vs-LDA difference on **per-patient θ** as a **fit-scale + β-learning**
effect, not a prior-family verdict:

- LDA is ~2× more peaked per patient (top_mass p50 ≈ 0.51, ~2.8 effective topics)
  than STM-A1 (≈ 0.27, ~8.5) — so **LDA can be too sparse for patient comorbidity**.
- 0038: STM is the **more faithful concentration recoverer at fixed β** (MAE 0.019
  vs 0.033) — even on *Dirichlet*-planted data — and LDA's α-optimization reads
  **cool/under-concentrated**, not hot. The true per-patient concentration is
  **unpinned** (bracket [0.27, 0.51]); the arbiter is **held-out within-document
  token likelihood** (0038), not either prior's default.

## What survives from 0028, and what does not

- **Survives (as a tendency, not destiny):** the mechanism — Dirichlet(α<1) seeks
  simplex vertices (sparse), logistic-normal is interior (smoother). Real, but with
  proper init + scale calibration the logistic-normal *does* reach sub-phenotype
  resolution.
- **Does not survive:** the causal attribution ("prior family, not stabilization")
  and the recommendation that follows ("must adopt Dirichlet/PLDA because STM
  cannot"). The collapse was fixable, and was fixed.
- **Never actually tested:** 0028's decisive **non-gated dementia-alone 2×2**
  (exp 0006 LDA vs 0007 STM) was **not** re-run on the stabilized STM, and **gated
  LDA / PLDA was never built**. So no clean same-arm prior-family comparison exists.

## Corrected takeaway

Engine choice (gated LDA vs stabilized gated STM) is **open**, not settled by prior
family. STM's correlated Σ — a **patient-level topic co-occurrence correlation**
(across-patient covariance of the per-document η; see [ADR 0033](../decisions/0033-stm-full-covariance-sigma.md)/[0034](../decisions/0034-stm-blockwise-unit-diagonal-correlation-sigma.md))
— is a **positive asset that gated LDA structurally lacks**. Resolve the choice
with a **head-to-head not yet run**: gated LDA vs stabilized gated STM on the same
rare arm, judged by held-out LL (0038's arbiter), rare-phenotype recovery, and
downstream case-finding utility.

**Do not cite 0028 as "STM cannot recover rare phenotypes" or as a reason to avoid
the logistic-normal engine.**
