---
id: 25
slug: stm-comorbid-fullcov-gated-pdcompletion
status: pending
model_class: stm
cohort: cancer_or_dementia
cohort_def: cancer_or_dementia
prior_obs_days: 0
person_mod: 4
doc_unit: patient_cohort
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 50
background_k: 30
foreground: "cancer:10,dementia:10"
group_var: source_cohort
max_iter: 100
sigma_init: 1.0
reference_topic: true
spectral_init: true
spectral_method: dense
min_pair_support: 10
---

# STM comorbid (cancer + dementia): gated multi-group Σ via positive-definite completion

Re-runs the exp 0021 gated multi-group configuration (K=50 = 30 background + 10
cancer + 10 dementia, `group_var=source_cohort`, `min_pair_support=10`) with the
new gated-Σ assembly: the maximum-determinant positive-definite completion replaces
the `min_pair_support` zero-pin plus post-hoc `nearest_spd` flooring, and BOTH
post-M-step regularizers (the inverse-Wishart prior and `sigma_diag_shrink`) are
removed. This is the resolution run for the SPD-assembly conditioning pathology
documented across exps 0021-0024 (insight
[0032](../insights/0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md))
and specified in the
[gated-Σ PD-completion design](../superpowers/specs/2026-06-30-stm-gated-sigma-pd-completion-design.md).
The frontmatter is exp 0021's, minus the three regularizer keys, which no longer
exist.

## Why

Exp 0021 established the scientific payoff — gated full-Σ recovers Alzheimer's vs
vascular dementia in a 19% minority arm — but exposed the SPD-assembly
ill-conditioning ADR 0033 predicted: `Σ_eig[min=1e-6 max=32.8 cond=3.28e7]`, with
the minimum eigenvalue pinned exactly at `SIGMA_FLOOR=1e-6`. Four lever-tuning
experiments tried to condition the assembled matrix after the fact and all failed
(insight 0032, Findings 4-6):

- exp 0022 (inverse-Wishart prior, ν=100) — left cond at 3.05e7; the N-weighted
  prior cannot reach the well-supported background↔foreground couplings that drive
  the near-singularity.
- exp 0023 (`sigma_diag_shrink=0.2`) — fixed the min eigenvalue (1e-6 → 1.0) by
  decorrelating, but the off-diagonal coupling was load-bearing for variance
  stability, so a single dementia topic's η-variance ran away to 2.23e7 (the
  max-eigenvalue end), ELBO −2.252e6.
- exp 0024 (both levers) — the runaway worsened to 6.08e8; the IW anchor is
  N-weighted and `scale=2` points the wrong way for stability.

The diagnosis: the conditioning problem has two heads — a min-eigenvalue
near-singularity from off-diagonal inconsistency AND a max-eigenvalue variance
runaway — and the off-diagonal correlations are simultaneously the SOURCE of the
near-singularity and the STABILIZER of the variances, so no post-hoc knob conditions
the matrix. The defect is upstream, in what the assembly pins the unobserved cells
to.

The fix reframes the assembly as Gaussian-graphical-model covariance selection:
zero the PRECISION (Σ⁻¹), not the COVARIANCE, on unobserved cross-pairs. Σ⁻¹_ij = 0
encodes conditional independence of topics i and j given all others — the
minimum-assumption statement for a pair with no joint data — and, unlike a zeroed
covariance, it is consistent with arbitrary marginal correlations through shared
neighbours (the background block). The covariance matrix that fixes the observed
entries and zeroes the precision on the rest is the maximum-determinant
positive-definite completion (Dempster 1972; Grone, Johnson, Sá & Wolkowicz 1984),
unique whenever the observed entries admit any PD completion, well-conditioned by
construction. The non-gated case (exp 0020, cond 13.3) is the fully-connected
special case where the completion is a no-op — which is why it was already clean.

The completion is implemented as
[`pd_complete`](../../spark-vi/spark_vi/models/topic/_linalg.py#L32-L105) using
convergent coordinate-wise iterative proportional scaling (Speed & Kiiveri 1986),
with a Higham 2002 `nearest_spd` PSD projection as the fallback when the observed
part admits no PD completion (the thin-group regime). On a decomposable pattern it
hits the Grone et al. 1984 / Lauritzen 1996 closed form in one sweep. The M-step
([stm.py:709-727](../../spark-vi/spark_vi/models/topic/stm.py#L709-L727)) now forms
the observed entries as S/N over the N≥`min_pair_support` mask (the diagonal forced
observed, an absent topic's variance lazy-kept at its current Σ[k,k]) and calls
`pd_complete`. The IW prior (`sigma_prior_scale`, `sigma_prior_count`) and
`sigma_diag_shrink` are fully removed — parameters, plumbing, and tests.

## Hypothesis

The gated multi-group Σ assembled by PD completion is well-conditioned at BOTH
spectral ends, with no variance runaway and no loss of the exp 0021 science:

(a) **Condition number drops to O(1e1-1e3)** — orders of magnitude below the 3.28e7
    of exp 0021 (and the 2.23e7 / 3.07e8 of exps 0023/0024), close to the non-gated
    exp 0020 regime (cond 13.3).
(b) **`sigma_eig_min` lifts well off the 1e-6 floor** — the near-singularity is gone
    because the completion fills unobserved cross-pairs with their
    conditional-independence-implied values rather than a transitively inconsistent
    zero, so no eigendirection is forced to the floor.
(c) **`sigma_eig_max` is back to O(1-10)** — both ends controlled simultaneously. The
    completion never decorrelates (it preserves the observed off-diagonals exactly),
    so the off-diagonal coupling that stabilizes weakly-identified variances is
    intact and no topic's η-variance runs away (contrast the 2.23e7 / 6.08e8 of
    exps 0023/0024).
(d) **ELBO comparable to exp 0021's −1.5897e6** — the design plan estimated ≈ −1.59e6;
    not exp 0024's degraded −1.836e6 or exp 0023's −2.252e6.
(e) **Dementia sub-phenotypes preserved** — topic with Amnesia + Alzheimer's + MCI
    (exp 0021 topic 41) and topic with atherosclerosis + AFib + dementia (exp 0021
    topic 49); per-block NPMI near exp 0021 levels (background ≈ 0.19, cancer ≈ 0.18,
    dementia ≈ 0.17); all 50 topics resolved.
(f) **Trustworthy cancer↔dementia R sub-block** — the (10×10) cancer-foreground ↔
    dementia-foreground sub-block of the saved `correlation.npy` holds the
    CI-implied values (correlated only through shared background comorbidity), not
    polluted by a near-singular direction (exp 0021's inflated max_offdiag 0.80 was
    almost entirely that pollution; the genuine couplings in this cohort are ≤ 0.18,
    per exp 0023's decorrelated readout).

## Method

Identical fit to exp 0021 — same cohort, K, block sizes, gating variable, init, and
`min_pair_support` — with the only change being the Σ assembly internals (zero-pin
+ `nearest_spd` → `pd_complete`) and the absence of the two regularizers. The gating
variable `source_cohort` is again deliberately absent from `covariate_formula`
(ADR 0026; see exp 0004). Comorbid patients contribute one document per cohort, so a
cancer↔dementia cross-foreground entry gains support only from co-activating
patients; thin cross-pairs (N < 10) are now FREE for the completion rather than
zero-pinned. `min_pair_support` keeps its meaning — the small-cell/robustness
threshold deciding which cross-pairs are trustworthy — but now selects observed-vs-free
for the completion.

## What to watch

- **`stm_sigma_diagnostic` per-iter Σ trace** — `Σ_eig[min … max … cond]` and
  `max_abs_offdiag_corr`. Target: cond O(1e1-1e3), min eig well off 1e-6, max eig
  O(1-10), no per-iter blowup at either end. This is the head-to-head against exps
  0021-0024's diagnostics.
- **Per-topic Σ_ii (variance) trace** — confirm no single topic's variance runs away
  (exps 0023/0024 saw 2.2e7 / 6.08e8). With the off-diagonal coupling preserved,
  every Σ_ii should stay O(1-10).
- **ELBO trajectory** — should settle near exp 0021's −1.5897e6, not degrade.
- **Cancer↔dementia R sub-block** (`correlation.npy`, the 10×10 cross-foreground
  block) — the CI-implied completed values; check they are small and sensible
  (≤ ~0.18) rather than inflated.
- **Per-block topic quality / NPMI** — each foreground block keeps block-distinctive
  phenotypes; the dementia sub-phenotype split survives.

## Decision

- **cond O(1e1-1e3) + both eigenvalue ends controlled + no variance runaway + ELBO
  ≈ −1.59e6 + dementia sub-phenotypes preserved + trustworthy cancer↔dementia R**
  → the PD completion is validated; it resolves the four-experiment conditioning
  failure by construction. Make the completion the gated-Σ default (it already is,
  per the design), close the conditioning thread in insight 0032, and record the
  resolution. The zero-pin + lever-tuning approach is retired.
- **A topic STILL runs away (max eig blows up despite the completion)** → the runaway
  was NOT caused by the cross-block zero-pin inconsistency. Re-examine via
  `stm_sigma_diagnostic`: identify the runaway topic and whether its own variance
  (Σ_ii data term) or a genuine well-supported direction is exploding, independent of
  the completion. This would point back to the per-doc E-step / spectral-init basin
  (insight 0029/0030), not the assembly.
- **cond still high but driven by the MIN end (a floored eigendirection)** → the
  observed part was not PD-completable and the Higham fallback engaged. Inspect which
  observed block is internally inconsistent (likely a thin within-group block, not a
  cross-block); this is the rare-disease non-PD-observed regime the fallback handles,
  but a persistently floored min eigenvalue means the fallback compromise is large —
  inspect the offending observed block's support.

## Run

```
make exp ID=25
```

Compare the full Σ diagnostic trace head-to-head with exps 0021 (zero-pin baseline),
0023 (diag-shrink), and 0024 (both levers): the delta is the completion effect. The
non-gated exp 0020 (cond 13.3, no completion needed) is the well-conditioned
reference both ends should approach.
