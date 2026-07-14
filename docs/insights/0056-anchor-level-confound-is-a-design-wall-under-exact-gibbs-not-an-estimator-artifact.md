# Insight 0056 — The anchor-level confound is a DESIGN wall (exact-Gibbs posterior does not contract with N), not an estimator artifact; design-wall directions are wide-but-STATIONARY, so the read-out can report them UNRESOLVED without an export hole

**Date:** 2026-07-14
**Branch:** pg-stm
**Topic:** pg-stm | dag-offsets | identifiability | gibbs | design-wall | read-out
**Status:** Observed
**Relates to:** insight 0050 (only increments identified), 0052 (information / estimator /
DESIGN wall taxonomy), 0054 (background-only members flip the flag but anchor levels stay
intercept-confounded), 0055 (the compiler reads the confound off the closure Gram statically).
This is the empirical **arbiter** Fable's step-0 asked for: classify the anchor-level confound
as an estimator wall (more data fixes it) vs a design wall (no engine recovers it) BEFORE the
read-out engine's promises are written.

**Context:** The identifiability compiler (0055) reads increment-identifiability off the closure
Gram statically. Fable flagged that the model's offset coefficients are fit through a RIDGE over a
SHARED design moment (`XtX` shared across all K-1 stick columns) with a CO-SAMPLED latent `psi`, so
a static per-column rank argument is muddied — the empirical Gibbs contraction check is the arbiter.
Built a faithful full Gibbs sweep (the validated `pg_stm_gibbs` gated sweep with ONLY the Gamma
point-estimate replaced by a proper matrix-normal offset DRAW `C ~ MN(dag_offset_ridge mean, A^{-1},
Sigma)`, `A = W'W + diag(depth-scaled penalty)`, drawn AND fed back each sweep = proper full Gibbs)
and ran it on a K-reduced insight-0054 corpus (anchor A with direct docs + subtype A1; anchor B with
NO direct docs + subtype B1) across a 16x doc-count ladder, same attestation pattern.
(`scratchpad/design_wall_gibbs_probe.py`.)

**Findings — the anchor level is a design wall, decisively:**

1. **Only the increment contracts; every anchor LEVEL is flat.** Posterior SD across a 16x N-ladder
   (a 1/sqrt(N) estimator wall would shrink 4x):
   - `A1` increment (subtype offset over its parent): SD shrank **3.16x** -> contracts ~1/sqrt(N),
     IDENTIFIED.
   - anchor `A` level: **0.86x** (flat); `B` level (`= B[B]+B[B1]`): **0.85x** (flat); the B/B1 split
     (`= B[B]-B[B1]`, the closure-Gram null-space): **0.96x** (flat). None contract — **more data does
     not resolve them.**

2. **Why (analytic, exact):** the augmented design `[intercept, A, B, A1, B1]` on this attestation has
   **nullity 2**: (a) the partition identity `intercept == A + B` (every doc is A-group or B-group)
   confounds anchor LEVELS with the intercept; (b) `B1 col == B col` (no direct-B docs) confounds the
   B/B1 split. **Both null vectors load exactly zero on the A1-increment coordinate**, so the increment
   is the SOLE offset coordinate in the design's row space — which is precisely the only coordinate that
   contracted. The Gibbs result and the linear algebra agree to the coordinate.

3. **Design-wall directions are WIDE-BUT-STATIONARY (Fable outcome a, not b).** Split-chain
   `improved_rhat` (Vehtari 2021) on `||DIFF||` across 2 chains at the largest N = **1.01** under a
   proper prior (`lam_base=0.25`, SD 8.1) AND **1.00** under a near-flat prior (`lam_base=1e-3`, SD 130).
   No wandering (outcome b) under either. The width tracks the PRIOR almost exactly:
   `8.08 * sqrt(0.25/1e-3) = 128 ~ 130` — **the posterior along the design wall IS the prior**
   (data-independent, N-independent). That is the definitive design-wall signature.

**Consequences:**
- **Classification settled: the anchor-level confound is a design wall.** The exact Gibbs — which has
  no mean-field under-dispersion to blame — leaves the level directions exactly as wide as the prior,
  invariant to a 16x data increase. No estimator (Gibbs, structured VI, anything) recovers a design-wall
  direction; only a change of DESIGN (attest the missing node directly, or add a distinguishing
  covariate) can.
- **The static compiler is a valid pre-fit surrogate.** Gibbs confirms what `closure_gram` (the B/B1
  null) and `foreground_grams`-with-intercept (the level-vs-intercept collinearity) compute without a
  fit — compiler and gold-standard posterior agree.
- **Read-out engine (step 2) — acceptance must be PER-COORDINATE-CLASS.** Increment directions get a
  number + nominal coverage; anchor-level / null-space directions report **UNRESOLVED** (a wide
  interval), never a point. Because the design-wall posterior is stationary (not divergent), reporting
  its width is honest and safe — **no export hole**. This extends insight 0045's lesson (mean-field
  under-dispersion and exact-Gibbs WIDTH are two honest encodings of the same weak identification) from
  the Sigma read-out to the DAG offsets.
- **Reusable artifact:** the matrix-normal offset draw `C ~ MN(dag_offset_ridge, A^{-1}, Sigma)` built
  for this probe IS the step-2 read-out kernel; promoting it into the library (with the 0055
  reparameterization-penalty fix on merged nodes) is the read-out engine's first task.

**Does not claim:** anything about real data (synthetic, model-matched generator). The probe dropped the
background-only members (they are `z=0` on the offset block — cannot touch a null direction; insight 0054
covers their flag-flip role) and reduced K (the design wall is dimension-independent); both are
attestation-preserving simplifications, verified not to change the null-space. Whether a design-wall
direction is worth reporting as UNRESOLVED vs suppressing entirely is a reporting-layer choice (deferred),
not a core-engine claim.
