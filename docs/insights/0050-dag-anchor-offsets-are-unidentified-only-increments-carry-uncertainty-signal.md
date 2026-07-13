# Insight 0050 — DAG anchor offsets are un-identified under a partitioning gate; only parent→child increments carry a sample-size signal

**Date:** 2026-07-13
**Branch:** pg-stm
**Topic:** pg-stm | dag-offsets | identifiability | diagnostics
**Status:** Observed
**Relates to:** the DAG/ontology PG-STM model core v1 (Task 6, offset-interval coverage); the Σ read-out identifiability thread (insights 0044 / 0047 / 0048).

**Context:** Building the DAG/ontology PG-STM model core (v1, additive-η mean-offsets). Test 4
was written to check that the ridge-posterior offset intervals are calibrated (~90% coverage)
and that a data-scarce node's intervals are *wider* than a well-populated node's — the
"ship-the-posterior + honest-uncertainty" deliverable. The first construction compared the two
**anchor** offsets (one populated at 600 docs, one scarce at 60) and failed decisively:
coverage 1.00 (band 0.80 to 0.98) and scarce/populated width ratio exactly 1.00 (needed > 1.5).

**Finding:** The failure is a real identifiability property, not a bug. In the additive model
`μ_d = Γᵀx_d + Σ_{u∈closure(v_d)} B[u]`, the individual **anchor** offsets are collinear with
the covariate intercept. Under a *partitioning* gate — every document belongs to exactly one
anchor group — the anchor closure-indicator columns satisfy `col_root ≡ col_A + col_B` for
every document, and the intercept (`x=[1]`) equals `col_root`. This is the classic
dummy-variable trap: among {intercept, root, anchor_A, anchor_B} there are two exact
redundancies. The ridge splits the point estimate arbitrarily, but the **per-node diagonal**
of the ridge posterior covariance `σ²·(WᵀW+diag(penalty))⁻¹` is then dominated by the shared
*unidentified* direction, so a 600-doc anchor and a 60-doc anchor receive numerically identical
interval widths (measured `Ainv` diagonals ≈ 249.94 for both, despite `XtX` diagonals of 600 vs
60). Coverage inflates to 1.00 because those intervals are uniformly too wide.

What **is** identified is each node's **increment relative to its parent** — a subtype offset
distinguished by that node's *own* documents against anchor-only documents. Provided the corpus
contains anchor-only docs (so `col_anchor ≠ col_subtype1 + col_subtype2`), the subtype columns
escape the trap and their posterior variance scales with each subtype's own document count.
Re-targeting Test 4 at a populated-vs-scarce **subtype** pair under one anchor makes the
scarce→wider property real and coverage meaningful.

**Consequences:**
- The honest uncertainty read-out for the DAG offset layer is defined on **increments**
  (parent→child specialization), which is exactly what the additive-over-closure model
  parameterizes — the individual anchor levels are not a reportable quantity on their own.
- Task 4's earlier subtype-offset recovery (r≈0.76 through a 2-level closure) already relied on
  this: the subtype increment is identified because subtype docs are distinguished from
  anchor-only docs.
- This is the mean-offset (Piece A) analogue of the recurring theme that only *contrasts* /
  identified directions are reportable — cf. the Σ read-out thread ([[project_pg_fullbayes_engine]]),
  where mean-field attenuation and label-switching likewise make only certain functionals
  trustworthy.

**Does not claim:** anything about real-data coverage (unmeasurable — Task 7's real-data
spurious-edge shrinkage check is the transfer-side guard), nor that the ridge-conditional
posterior at the VI ψ-mean is exact (it omits ψ uncertainty — a stated approximation).
