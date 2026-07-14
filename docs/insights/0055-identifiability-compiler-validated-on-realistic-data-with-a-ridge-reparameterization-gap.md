# Insight 0055 — The identifiability compiler is validated on realistic data (rides on the fit's exact moment; fragility spectrum is quantitative); the one seam is a depth-scaled-ridge reparameterization gap

**Date:** 2026-07-14
**Branch:** pg-stm
**Topic:** pg-stm | dag-offsets | identifiability | compiler | validation
**Status:** Observed
**Relates to:** the identifiability compiler (spec/plan 2026-07-14, module `dag_identify.py`);
insight 0050 (only increments identified), 0052 (information/estimator/design walls), 0054
(background-only members flip the flag but anchor levels stay confounded; no-branching nodes
collapsible). This is the compiler turning 0050/0054 from per-fit surprises into a deterministic
pre-fit computation.

**Context:** After building the v1 identifiability compiler (reads increment-identifiability off the
closure-indicator Gram `G = sum_d z_d z_d^T` and rewrites the DAG to its identified quotient), it had
been exercised only on tiny hand-built attestation patterns and one trivial identity fit. Ran a set of
validation + adversarial experiments on realistic-overlap DAG-offset corpora (including the exact
insight-0054 corpus: anchor A with direct docs + subtype, anchor B with NO direct docs + subtype B1,
background-only members).

**Findings — the core does what the design hypothesized, quantitatively:**

1. **"If you can fit, you can compile" is literal.** `closure_gram(dag, doc_nodes)` equals the offset
   block of the augmented design moment the fit accumulates (`w = [x ; offset_indicator]`,
   `XtX[P:, P:]`) to machine zero. The compiler adds no new moment — it reads the one the fit already
   forms.
2. **The compiler reproduces the 0054 collapse deterministically.** On the insight-0054 corpus it
   auto-collapses the no-direct-docs anchor B into its sole subtype B1 (`z_B == z_B1`, `node_map[B]
   == node_map[B1]`), keeps the identified A/A1 distinction (A has direct docs -> `d(A,A1) = 400`),
   reports zero flagged residual, and the correctness invariant holds exactly. What took a controller
   isolation-probe to diagnose by hand, the compiler now emits from a count reduce.
3. **The fragility spectrum is quantitative.** As an anchor gains own (distinguishing) documents, the
   column-equality metric `d(anchor, subtype)` equals exactly that count; the collapse decision flips
   at a strict `d < tol`; and the smallest eigenvalue of the (anchor, subtype) block grows
   monotonically (own-docs 0/5/25/100 -> min-eig 0.0/2.5/12.3/47.5). The small-but-nonzero eigenvalues
   are a real, monotone measure of identification strength -- the raw material for the deferred
   reporting layer's "fragile" tier, and (later) a cheap pre-fit surrogate for the read-out engine's
   interval widths.
4. **The safety split holds under adversarial attestation.** Two NON-adjacent subtypes under different
   anchors that are only ever co-attested (identical support columns, a many-to-many pattern) are
   FLAGGED (`flagged_dim >= 1`), NOT merged -- the exact spurious sibling/cross-structural merge the
   design set out to prevent. Deep multi-level no-docs chains collapse whole; the `d < tol` boundary
   has no off-by-one; mixed collapse+keep+flag composes.

**The one seam (a real, small finding):** the quotient's merged node carries the identified sum only
**up to a depth-scaled-ridge reparameterization gap**, not to machine precision. Fitting the original
DAG splits the un-identified direction arbitrarily (B_B, B_B1); fitting the quotient yields one merged
node whose offset equals their vector sum in DIRECTION (correlation ~1) but not exactly, because the
depth-scaled ridge penalty differs between the split chain (a penalty on each of two nodes) and the
merged node (one penalty). Under the light `lam_base = 1e-3` the gap is tiny (< 1e-2 on a ~1.5-norm
signal), but it is nonzero and systematic.

**Consequences:**
- The compiler is a validated pre-fit stage: it preserves everything identified, removes the arbitrary
  un-identified split, and refuses every spurious merge, on realistic synthetic data.
- The reparameterization gap matters only if a downstream consumer needs the merged offset to equal the
  EXACT pre-collapse sum (e.g., the step-2 read-out engine reporting a merged node's increment against
  a target). The fix is local: set the merged node's penalty to match the chain it replaces (sum the
  depth-scaled penalties, or use the min-depth node's penalty), so quotient-then-fit == fit-then-sum on
  the penalty too, not just on the design moment. Worth a note when the read-out engine lands.

**Does not claim:** anything about real-data recovery or transfer (synthetic proves math-correctness
only); nor that the merged node's offset is itself recoverable (that inherits the 0054 design-wall on
absolute levels) -- only that the compiler faithfully carries whatever the original fit put on the
un-identified direction, up to the ridge reparameterization.
