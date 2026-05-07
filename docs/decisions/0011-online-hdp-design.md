# ADR 0011 — Online HDP v1 Design Decisions

**Status:** Accepted
**Date:** 2026-05-06
**Related:** ADR 0007 (VIModel inference capability), ADR 0008 (Vanilla LDA),
ADR 0009 (MLlib shim), ADR 0010 (concentration-parameter optimization).

## Context

The OnlineHDP port of Wang/Paisley/Blei 2011 to the spark_vi framework
involves several scope decisions that don't follow obviously from the
algorithmic reference. Recording them here so future revisits know what
was deliberate vs. accidental.

## Decisions

### Skip the lazy-lambda sparse-vocabulary update in v1

Wang's reference uses per-word timestamps + a running cumulative log-discount
to amortize natural-gradient (1−ρ) shrinkage across vocabulary words touched
by a minibatch. At our clinical scale (V ≈ 5-10k concept_ids, much smaller
than NLP V≈100k) full-V digamma per minibatch is cheap, and skipping the
lazy update simplifies the distributed E-step considerably. Revisit only
if profiling shows the full-V digamma is the bottleneck.

### Hold γ and α fixed at user-set values

HDP concentration parameters are part of the model's core appeal — γ
controls how many topics get discovered, α controls how many topics each
doc uses. Optimizing them is a real win, especially for γ. But the math
is its own piece of work, and ADR 0010 already templates the Newton
machinery on LDA. Punt to a follow-on ADR + spec pair after v1 lands.

### No in-loop optimal_ordering

Wang's reference re-sorts topics by descending λ row-sum after every
M-step. Useful for visualization, breaks reproducible topic indices
across runs, and not required by the algorithm. VanillaLDA does not do
this either. If we want it later, it should be a post-fit
`reorder_by_usage()` helper, not an in-loop side-effect.

### Real frozen-globals HDP doc-CAVI for `transform()`, not LDA-collapse

Wang's reference exposes `infer_only` which collapses the trained HDP into
an LDA-equivalent (computing α from sticks, treating λ as the LDA topic-word
matrix) and runs ordinary LDA E-step. We deliberately implement the full
HDP doc-CAVI for `transform()` instead. Reasons:

1. **Held-out evaluation accuracy.** LDA-collapse loses the doc-stick
   structure (the per-doc π_jt, c_jt latent variables collapse into a flat
   Dirichlet prior). Real HDP CAVI gives the actual variational posterior
   under the real model.
2. **Future patient-train / visit-infer split.** That enhancement
   (TOPIC_STATE_MODELING.md L507-523) requires real frozen-globals HDP
   inference at visit granularity; the LDA collapse can't represent it.
3. **Predictive modeling and on-device serving** (eventual goal) need the
   full posterior, not a derivative.

We don't need LDA-shaped HDP outputs, so the LDA-collapse helper is
dropped from scope entirely (not even deferred).

### Keep the iter < 3 warmup trick as default

Wang's reference drops the prior-correction terms (E[log β], E[log π]) in
the var_phi / phi updates for the first three iterations of doc-CAVI.
This is undocumented in the AISTATS paper but preserved in both Wang's
Python and intel-spark's Scala port. We keep it; v2 will run an ablation
(`warmup_iters=0`) to check whether it earns its keep.

### Match-LDA Gamma init for λ; paper-following init for corpus sticks

`λ` initializes via `Gamma(gamma_shape=100, scale=1/100)`, matching
VanillaLDA. Departs from Wang's reference (`Gamma(1,1) · D · 100 / (T·V) − η`)
which is undocumented and not derived. Match-LDA is the boring validated
choice.

`(u, v)` initializes at the prior mean: `u = 1`, `v = γ`. Departs from
Wang's reference (`v = [T-1, ..., 1]` as "make a uniform at beginning")
which is an undocumented bias toward low topic indices.

### doc_z_term ELBO calculation includes count-weighting (deviation from Wang)

The doc-level Z-term entropy + cross-entropy must sum over word *tokens*,
not unique word types. Since `phi[w, k]` stores per-unique-word probabilities,
the sum requires a `* counts[:, None]` factor. Wang's reference Python
omits this factor — a bug shared with the intel-spark Scala port. Our
per-iter doc-ELBO monotonicity test (`test_doc_e_step_per_iter_elbo_nondecreasing`)
caught this during Task 5 implementation; the fix is documented inline at
`_doc_e_step`'s ELBO calculation block. Without the fix, the (a, b) update
(which uses count-weighted phi_sum) maximizes a different objective than
the reported ELBO, causing post-convergence drift.

### Defer MLlib shim and driver scripts to v2

Following the LDA precedent (ADR 0009 added the shim *after* the model
was built and validated), the `spark_vi.mllib.HDP` shim plus
`analysis/local/` and `analysis/cloud/` driver scripts are out of v1
scope. They land in their own ADR + spec once the inner model is unit-
and integration-tested. The second-data-point hypothesis from ADR 0009
(does the LDA shim shape generalize to HDP?) gets resolved at that
time.

## Consequences

- v1 ships a focused, testable model with a clear scope. No half-baked
  optimization knobs, no MLlib-specific machinery to debug alongside the
  math.
- The framework's second concrete model lands; the LDA → HDP
  generalization can inform v2 framework refactors if needed.
- A clear v2 roadmap exists: shim + drivers + γ/α optimization + lazy
  lambda update + warmup ablation + held-out perplexity track. Each is
  its own ADR + spec.

## References

- Wang, Paisley, Blei (2011), "Online Variational Inference for the
  Hierarchical Dirichlet Process," AISTATS.
