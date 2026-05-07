# ADR 0008 — Vanilla LDA design choices

**Status:** Accepted
**Date:** 2026-04-30

## Context

The framework needed a real multi-parameter VIModel exercising the contract
end-to-end against synthetic data with known ground truth, before the more
ambitious `OnlineHDP` work begins. Vanilla LDA fills that role and provides
a natural correctness oracle via Spark MLlib's reference implementation.

## Decisions

### Algorithm: Hoffman 2010 + Lee/Seung 2001

We implement Online LDA with stochastic variational inference (Hoffman et
al. 2010, 2013). The CAVI inner loop uses the Lee/Seung 2001 trick to avoid
materializing the full (K, n_unique_tokens) phi matrix per document; we
carry only gamma_d (K,) and an implicit phi_norm (n_unique,). Memory is
O(K + n_unique) per doc rather than O(K * n_unique).

This matches MLlib's `OnlineLDAOptimizer.variationalTopicInference`
implementation, which keeps comparison fair.

### Hyperparameter defaults match MLlib

`alpha = 1/K`, `eta = 1/K`, `gamma_shape = 100`, `cavi_max_iter = 100`,
`cavi_tol = 1e-3`. Aligning with `pyspark.ml.clustering.LDA` defaults means
any divergence in head-to-head results is attributable to actual algorithmic
differences, not parameter drift.

### Symmetric alpha only (deferred: asymmetric + optimizeDocConcentration)

MLlib supports asymmetric alpha and a Newton-Raphson update on it
(`optimizeDocConcentration`). Both are off by default. We defer both:
adding them would introduce a second-order optimization step that's a
meaningful complication for a v1 whose purpose is framework validation.
Listed as future work.

The case for *eventually* supporting asymmetric α on θ (paired with a
symmetric prior on β) is made canonically in Wallach, Mimno, McCallum
(NIPS 2009), ["Rethinking LDA: Why Priors
Matter"](https://mimno.infosci.cornell.edu/papers/NIPS2009_0929.pdf):
asymmetric θ-prior plus symmetric β-prior beats both-symmetric on
held-out perplexity and topic quality, while asymmetric β-prior offers
no real benefit. They learn α from data via empirical Bayes; the
simulator now supports a related but distinct path — feeding in a
fixed asymmetric base measure from external metadata (the upstream
HF-dataset's per-topic usage values) — for generating realistic
long-tailed synthetic corpora. See `scripts/simulate_lda_omop.py`.

### `BOWDocument` as canonical row type

A small frozen dataclass at `spark_vi.core.types.BOWDocument` carrying
`(indices: int32[], counts: float64[], length: int)`. Lives in `core/`
rather than `models/lda.py` because it's the natural row type for any
topic-style model (HDP will reuse it). Naming describes the *data shape*
("bag of words"), not an intended use.

### MLlib parity expectations

We do **not** expect bit-exact agreement. Different RNG, different
convergence cutoffs, different float precision in places. The agreement
gate is prevalence-aligned topic similarity (mean diagonal JS divergence
after sorting both by topic prevalence), not numerical equality. See
`RISKS_AND_MITIGATIONS.md`.

## Relation to prior ADRs

- [ADR 0005](0005-mini-batch-sampling.md) — already aligned mini-batch
  scaling to MLlib's `corpus_size / batch_size` convention; this ADR
  inherits that decision unchanged.
- [ADR 0006](0006-unified-persistence-format.md) — `VIResult` as canonical
  state; LDA stores its vocab map in `metadata["vocab"]` per documented
  driver convention, no `VIResult` schema changes needed.
- [ADR 0007](0007-vimodel-inference-capability.md) — `VanillaLDA.infer_local`
  is the first non-trivial implementation of the new optional capability.

## Lessons learned during implementation

### The `expElogbeta` factor in `update_global`

Initial implementation of `update_global` aggregated only the per-doc
`expElogthetad` factor in `local_update` and applied a Robbins-Monro step
without re-multiplying by `expElogbeta`. This produced lambda sums an
order of magnitude higher than expected and topic recovery JS divergence
~0.25 nats — symptoms that looked like generic small-corpus seed-fragility
but were actually a math bug.

The CAVI implicit-phi parameterization is `phi_dnk ∝ expElogthetad[k] *
expElogbeta[k, w_dn]`. The doc-side accumulation in `local_update`
captures the first factor; the second must be re-introduced when the
aggregated sufficient statistic is used to push lambda toward its target.
MLlib's `OnlineLDAOptimizer.submitMiniBatch` does this with a single
post-aggregation `*:* expElogbeta.t`; we now match that.

The MLlib parity test (Task 15) is the rigorous gate that catches
regressions of this kind: with matched hyperparameters the diagonal mean
JS divergence runs ~0.01 nats, so any factor-of-N mismatch on either side
is a strong, unambiguous signal.

## Future work

- Asymmetric alpha + `optimizeDocConcentration`.
- Per-iteration ELBO trace for MLlib (refit at growing maxIter).
- Notebook tutorial walking through the implementation alongside the math.
- The real `OnlineHDP` (this spec is the warm-up).
