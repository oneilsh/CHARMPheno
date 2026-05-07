# ADR 0012 — MLlib Estimator/Transformer shim for OnlineHDP

**Status:** Accepted
**Date:** 2026-05-07
**Related:** ADR 0009 (LDA MLlib shim), ADR 0011 (Online HDP v1 design).

## Context

ADR 0011 deferred the `OnlineHDP` MLlib shim and cloud driver scripts to
v2 ("Following the LDA precedent (ADR 0009 added the shim *after* the
model was built and validated)"). The model has now landed on `main`.
We want the same MLlib-shaped Estimator/Model surface for HDP that
ADR 0009 gave us for LDA so that:

- Pipelines can compose CountVectorizer → OnlineHDPEstimator the same
  way they compose CountVectorizer → VanillaLDAEstimator.
- The cloud driver `analysis/cloud/hdp_bigquery_cloud.py` mirrors
  `lda_bigquery_cloud.py` instead of inventing its own runner-based shape.
- The "second data point" question from ADR 0009 (does the shim shape
  generalize beyond LDA?) finally gets resolved.

## Decisions

### Subclass `pyspark.ml.base.{Estimator, Model}`, mirror ADR 0009 structure

`OnlineHDPEstimator` / `OnlineHDPModel` parallel `VanillaLDAEstimator` /
`VanillaLDAModel`. Same Params-mixin pattern (`_OnlineHDPParams`), same
Vector-column in/Vector-column out contract, same UDF-based `_transform`,
same persistence deferral. We keep the two shims as siblings rather than
introducing a generic `VIModel → MLlib` adapter — the abstraction would
be premature with two concrete data points and the constructor surfaces
genuinely diverge (HDP has T, K, γ; LDA has K, η).

### Param naming: `k` = corpus truncation T; `docTruncation` = doc truncation K

MLlib's `pyspark.ml.clustering.LDA` exposes `k` as "number of topics".
For HDP the analogous knob is the corpus truncation T (upper bound on
topics the model can discover), so we put T on `k`. The doc-level
truncation K has no MLlib analog — we expose it as a new camelCase
extra `docTruncation`. Both default values stay in OnlineHDP's
constructor (T=150 via the wrapper convention; K=15).

Naming pitfall already recorded in `project_hdp_truncation_naming` memory:
the Wang paper uses K=corpus / T=doc, but Wang's reference code and the
intel-spark Scala port both invert it (T=corpus / K=doc). CHARMPheno
follows the code convention. The shim's `k`-as-corpus-truncation choice
is consistent with that, not the paper notation.

### Param naming: `corpusConcentration` for γ (new)

HDP has *two* concentration scalars: α on the doc-level stick (matches
LDA's `docConcentration`) and γ on the corpus-level stick (no LDA
analog). We add a new camelCase extra `corpusConcentration` for γ.
`docConcentration` stays scalar-only (length-1 list per
`TypeConverters.toListFloat`); rejecting vector α is the only validator
rejection on this knob in v1, since ADR 0011 punts both γ and α
optimization to v3.

### No `optimizeDocConcentration` / `optimizeTopicConcentration` / `optimizeCorpusConcentration` flags

ADR 0011 deferred all γ/α optimization. The shim does not surface those
flags — adding them as `False`-only Params would be misleading. They
land in a follow-up ADR + spec along with the underlying Newton machinery.
This is the one substantive surface divergence from ADR 0009, where the
LDA shim *does* carry `optimizeDocConcentration` because the underlying
optimization landed in ADR 0010.

### Reject unsupported configurations explicitly

Three rejections at fit time, mirroring ADR 0009:

- `optimizer != "online"` (we are SVI-only, Wang/Paisley/Blei 2011 stochastic VI).
- Vector `docConcentration` (HDP α is scalar; ADR 0011 doesn't optimize it).
- Vector `topicConcentration` (HDP η is scalar symmetric Dirichlet on the topic-word prior).

Silent fallback would mislead about what users are getting.

### Model surface: `topicsMatrix`, `describeTopics`, plus HDP-specific accessors

`topicsMatrix()` returns `DenseMatrix(V, T)` — same orientation as
MLlib LDA's `topicsMatrix`, indexed by (vocab term, topic). Returns
the *full* T topics, not active-only — filtering inactive topics is a
caller decision. `describeTopics(maxTermsPerTopic)` matches MLlib's
DataFrame schema `(topic, termIndices, termWeights)`.

HDP-specific accessors that have no LDA analog:

- `corpusStickWeights() → np.ndarray (T,)`: the `E[β_t]` vector under
  the mean-field variational posterior, derived from the `(u, v)`
  Beta factors. Exact for the variational mean (independent factors
  let expectation distribute through the stick-breaking product).
  Surfaces the effective topic prior so callers can rank/filter active
  topics.
- `activeTopicCount(mass_threshold=0.95) → int`: smallest count of
  topics whose top-ranked `E[β_t]` sum to ≥ `mass_threshold`. Default
  0.95 — PCA's "explained-variance" analog. Truncation-independent
  (same answer for any T ≥ effective topic count), in contrast to a
  fixed threshold like `1/(2T)` which scales with the truncation knob.
  `OnlineHDP.iteration_summary` uses the same parameterized definition,
  so live-training and post-fit "active" reports agree.

Trained-scalar accessors are exposed as **methods** (`trainedAlpha()`,
`trainedCorpusConcentration()`, `trainedTopicConcentration()`) rather
than properties, for two reasons:

1. The MLlib idiom: `pyspark.ml.clustering.LDAModel` exposes
   `docConcentration()` / `topicConcentration()` as methods.
2. A `@property` whose name collides with a Param descriptor breaks
   `_set` / `_setDefault` resolution: when `Estimator._fit` copies Param
   values to the Model with `out_model._set(topicConcentration=val)`,
   PySpark internally does `getattr(self, "topicConcentration")`
   expecting a Param, but the @property would shadow the descriptor and
   return the trained scalar instead. The LDA shim avoids this only
   because no LDA test path explicitly sets the colliding Params; HDP
   v1 callers may legitimately set `topicConcentration` directly. The
   "trained" prefix sidesteps the collision entirely. (V1 returns
   constructor inputs unchanged — no optimization per ADR 0011 — so
   a future v3 that adds optimization can keep these method names.)

### `_transform` UDF returns θ length-T

The Model's `transform` applies a UDF over the Vector column that runs
`OnlineHDP.infer_local`, returning the per-doc θ as a `DenseVector` of
length T (corpus topics). This matches the LDA shim's "topic distribution
column is length-K" contract: the user-facing dimension is T because
the doc-level K is a latent-factor truncation that doesn't survive
projection to corpus topics. Broadcast unpersist in `finally`, same
discipline as the LDA shim's `_transform`.

### Persistence (`MLReadable` / `MLWritable`) deferred — same as ADR 0009

The driving v2 use case is comparison and Pipeline ergonomics, not
`Pipeline.save()`. Until a concrete user materializes, users persist
via `VIResult.export_zip` and reconstruct.

### `logLikelihood` / `logPerplexity` stubbed — same as ADR 0009

Held-out perplexity for variational HDP requires deriving a held-out
ELBO bound; non-trivial and there is no concrete user. Stubs raise
`NotImplementedError` and point to `VIResult.elbo_trace`. Per the
post-merge conversation around ADR 0011, perplexity is on the v3
roadmap as its own branch covering both LDA and HDP, alongside
coherence metrics.

## Relation to prior ADRs

- [ADR 0009](0009-mllib-shim.md) — the LDA shim that this one parallels
  in shape. The "generic VIModel adapter" question deferred there gets a
  clear answer here: keep the two shims as siblings; the MLlib surface
  difference (HDP carries T, K, γ; LDA carries K, η + optimize flags)
  is large enough that a generic adapter would only abstract over the
  small mechanical pieces.
- [ADR 0011](0011-online-hdp-design.md) — "Defer MLlib shim and driver
  scripts to v2" is what this ADR closes out.

## Future work

- `MLReadable` / `MLWritable` when a concrete user wants `Pipeline.save()`.
- γ / α optimization (paired ADR with the Newton machinery), at which
  point `optimizeDocConcentration`, `optimizeCorpusConcentration` flags
  become first-class on the shim.
- Held-out `logLikelihood` / `logPerplexity` (own branch, both LDA and HDP).
