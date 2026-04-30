# Vanilla LDA — Design Spec

**Date:** 2026-04-30
**Status:** Draft, pending user review
**Scope:** Implementation of vanilla Latent Dirichlet Allocation as a `VIModel` in the `spark_vi` framework, an evaluation harness in `charmpheno/evaluate/` that produces a head-to-head comparison against Spark MLlib's reference `LDA` implementation, and the supporting framework extensions (optional `infer_local` capability, `BOWDocument` row type) needed to make those work cleanly. Does **not** cover the real `OnlineHDP` implementation, the temporal state model, asymmetric-α optimization, or MLlib `Estimator/Transformer` adoption — those are explicitly deferred.

---

## Context

The repo's `spark_vi` framework currently has one fully-implemented `VIModel` — the `CountingModel` (Beta-Bernoulli) used as a proof-of-life. The `OnlineHDP` exists only as a stub with `NotImplementedError` bodies. The framework's contract methods, mini-batch SVI machinery, broadcast lifecycle, persistence format, and test patterns are all working and battle-tested via the bootstrap walkthrough.

What's missing is a *real* multi-parameter probabilistic model running end-to-end through that framework, exercised against synthetic data with known ground truth. Vanilla LDA fills that gap before the more ambitious HDP work begins. It also provides a natural correctness oracle: Spark MLlib ships a reference Online LDA implementation, and a head-to-head comparison on the same data tells us whether discrepancies (if any) live in our framework, our model, or the data pipeline — three distinct failure modes that HDP's discovered-K complexity would otherwise conflate.

The synthetic data pipeline (`scripts/simulate_lda_omop.py`) already produces OMOP-shaped parquet from a fixed β with per-token `true_topic_id` oracle. The empty `charmpheno/evaluate/` namespace is staged for exactly this kind of recovery-validation machinery.

This spec was preceded by a brainstorming session that settled the major design forks. Open threads not addressed here are documented in the [Future Work](#future-work) section.

## Goals

1. **Ship a real multi-parameter `VIModel`.** `VanillaLDA` implementing Hoffman 2010 Online LDA with the Lee/Seung 2001 trick to avoid materializing φ.
2. **Add inference as a first-class framework capability.** Optional `infer_local(row, global_params)` hook on `VIModel`, plus `VIRunner.transform(rdd, global_params)`. Pure function of inputs — no instance-state dependence.
3. **Validate against a reference implementation.** Comparison harness that runs both ours and `pyspark.ml.clustering.LDA` on the same prepared input and produces a three-panel JS-similarity biplot (ours vs. truth, MLlib vs. truth, ours vs. MLlib), each prevalence-ordered.
4. **Preserve the one-way dependency invariant.** `charmpheno → spark_vi`, never reverse. The LDA model and `BOWDocument` type stay in `spark_vi`; data prep and clinical evaluation stay in `charmpheno/`.
5. **Don't paint into MLlib-Estimator corners.** All design choices keep a future MLlib `Estimator/Transformer` shim mechanical — specifically by enforcing `infer_local` purity.
6. **Match MLlib's defaults for fair comparison.** Hyperparameter defaults, CAVI convergence threshold, and mini-batch math align with `OnlineLDAOptimizer` so any divergence in results is attributable to actual algorithmic differences, not parameter drift.

## Non-goals

- Asymmetric document concentration α and `optimizeDocConcentration` Newton-Raphson update. Out of scope for v1; deferred to future work.
- Real `OnlineHDP` implementation. This spec is the warm-up; HDP is the next major target.
- Temporal state model (Sparse OU on top of topic mixtures). Architecture-doc territory; orthogonal to this spec.
- MLlib `Estimator/Transformer` / `Pipeline` adoption. The compat path is left clean but unused.
- Per-iteration ELBO trace from MLlib. Spark's `OnlineLDAOptimizer` doesn't expose one; we record what's available and document the gap.
- Math curriculum / derivation document. The user already has a working understanding of SVI; the implementation is the focus, not pedagogy.
- Bit-exact agreement with MLlib. Different RNG, different convergence cutoffs, different float precision in places. The agreement gate is prevalence-aligned topic similarity, not numerical equality.

---

## Architecture overview

Three deliverables across two packages and one analysis directory:

**`spark_vi/` (framework changes):**
- `spark_vi/models/lda.py` — new `VanillaLDA(VIModel)` class.
- `spark_vi/core/types.py` (new) — `BOWDocument` dataclass.
- `spark_vi/core/model.py` — extend `VIModel` with optional `infer_local` capability.
- `spark_vi/core/runner.py` — add `VIRunner.transform`.

**`charmpheno/` (clinical data prep + evaluation):**
- `charmpheno/omop/local.py` (extend) or new sibling — `to_bow_dataframe` function.
- `charmpheno/evaluate/lda_compare.py` (new) — pure-function orchestration of both implementations.
- `charmpheno/evaluate/topic_alignment.py` (new) — JS-similarity, prevalence ordering, biplot data preparation.

**`analysis/local/` (drivers):**
- `analysis/local/fit_lda_local.py` — fit `VanillaLDA` end-to-end.
- `analysis/local/compare_lda_local.py` — orchestrate the comparison and produce figures.

**`docs/` (records):**
- ADR 0007 — `VIModel` inference capability.
- ADR 0008 — Vanilla LDA design choices.
- Updates to `SPARK_VI_FRAMEWORK.md` and `RISKS_AND_MITIGATIONS.md`.

Tests added at every layer; ladder detailed below.

---

## Component 1 — `VIModel` inference capability

### Contract extension

`spark_vi/core/model.py`: add an optional `infer_local` method to the `VIModel` ABC. No base implementation — it raises `NotImplementedError` with a model-named message:

```python
def infer_local(self, row, global_params):
    """Optional capability: per-row variational posterior given fixed global params.

    Models with local latent variables (LDA, HDP) override this. Models without
    (CountingModel) leave it unimplemented.

    MUST be a pure function of (row, global_params) — no dependence on instance
    state from training. This invariant keeps a future MLlib Estimator/Transformer
    shim mechanical.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement local inference. "
        f"Models without per-row latent variables cannot be used with VIRunner.transform()."
    )
```

Default-to-error rather than default-to-NaN/None: silent fallback masks a real user error ("I called transform on a model that can't do inference"). The error message names the concrete subclass for debuggability.

### `VIRunner.transform`

`spark_vi/core/runner.py`: add a new method paralleling `fit` but doing only the per-row inference pass — no reduce, no global update, no checkpoint.

```python
def transform(self, rdd, global_params):
    """Apply trained global params to infer per-row posteriors.

    Returns RDD of whatever `model.infer_local` returns. One pass, no reduce.
    """
    bc = self.spark_context.broadcast(global_params)
    try:
        model = self.model
        return rdd.map(lambda row: model.infer_local(row, bc.value))
    finally:
        bc.unpersist(blocking=False)
```

Broadcast lifecycle parallels `fit`. Eager `unpersist` matches the discipline established in the existing `test_broadcast_lifecycle.py`.

### MLlib-compat invariant

Encoded in the docstring: `infer_local` is a pure function of `(row, global_params)`. The model object holds *only* hyperparameters; trained state lives in `global_params` (a dict carried in `VIResult`). This invariant means a future `MLlibVanillaLDAModel(Transformer)` can hold a captured `global_params` reference and delegate row-by-row to `infer_local` without any framework change.

---

## Component 2 — `BOWDocument` row type

### Dataclass

`spark_vi/core/types.py` (new module):

```python
@dataclass(frozen=True, slots=True)
class BOWDocument:
    """Bag-of-words document: sparse-vector representation for topic-style models.

    Invariants:
      indices is sorted, all values in [0, vocab_size).
      counts has the same length as indices, all values > 0.
      length == counts.sum() — total tokens (with repeats), pre-computed.
    """
    indices: np.ndarray  # int32, shape (n_unique,)
    counts: np.ndarray   # float64, shape (n_unique,)
    length: int

    @classmethod
    def from_spark_row(cls, row, features_col: str = "features") -> "BOWDocument":
        """Construct from a DataFrame row whose `features` column is a SparseVector."""
        sv = row[features_col]
        return cls(
            indices=np.asarray(sv.indices, dtype=np.int32),
            counts=np.asarray(sv.values, dtype=np.float64),
            length=int(sv.values.sum()),
        )
```

`frozen=True` for hash/serialization safety across Spark partitions; `slots=True` for memory efficiency at scale (millions of patients).

The type lives in `spark_vi/core/types.py` rather than `spark_vi/models/lda.py` because it's the canonical row type for any topic-style model — `OnlineHDP` will reuse it. Naming `BOWDocument` (rather than `LDADocument` or `TopicDocument`) describes the *data shape*, not an intended use.

### Data flow

One shared prep path produces both an MLlib-ready DataFrame and an RDD of `BOWDocument`s, guaranteeing both LDA implementations consume byte-identical input:

```
parquet (4-col OMOP)
       │
       ▼
charmpheno.omop.load_omop_parquet  →  DataFrame(person_id, visit_occurrence_id, concept_id, concept_name)
       │
       ▼
charmpheno.omop.to_bow_dataframe   →  (DataFrame(person_id, features: SparseVector), vocab_map: dict[int, int])
       │
       ├── (MLlib)  pyspark.ml.clustering.LDA().fit(df) → LDAModel
       │
       └── (ours)   df.rdd.map(BOWDocument.from_spark_row) → RDD[BOWDocument]
                    → VIRunner(VanillaLDA(K, V, ...)).fit(rdd, config) → VIResult
```

### `to_bow_dataframe`

New function in `charmpheno/omop/` (sibling of `load_omop_parquet`):

```python
def to_bow_dataframe(
    df: DataFrame,
    doc_col: str = "person_id",
    token_col: str = "concept_id",
) -> tuple[DataFrame, dict[int, int]]:
    """Group rows into bag-of-words documents and build a contiguous vocab map.

    Returns:
      bow_df: DataFrame[doc_col, features: SparseVector]
      vocab_map: dict[concept_id, idx]  where idx in [0, V)
    """
```

Internals:
1. `df.groupBy(doc_col).agg(F.collect_list(token_col).alias("tokens"))`.
2. `pyspark.ml.feature.CountVectorizer(inputCol="tokens", outputCol="features")` — fit once.
3. Extract `vocab_map` from the fitted `CountVectorizerModel.vocabulary` (a list where position equals index).

Using Spark's `CountVectorizer` rather than rolling our own gives us battle-tested behavior (empty-doc handling, deterministic vocab ordering, `SparseVector` interop with MLlib LDA) for free.

Lives in `charmpheno/omop/` because it's a *loader-family* function — it transforms an OMOP-shaped DataFrame into a different DataFrame shape, the same way `load_omop_parquet` transforms a parquet path into an OMOP DataFrame.

### Vocab persistence

`VIResult.metadata` is a free-form dict (per ADR 0006). Convention enforced by drivers: `metadata["vocab"] = list[int]` ordered such that `vocab[idx] = concept_id`. This makes λ interpretable across save/load round trips without baking data-shape knowledge into `spark_vi`. No changes to `VIResult` are needed.

---

## Component 3 — `VanillaLDA` model

### Generative model

For each document d (= patient) in a corpus of D docs:
- θ_d ~ Dirichlet(α · 1_K)
- For each token n in 1..N_d:
  - z_dn ~ Categorical(θ_d)
  - w_dn ~ Categorical(β_{z_dn})

Globally, for each topic k in 1..K:
- β_k ~ Dirichlet(η · 1_V)

Variational mean-field approximation:
- q(β_k) = Dirichlet(λ_k)               # global; λ has shape (K, V)
- q(θ_d) = Dirichlet(γ_d)               # local; γ_d has shape (K,)
- q(z_dn) = Categorical(φ_dn)           # local; collapsed via Lee/Seung trick

### Hyperparameters

```python
VanillaLDA(
    K: int,                          # number of topics (required)
    vocab_size: int,                 # V (required)
    alpha: float | None = None,      # symmetric doc concentration; default 1/K
    eta: float | None = None,        # topic concentration; default 1/K
    gamma_shape: float = 100.0,      # init Gamma shape for λ
    cavi_max_iter: int = 100,        # per-doc CAVI iteration cap
    cavi_tol: float = 1e-3,          # per-doc convergence threshold (matches MLlib)
)
```

Defaults match `pyspark.ml.clustering.LDA` and `OnlineLDAOptimizer` for fair head-to-head comparison. Symmetric α only — asymmetric α and `optimizeDocConcentration` are deferred (see Non-goals).

`VIConfig` already covers: `corpus_size`, `n_iterations`, `mini_batch_fraction`, `sample_with_replacement`, `random_seed`, `learning_rate_tau` (τ_0), `learning_rate_kappa` (κ), `checkpoint_dir`, `checkpoint_interval`. No additions needed.

### Contract method implementations

**`initialize_global(data_summary=None) → {"lambda": (K, V) ndarray}`**
λ_init drawn from `Gamma(shape=gamma_shape, scale=1/gamma_shape)`. Seeded via `VIConfig.random_seed`. `data_summary` is unused but accepted for contract conformance.

**`local_update(rows, global_params) → {"lambda_stats": (K, V) ndarray, "doc_loglik_sum": float, "doc_theta_kl_sum": float, "n_docs": int}`**
Per-partition E-step:
1. Pre-compute `expElogbeta = exp(digamma(λ) - digamma(λ.sum(axis=1, keepdims=True)))` once.
2. For each `BOWDocument` in `rows`:
   - Initialize γ_d ~ `Gamma(gamma_shape, 1/gamma_shape)` (K-vector).
   - Iterate the Lee/Seung CAVI loop until convergence (γ_d only; never materialize φ_d):
     ```
     expElogthetad = exp(digamma(γ_d) - digamma(γ_d.sum()))
     phi_norm      = expElogbeta[:, indices].T @ expElogthetad + 1e-100
     γ_d_new       = α + expElogthetad * (expElogbeta[:, indices] @ (counts / phi_norm))
     ```
     Stop when `mean(|γ_d_new - γ_d|) < cavi_tol` or after `cavi_max_iter`.
   - Accumulate suff stats: `lambda_stats[:, indices] += outer(expElogthetad_final, counts / phi_norm_final)`.
   - Accumulate doc-level data-likelihood contribution: `doc_loglik_sum += sum(counts * log(phi_norm_final))`.
   - Accumulate per-doc KL contribution: `doc_theta_kl_sum += dirichlet_kl(γ_d, α · 1_K)`.
3. Return the partition-local sum as a dict.

**`combine_stats(a, b) → dict`**
Element-wise sum of `lambda_stats`; sum of scalar `doc_loglik_sum` and `n_docs`. Default ndarray combine works once we declare these field shapes.

**`update_global(global_params, target_stats, learning_rate) → dict`**
λ_new = (1 − ρ_t) · λ + ρ_t · (η + expElogbeta · `target_stats["lambda_stats"]`).

The `expElogbeta` (= `exp(digamma(λ) − digamma(λ.sum(axis=1, keepdims=True)))`) multiplication is the per-topic-per-vocab factor of `φ_dnk` that was deliberately *not* included in `local_update`'s per-doc accumulation. `φ_dnk = expElogthetad_k · expElogbeta_{k, w_dn} / phi_norm` — only `expElogthetad` is per-doc; `expElogbeta` is the same across all docs in a mini-batch, so factoring it out of the per-doc sum and applying it once at the driver after aggregation is correct and matches MLlib's `OnlineLDAOptimizer` exactly (`statsSum *:* expElogbeta.t` before `updateLambda`).

`target_stats` is already pre-scaled by `corpus_size / batch_size` upstream in `VIRunner` (per ADR 0005). `expElogbeta` is computed from the same λ that `local_update` saw, so the reference frame is consistent.

**`compute_elbo(global_params, aggregated_stats) → float`**
Real ELBO, not a surrogate (matches the post-Detour-2 standard from `CountingModel`). Three terms:
- Data likelihood `E_q[log p(w | β, θ, z)]` ← `aggregated_stats["doc_loglik_sum"]` (computed inside `local_update`, escaped via the suff-stat dict).
- Doc-level KL `E_q[log p(θ | α)] − E_q[log q(θ | γ)]` ← `−aggregated_stats["doc_theta_kl_sum"]`, summed across docs in the partition.
- Global KL `E_q[log p(β | η)] − E_q[log q(β | λ)]` ← closed form from λ alone, computed on the driver.

The data-likelihood term needs per-document quantities computed inside `local_update` to escape the partition. Extending the suff-stat dict (rather than a second pass) keeps everything within the existing single-pass `mapPartitions` pattern, matching MLlib's approach.

**`infer_local(row, global_params) → {"gamma": (K,), "theta": (K,)}`**
Same CAVI inner loop as `local_update`, but for one document, returning normalized θ_d = γ_d / γ_d.sum() alongside γ_d. Pure function of `(row, global_params)`.

### Module docstring

A short top-of-file docstring (~20 lines): one-paragraph generative model summary, the Lee/Seung trick stated as a 3-line equation block, references to Hoffman 2010 and Lee/Seung 2001, and a symbol table (K, V, D, N_d, λ, γ, φ, α, η, expElogbeta, expElogthetad, phi_norm) so anyone reading the code knows what each variable means. No derivation.

---

## Component 4 — Evaluation harness

### `charmpheno/evaluate/lda_compare.py`

Pure orchestration. Runs both implementations against the same prepared input and returns timing + ELBO + topic-word matrices.

```python
@dataclass
class LDARunArtifacts:
    topics_matrix: np.ndarray           # (K, V), rows = E[β_k] (normalized)
    topic_prevalence: np.ndarray        # (K,), total mass = sum_d theta_d,k
    elbo_trace: list[float] | None      # ours: every iter; mllib: None
    per_iter_seconds: list[float]
    wall_time_seconds: float
    final_log_likelihood: float | None  # mllib: from .logLikelihood(df)

def run_ours(rdd: RDD[BOWDocument], vocab_size: int, K: int,
             config: VIConfig) -> LDARunArtifacts: ...

def run_mllib(df: DataFrame, vocab_size: int, K: int,
              config: VIConfig) -> LDARunArtifacts: ...
```

**Asymmetry note:** `pyspark.ml.clustering.LDA` does not expose a per-iteration ELBO/logLikelihood trace from `OnlineLDAOptimizer`; only a final `LDAModel.logLikelihood(df)`. We record what's available and document the gap. The workaround for a comparable trace (refit at growing `maxIter` values) is out of scope for v1.

For "ours", topic-word distributions come from λ normalized per row: `E[β_k] = λ_k / λ_k.sum()`. For MLlib, `LDAModel.topicsMatrix()` returns V×K with un-normalized counts; we transpose and normalize identically.

For "ours" prevalence, run `VIRunner.transform(rdd, λ)` and sum normalized θ. For MLlib, `LDAModel.transform(df)` produces a `topicDistribution` column, summed analogously. Both paths sum *normalized θ*, not raw γ — apples-to-apples.

### `charmpheno/evaluate/topic_alignment.py`

Pure numpy. No Spark, no plotting.

```python
def js_divergence_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise JS divergence between rows of A (K_a × V) and B (K_b × V).
    Returns (K_a, K_b) matrix in nats. Symmetric in (A, B)."""

def order_by_prevalence(topics: np.ndarray, prevalence: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Sort topics descending by prevalence. Returns (sorted_topics, perm)."""

def alignment_biplot_data(
    topics_a: np.ndarray, prevalence_a: np.ndarray,
    topics_b: np.ndarray, prevalence_b: np.ndarray,
) -> dict:
    """Order both, compute JS matrix in the ordered frame.
    Returns {js_matrix, perm_a, perm_b, prevalence_a_sorted, prevalence_b_sorted}.
    Diagonal-dominance after this ordering = topic agreement."""

def ground_truth_from_oracle(
    df: DataFrame, vocab_map: dict[int, int], K_true: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (true_beta, true_prevalence) by aggregating the
    true_topic_id column. true_beta normalized per-row.

    More robust than reading the simulator's beta sidecar: this captures
    the empirical realization in this particular synthetic dataset (finite-
    sample noise included), which is what we should hold recovery accountable to.
    """
```

---

## Component 5 — Driver scripts

### `analysis/local/fit_lda_local.py`

Single-purpose: fit `VanillaLDA` on a parquet path, save `VIResult` + sidecar vocab.

```
fit_lda_local.py
  --input data/simulated/omop_N1000_seed42.parquet
  --K 10
  --iterations 200
  --mini-batch-fraction 0.1
  --tau0 1024 --kappa 0.7
  --seed 42
  --checkpoint-dir <user-chosen path under data/>
  --output <user-chosen path under data/>
```

`--output` is required (no default), matching the existing `fit_charmpheno_local.py` convention. Users typically point it under `data/runs/...` since `data/` is already in `.gitignore`.

Body shape:
1. Build `SparkSession`.
2. `df = load_omop_parquet(input)`; `bow_df, vocab_map = to_bow_dataframe(df)`.
3. `rdd = bow_df.rdd.map(BOWDocument.from_spark_row)`.
4. `model = VanillaLDA(K=K, vocab_size=len(vocab_map), ...)`.
5. `result = VIRunner(model, sc).fit(rdd, config)`.
6. `result.metadata["vocab"] = [concept_id for concept_id, idx in sorted(vocab_map.items(), key=lambda x: x[1])]`.
7. `result.save(output)`.

### `analysis/local/compare_lda_local.py`

Orchestrates the head-to-head comparison.

```
compare_lda_local.py
  --input data/simulated/omop_N1000_seed42.parquet
  --K 10
  --iterations 200
  --K-true 10
  --output <user-chosen path under data/>
```

Body shape:
1. Spark session, `load_omop_parquet`, `to_bow_dataframe` once.
2. Branch into `lda_compare.run_ours(rdd, ...)` and `lda_compare.run_mllib(df, ...)`.
3. Pull ground truth: `true_beta, true_prevalence = ground_truth_from_oracle(df_raw, vocab_map, K_true)`.
4. Build three biplot data dicts via `topic_alignment.alignment_biplot_data(...)`.
5. Render the three-panel matplotlib figure (ours vs. truth, MLlib vs. truth, ours vs. MLlib) with shared JS color scale and prevalence marginal bars.
6. Write PNG, perf-table CSV, and `artifacts.npz` to output dir.

### Output directory layout

Drivers populate the user-supplied `--output` directory with the following:

```
<output>/                       # fit_lda_local
├── manifest.json
├── lambda.npy
├── elbo_trace.npy
└── checkpoints/...

<output>/                       # compare_lda_local
├── biplots.png
├── perf_table.csv
└── artifacts.npz
```

Drivers stay thin — argparse, `SparkSession`, function calls. All real logic lives in `spark_vi/` and `charmpheno/`. Notebook-as-thin-driver discipline applied to scripts.

---

## Tests

Three-tier ladder matching the existing `make test` / `make test-all` / `make test-cluster` discipline.

### Tier 1 — Unit tests (no Spark, fast — `make test`)

**`spark_vi/tests/test_bow_document.py`**
- Invariants: indices sorted, indices in `[0, V)`, counts > 0, `length == counts.sum()`.
- `from_spark_row` round-trip from a `SparseVector` fixture.
- Frozen-dataclass behavior: hash, equality, immutability.

**`spark_vi/tests/test_lda_math.py`** — pure numpy, hand-checked numbers.
- CAVI fixed-point on a 2-topic toy: K=2, V=3, peaked β. Run `_cavi_inner_loop` to convergence, assert γ aligns with the dominant topic and `meanGammaChange < cavi_tol`.
- Lee/Seung equivalence: implement explicit-φ once for the test only, run it and the production implicit-φ path on identical input, assert γ_d and the suff-stat row agree to 1e-10.
- ELBO closed-form sanity: at uniform γ_d and uniform λ on a uniform-prior model, ELBO matches an analytically derived constant. Same pattern as `CountingModel`'s post-Detour-2 tightness test.
- ELBO monotone in CAVI: each CAVI iteration on a single document increases the per-doc bound. Iterate, record bound, assert non-decreasing.
- Symmetric Dirichlet KL: `_dirichlet_kl(α·1_K, α·1_K)` == 0; peaked γ_d gives KL > 0.

**`spark_vi/tests/test_lda_contract.py`** — `VIModel` contract conformance.
- All abstract methods present and signatures match the base class.
- `combine_stats` associativity: three random suff-stat dicts, assert `combine(a, combine(b, c)) == combine(combine(a, b), c)` elementwise. Critical for `treeReduce` correctness (per Detour 3).
- `update_global` natural-gradient interpolation: `learning_rate=0` → λ unchanged; `learning_rate=1` → λ jumps fully.
- `infer_local` purity: same row, same global_params → identical output (pure function check).

### Tier 2 — Integration tests (Spark local, hermetic fixtures — `make test-all`)

**`spark_vi/tests/test_lda_integration.py`**
- Hermetic fixture: build a tiny synthetic LDA dataset in-test (D=200, V=50, K=5, fixed seed). Don't depend on `simulate_lda_omop.py`'s output — keep the test self-contained.
- Recovery threshold: fit `VanillaLDA` via `VIRunner` with mini-batch, K=K_true, run to convergence. Assert mean diagonal JS divergence (after prevalence-sorting against ground truth β) < 0.1 nats. Threshold pinned empirically on first green run with deterministic seed.
- ELBO monotone trend: smoothed ELBO trace (10-iteration moving average) is non-decreasing. Hard monotonicity is too strict for stochastic VI.
- `transform` shape and stochasticity: run on held-out docs, assert shape `(K,)` per row and `theta.sum() ≈ 1`.

**`spark_vi/tests/test_broadcast_lifecycle.py`** — extend the existing module.
- Add a `transform` case mirroring the `fit` test pattern: broadcast-once, unpersist-once, no leaks.

**`charmpheno/tests/test_to_bow_dataframe.py`**
- Determinism: same input → same vocab ordering across two runs.
- Vocab completeness: every distinct `concept_id` ends up in the vocab map exactly once.
- SparseVector shape: each row's sparse vector has length == vocab_size, indices sorted, no duplicates.

**`charmpheno/tests/test_topic_alignment.py`** — pure numpy, fast but lives here because it's clinical-evaluation logic.
- JS matrix correctness on hand-crafted distributions (identical rows → diagonal == 0; orthogonal distributions → JS == log 2).
- Prevalence ordering: ascending input → descending-sort permutation.
- `alignment_biplot_data` end-to-end on a 3×3 toy.

**`charmpheno/tests/test_lda_compare.py`** — smoke only.
- Run both `run_ours` and `run_mllib` against a tiny fixture for 5 iterations.
- Assert artifact shape `(K, V)` and that nothing crashes. No correctness assertions; that's what the integration test covers.

### Tier 3 — Cluster (`make test-cluster`)

No additions for this work. The existing cluster gate stays as-is.

### Test count and runtime targets

- Tier 1: ~25 new tests. Runtime adds seconds, not minutes.
- Tier 2: ~10 new tests. Sub-minute total even with Spark startup amortized.
- Suite grows from 59 to ~94 tests. The `make test` ten-second target stays in reach.

The recovery-threshold test is the only non-deterministic gate. Practice: pin threshold + seed empirically on first green run; tighten on first observed flake rather than pre-engineering for hypothetical machine variation.

---

## Documentation

Minimal, focused on design records and operational accuracy. No pedagogical apparatus.

### Two new ADRs

- **ADR 0007 — VIModel inference capability.** Captures the decision to add `infer_local` as an optional capability rather than a required abstract method, the `(row, global_params)` purity invariant, and the deferred-but-non-foreclosed path to MLlib `Estimator/Transformer` compatibility.
- **ADR 0008 — Vanilla LDA design choices.** Captures: Hoffman 2010 + Lee/Seung 2001 as algorithm choice, hyperparameter defaults aligned with MLlib for fair comparison, symmetric α only (deferring asymmetric + `optimizeDocConcentration`), `BOWDocument` as canonical row type. References ADRs 0005 and 0006 to make the chain explicit.

### Updates to existing living docs

- **`SPARK_VI_FRAMEWORK.md`** — add `VanillaLDA` to "Implemented Models", document `infer_local` capability + `VIRunner.transform`, point to ADR 0007.
- **`RISKS_AND_MITIGATIONS.md`** — new "MLlib parity expectations" entry: agreement gate is prevalence-aligned topic similarity, not numerical equality. Different RNG, different convergence cutoffs, different float precision in places.

### `lda.py` module docstring

~20 lines: generative model summary, Lee/Seung trick as a 3-line equation block, references to Hoffman 2010 and Lee/Seung 2001, symbol table.

---

## Future work

Listed here to keep the implementation scope crisp:

- **Asymmetric α and `optimizeDocConcentration`.** Newton-Raphson update on α; meaningful complication. MLlib has it off by default; we don't need it for v1.
- **Per-iteration ELBO trace from MLlib.** Refit at growing `maxIter` for a comparable trace; diagnostic-only, not blocking.
- **MLlib `Estimator/Transformer` shim.** Wrapper at a separate compat layer once the core is stable. Already designed not to be foreclosed.
- **`elbo_eval_interval` field on `VIConfig`.** First-class skip-cadence kwarg for expensive ELBOs (parked thread from the bootstrap walkthrough).
- **Combined mini-batch + auto-checkpoint integration test.** Both features work individually; cross-feature interaction not explicitly tested (parked thread).
- **LDA notebook tutorial (`notebooks/tutorials/02_lda_walkthrough.ipynb`).** Runnable companion if pedagogical depth is wanted later. Out of scope for this work.
- **Real `OnlineHDP` implementation.** This spec is the warm-up; HDP is the next major target.

---

## Open questions

None at spec time — all design forks were resolved during the brainstorming session.

## References

- Hoffman, M. D., Blei, D. M., & Bach, F. (2010). *Online learning for latent Dirichlet allocation*. NIPS.
- Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). *Stochastic variational inference*. JMLR.
- Lee, D. D., & Seung, H. S. (2001). *Algorithms for non-negative matrix factorization*. NIPS.
- Spark MLlib `OnlineLDAOptimizer`: `mllib/src/main/scala/org/apache/spark/mllib/clustering/LDAOptimizer.scala`.
- ADR 0005 — Mini-batch sampling matching MLlib `OnlineLDAOptimizer`.
- ADR 0006 — Unified persistence format: `VIResult` as canonical state.
- `docs/architecture/SPARK_VI_FRAMEWORK.md` — framework contract and implemented models.
- `docs/architecture/RISKS_AND_MITIGATIONS.md` — risk register.
