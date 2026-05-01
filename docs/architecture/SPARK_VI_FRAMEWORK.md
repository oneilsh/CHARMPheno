# spark-vi: A PySpark Framework for Distributed Variational Inference

## Table of Contents

- [Executive Summary](#executive-summary)
- [Background & Motivation](#background--motivation)
  - [The Distributed VI Pattern](#the-distributed-vi-pattern)
  - [Why a Framework](#why-a-framework)
  - [Design Philosophy](#design-philosophy)
- [Framework Architecture](#framework-architecture)
  - [Core Computational Pattern](#core-computational-pattern)
  - [The VIModel Base Class](#the-vimodel-base-class)
  - [Data Contract: User-Facing vs. Internal](#data-contract-user-facing-vs-internal)
  - [The VIRunner](#the-virunner)
  - [VIConfig](#viconfig)
  - [VIResult and Model Export](#viresult-and-model-export)
- [Notebook Diagnostics and Output](#notebook-diagnostics-and-output)
  - [Live Training Display](#live-training-display)
  - [Batch Mode](#batch-mode)
  - [Model-Level Display](#model-level-display)
- [Pre-Built Models](#pre-built-models)
  - [Online HDP (Topic Model)](#online-hdp-topic-model)
  - [Sparse OU Estimator (Continuous-Time Dynamics)](#sparse-ou-estimator-continuous-time-dynamics)
  - [Third Model (TBD)](#third-model-tbd)
- [Project Structure](#project-structure)
- [Tooling and Development](#tooling-and-development)
- [Design Decisions](#design-decisions)
- [Future Directions](#future-directions)
- [References](#references)

---

## Executive Summary

**spark-vi** is a PySpark-native framework for distributed variational inference on
large datasets. Model authors subclass a simple base class and implement the
model-specific math — local variational updates, sufficient statistics, and global
parameter updates. The framework handles all Spark orchestration, training loop
management, convergence diagnostics, and notebook-first visualization.

The framework ships with pre-built models for clinical data analysis — an Online
Hierarchical Dirichlet Process for phenotype discovery and a Sparse
Ornstein-Uhlenbeck estimator for continuous-time dynamics — but the abstractions are
general-purpose and accommodate any model that fits the distributed VI pattern.

```mermaid
graph LR
    A(["User DataFrame<br/>long format"]) --> B(["Model.prepare_data<br/>reshape & validate"])
    B --> C(["VIRunner.fit<br/>distributed VI loop"])
    C --> D(["VIResult"])
    D --> E(["Notebook Display<br/>_repr_html_"])
    D --> F(["Model Export<br/>JSON + .npy"])
    D --> G(["Downstream Use<br/>transform / simulate / forecast"])

    style A fill:#e8d4f0,stroke:#7b2d8e,stroke-width:2px
    style B fill:#fff3cd,stroke:#d4a017,stroke-width:2px
    style C fill:#fff3cd,stroke:#d4a017,stroke-width:2px
    style D fill:#d4edda,stroke:#28a745,stroke-width:2px
    style E fill:#cce5ff,stroke:#0069d9,stroke-width:2px
    style F fill:#cce5ff,stroke:#0069d9,stroke-width:2px
    style G fill:#cce5ff,stroke:#0069d9,stroke-width:2px
```

**Key properties:**

- **Simple contract.** Three methods define a model. Everything else is duck-typed.
- **PySpark-native.** NumPy/SciPy on workers, Spark for orchestration. No JVM code,
  no special cluster permissions.
- **Notebook-first.** Live training diagnostics, rich HTML display for fitted models,
  matplotlib plotting. Explicit batch mode for submitted jobs.
- **Privacy-friendly export.** Fitted models contain only population-level parameters
  (no patient data) and serialize to human-inspectable JSON + NumPy arrays.

---

## Background & Motivation

### The Distributed VI Pattern

Variational inference for Bayesian models with conditionally conjugate or
gradient-amenable structure decomposes into a natural distributed pattern:

```mermaid
graph LR
    subgraph CLUSTER ["Spark Cluster"]
        direction LR
        G(["Global<br/>Parameters"]) -->|broadcast| W1(["Worker 1<br/>local step"])
        G -->|broadcast| W2(["Worker 2<br/>local step"])
        G -->|broadcast| WN(["Worker N<br/>local step"])
        W1 -->|summary stats| AG(["Aggregate<br/>global step"])
        W2 -->|summary stats| AG
        WN -->|summary stats| AG
        AG -->|iterate| G
    end

    AG -->|converged| EX(["Export"])
    EX --> MH(["Model Hosting<br/>Service"])
    EX --> PH(["On-Device<br/>Inference"])

    style G fill:#fff3cd,stroke:#d4a017,stroke-width:2px
    style AG fill:#fff3cd,stroke:#d4a017,stroke-width:2px
    style W1 fill:#e8d4f0,stroke:#7b2d8e,stroke-width:1px
    style W2 fill:#e8d4f0,stroke:#7b2d8e,stroke-width:1px
    style WN fill:#e8d4f0,stroke:#7b2d8e,stroke-width:1px
    style EX fill:#d4edda,stroke:#28a745,stroke-width:2px
    style MH fill:#cce5ff,stroke:#0069d9,stroke-width:2px
    style PH fill:#cce5ff,stroke:#0069d9,stroke-width:2px
```

Each iteration:

1. **Broadcast** global variational parameters to all workers
2. **Local update** — each worker runs variational E-step on its data partition,
   conditioned on global parameters, producing sufficient statistics
3. **Aggregate** — sufficient statistics are summed across workers via `treeAggregate`
4. **Global update** — driver updates global variational parameters from aggregated
   statistics

This is Spark MLlib's pattern for Online LDA (Hoffman et al., 2010), and it
generalizes far beyond topic models. Any model where:

- The local variational update depends only on local data + global parameters
- The information needed from local updates can be summarized as additive sufficient
  statistics

...fits this pattern. This includes most models in the exponential family with
conjugate priors, and extends to non-conjugate models via gradient-based updates
(where "sufficient statistics" become gradient contributions).

### Why a Framework

The two-stage clinical modeling pipeline described in this project's
[research design](TOPIC_STATE_MODELING.md) — Online HDP for phenotype discovery, Sparse OU
for continuous-time dynamics — both follow this exact pattern. The realization that
two very different models (discrete bag-of-words vs. continuous time series, closed-form
vs. gradient-based) share the same computational skeleton motivates a general framework.

The viability of this pattern at scale is already demonstrated by existing
implementations: Spark MLlib's `OnlineLDAOptimizer` uses exactly this
broadcast→mapPartitions→treeAggregate loop for LDA topic modeling, and the
intel-spark TopicModeling project implements Online HDP in Scala using the same
distribution strategy. Our contribution is not inventing the pattern, but
recognizing its generality and providing a framework that makes it accessible for
arbitrary Bayesian models in pure PySpark.

Rather than implementing each model's Spark orchestration from scratch, we factor out
the common infrastructure:

| Concern | Framework handles | Model author handles |
|---|---|---|
| Spark broadcast/aggregate | Yes | No |
| Partition data management | Yes | No |
| Training loop & convergence | Yes | No |
| Learning rate scheduling | Yes | No |
| Notebook diagnostics | Yes | No |
| Model export/serialization | Yes | No |
| Local variational updates | No | Yes |
| Global parameter updates | No | Yes |
| Sufficient statistics shape | No | Yes |
| Model-specific methods | No | Yes |

### Design Philosophy

**PyTorch-style duck typing.** The framework defines a minimal required contract (three
methods). Everything beyond that — `transform`, `simulate`, `forecast`,
`interaction_graph`, custom plotting — is just methods on your class. If it exists,
call it. No registration, no capabilities dict.

**Notebook-first, batch-capable.** The primary interface is Jupyter/Databricks
notebooks with live diagnostics. Batch mode is explicitly opted into, not auto-detected.

**PySpark-native.** Runs anywhere you can submit a PySpark job. NumPy and SciPy on
workers. No custom JVM code, no special cluster permissions.

**YAGNI.** The base class contract is deliberately minimal. We start with three models
and let real usage reveal what abstractions are needed, rather than designing a
sophisticated type hierarchy up front.

---

## Framework Architecture

### Core Computational Pattern

Each training iteration executes:

$$
\lambda^{(t+1)} = (1 - \rho_t)\,\lambda^{(t)} + \rho_t\;\hat{\lambda}\!\left(\sum_{p=1}^{P} s_p\right)
$$

where $\lambda^{(t)}$ are global variational parameters at iteration $t$,
$\rho_t$ is the learning rate, $s_p$ are sufficient statistics from partition $p$,
and $\hat{\lambda}$ is the natural gradient update. The summation over partitions is
computed via `treeAggregate`.

### The VIModel Base Class

Model authors subclass `VIModel` and implement:

**Required methods:**

- **`data_schema() -> dict`** — Declares the internal data format. Column names and
  types the framework will provide to `local_update`.

- **`initialize_global(data_summary: dict) -> dict[str, np.ndarray]`** — Called once
  before training. `data_summary` contains lightweight metadata computed in a pre-pass
  (vocabulary size, feature dimensions, observation count, etc.). Returns initial
  global variational parameters as a dict of NumPy arrays.

- **`local_update(partition_data, global_params) -> dict[str, np.ndarray]`** — Called
  on each worker partition each iteration. Receives a chunk of data and the current
  broadcast global parameters. Returns sufficient statistics as a dict of NumPy arrays.
  Whatever happens inside — one pass, iterative loop, local optimization — is the model
  author's business.

- **`update_global(global_params, target_stats, learning_rate) -> dict[str, np.ndarray]`** —
  Called on the driver after aggregation. Receives the current global parameters, the
  pre-scaled target sufficient statistics, and the Robbins-Monro learning rate.
  Returns updated global parameters. The `target_stats` dict is the natural-gradient
  target — a model should compute `lambda_hat = prior_natural_params + target_stats`
  and interpolate against `global_params` via `learning_rate`. When the runner is in
  mini-batch mode it has multiplied the raw aggregated stats by `corpus_size /
  batch_size` so this same arithmetic produces an unbiased corpus-equivalent estimate;
  in full-batch mode `target_stats` equals the raw aggregated stats. Models do not
  need to know which mode is active.

**Optional methods:**

- **`compute_elbo(global_params, aggregated_stats) -> float`** — If overridden, the
  framework records this each iteration for ELBO-based convergence monitoring. The
  `aggregated_stats` here are the *raw* aggregated sufficient statistics (not pre-scaled
  to a corpus-level target), suitable for ELBO terms representing observed evidence.
  Default returns NaN, which callers treat as "ELBO not available."

  **ELBO-term placement pattern.** When an ELBO term depends on per-record local
  state (γ_d, per-doc normalizers, sample-level expectations) that is already in scope
  inside `local_update`, accumulate the term *there* as a scalar entry in the returned
  suff-stats dict; the runner sums it across partitions via `combine_stats` and
  `compute_elbo` just reads the scalar back out. Re-deriving such a term inside
  `compute_elbo` would force you to either re-run the local E-step or stash per-record
  state into a corpus-sized array. `compute_elbo` is the right home only for terms that
  depend on `global_params` alone (typically global Dirichlet/Gaussian KL terms), which
  are cheap to evaluate once on the driver. See `VanillaLDA` for a worked example.

- **`combine_stats(stats_a, stats_b) -> dict[str, np.ndarray]`** — Override if
  sufficient statistics don't aggregate by elementwise addition. Default: sum all
  arrays in the dict.

- **`prepare_data(df, **kwargs) -> DataFrame`** — For pre-built models, reshapes a
  user-friendly long-format DataFrame into the internal format. Custom model authors
  can skip this and provide data in internal format directly.

- `infer_local(row, global_params)` (optional) — per-row variational
  posterior under fixed global params. Models with local latent variables
  (LDA, HDP) override this; models without (CountingModel) leave it
  unimplemented. Default raises `NotImplementedError`. Must be a pure
  function of `(row, global_params)`. See [ADR 0007](../decisions/0007-vimodel-inference-capability.md).

**Example — Online HDP implementation sketch:**

```python
class OnlineHDP(VIModel):
    def __init__(self, T=150, K=15, alpha=1.0, gamma=1.0, eta=0.01):
        self.T = T  # truncation level
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta

    def data_schema(self):
        return {"word_ids": "array<int>", "word_counts": "array<float>"}

    def prepare_data(self, df, doc_col, word_col, timestamp_col=None):
        # Group long-format (doc_id, word) rows into per-document arrays
        # Returns DataFrame matching data_schema
        ...

    def initialize_global(self, data_summary):
        V = data_summary["vocab_size"]
        return {
            "lambda": np.random.gamma(100., 1./100., (self.T, V)),
            "stick_a": np.ones(self.T),
            "stick_b": np.full(self.T, self.gamma),
        }

    def local_update(self, partition_data, global_params):
        # Run doc-level variational E-step over all docs in partition
        # Returns word-topic counts, stick counts, doc count
        ...

    def update_global(self, global_params, target_stats, learning_rate):
        # lambda_hat = prior + target_stats; interpolate against global_params.
        ...

    def print_topics(self, n_words=10, vocabulary=None):
        # Display top words per discovered topic
        ...

    def transform(self, df, **kwargs):
        # Infer per-document topic proportions on new data
        ...

    def simulate(self, n_docs, avg_length=20):
        # Generate synthetic documents from learned topics
        ...
```

### Data Contract: User-Facing vs. Internal

The data pipeline has two layers:

```mermaid
graph LR
    A(["User DataFrame<br/>(long format)"]) -->|"model.prepare_data(df, doc_col=..., word_col=...)"| B(["Internal DataFrame<br/>(matches data_schema)"])
    B -->|"framework partitions & converts"| C(["Partition arrays<br/>(NumPy on workers)"])

    style A fill:#e8d4f0,stroke:#7b2d8e,stroke-width:2px
    style B fill:#fff3cd,stroke:#d4a017,stroke-width:2px
    style C fill:#d4edda,stroke:#28a745,stroke-width:2px
```

**User-facing (pre-built models):** Long-format DataFrames. Users pass column names
as keyword arguments. The model's `prepare_data` handles reshaping.

```python
# HDP: one row per (visit, diagnosis_code)
runner.fit(visits_df, doc_col="visit_id", word_col="icd_code")

# OU: one row per (patient, timepoint, measurement)
runner.fit(trajectories_df, entity_col="patient_id",
           timestamp_col="date", value_cols=["topic_1", "topic_2"])
```

**Internal (framework level):** The DataFrame produced by `prepare_data` matches the
model's `data_schema()`. The framework handles partitioning and conversion to NumPy
arrays on workers.

**Custom models:** Authors can skip `prepare_data` and provide data already in
internal format.

### The VIRunner

The main user-facing object. Owns the training loop and all Spark mechanics.

```python
runner = VIRunner(spark, model, config=VIConfig())
result = runner.fit(df, **data_kwargs)
```

**What `fit` does each iteration:**

1. If `mini_batch_fraction` is set, sample a fresh mini-batch from the input RDD
   (with replacement by default; per-iteration seed derived from `random_seed`).
   Cache the batch and call `count()` to obtain `batch_size`. If empty, skip to
   the next iteration. Otherwise compute `stats_scale = corpus_size / batch_size`.
   Otherwise (full-batch mode) `batch = full_rdd` and `stats_scale = 1.0`.
2. Broadcast `global_params` to all workers
3. `mapPartitions` $\rightarrow$ `model.local_update` $\rightarrow$ collect
   sufficient stats
4. `treeAggregate` with `model.combine_stats` (default: elementwise sum). Implemented
   in PySpark via `RDD.treeReduce(model.combine_stats)` — equivalent to `treeAggregate`
   for our associative + commutative combiner, with a slightly simpler API.
5. Pre-scale aggregated stats by `stats_scale` to form `target_stats`; the raw
   aggregated stats are also retained for ELBO computation.
6. `model.update_global(global_params, target_stats, learning_rate)` on driver
7. `model.compute_elbo(global_params, aggregated_stats)` for diagnostics; log
   metrics (parameter deltas, ELBO if available, wall time)
8. Push metrics to notebook display (or logger in batch mode)
9. Check convergence, stop early if threshold met

Beyond `fit`, `VIRunner` also exposes `transform(rdd, global_params)` —
a one-pass per-row inference orchestrator. It broadcasts the trained
global params, maps `model.infer_local` over the RDD, and unpersists
the broadcast on completion. Models that don't implement `infer_local`
produce a clear NotImplementedError when the resulting RDD is collected.

**Output modes:**

- `VIRunner(spark, model, output="notebook")` — **default.** Live-updating display
  in Jupyter/Databricks.
- `VIRunner(spark, model, output="batch")` — metrics go to Python `logging` at
  INFO level.

### VIConfig

Minimal training hyperparameters with sensible defaults:

```python
@dataclass(frozen=True)
class VIConfig:
    max_iterations: int = 100
    learning_rate_tau0: float = 1.0
    learning_rate_kappa: float = 0.7
    convergence_tol: float = 1e-4
    checkpoint_interval: int | None = None
    checkpoint_dir: Path | str | None = None
    mini_batch_fraction: float | None = None     # None → full-batch
    sample_with_replacement: bool = True
    random_seed: int | None = None
```

**Auto-checkpoint fields** (`checkpoint_interval`, `checkpoint_dir`) trigger
`VIRunner.fit` to save a `VIResult` to `checkpoint_dir` every
`checkpoint_interval` iterations, overwriting the previous checkpoint (last
one is the only one needed for resume). The two fields are coupled: setting
one without the other raises `ValueError` at construction. The on-disk format
is the same `manifest.json + params/*.npy` layout used for `save_result`, so
auto-checkpoints can be loaded directly via `load_result` or fed into
`VIRunner.fit(rdd, resume_from=path)` to continue training. See ADR 0006.

**Mini-batch fields** (`mini_batch_fraction`, `sample_with_replacement`, `random_seed`)
adopt the MLlib `OnlineLDAOptimizer` `subsamplingRate` pattern. With
`mini_batch_fraction=f`, each iteration samples `RDD.sample(withReplacement=…,
fraction=f, seed=…)` from the input. The runner discovers `corpus_size` once
via `data_rdd.count()` at fit start; per-iteration `batch_size` comes from
`batch.count()` (the batch RDD is cached between count and `mapPartitions` to
avoid recomputing the sample). The natural-gradient target seen by
`update_global` is pre-scaled by `corpus_size / batch_size`, matching MLlib's
exact convention. With `mini_batch_fraction=None` (the default), the runner
processes the full RDD every iteration and `stats_scale = 1.0`. See
`docs/decisions/0005-mini-batch-sampling.md` for the design rationale.

The default learning rate follows the Robbins-Monro schedule:

$$
\rho_t = (\tau_0 + t)^{-\kappa}
$$

with $t \in \{1, 2, \ldots\}$ (Hoffman et al., 2013). The implementation indexes
iterations from 0, so it equivalently computes $\rho_t = (\tau_0 + t + 1)^{-\kappa}$
with $t \in \{0, 1, \ldots\}$; both forms produce the same sequence of step sizes.
For $\kappa \in (0.5, 1]$ the schedule satisfies $\sum_t \rho_t = \infty$ and
$\sum_t \rho_t^2 < \infty$, guaranteeing convergence of the stochastic natural
gradient. `VIConfig` accepts the wider range $\kappa \in (0, 1]$ for
experimentation; values in $(0, 0.5]$ violate the Robbins-Monro condition and
are not guaranteed to converge.

### VIResult and Model Export

`fit` returns a `VIResult`:

```python
@dataclass(frozen=True)
class VIResult:
    global_params: dict[str, np.ndarray]  # fitted variational parameters
    elbo_trace: list[float]               # per-iteration ELBO values
    n_iterations: int                     # iterations completed (incl. resume)
    converged: bool                       # True iff convergence criterion met
    metadata: dict[str, Any]              # model class, timestamps, git sha, ...
```

`VIResult` is the canonical record of training-run state — the same dataclass round-trips
for both completed runs and in-progress checkpoints written during a fit. `converged=True`
indicates a finished, converged run; `converged=False` covers both runs that exhausted
`max_iterations` without converging and interim checkpoints. See ADR 0006 for the
unification rationale.

**Export format:** A directory containing:
- `manifest.json` — `format_version`, `elbo_trace`, `n_iterations`, `converged`,
  `metadata`, `param_names`
- `params/*.npy` — one file per entry in `global_params`

Human-inspectable, no pickle. No patient data — only population-level distributional
parameters. The `format_version` field provides a migration handle: `load_result` raises
`ValueError` for unknown versions.

```python
from spark_vi.io.export import save_result, load_result

save_result(result, "path/to/model")
loaded = load_result("path/to/model")  # returns a VIResult
```

Loaded `VIResult`s can be passed straight to `VIRunner.fit(rdd, resume_from=path)`
to continue training, or consumed downstream by analysis code that operates on
NumPy arrays directly without Spark.

---

## Notebook Diagnostics and Output

### Live Training Display

During `fit()` in notebook mode, the runner shows a live-updating display:

- Current iteration and elapsed time
- Convergence metric (parameter change magnitude)
- ELBO value if the model computes it
- A simple line plot that updates in-place (via `IPython.display` or `ipywidgets`)

### Batch Mode

Same information goes to Python `logging` at INFO level. User opts in explicitly:

```python
runner = VIRunner(spark, model, output="batch")
```

### Model-Level Display

Pre-built models implement `_repr_html_()` for rich notebook rendering (following the
Pandas convention):

- A fitted **HDP** displays a topic summary table (top words per topic, topic weights)
- A fitted **OU** displays the interaction matrix and key eigenvalues

Plain `__repr__` gives a text summary for batch/terminal use.

Plotting methods (`plot_topics()`, `plot_interaction_matrix()`, `plot_convergence()`)
use matplotlib and are just regular methods on the model — no framework involvement.

---

## Pre-Built Models

### Online HDP (Topic Model)

Nonparametric Bayesian topic model using the Hierarchical Dirichlet Process
(Teh et al., 2006) with online variational inference (Wang et al., 2011).

| Aspect | Detail |
|---|---|
| **Data format** | Long-format: `(doc_id, word, [timestamp])` |
| **Latent structure** | Discrete topic assignments, per-document topic proportions, corpus-level topic-word distributions |
| **Sufficient stats** | Word-topic count matrices, stick-breaking counts |
| **Global params** | Topic-word distributions $\lambda$, stick parameters $(a, b)$ |

**User-facing methods:**

- `prepare_data(df, doc_col, word_col, timestamp_col=None)` — reshape long format
- `transform(df)` — infer per-document topic proportions
- `simulate(n_docs, avg_length)` — generate synthetic documents
- `print_topics(n_words, vocabulary)` — readable topic summaries
- `plot_topics(n_words)` — visual topic display

**Clinical use case:** Discover phenotypes from diagnosis codes. Each topic is a
cluster of co-occurring codes representing a clinical pattern.

### Sparse OU Estimator (Continuous-Time Dynamics)

$L_1$-penalized maximum likelihood estimation of a multivariate Ornstein-Uhlenbeck
process (Gaiffas & Matulewicz, 2019):

$$
d\mathbf{x}(t) = A\bigl(\mathbf{x}(t) - \boldsymbol{\mu}\bigr)\,dt + \Sigma^{1/2}\,d\mathbf{W}(t)
$$

where $A$ is the drift (interaction) matrix with $L_1$ penalty for sparsity,
$\boldsymbol{\mu}$ is the mean-reversion target, and $\Sigma$ is the diffusion matrix.

| Aspect | Detail |
|---|---|
| **Data format** | Long-format: `(entity_id, timestamp, value_1, value_2, ...)` |
| **Latent structure** | Drift matrix $A$, diffusion $\Sigma$, mean-reversion target $\boldsymbol{\mu}$ |
| **Sufficient stats** | Gradient contributions for $A$, $\boldsymbol{\mu}$, $\Sigma$ accumulated over entity trajectories |
| **Global params** | $A$ (sparse), $\boldsymbol{\mu}$, $\Sigma$ |

**User-facing methods:**

- `prepare_data(df, entity_col, timestamp_col, value_cols)` — reshape long format
- `transform(df)` — smoothed state estimates
- `forecast(state, horizon)` — predicted trajectory with uncertainty
- `simulate(n_entities, n_timepoints)` — generate synthetic trajectories
- `interaction_graph()` — NetworkX or adjacency representation of $A$

**Implementation note:** When the input is topic proportions (which live on the
simplex), an isometric log-ratio (ILR) transform maps them to unconstrained
$\mathbb{R}^{K-1}$ space before OU modeling. The `prepare_data` step handles this
transparently, and results are mapped back to the simplex for interpretation.

**Clinical use case:** Model causal dynamics between phenotypes discovered by the HDP.
The sparse $A$ matrix reveals which phenotypes drive or inhibit other phenotypes,
with timescales from eigenanalysis.

### `VanillaLDA`

Hoffman 2010 Online LDA with the Lee/Seung 2001 implicit-phi trick. Mini-
batch SVI, default hyperparameters aligned with `pyspark.ml.clustering.LDA`
for fair head-to-head comparison. See [ADR 0008](../decisions/0008-vanilla-lda-design.md)
for design choices and the [LDA design spec](../superpowers/specs/2026-04-30-vanilla-lda-design.md)
for the full algorithmic detail.

Consumes `RDD[BOWDocument]`. Provides `infer_local` for per-document theta
inference; combinable with `VIRunner.transform` to score new documents
under trained global params.

### Third Model (TBD)

Chosen during implementation based on which dimension of the abstraction feels
underexercised after the HDP and OU. Candidates:

- **Bayesian Gaussian Mixture** — simplest possible VI model, good for tutorials and
  documentation. Tests the abstraction with minimal model complexity.
- **Probabilistic PCA / Factor Analysis** — continuous latent factors, different
  sufficient statistics structure from both HDP and OU.
- **Hierarchical Poisson Factorization** — count data with a different likelihood
  family. Relevant for clinical event count data.

---

## Project Structure

```
spark_vi/
├── core/
│   ├── model.py            # VIModel base class
│   ├── runner.py           # VIRunner orchestration + training loop
│   ├── config.py           # VIConfig
│   └── result.py           # VIResult, save/load
├── models/
│   ├── hdp.py              # Online HDP
│   ├── sparse_ou.py        # Sparse OU estimator
│   └── ...                 # Third model TBD
├── diagnostics/
│   ├── notebook.py         # Live notebook display
│   ├── batch.py            # Logging-based output for batch mode
│   └── plotting.py         # Shared matplotlib utilities
└── tests/
    ├── test_runner.py      # Framework orchestration tests
    ├── test_hdp.py         # HDP-specific tests
    ├── test_sparse_ou.py   # OU-specific tests
    └── synthetic/          # Synthetic data generators for validation
```

---

## Tooling and Development

**Dependency management:** `uv` for fast, reproducible dependency resolution.
`pyproject.toml` as the single source of truth for package metadata, dependencies,
and build configuration.

**Dependencies (v1):**

| Category | Packages |
|---|---|
| Runtime | `pyspark`, `numpy`, `scipy`, `matplotlib` |
| Optional | `ipywidgets` (notebook display), `networkx` (OU graph export) |
| Dev | `pytest`, `ruff` |

**Makefile targets:**

| Target | Command |
|---|---|
| `make install` | `uv sync` — set up local dev environment |
| `make test` | Run pytest suite |
| `make lint` | `ruff check` + `ruff format --check` |
| `make build` | Build distributable package |
| `make clean` | Remove build artifacts |

**Testing strategy:**

- **Unit tests** for base class contract (mock models returning known sufficient stats)
- **Integration tests** per model with small synthetic data (PySpark local mode)
- **Synthetic data generators** in `tests/synthetic/` double as validation tools

**Target:** PyPI-publishable package.

---

## Design Decisions

### Flat base class over hierarchical model graph

We considered three abstraction strategies:

1. **Single base class, flat contract** — one `VIModel` with three methods
2. **Hierarchical model graph** — models declare a DAG of latent variable groups,
   framework traverses the graph coordinating updates
3. **Single base class with composable mixins** — flat contract plus opt-in mixins
   (`TemporalMixin`, `HierarchicalStatsMixin`, etc.)

**Chose (1)** for v1. Approach (2) is elegant but risks over-engineering before we
have enough models to know what the right graph structure is. Approach (3) is a
natural evolution path if the flat contract proves limiting. Both are noted as future
directions.

### PyTorch-style duck typing over capabilities registration

Models expose whatever methods make sense (`transform`, `simulate`, `forecast`,
custom plotting, etc.) as regular Python methods. No registration, no capabilities
dict. If a method exists, call it. If it doesn't, `AttributeError`. This matches
how PyTorch `nn.Module` subclasses work in practice — users add `generate()`,
`encode()`, etc. as needed.

### Explicit batch mode over auto-detection

`VIRunner(spark, model, output="batch")` rather than auto-detecting the environment.
Explicit is better than implicit.

### Simple local update contract

`local_update(partition_data, global_params) -> sufficient_stats` — one call, one
return. If the local step needs internal iteration, that's the model author's concern.
The framework doesn't manage local convergence, iteration budgets, or local ELBO
contributions. This keeps the contract minimal and the framework simple.

### Additive sufficient statistics as default

Most VI models produce additive sufficient statistics. The default `combine_stats`
sums arrays elementwise via `treeAggregate`. Models with non-additive aggregation
override `combine_stats`.

### Long-format DataFrames as user convention

Pre-built models accept long-format DataFrames with column name kwargs. This is the
most natural format for tabular clinical data regardless of source system (one row
per visit-diagnosis, one row per patient-timepoint-measurement) and requires no
user-side reshaping. The model's `prepare_data` handles conversion to internal format.

### JSON + NumPy export over pickle

Fitted models serialize to a directory of `config.json` + `params/*.npy`. No pickle
(security concerns, version fragility). Human-inspectable. Contains only
population-level parameters — no patient data.

---

## Future Directions

These are ideas surfaced during design that are explicitly **out of scope for v1** but
worth pursuing:

### Bayesian Latent-Class Phenotype Estimation Models

The framework is well-suited to Bayesian latent-class phenotype-estimation models in
the style of Hubbard et al. (2021, *Estimating Patient Phenotypes and Outcome-Exposure
Associations in the Presence of Missing Data in Electronic Health Records*), where a
pre-specified phenotype (e.g., pediatric type 2 diabetes) is estimated from noisy and
informatively-missing EHR indicators. This class of models is structurally lighter
than the HDP: the per-patient E-step is a closed-form Bayes-rule calculation over a
small finite latent space (not an iterative coordinate-ascent loop), the global state
is a handful of class-prevalence and per-class emission parameters (kilobytes rather
than megabytes), and the M-step is a closed-form conjugate update on the aggregated
sufficient statistics. Such models would run comfortably in spark-vi with no
architectural changes and would give the framework a concrete second-model-family
demonstration complementary to the HDP — an unsupervised-discovery model and a
pre-specified-phenotype-estimation model sharing the same infrastructure. Extensions
to model MNAR missingness (with dependence on the unobserved indicator value) fit
naturally via data augmentation, which remains compatible with the
broadcast→aggregate→update pattern.

### Autodiff on Workers

Allow JAX or PyTorch on workers for automatic differentiation of the local ELBO. This
would open up models where closed-form local updates don't exist — the model author
defines the local ELBO as a differentiable function, and the framework computes
gradients automatically. Falls back to NumPy when autodiff libraries aren't available.

### Rich Local Update Contract

The framework could manage local convergence: iteration budgets per partition, local
ELBO contribution tracking, adaptive local computation. This would help models with
expensive local steps (e.g., amortized inference with neural networks) but adds
complexity to the contract.

### Hierarchical Model Graph (Approach B)

Models declare a DAG of latent variable groups, each with its own local/global split.
The framework traverses the graph, coordinating updates — essentially plate notation
mapped to computation. Could enable automatic scheduling of coordinate ascent for
complex models.

### Composable Mixins (Approach C)

Opt-in mixin classes that add structured behavior to the flat base class:
`TemporalMixin` for time-indexed observations, `HierarchicalStatsMixin` for
tree-structured aggregation, `NaturalGradientMixin` for natural parameter updates.
Framework recognizes mixins and adjusts orchestration accordingly.

### Stochastic Variational Inference

**Implemented as of ADR 0005.** `VIConfig.mini_batch_fraction` subsamples the input
RDD each iteration, with the runner pre-scaling aggregated stats by
`corpus_size / batch_size` to form an unbiased natural-gradient target. Enables
stochastic VI (Hoffman et al., 2013) for datasets too large for full-pass
iterations. The Robbins-Monro decay schedule is principled in this regime; for
full-batch models (`mini_batch_fraction=None`), step sizes still decay and
eventually stall — see RISKS_AND_MITIGATIONS.md for the mitigation.

### ELBO Evaluation Scheduling and Checkpointing

**Checkpointing implemented as of ADR 0006.** `VIConfig.checkpoint_interval` paired
with `VIConfig.checkpoint_dir` causes `VIRunner.fit` to auto-save a `VIResult` every
N iterations to a directory, overwriting the previous checkpoint.
`VIRunner.fit(rdd, resume_from=path)` loads a saved checkpoint and continues training
with the Robbins-Monro counter preserved across the boundary. The format is the same
`manifest.json + params/*.npy` layout used for `save_result`.

Configurable `elbo_eval_interval` (ELBO computation may be expensive and needn't run
every iteration) is still future work — currently every iteration computes ELBO. For
models where ELBO is expensive, return NaN from `compute_elbo` to skip the cost while
still hitting the convergence-check call site.

### Differential Privacy

The `treeAggregate` step — summing sufficient statistics across partitions — is the
natural injection point for differentially private noise. Adding calibrated Gaussian
or Laplace noise to aggregated statistics before the global update would provide
formal DP guarantees. The framework's structure is naturally compatible; the research
challenge is calibrating noise per model family to maintain utility. Generative models
(HDP, OU) are particularly well-suited since their outputs are population-level
distributions, not individual records.

### On-Device Inference

Trained models are compact (a 100-topic HDP over ~70k codes is ~30-60MB; an OU drift
matrix for 100 topics is trivially small) and inference is lightweight matrix algebra.
This opens the possibility of shipping trained models directly to patient devices —
phones, tablets, browsers — where inference runs locally against the patient's own
data without transmitting it externally. The population-level model arrives on-device;
the sensitive patient data never leaves. Implementation would require a lightweight
inference runtime (Swift/Kotlin native, WASM/JS for web, or a minimal C library),
but the math is simple enough that no ML framework is needed on-device.

### Interoperability with Clinical Data Standards

The framework trains on whatever vocabulary the data uses (e.g., ICD-10, SNOMED CT,
local codes). At inference time, a thin vocabulary mapping layer could translate
incoming records from different coding systems to the model's vocabulary. For
standard terminologies, static mapping tables (such as those maintained by OMOP)
would suffice. Unmappable codes can be gracefully skipped — inference from a partial
set of codes is still useful. This is fundamentally easier than full data
harmonization because inference tolerates missing codes in a way that training
pipeline construction does not.

### Federated Learning Across Sites

The framework's computational pattern — local sufficient statistics aggregated into
global updates — maps directly to a federated learning setting. "Local" currently means
"Spark partition," but nothing requires partitions to be co-located. If multiple
organizations each compute sufficient statistics on their own data, those statistics
can be summed and used for the global update, producing a result mathematically
identical to training on the combined data. This is federated averaging applied to
variational inference rather than SGD, and is arguably better suited to federation:
sufficient statistics are additive and interpretable (not opaque weight deltas), no
patient data crosses organizational boundaries (stats are population-level aggregates
by construction), and the convergence theory for stochastic natural gradient VI already
covers the non-IID case where each "minibatch" comes from a different data distribution
— the core difficulty that makes federated SGD fragile. The practical question is
communication overhead (how many rounds of stat exchange are needed), but for models
like the HDP with large per-site datasets, even a few rounds should suffice.
Federated nonparametric topic modeling for multi-site clinical data would be a
meaningful research contribution in its own right.

## Macrovisit pre-processing

A prerequisite for meaningful federation is that the document unit is comparable
across sites. Clinical encounter definitions vary significantly across institutions
(see the N3C macrovisit work by Pfaff et al., 2023, which merges heterogeneous
micro-encounters into standardized composite visits). A federated deployment would
need either a shared macrovisit construction step at each site, or sufficient
robustness to document-level variation — an open question worth investigating.

### Phenotype-Profile-Driven Capabilities

Several capabilities become possible once interpretable per-patient phenotype
profiles are available as a first-class output. These are noted here as
forward-looking directions for the framework and the models built on it.

- **Phenotype-based patient similarity.** Per-patient phenotype profiles
  are dense, interpretable vector representations and admit principled
  similarity comparisons (cosine on the appropriate transform of the
  profile is well-defined; see [TOPIC_STATE_MODELING.md](TOPIC_STATE_MODELING.md)).
  Privacy-preserving variants — federated nearest-neighbor protocols,
  homomorphic encryption applied to similarity computation, and
  differentially-private profile release — are all compatible with the
  profile representation. Phenotype profiles are more
  protective against identity disclosure than raw clinical records but
  can still convey sensitive attribute information.

- **Trajectory simulation.** Coupling phenotype profiles with a dynamic
  model (such as the OU stage discussed in
  [TOPIC_STATE_MODELING.md](TOPIC_STATE_MODELING.md)) enables
  posterior-predictive simulation of plausible patient trajectories over
  time, supporting aggregate risk exploration and what-if analysis at the
  individual-patient level.

- **Phenotype-conditioned outcome modeling.** Once phenotype profiles are
  available, they form a natural feature substrate for downstream outcome
  models, with the interpretability and uncertainty quantification of the
  phenotyping stage carrying over to the outcome stage.

- **Structured summaries for clinical review.** An interpretable phenotype
  profile is a compact, clinically-readable summary of a patient's history
  that can support the review of complex multimorbid records.

- **Cross-references to technical extensions.** Several longer-horizon
  technical directions are captured in
  [TOPIC_STATE_MODELING.md](TOPIC_STATE_MODELING.md) under Open Questions
  & Future Work, including distance-dependent / autoregressive HDP for
  continuous-time topic dynamics, nested HDP for hierarchical cohort
  pooling, Fisher embeddings as model-aware patient signatures, and OU
  dynamics with nonlinear drift.

---

## Prior Art

The distributed VI pattern (broadcast globals → local updates → aggregate sufficient
statistics → global update) is not novel to this framework. What spark-vi contributes
is a **reusable PySpark implementation** of this pattern with a minimal, flexible model
contract — not the pattern itself.

### Existing frameworks

**AMIDST** (Masegosa et al., 2019) is the most direct prior art: a Java framework for
scalable VI on Bayesian networks with Flink/Spark backends. AMIDST demonstrated the
pattern at scale (billion-parameter models on 16-node clusters) for GMMs, HMMs, Kalman
filters, factor analysis, and LDA. Key differences from spark-vi:

| Dimension | AMIDST | spark-vi |
|---|---|---|
| Model definition | Compose from ~6 built-in distribution types | Arbitrary code in 3-method contract |
| Nonparametric models (HDP) | Not supported | Core pre-built model |
| Non-VI models (penalized MLE) | Not supported | Supported (OU estimator) |
| Maintenance | Dormant since 2018; creator moved to other research | Active |
| Language/ecosystem | Java | Python/PySpark/NumPy |

AMIDST's restriction to directed graphical models with VMP-compatible distributions is
the fundamental architectural difference. spark-vi's duck-typed contract allows any
model where local computation produces additive sufficient statistics — including
models like the sparse OU estimator that use penalized maximum likelihood rather than
variational inference. AMIDST cannot host such models without core modifications.

**Pyro/NumPyro** support SVI and can distribute via Horovod or JAX `pmap`, but target
GPU-centric workflows. They assume data fits in memory (or is handled by PyTorch
DataLoaders), not that data lives in a distributed data lake processed as Spark
partitions. For organizations where data is already in Spark (e.g., clinical data
warehouses on Databricks/Synapse), requiring a separate GPU pipeline is a significant
adoption barrier.

**ForneyLab** (Akbayrak et al., 2022) embeds stochastic VI into an automated
message-passing framework in Julia — more automated than spark-vi (model authors
specify factor graphs, not update equations) but single-machine only.

**Stan, PyMC, Turing.jl** implement ADVI (black-box VI without natural gradients) and
are single-machine. They do not exploit conjugate structure or distribute computation.

### Positioning

The strongest case for spark-vi is not "a better AMIDST" but rather: the clinical
modeling pipeline (HDP → OU) requires a framework flexible enough to host both
nonparametric variational inference and penalized MLE for continuous-time processes,
and no existing framework — active or dormant — accommodates both.

---

## References

- Akbayrak, S., Şenöz, İ., Sarı, A., & de Vries, B. (2022). Probabilistic Programming
  with Stochastic Variational Message Passing. *International Journal of Approximate
  Reasoning*, 148, 235-252.
- Blei, D. M., & Lafferty, J. D. (2006). Dynamic Topic Models. *ICML*.
- Gaiffas, S., & Matulewicz, G. (2019). Sparse inference of the drift of a
  high-dimensional Ornstein-Uhlenbeck process. *Journal of Multivariate Analysis*, 169.
- Hoffman, M. D., Blei, D. M., & Bach, F. R. (2010). Online Learning for Latent
  Dirichlet Allocation. *NeurIPS*.
- Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). Stochastic Variational
  Inference. *JMLR*, 14.
- Masegosa, A. R., et al. (2019). AMIDST: A Java Toolbox for Scalable Probabilistic
  Machine Learning. *Knowledge-Based Systems*, 163, 595-597.
- Masegosa, A. R., et al. (2017). Scaling up Bayesian Variational Inference Using
  Distributed Computing Clusters. *International Journal of Approximate Reasoning*, 88.
- Masegosa, A. R. & Gómez-Olmedo, M. (2025). Toward Variational Structural Learning
  of Bayesian Networks. *IEEE Access*, 13, 26130-26141.
- Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). Hierarchical
  Dirichlet Processes. *JASA*, 101(476).
- Wang, C., Paisley, J., & Blei, D. M. (2011). Online Variational Inference for the
  Hierarchical Dirichlet Process. *AISTATS*.
