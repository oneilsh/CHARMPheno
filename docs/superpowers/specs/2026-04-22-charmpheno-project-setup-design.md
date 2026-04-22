# CHARMPheno — Project Setup & Development Workflow Design

**Date:** 2026-04-22
**Status:** Draft, pending user review
**Scope:** Repository layout, development tooling, data strategy, I/O boundaries, testing discipline, and bootstrap sequence for the CHARMPheno project. Does **not** cover HDP implementation details, research methodology, or milestone planning — those live in `docs/architecture/` and are out of scope here.

---

## Context

CHARMPheno is a computational phenotyping project whose research vision is captured in three existing documents:

- [`docs/architecture/TOPIC_STATE_MODELING.md`](../../architecture/TOPIC_STATE_MODELING.md) — HDP-based phenotype discovery approach
- [`docs/architecture/SPARK_VI_FRAMEWORK.md`](../../architecture/SPARK_VI_FRAMEWORK.md) — reusable PySpark framework for distributed variational inference
- [`docs/architecture/RISKS_AND_MITIGATIONS.md`](../../architecture/RISKS_AND_MITIGATIONS.md) — known risks and constraints

These vision documents describe *what* is being built and *why*. This spec describes *how the codebase is organized and developed* — the scaffolding that makes the vision buildable, testable, and deployable in both local and cloud notebook environments.

The project will run in two environments:
- **Local development** (author's workstation): PySpark in local mode, small synthetic datasets, fast iteration.
- **Cloud notebook environment** (a GCS-backed Dataproc + Jupyter workbench): production-scale training against OMOP clinical data in BigQuery.

The setup must make code portable between these with minimal friction, and must prevent clinical-data artifacts from entering source control.

## Goals

1. **Single-repo, forward-compatible layout.** Two internal Python packages (`spark-vi`, `charmpheno`) plus analysis scripts and notebooks, structured so the packages can later be split into dedicated repositories with minimal work.
2. **Fast local iteration.** Default test suite runs in under ten seconds.
3. **Dual-delivery packaging.** Both packages buildable as pip-installable wheels *and* as flat `.zip` archives for Spark executor distribution.
4. **Explicit, boringly-auditable I/O.** OMOP data-loading primitives are narrow, environment-unaware functions. No auto-detection or environment-sniffing layers.
5. **Hard data-safety invariants.** Clinical data never enters the working tree; pre-commit hooks enforce this mechanically.
6. **Living architectural docs.** The vision docs are treated as authoritative contract, updated in lockstep with code. Drift is a flag, not a norm.
7. **Understanding as a first-class deliverable.** Both the codebase organization and the math-heavy code ship with exposition: package boundaries and module responsibilities are explicit and navigable from top-level docs; non-obvious math is explained in docstrings, linked `docs/architecture/` sections, and tutorial notebooks. Complementing this, **organization is treated as living, not frozen** — major refactors are an expected part of the project's evolution, not an exceptional event.

## Non-goals

- HDP algorithm implementation details (deferred to a later spec).
- Real BigQuery data access (scaffolded only; implementation deferred).
- Continuous integration setup (manual testing until a later decision).
- The Sparse OU (Stage 2) model (explicitly deferred; may later appear inside `charmpheno/` as an extension example, not inside `spark-vi/`).
- Recovery-metric regression testing as a gating signal (kept as a reported diagnostic only; real quantitative validation is deferred to cluster-scale work).

## Architecture

The Architecture section below describes the **final intended structure** of the repository and packages. The Bootstrap section further down describes what of that structure actually lands in the first pass — many module directories exist as placeholder packages with stub implementations until dedicated follow-on specs fill them in.

### Repository skeleton

```
CHARMPheno/
├── README.md
├── AGENTS.md                       # orientation for coding agents
├── Makefile                        # top-level orchestrator
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version                 # 3.10 floor
│
├── docs/
│   ├── architecture/               # living vision documents
│   │   ├── TOPIC_STATE_MODELING.md
│   │   ├── SPARK_VI_FRAMEWORK.md
│   │   └── RISKS_AND_MITIGATIONS.md
│   ├── decisions/                  # ADRs (architecture decision records)
│   │   ├── README.md
│   │   ├── 0001-single-repo-layout.md
│   │   ├── 0002-package-boundaries.md
│   │   ├── 0003-explicit-omop-io.md
│   │   └── 0004-pure-python-zip-invariant.md
│   └── superpowers/specs/          # design docs (this file lives here)
│
├── spark-vi/                       # poetry project #1
│   ├── pyproject.toml
│   ├── Makefile
│   ├── README.md
│   ├── spark_vi/                   # flat-layout, zip-compatible package
│   │   ├── __init__.py
│   │   ├── core/                   # VIModel, VIRunner, VIConfig, VIResult
│   │   ├── models/                 # OnlineHDP (generic / domain-agnostic)
│   │   ├── diagnostics/            # ELBO tracker, live display, checkpointing
│   │   └── io/                     # model export (JSON + .npy)
│   ├── tests/
│   │   ├── conftest.py             # local Spark fixture
│   │   └── data/                   # tiny committed test data
│   └── dist/                       # gitignored; wheel, sdist, zip
│
├── charmpheno/                     # poetry project #2 (depends on spark-vi)
│   ├── pyproject.toml
│   ├── Makefile
│   ├── README.md
│   ├── charmpheno/
│   │   ├── __init__.py
│   │   ├── phenotype/              # CharmPhenoHDP (wraps spark_vi OnlineHDP)
│   │   ├── omop/                   # load_omop_parquet, load_omop_bigquery, validate
│   │   ├── evaluate/               # recovery metrics (reported, not asserted)
│   │   ├── export/                 # downstream export artifacts
│   │   └── profiles/               # per-patient profile construction + viz
│   ├── tests/
│   │   ├── conftest.py
│   │   └── data/
│   └── dist/                       # gitignored
│
├── analysis/                       # runnable, environment-specific entrypoints
│   ├── local/                      # local-dev scripts (parquet I/O)
│   └── cloud/                      # cloud-notebook scripts (BigQuery I/O)
│
├── notebooks/                      # committed, output-stripped, thin drivers
│   ├── tutorials/                  # pedagogical exposition
│   └── exploratory/                # gitignored — free-form play
│
├── scripts/
│   ├── fetch_lda_beta.py           # HF streaming + top-K filter → parquet
│   ├── simulate_lda_omop.py        # β + θ prior → synthetic OMOP parquet
│   ├── check_no_data_files.sh      # pre-commit helper
│   └── setup_dev.sh                # one-shot local bootstrap
│
└── data/                           # gitignored entirely
    ├── cache/                      # filtered β parquet
    └── simulated/                  # simulator outputs
```

### Package boundaries and delivery

- **`spark-vi`** is a pure-Python, domain-agnostic PySpark framework for distributed variational inference.
  - Public API: `spark_vi.core.{VIModel, VIRunner, VIConfig, VIResult}`.
  - Ships a generic `spark_vi.models.OnlineHDP` suitable for any bag-of-words / topic-model workload.
  - **Must not import `charmpheno` or any clinical, OMOP, or BigQuery code.** Enforced by convention + reviewer agent; no structural mechanism required beyond not doing it.
- **`charmpheno`** is the clinical specialization.
  - Depends on `spark-vi` (conservative minor-version pin).
  - Wraps `spark_vi.models.OnlineHDP` with OMOP semantics, concept-vocabulary handling, recovery metrics, downstream export, per-patient profile construction.
  - Owns the I/O primitives that know about OMOP shape.
- **`analysis/`** holds thin end-to-end entrypoint scripts that compose the two packages for specific runs. `analysis/local/` and `analysis/cloud/` split by environment — same structural logic, different I/O calls.
- **`notebooks/`** are thin drivers only. Algorithms never live in notebook cells. `notebooks/tutorials/` is exposition; `notebooks/exploratory/` is gitignored.

### Dual delivery (pip and zip, per package)

Each package emits four artifacts on `make build && make zip`:

| Artifact | Consumed where |
|---|---|
| `<pkg>-X.Y.Z-py3-none-any.whl` | Driver / notebook via `pip install` |
| `<pkg>-X.Y.Z.tar.gz` | PyPI publication (future) |
| `<pkg>.zip` | Spark executors via `--py-files` or `sc.addPyFile()` |
| `dist/` contents | Synced to cloud-notebook environment's object storage per release |

The zip target is a flat `zip -r dist/<pkg>.zip <pkg>/` — which requires the **pure-Python, flat-layout invariant** (ADR 0004). No C extensions, no build-time codegen, no conditional imports of non-standard deps at module import time. This invariant is what makes `--py-files` delivery possible; breaking it silently breaks remote execution.

### I/O primitives (explicit, not auto-detecting)

Two loaders, each obvious about where data comes from:

```python
# charmpheno/omop/local.py
def load_omop_parquet(path: str, *, spark: SparkSession) -> DataFrame: ...

# charmpheno/omop/bigquery.py
def load_omop_bigquery(*, spark: SparkSession, cdr_dataset: str,
                       concept_types=("condition",), limit=None) -> DataFrame: ...
```

Both return the canonical OMOP-shaped Spark DataFrame:

```
person_id:int, visit_occurrence_id:int, concept_id:int, concept_name:str
```

A `charmpheno.omop.validate(df)` function asserts this shape and fails loudly. Scripts call it at the top of every entrypoint. No environment sniffing layer chooses between the two — the caller imports the one it wants.

Rationale (ADR 0003): environment-switching I/O hides behavior, invites magic, and complicates testing. Explicit primitives are boring and auditable.

### Test data strategy

Three distinct roles, kept strictly separate:

**1. Ground-truth β from prior LDA work.**
Source of truth: Hugging Face dataset `oneilsh/lda_pasc`. `scripts/fetch_lda_beta.py` streams it (no full download), keeps top-K (default 1000) concepts per topic by `term_weight`, renormalizes each topic row, writes `data/cache/lda_beta_topk.parquet` (~15 MB at K=1000).

**2. Synthetic OMOP via LDA generative process.**
`scripts/simulate_lda_omop.py` draws θ ∼ Dirichlet(α) per patient (with configurable sparse-active-topics structure), samples visits and codes per visit using the filtered β, and writes a parquet with columns `person_id, visit_occurrence_id, concept_id, concept_name, true_topic_id`. The `true_topic_id` column is oracle metadata — it's in the fixture, not in the model's input path. Training code never reads it.

**3. Unit-test data.**
Tiny, hand-crafted, committed under each package's `tests/data/`. No more than ~50 rows per file. When a unit test fails, the human should understand why in seconds. Unit tests never touch `data/cache/` or `data/simulated/`.

### Git and repo hygiene

- `data/` is globally gitignored. Committed sample data lives only under `tests/*/data/`.
- `.pre-commit-config.yaml` runs: `nbstripout`, `check-added-large-files` (1 MB cap), `trailing-whitespace`, `end-of-file-fixer`, `check-merge-conflict`, and a local hook that rejects `.parquet`/`.csv`/`.feather`/`.arrow`/`.npz` files outside `tests/*/data/` or `docs/`.
- Generic data-hygiene rule: work with clinical data only in its approved environment; clinical data must never enter the working tree.
- The repo supports the standard cloud-notebook pattern of syncing built wheels and zips to an object-storage location per release; the exact commands live in README/tutorials, not in library code.

### Testing strategy

Three tiers, marker-gated:

| Tier | Marker | Duration target | Triggered by |
|---|---|---|---|
| Unit | (none) | < 10s total | `make test` (default loop) |
| Integration | `@pytest.mark.slow` | 1–3 min | `make test-all` |
| Cluster | `@pytest.mark.cluster` | varies | `make test-cluster`, manual only |

`pyproject.toml` default: `addopts = "-m 'not slow and not cluster'"`. Unit + integration together must finish under five minutes locally; if they creep past, downsize N or split further rather than loosen the budget.

**Unit tests** (both packages) exercise the contract against tiny fixtures and in-memory toy data. No external I/O. For `spark-vi`: a trivial `CountingModel` exercises the VIModel contract end-to-end. For `charmpheno`: OMOP shape validation, tiny HDP convergence, Hungarian-alignment identity.

**Integration tests** consume a small simulator output (e.g., N=1000 patients), run a real HDP fit, and assert that ELBO improves, the fit doesn't OOM, per-patient θ is non-degenerate. Recovery score vs. ground-truth β is **reported as a diagnostic** (logged + sidecar), not asserted — given the scale mismatch between the β's original training corpus (hundreds of millions of conditions) and laptop-scale simulations (thousands of patients), strict recovery thresholds would be noisy and misleading locally.

**Cluster tests** validate deployment shape (wheels install from cloud storage, executors import from zips, version strings match across partitions), BigQuery loader shape, and real-scale HDP sanity. Three tests total, no more.

### Understanding as first-class (in committed artifacts)

"Understanding" here covers two dimensions — the codebase's organization, and its math-heavy implementations. Both are expected to be legible.

**Codebase organization:**

- The top-level README and AGENTS.md name the major boundaries (packages, modules, analysis vs. library code) and explain *why* they exist as they do.
- Every package and top-level subpackage ships a short README or `__init__.py` docstring that answers: what lives here, what depends on it, what it depends on.
- `docs/decisions/` (ADRs) records the *why* of each significant organizational choice. When a new layout choice is made, a new ADR records it; when a later refactor supersedes an earlier decision, the new ADR names what it supersedes.
- `docs/architecture/` describes the intended end-state architecture. Where current code differs (e.g., stubs, deferred pieces), that's acknowledged explicitly rather than papered over.
- **Refactors are expected.** As the project evolves, module boundaries, package structure, and internal interfaces will change. Refactors are recorded in ADRs, not treated as exceptional events — the scaffolding, tests, and docs are designed so that cross-cutting reorganization remains tractable.

**Math-heavy code:**

- Docstrings explain *why* a formula takes its shape, not just what it computes. Derivations link to anchored sections in `docs/architecture/`.
- `notebooks/tutorials/NN_<concept>.ipynb` builds intuition for concepts a reader has to understand to maintain the code — toy examples, visualizations, short narrative. Output-stripped on commit.
- When a non-obvious math function is written, the docstring's "why" comes before the implementation. If it can't be explained, the implementation pauses until it can.

**For both dimensions:** drift between `docs/architecture/` (or `docs/decisions/`) and code is a flag — either update the doc in the same change or write a new ADR recording the departure. No silent divergence.

### ADR convention

`docs/decisions/NNNN-<slug>.md`, ~200 words each: Context, Decision, Alternatives, Consequences. Written in project-level voice, impersonal. The first four ADRs record the decisions made in this spec (Sections 1–3 of the brainstorm above).

### Development tooling

- Python: `>=3.10,<3.13`. Floor matches common Dataproc images; ceiling avoids unknown image drift.
- PySpark: `>=3.5,<4.0`.
- Java 17 for local Spark (Makefile autodetects macOS Homebrew and Linux paths).
- Poetry for both packages.
- One venv per package; `spark-vi` installed editable into the `charmpheno` venv for dev workflow.

## Bootstrap sequence

The bootstrap gets from "no git repo, brpmatch reference folder still present" to "both packages installable, `make test` green, synthetic data generatable, end-to-end smoke runs on local Spark." No CharmPheno science yet — just the ground to stand on. Each phase is a small, reviewable commit.

**Phase 1 — Repo init & hygiene.**
Lift brpmatch templates (Makefile, pyproject, conftest.py, .gitignore) into a temporary `_templates/` dir. Remove `brpmatch/` and `attic/`. Move the three architecture docs into `docs/architecture/`. Create empty target dirs. Write `.gitignore`, `.pre-commit-config.yaml`, `.python-version`, `AGENTS.md`, top-level `Makefile`, README update. `git init`, initial commit, push to fresh GitHub repo (private to start). Install pre-commit hooks. Write ADRs 0001–0004. Remove `_templates/` as each piece lands in its final home.

**Phase 2 — Data pipeline.**
Implement `scripts/fetch_lda_beta.py` (HF streaming + top-K filter + renormalize → parquet) and `scripts/simulate_lda_omop.py` (β + θ prior + visit/code length distributions → synthetic OMOP parquet). Wire through top-level Makefile as `make data`. Produce small fixtures under each package's `tests/data/`. Phase 2 is feasible before the framework because it uses only pandas/pyarrow/numpy — no PySpark required for the generator itself.

**Phase 3 — spark-vi skeleton.**
`spark-vi/` project with minimum viable `VIModel` / `VIRunner` / `VIConfig` / `VIResult`. A trivial `CountingModel` (coin-flip posterior, tens of lines) exercises the full contract end-to-end — this is what `make test` runs against. `spark_vi.models.OnlineHDP` exists as a stub class with the intended public signature and a `NotImplementedError`-raising body; its real implementation is a dedicated follow-on spec. Tests include: contract shape (does a minimal subclass work?), broadcast lifecycle (`unpersist()` called on prior broadcasts — see RISKS §Framework Risks), checkpoint round-trip, export format round-trip. Both `make build` and `make zip` produce artifacts that import in a fresh venv.

*Reference material for implementers:* consult `docs/architecture/SPARK_VI_FRAMEWORK.md` §Framework Architecture and §Prior Art & Positioning before starting. The Spark MLlib `OnlineLDAOptimizer` source is the most faithful reference implementation of the broadcast → mapPartitions → treeAggregate pattern (cited at [SPARK_VI_FRAMEWORK.md:119, 139](../../architecture/SPARK_VI_FRAMEWORK.md#L119)); the intel-spark TopicModeling Scala implementation is a second reference with a documented gotcha to avoid ([TOPIC_STATE_MODELING.md:475-478](../../architecture/TOPIC_STATE_MODELING.md#L475)). Theoretical grounding: Hoffman et al. 2010/2013 (both cited).

**Phase 4 — charmpheno skeleton.**
`charmpheno/` project with pyproject, editable dev dep on `spark-vi`. Implement `charmpheno.omop.load_omop_parquet` and `validate` as working code. `CharmPhenoHDP` is added as a thin wrapper class around the Phase 3 `spark_vi.models.OnlineHDP` stub — the wrapper's public surface is real, the underlying HDP remains stubbed until its dedicated spec. `load_omop_bigquery` exists as a signature with `NotImplementedError` and a docstring describing the planned implementation. Unit tests green.

**Phase 5 — End-to-end smoke.**
`analysis/local/fit_charmpheno_local.py` loads the simulator parquet, constructs a `CharmPhenoHDP` around the stub `OnlineHDP`, runs the `CountingModel` (or a similarly trivial model) through `VIRunner` against the data, and writes a model artifact. The goal is to prove the data → framework → export path works end-to-end; it is explicitly not yet running real HDP math. An `@pytest.mark.slow` integration test covers the same path. `make test-all` green.

**Phase 6 — First tutorial.**
`notebooks/tutorials/01_project_setup.ipynb` covers: clone → local env setup (Python, Java 17) → `make install` → `make data` → `make test` → run the end-to-end smoke → short appendix showing how the cloud-deployed notebook header differs. Serves as onboarding, not math exposition. Math-heavy tutorials come later, one per non-obvious concept as it is implemented.

### End state after bootstrap

- Fresh machine can clone, `make install`, `make data`, `make test`, run the end-to-end smoke.
- Both packages build wheels and zips cleanly.
- ADRs 0001–0004 committed; architecture docs live in `docs/architecture/`.
- Pre-commit hooks active; all guard rails in place.
- One tutorial orients a new contributor to the codebase.
- HDP algorithm, BigQuery loader, recovery metric, and all science work start from this foundation.

### Post-bootstrap workflow

Each subsequent piece of work follows the same flow:
- New design doc in `docs/superpowers/specs/` (via the brainstorming skill).
- Implementation plan via the writing-plans skill.
- New ADR in `docs/decisions/` if the work introduces or shifts an architectural boundary.
- Tutorial notebook in `notebooks/tutorials/` if the work introduces a new math concept.

## Risks and mitigations (for this setup design)

- **Drift between docs, code, and tutorials.** Mitigated by (a) "doc update in same commit" rule for architectural changes, (b) anchor comments in code pointing to `docs/architecture/` sections, (c) periodic explicit reconciliation at milestones.
- **Zip-compatibility invariant broken silently.** Mitigated by `make zip` being part of every release build; if a future dep introduces C extensions or build-time codegen, the zip target fails and the constraint is re-negotiated explicitly.
- **Integration tests creep past the time budget.** Mitigated by a hard stated budget (under five minutes combined) and splitting tiers rather than loosening.
- **Notebook output accidentally committed.** Mitigated by `nbstripout` pre-commit hook and the `check-added-large-files` hook (1 MB cap), both of which fire before a commit lands.

## Open questions deferred to later specs

- Real Online HDP implementation math and Spark data-flow (its own spec; references the MLlib and intel-spark prior implementations).
- Real BigQuery loader (its own spec; will include query pushdown considerations).
- Recovery-metric methodology and what threshold to use at cluster scale (deferred).
- CI platform decision (GitHub Actions for unit+integration; cluster tests stay manual).
- Sparse OU as a demonstration of extending `VIModel` in `charmpheno/` (separate spec if pursued).
