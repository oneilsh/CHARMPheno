# AGENTS.md

Orientation for LLM-based coding agents working on CHARMPheno. Humans can read
it too; it is not a user-facing README.

## Read these before suggesting architectural changes

The architectural vision lives in [`docs/architecture/`](docs/architecture/):
- `TOPIC_STATE_MODELING.md` — research design for phenotype discovery.
- `SPARK_VI_FRAMEWORK.md` — framework design (VIRunner, VIModel contract).
- `RISKS_AND_MITIGATIONS.md` — known risks, constraints, deployment notes.

These are **living** documents. If a change contradicts them, either update the
relevant section in the same commit, or write an ADR in
[`docs/decisions/`](docs/decisions/) recording the departure. Never silently
diverge.

## Understanding is a first-class deliverable

**Codebase organization** and math-heavy code are both expected to be legible.

- Top-level README and this file name the major boundaries. Each package and
  top-level subpackage carries a short README or `__init__.py` docstring
  answering: what lives here, what depends on it, what it depends on.
- ADRs in `docs/decisions/` record the *why* of significant organizational
  choices. New decisions get new ADRs; later refactors supersede earlier ADRs
  by name.
- **Refactors are expected.** As the project evolves, module boundaries and
  interfaces will change. Refactors are recorded in ADRs, not treated as
  exceptional events.
- Docstrings for non-obvious math explain *why* the formula takes its shape,
  not just what it computes. Derivations link to anchored sections in
  `docs/architecture/` rather than being duplicated.
- `notebooks/tutorials/NN_<concept>.ipynb` builds intuition for concepts a
  reader has to understand to maintain the code. Output-stripped on commit.
- Before implementing a non-obvious math function, write the docstring's
  "why" first. If it can't be explained, the implementation pauses until
  it can.

## Project layout and boundaries

- `spark-vi/` is a pure-Python, domain-agnostic framework.
  - Public API: `spark_vi.core.{VIModel, VIRunner, VIConfig, VIResult}`.
  - Ships generic `spark_vi.models.OnlineHDP` (bag-of-words, not clinical).
  - **Must never import `charmpheno`** or clinical / OMOP / BigQuery code.
- `charmpheno/` is the clinical specialization.
  - Wraps `spark_vi.models.OnlineHDP` with OMOP semantics, concept vocab,
    downstream export, recovery-vs-ground-truth metrics.
  - May depend on `spark-vi`; the reverse is forbidden.
- `analysis/` holds runnable end-to-end scripts (thin).
- `notebooks/` holds thin drivers that import from `analysis/` or packages.
  Algorithms never live in notebook cells.

## Packaging invariants

- Both packages must stay **pure-Python, flat-layout** so `make zip` produces
  a `--py-files`-compatible archive for Spark executors.
- No C extensions. No build-time code generation. No conditional imports
  requiring non-standard dependencies at import time.
- Dual delivery: `make build` (wheel + sdist) AND `make zip` (flat archive).
  Both targets must stay green on every commit touching the packages.

## Data and repo hygiene

- `data/` is globally gitignored. Committed sample data lives only under
  `tests/*/data/` and is capped at ~50 rows per file.
- `.pre-commit-config.yaml` enforces: nbstripout, max-file-size (1 MB),
  no `.parquet` / `.csv` / `.feather` / `.arrow` / `.npz` files outside
  `tests/*/data/` or `docs/`.
- Work with clinical data only in its approved environment; do not check
  patient-level data into the working tree under any circumstances.

## Decision log

`docs/decisions/NNNN-<slug>.md` records architectural choices. When making a
new architectural call, check whether it refines, supersedes, or conflicts
with an existing ADR, and record accordingly.

## Testing expectations

- Default `make test` runs unit tests only and must finish in under ~10s.
- `@pytest.mark.slow` for integration tests (simulator data, local Spark,
  minutes-scale). Run via `make test-all`.
- `@pytest.mark.cluster` for tests that require a real Dataproc cluster.
  Manual only, triggered by `make test-cluster`.

## When you finish a change

Before declaring work complete:
- Tests pass (`make test` minimum; `make test-all` if integration surfaces
  changed).
- Relevant `docs/architecture/*.md` section updated if the change was
  architectural, or an ADR records the exception.
- Docstrings for new math functions explain *why*, not just *what*.
- No data files, secrets, or large binaries staged.
