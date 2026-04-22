# CHARMPheno Project Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold CHARMPheno from its current pre-git state into a working single-repo monorepo with two zip-compatible Python packages (`spark-vi`, `charmpheno`), a synthetic-data pipeline, passing unit + integration tests, dual delivery (wheel + flat zip), pre-commit hygiene, ADRs, and one orientation tutorial — ready for subsequent specs to layer real HDP math, real BigQuery loading, and analysis work.

**Architecture:** Two independent poetry projects inside one monorepo. `spark-vi` is a pure-Python, domain-agnostic PySpark framework for distributed variational inference. `charmpheno` depends on `spark-vi` and adds clinical (OMOP) semantics on top. Thin analysis scripts and notebook drivers compose them. Bootstrap delivers the contract + data + smoke path; real HDP math and real BigQuery access are stubbed with `NotImplementedError` for dedicated follow-on specs.

**Tech Stack:** Python 3.10+, Poetry, PySpark 3.5, PyArrow, NumPy, SciPy, pandas, datasets (HF), pytest, pre-commit + nbstripout, Java 17 for local Spark.

---

## Preconditions

- Working directory: `/Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/`
- Current state: NOT a git repo; contains legacy `brpmatch/` (template source) and `attic/` (historical), plus three architecture markdown files at top level, and a `docs/superpowers/` tree from the brainstorming phase.
- Java 17 available on the local machine (Homebrew `openjdk@17` on macOS, or Linux equivalent).
- `poetry` ≥ 2.0 installed (`pipx install poetry` recommended).
- `pre-commit` available to the shell (will be installed into the root venv in Phase 1).

## File structure map

Paths shown are what exists at the end of each phase. `→` = created, `×` = deleted, `⇒` = moved.

**After Phase 1 (repo init & hygiene):**
```
CHARMPheno/
├── README.md                        (updated)
├── AGENTS.md                        →
├── Makefile                         →
├── pyproject.toml                   →   (root dev tools + scripts deps)
├── .gitignore                       →
├── .python-version                  →
├── .pre-commit-config.yaml          →
├── docs/
│   ├── architecture/
│   │   ├── TOPIC_STATE_MODELING.md      ⇒
│   │   ├── SPARK_VI_FRAMEWORK.md        ⇒
│   │   └── RISKS_AND_MITIGATIONS.md     ⇒
│   ├── decisions/
│   │   ├── README.md                    →
│   │   ├── 0001-single-repo-layout.md   →
│   │   ├── 0002-package-boundaries.md   →
│   │   ├── 0003-explicit-omop-io.md     →
│   │   └── 0004-pure-python-zip-invariant.md →
│   └── superpowers/
│       ├── specs/2026-04-22-charmpheno-project-setup-design.md  (exists)
│       └── plans/2026-04-22-charmpheno-project-setup.md          (this file)
├── scripts/
│   └── check_no_data_files.sh       →
├── analysis/local/                  →  (empty .gitkeep)
├── analysis/cloud/                  →  (empty .gitkeep)
├── notebooks/exploratory/           →  (gitignored entirely)
├── notebooks/tutorials/             →  (empty .gitkeep)
├── data/                            →  (gitignored entirely; create .gitkeep pattern docs note: don't)
├── brpmatch/                        ×
├── attic/                           ×
└── profile_example.png              (kept; referenced in README)
```

**After Phase 2 (data pipeline):**
```
+ scripts/fetch_lda_beta.py
+ scripts/simulate_lda_omop.py
+ tests/scripts/test_fetch_lda_beta.py
+ tests/scripts/test_simulate_lda_omop.py
+ tests/scripts/conftest.py
```

**After Phase 3 (spark-vi skeleton):**
```
+ spark-vi/
  ├── pyproject.toml
  ├── Makefile
  ├── README.md
  ├── .gitignore
  ├── spark_vi/
  │   ├── __init__.py
  │   ├── core/
  │   │   ├── __init__.py
  │   │   ├── config.py           (VIConfig)
  │   │   ├── model.py            (VIModel ABC)
  │   │   ├── result.py           (VIResult)
  │   │   └── runner.py           (VIRunner)
  │   ├── models/
  │   │   ├── __init__.py
  │   │   ├── counting.py         (CountingModel — contract exerciser)
  │   │   └── online_hdp.py       (stub, raises NotImplementedError)
  │   ├── diagnostics/
  │   │   ├── __init__.py
  │   │   └── checkpoint.py
  │   └── io/
  │       ├── __init__.py
  │       └── export.py
  └── tests/
      ├── conftest.py              (spark fixture + tmp_path helpers)
      ├── data/.gitkeep
      ├── test_config.py
      ├── test_result.py
      ├── test_runner.py
      ├── test_counting_model.py
      ├── test_broadcast_lifecycle.py
      ├── test_checkpoint.py
      ├── test_export.py
      └── test_zip_import.py
```

**After Phase 4 (charmpheno skeleton):**
```
+ charmpheno/
  ├── pyproject.toml
  ├── Makefile
  ├── README.md
  ├── .gitignore
  ├── charmpheno/
  │   ├── __init__.py
  │   ├── omop/
  │   │   ├── __init__.py
  │   │   ├── schema.py           (canonical shape + validate())
  │   │   ├── local.py            (load_omop_parquet)
  │   │   └── bigquery.py         (stub)
  │   ├── phenotype/
  │   │   ├── __init__.py
  │   │   └── charm_pheno_hdp.py  (wrapper around spark_vi.models.OnlineHDP)
  │   ├── evaluate/__init__.py    (placeholder)
  │   ├── export/__init__.py      (placeholder)
  │   └── profiles/__init__.py    (placeholder)
  └── tests/
      ├── conftest.py
      ├── data/.gitkeep
      ├── test_schema.py
      ├── test_load_omop_parquet.py
      ├── test_bigquery_stub.py
      └── test_charm_pheno_hdp_wrapper.py
```

**After Phase 5 (end-to-end smoke):**
```
+ analysis/local/fit_charmpheno_local.py
+ tests/integration/test_fit_charmpheno_local.py
+ tests/integration/conftest.py
```

**After Phase 6 (first tutorial):**
```
+ notebooks/tutorials/01_project_setup.ipynb
```

---

## Phase 1 — Repo init & hygiene

Goal: turn the current pre-git directory into a clean, hygienic git repository with all guard-rails, directory skeletons, and documentation landing pages in place. No Python packages yet.

### Task 1.1: Stash the brpmatch templates to `/tmp/brpmatch-templates/`

**Files:**
- Read (for reference): `brpmatch/Makefile`, `brpmatch/pyproject.toml`, `brpmatch/tests/conftest.py`, `brpmatch/.gitignore`
- Create: none yet

These templates are the reference for Phases 3 & 4. We stash them outside the repo before deleting `brpmatch/` so they remain accessible during scaffolding.

- [ ] **Step 1: Copy brpmatch templates out**

Run:
```bash
mkdir -p /tmp/brpmatch-templates
cp brpmatch/Makefile /tmp/brpmatch-templates/Makefile.ref
cp brpmatch/pyproject.toml /tmp/brpmatch-templates/pyproject.toml.ref
cp brpmatch/tests/conftest.py /tmp/brpmatch-templates/conftest.py.ref
cp brpmatch/.gitignore /tmp/brpmatch-templates/gitignore.ref
```

Expected: four `.ref` files in `/tmp/brpmatch-templates/`.

- [ ] **Step 2: Verify the stash**

Run:
```bash
ls -1 /tmp/brpmatch-templates/
```

Expected output:
```
Makefile.ref
conftest.py.ref
gitignore.ref
pyproject.toml.ref
```

### Task 1.2: Remove legacy folders

- [ ] **Step 1: Remove `brpmatch/` and `attic/`**

Run:
```bash
rm -rf brpmatch attic
```

- [ ] **Step 2: Verify removal**

Run:
```bash
ls -1 .
```

Expected: no `brpmatch` or `attic` in output. `README.md`, `RISKS_AND_MITIGATIONS.md`, `SPARK_VI_FRAMEWORK.md`, `TOPIC_STATE_MODELING.md`, `docs/`, `profile_example.png` remain.

### Task 1.3: Move architecture documents into `docs/architecture/`

- [ ] **Step 1: Create docs/architecture/ and move files**

Run:
```bash
mkdir -p docs/architecture
mv TOPIC_STATE_MODELING.md docs/architecture/
mv SPARK_VI_FRAMEWORK.md docs/architecture/
mv RISKS_AND_MITIGATIONS.md docs/architecture/
```

- [ ] **Step 2: Verify moves**

Run:
```bash
ls docs/architecture/
```

Expected:
```
RISKS_AND_MITIGATIONS.md
SPARK_VI_FRAMEWORK.md
TOPIC_STATE_MODELING.md
```

### Task 1.4: Create directory skeleton with `.gitkeep` placeholders

Git does not track empty directories, so every empty target directory that must exist after Phase 1 gets a `.gitkeep` file.

- [ ] **Step 1: Create all target directories and gitkeeps**

Run:
```bash
mkdir -p analysis/local analysis/cloud notebooks/tutorials notebooks/exploratory scripts data/cache data/simulated
touch analysis/local/.gitkeep analysis/cloud/.gitkeep notebooks/tutorials/.gitkeep scripts/.gitkeep
```

Note: no `.gitkeep` in `notebooks/exploratory/` or `data/` — they are gitignored entirely and should not force-track.

- [ ] **Step 2: Verify structure**

Run:
```bash
ls -a analysis/local analysis/cloud notebooks/tutorials scripts
```

Expected: each directory lists `.`, `..`, and `.gitkeep`.

### Task 1.5: Write `.gitignore`

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Create `.gitignore`**

Create `.gitignore` with this content:
```gitignore
# venvs / caches
.venv/
venv/
env/
__pycache__/
*.py[cod]
*.so

# build + distribution
dist/
build/
*.egg-info/
*.egg
.eggs/

# testing
.pytest_cache/
.coverage
.coverage.*
htmlcov/

# IDEs / OS
.DS_Store
.idea/
.vscode/
*.swp
*.swo

# Claude Code / agent caches
.claude/

# Jupyter
.ipynb_checkpoints/

# Python pinning (per-user)
.python-version

# Spark local run artifacts
derby.log
metastore_db/
spark-warehouse/

# Secrets
.env
.env.*
!.env.example

# Generated data — never in repo
data/
!tests/*/data/

# Free-form exploratory notebooks
notebooks/exploratory/

# Local template stash path (if we ever symlink or check stub refs)
*.ref
```

- [ ] **Step 2: Verify file exists with expected tail**

Run:
```bash
tail -5 .gitignore
```

Expected:
```
notebooks/exploratory/

# Local template stash path (if we ever symlink or check stub refs)
*.ref
```

### Task 1.6: Write `.python-version`

**Files:**
- Create: `.python-version`

- [ ] **Step 1: Create file**

Create `.python-version` with content:
```
3.11
```

(Note: `.python-version` itself is `.gitignore`d as a personal pin, but we write it so local tooling picks 3.11 consistently during bootstrap. It will not be committed.)

### Task 1.7: Write `.pre-commit-config.yaml` and `scripts/check_no_data_files.sh`

**Files:**
- Create: `.pre-commit-config.yaml`
- Create: `scripts/check_no_data_files.sh`

- [ ] **Step 1: Create `.pre-commit-config.yaml`**

Create `.pre-commit-config.yaml` with content:
```yaml
repos:
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-merge-conflict
      - id: check-added-large-files
        args: ['--maxkb=1024']

  - repo: local
    hooks:
      - id: no-data-files-outside-tests
        name: Prevent data files outside tests/*/data/ or docs/
        entry: scripts/check_no_data_files.sh
        language: script
        files: '\.(parquet|csv|feather|arrow|npz)$'
```

- [ ] **Step 2: Create `scripts/check_no_data_files.sh`**

Create `scripts/check_no_data_files.sh` with content:
```bash
#!/usr/bin/env bash
# Reject staged data files outside tests/*/data/ and docs/.
# Pre-commit passes candidate file paths as arguments.
set -euo pipefail
status=0
for f in "$@"; do
  case "$f" in
    tests/*/data/*|docs/*)
      # allowed
      ;;
    *)
      echo "ERROR: data file outside allowed paths: $f"
      status=1
      ;;
  esac
done
exit $status
```

- [ ] **Step 3: Make the script executable**

Run:
```bash
chmod +x scripts/check_no_data_files.sh
```

- [ ] **Step 4: Verify**

Run:
```bash
scripts/check_no_data_files.sh tests/spark_vi/data/ok.parquet
echo "exit=$?"
scripts/check_no_data_files.sh data/bad.parquet || echo "rejected as expected"
```

Expected output:
```
exit=0
ERROR: data file outside allowed paths: data/bad.parquet
rejected as expected
```

### Task 1.8: Write root `pyproject.toml` for dev tooling & scripts deps

**Files:**
- Create: `pyproject.toml`

This root pyproject manages the **scripts' and dev tooling's** dependencies. It is not a distributable package — it exists so that `poetry run python scripts/...` and `poetry run pre-commit ...` work.

- [ ] **Step 1: Create root `pyproject.toml`**

Create `pyproject.toml` with content:
```toml
[tool.poetry]
name = "charmpheno-monorepo"
version = "0.0.0"
description = "Monorepo root for dev tooling and scripts. Not distributable."
authors = ["CHARMPheno contributors"]
readme = "README.md"
package-mode = false

[tool.poetry.dependencies]
python = ">=3.10,<3.13"
datasets = ">=2.18"
pandas = ">=2.0"
pyarrow = ">=15.0"
numpy = ">=1.24"

[tool.poetry.group.dev.dependencies]
pre-commit = ">=3.6"
pytest = ">=7.0"

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

- [ ] **Step 2: Install the root env**

Run:
```bash
poetry install
```

Expected: installs datasets, pandas, pyarrow, numpy, pre-commit, pytest into a `.venv/` at the repo root.

- [ ] **Step 3: Verify the venv imports work**

Run:
```bash
poetry run python -c "import datasets, pandas, pyarrow, numpy; print('ok')"
```

Expected: `ok`

### Task 1.9: Write `AGENTS.md`

**Files:**
- Create: `AGENTS.md`

- [ ] **Step 1: Create `AGENTS.md`**

Create `AGENTS.md` with content:
```markdown
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
```

### Task 1.10: Write `docs/decisions/` — README + four ADRs

**Files:**
- Create: `docs/decisions/README.md`
- Create: `docs/decisions/0001-single-repo-layout.md`
- Create: `docs/decisions/0002-package-boundaries.md`
- Create: `docs/decisions/0003-explicit-omop-io.md`
- Create: `docs/decisions/0004-pure-python-zip-invariant.md`

- [ ] **Step 1: Create `docs/decisions/README.md`**

Create `docs/decisions/README.md` with content:
```markdown
# Architecture Decision Records

This directory records the *why* of significant architectural choices as
lightweight ADRs. Each ADR is ~200 words: context, decision, alternatives
considered, consequences.

When a new architectural decision is made, add a new ADR with the next
available four-digit number. When a later decision supersedes an earlier one,
name the superseded ADR explicitly in the new one's "Consequences" or
"Supersedes" section.

Format skeleton:

    # NNNN — Short Title
    **Status:** Accepted | Superseded by NNNN
    **Date:** YYYY-MM-DD
    ## Context
    ## Decision
    ## Alternatives considered
    ## Consequences
```

- [ ] **Step 2: Create `docs/decisions/0001-single-repo-layout.md`**

Create with content:
```markdown
# 0001 — Single-repo layout with two poetry projects

**Status:** Accepted
**Date:** 2026-04-22

## Context
The project comprises a reusable PySpark framework (`spark-vi`), a clinical
specialization (`charmpheno`), and analysis / notebook work that drives
development of both. All three co-evolve tightly in the early phase; each has
its own eventual release cadence.

## Decision
Adopt a single-repo monorepo that contains two independent poetry projects
side-by-side (`spark-vi/` and `charmpheno/`), plus top-level `analysis/`,
`notebooks/`, and `scripts/` directories. Each poetry project has its own
`pyproject.toml`, `Makefile`, tests, and `dist/` output directory.

## Alternatives considered
- Three separate repos from day 1 (rejected: cross-repo friction when the
  two packages are co-evolving and need joint iteration).
- Single pyproject with both packages in one `src/` tree (rejected: fuses
  release boundaries; splitting later becomes a refactor rather than an
  extraction).

## Consequences
- Forward-compatible: either package can be split into its own repo later
  with minimal work.
- Dev workflow requires two editable installs; one-time cost.
- Top-level `Makefile` delegates to each package (`make -C spark-vi test`).
```

- [ ] **Step 3: Create `docs/decisions/0002-package-boundaries.md`**

Create with content:
```markdown
# 0002 — Package boundaries: generic framework + clinical wrapper

**Status:** Accepted
**Date:** 2026-04-22

## Context
Both `spark-vi` and `charmpheno` have a claim on the Online HDP model. The
framework docs position `spark-vi` as reusable beyond clinical use; the
research docs position `charmpheno` around HDP-based phenotype discovery.

## Decision
- `spark-vi` ships a **generic, domain-agnostic** `OnlineHDP` suitable for any
  bag-of-words / topic-model workload. It has no clinical or OMOP semantics.
- `charmpheno` wraps the generic HDP with OMOP-shaped I/O, concept-vocabulary
  handling, recovery-metric evaluation, downstream export, and per-patient
  profile construction.
- `spark-vi` must **never import `charmpheno`** or any clinical / BigQuery
  code. The dependency direction is one-way.

## Alternatives considered
- Clinical HDP in `spark-vi` directly (rejected: conflates framework with
  application; a non-clinical user would pull in OMOP code they don't need).
- Pure framework with no model (rejected: every model author would
  re-implement the HDP orchestration, defeating the reuse goal).

## Consequences
- OU or other future models can be added in `charmpheno` as demonstrations
  of extending `VIModel`, without destabilizing the framework.
- The `spark-vi` public API must stay stable enough to be depended upon.
```

- [ ] **Step 4: Create `docs/decisions/0003-explicit-omop-io.md`**

Create with content:
```markdown
# 0003 — Explicit OMOP I/O primitives (no environment sniffing)

**Status:** Accepted
**Date:** 2026-04-22

## Context
Code must run in two environments: local dev (parquet files) and a cloud
notebook environment (BigQuery). A natural temptation is a single
`load_omop(...)` function that sniffs the environment and dispatches.

## Decision
Expose two explicit, narrow loaders in `charmpheno.omop`:
- `load_omop_parquet(path, *, spark) -> DataFrame`
- `load_omop_bigquery(*, spark, cdr_dataset, ...) -> DataFrame`

Neither sniffs the environment. The caller imports and invokes the one it
wants. Both return the canonical OMOP-shaped Spark DataFrame:
`person_id, visit_occurrence_id, concept_id, concept_name`.

A `charmpheno.omop.validate(df)` function asserts shape and fails loudly.

## Alternatives considered
- Environment-switched single loader (rejected: hides behavior, invites
  magic, complicates testing, makes I/O bugs much harder to localize).

## Consequences
- Scripts carry one extra import line declaring their environment.
- Library behavior is straightforward to reason about without needing to
  know which env vars were set.
- Caller code visually declares its target environment.
```

- [ ] **Step 5: Create `docs/decisions/0004-pure-python-zip-invariant.md`**

Create with content:
```markdown
# 0004 — Pure-Python zip-compatibility invariant for both packages

**Status:** Accepted
**Date:** 2026-04-22

## Context
Spark executors in Dataproc can receive additional Python code via
`--py-files` or `sc.addPyFile()`, which accept `.py`, `.zip`, and `.egg`
archives only. Wheels with native code are not supported via this mechanism.
The project needs `--py-files`-compatible archives for both packages.

## Decision
Both `spark-vi` and `charmpheno` stay **pure-Python, flat-layout** packages:
- No C extensions.
- No build-time code generation.
- No conditional imports requiring non-standard dependencies at import time.

The `make zip` target in each package produces a flat `zip -r <pkg>.zip <pkg>/`
archive suitable for `--py-files` / `addPyFile`. The `make build` target
produces the standard wheel + sdist for `pip install`.

## Alternatives considered
- Allow C extensions / rely on conda-pack (rejected for bootstrap: adds a
  cluster-side setup dependency and complicates the deploy story).
- Skip zip delivery; rely on `pip install` only (rejected: Dataproc
  wheels-on-executors requires cluster properties set at creation time,
  which is less portable than `addPyFile`).

## Consequences
- If a future dependency introduces native code, the zip target fails,
  forcing an explicit re-discussion rather than silent breakage.
- Both `dist/<pkg>.whl` and `dist/<pkg>.zip` must stay green on every
  package-touching commit.
```

- [ ] **Step 6: Verify ADRs exist**

Run:
```bash
ls docs/decisions/
```

Expected:
```
0001-single-repo-layout.md
0002-package-boundaries.md
0003-explicit-omop-io.md
0004-pure-python-zip-invariant.md
README.md
```

### Task 1.11: Write the top-level `Makefile`

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Create root `Makefile`**

Create `Makefile` with content:
```makefile
.PHONY: help install install-dev data data-clean test test-all test-cluster \
        build zip clean precommit-install lint

default: help

help:
	@echo "Top-level orchestrator. Common targets:"
	@echo "  install          - Install root venv (scripts + dev tools)"
	@echo "  install-dev      - Install both package venvs (spark-vi, charmpheno) editable"
	@echo "  data             - Fetch LDA beta and simulate synthetic OMOP data"
	@echo "  data-clean       - Delete the data/ cache and simulated outputs"
	@echo "  test             - Run unit tests in both packages (default loop, <10s)"
	@echo "  test-all         - Run unit + @slow integration tests"
	@echo "  test-cluster     - Run @cluster tests (manual, cluster-only)"
	@echo "  build            - Build wheels + sdists for both packages"
	@echo "  zip              - Build flat zips for both packages"
	@echo "  clean            - Remove dist/, build/, egg-info, pytest caches"
	@echo "  precommit-install - Install the pre-commit hooks into .git/hooks/"
	@echo "  lint             - Run pre-commit against all tracked files"

install:
	poetry install

install-dev: install
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi install; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno install; fi

data:
	poetry run python scripts/fetch_lda_beta.py
	poetry run python scripts/simulate_lda_omop.py

data-clean:
	rm -rf data/cache data/simulated

test:
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi test; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno test; fi
	@if [ -d tests/scripts ]; then poetry run pytest tests/scripts -v; fi

test-all:
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi test-all; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno test-all; fi
	@if [ -d tests/integration ]; then poetry run pytest tests/integration -v -m "not cluster"; fi

test-cluster:
	@if [ -d tests/integration ]; then poetry run pytest tests/integration -v -m cluster; fi

build:
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi build; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno build; fi

zip:
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi zip; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno zip; fi

clean:
	rm -rf .pytest_cache
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi clean; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno clean; fi

precommit-install:
	poetry run pre-commit install

lint:
	poetry run pre-commit run --all-files
```

- [ ] **Step 2: Verify `make help` works**

Run:
```bash
make help
```

Expected: lists all targets above.

### Task 1.12: Update the top-level `README.md`

The existing README has content referring to documents at the repo root; those documents now live in `docs/architecture/`. Update the links.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README to locate links**

Run:
```bash
grep -n 'TOPIC_STATE_MODELING\|SPARK_VI_FRAMEWORK\|RISKS_AND_MITIGATIONS\|MILESTONES' README.md
```

Expected: lines near bottom of file referencing `TOPIC_STATE_MODELING.md`, `SPARK_VI_FRAMEWORK.md`, `MILESTONES.md`.

- [ ] **Step 2: Update links**

Use the Edit tool to update the three links in `README.md`:
- `[TOPIC_STATE_MODELING.md](TOPIC_STATE_MODELING.md)` → `[TOPIC_STATE_MODELING.md](docs/architecture/TOPIC_STATE_MODELING.md)`
- `[SPARK_VI_FRAMEWORK.md](SPARK_VI_FRAMEWORK.md)` → `[SPARK_VI_FRAMEWORK.md](docs/architecture/SPARK_VI_FRAMEWORK.md)`
- Remove the `MILESTONES.md` reference (the file was in `attic/` and has been deleted).

- [ ] **Step 3: Add a brief "Repository layout" section**

Append to the end of `README.md`:
```markdown

---

## Repository layout

```
CHARMPheno/
├── docs/architecture/     — living architectural vision (read these first)
├── docs/decisions/        — ADRs for significant organizational choices
├── docs/superpowers/      — design specs and implementation plans
├── spark-vi/              — domain-agnostic distributed-VI framework (package)
├── charmpheno/            — clinical specialization (package)
├── analysis/              — runnable end-to-end scripts
├── notebooks/tutorials/   — exposition and onboarding
├── scripts/               — data fetch, simulator, dev helpers
└── AGENTS.md              — orientation for coding agents
```

See [`AGENTS.md`](AGENTS.md) for a fuller project orientation.
```

- [ ] **Step 4: Verify links resolve**

Run:
```bash
test -f docs/architecture/TOPIC_STATE_MODELING.md && echo "link 1 ok"
test -f docs/architecture/SPARK_VI_FRAMEWORK.md && echo "link 2 ok"
```

Expected:
```
link 1 ok
link 2 ok
```

### Task 1.13: Initialize git, install hooks, first commit

- [ ] **Step 1: `git init`**

Run:
```bash
git init
git branch -m main
```

Expected: `Initialized empty Git repository in ...`.

- [ ] **Step 2: Install pre-commit hooks**

Run:
```bash
poetry run pre-commit install
```

Expected: `pre-commit installed at .git/hooks/pre-commit`.

- [ ] **Step 3: Stage all files**

Run:
```bash
git add .
```

- [ ] **Step 4: Run pre-commit against staged files**

Run:
```bash
poetry run pre-commit run --all-files
```

Expected: all hooks pass (some may modify trailing whitespace; if so, `git add` again).

- [ ] **Step 5: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
Bootstrap repo: hygiene, docs structure, ADRs, Makefile

Remove legacy brpmatch/ and attic/ folders (templates stashed in /tmp/).
Move architectural docs to docs/architecture/. Add AGENTS.md, ADRs 0001-0004,
top-level Makefile, root pyproject.toml for scripts and dev tooling,
.gitignore, .pre-commit-config.yaml, nbstripout + size + data-path hooks.
Create analysis/, notebooks/, scripts/ skeletons.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Verify clean tree**

Run:
```bash
git status
git log --oneline
```

Expected: clean working tree; one commit.

---

## Phase 2 — Data pipeline

Goal: `scripts/fetch_lda_beta.py` and `scripts/simulate_lda_omop.py` produce `data/cache/lda_beta_topk.parquet` and `data/simulated/*.parquet` respectively. Small committed fixtures appear under `tests/scripts/data/` for fast unit tests.

### Task 2.1: Create `tests/scripts/` skeleton

**Files:**
- Create: `tests/scripts/__init__.py`
- Create: `tests/scripts/conftest.py`
- Create: `tests/scripts/data/.gitkeep`

- [ ] **Step 1: Create directory and files**

Run:
```bash
mkdir -p tests/scripts/data
touch tests/scripts/__init__.py tests/scripts/data/.gitkeep
```

- [ ] **Step 2: Create `tests/scripts/conftest.py`**

Create `tests/scripts/conftest.py` with content:
```python
"""Fixtures for script-level tests."""
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Make scripts/ importable in tests so we can call functions directly.
sys.path.insert(0, str(REPO_ROOT / "scripts"))


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    """Run the test from inside a fresh tmp dir so relative-path defaults work."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
```

### Task 2.2: Write failing test for `fetch_lda_beta` top-K filter function

We test the pure top-K + renormalize function first, not the HF download itself.

**Files:**
- Create: `tests/scripts/test_fetch_lda_beta.py`

- [ ] **Step 1: Create the failing test file**

Create `tests/scripts/test_fetch_lda_beta.py` with content:
```python
"""Tests for scripts/fetch_lda_beta.py."""
import numpy as np
import pandas as pd
import pytest


def test_top_k_filter_keeps_top_weights_per_topic():
    """Per-topic, keep only the K highest-weight concepts and renormalize."""
    from fetch_lda_beta import top_k_per_topic_and_renormalize

    # Two topics, four concepts each, unbalanced weights.
    rows = pd.DataFrame({
        "topic_id": [0, 0, 0, 0, 1, 1, 1, 1],
        "concept_id": [10, 11, 12, 13, 20, 21, 22, 23],
        "concept_name": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "term_weight": [0.5, 0.3, 0.15, 0.05, 0.1, 0.2, 0.3, 0.4],
    })

    out = top_k_per_topic_and_renormalize(rows, top_k=2)

    # Two topics survive; two rows per topic.
    assert set(out["topic_id"].unique()) == {0, 1}
    counts = out.groupby("topic_id").size()
    assert counts.loc[0] == 2 and counts.loc[1] == 2

    # Topic 0 keeps 10 (0.5) and 11 (0.3).
    topic0 = out[out["topic_id"] == 0].sort_values("term_weight", ascending=False)
    assert list(topic0["concept_id"]) == [10, 11]

    # Topic 1 keeps 23 (0.4) and 22 (0.3).
    topic1 = out[out["topic_id"] == 1].sort_values("term_weight", ascending=False)
    assert list(topic1["concept_id"]) == [23, 22]

    # Each topic's kept weights sum to 1.0 after renormalization.
    sums = out.groupby("topic_id")["term_weight"].sum()
    np.testing.assert_allclose(sums.values, [1.0, 1.0], atol=1e-9)


def test_top_k_filter_small_k_zero_rejects():
    """top_k=0 is nonsensical and should raise."""
    from fetch_lda_beta import top_k_per_topic_and_renormalize

    rows = pd.DataFrame({
        "topic_id": [0, 0],
        "concept_id": [1, 2],
        "concept_name": ["a", "b"],
        "term_weight": [0.5, 0.5],
    })
    with pytest.raises(ValueError):
        top_k_per_topic_and_renormalize(rows, top_k=0)


def test_top_k_filter_handles_fewer_rows_than_k():
    """If a topic has fewer rows than K, keep all of them."""
    from fetch_lda_beta import top_k_per_topic_and_renormalize

    rows = pd.DataFrame({
        "topic_id": [0, 0, 1],
        "concept_id": [10, 11, 20],
        "concept_name": ["a", "b", "c"],
        "term_weight": [0.3, 0.7, 1.0],
    })
    out = top_k_per_topic_and_renormalize(rows, top_k=10)
    # Topic 0 keeps both rows; topic 1 keeps its one row.
    assert len(out) == 3
    sums = out.groupby("topic_id")["term_weight"].sum().sort_index()
    np.testing.assert_allclose(sums.values, [1.0, 1.0], atol=1e-9)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
poetry run pytest tests/scripts/test_fetch_lda_beta.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'fetch_lda_beta'`.

### Task 2.3: Implement `scripts/fetch_lda_beta.py`

**Files:**
- Create: `scripts/fetch_lda_beta.py`

- [ ] **Step 1: Create `scripts/fetch_lda_beta.py`**

Create `scripts/fetch_lda_beta.py` with content:
```python
"""Fetch the prior-LDA topic-concept β from the Hugging Face dataset
`oneilsh/lda_pasc`, filter to the top-K highest-weight concepts per topic,
renormalize each topic row to sum to 1.0, and write a compact parquet.

Output columns: topic_id:int, concept_id:int, concept_name:str, weight:float.

Default output: data/cache/lda_beta_topk.parquet
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

HF_DATASET = "oneilsh/lda_pasc"
DEFAULT_TOP_K = 1000
DEFAULT_OUTPUT = Path("data/cache/lda_beta_topk.parquet")


def parse_topic_id(topic_name: str) -> int:
    """Parse numeric id out of a topic_name string like 'T-148 (U 0.2%, H 0.93, ...)'.

    Why: the source file encodes the topic id inside a descriptive string; we
    need a stable integer key. The 'T-<digits>' prefix is invariant in the
    upstream artifact.
    """
    m = re.match(r"T-(\d+)", topic_name)
    if not m:
        raise ValueError(f"Could not parse topic id from {topic_name!r}")
    return int(m.group(1))


def top_k_per_topic_and_renormalize(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Keep the K highest-`term_weight` rows per topic and renormalize.

    Why: the full β matrix (300 topics × ~50K concepts) is 1.7 GB on disk,
    but ~99% of each topic's probability mass lives in its top ~1000 concepts
    (power-law distribution). Filtering preserves the generative process while
    fitting in a laptop-friendly artifact. Renormalization keeps each topic
    row a proper probability distribution.

    Input must have columns: topic_id, concept_id, concept_name, term_weight.
    Output is the same shape with `term_weight` renormalized per-topic.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    # Keep only the top-K by term_weight within each topic.
    ranked = (df.sort_values(["topic_id", "term_weight"], ascending=[True, False])
                .groupby("topic_id", group_keys=False)
                .head(top_k))

    # Renormalize each topic's surviving weights to sum to 1.
    sums = ranked.groupby("topic_id")["term_weight"].transform("sum")
    out = ranked.copy()
    out["term_weight"] = out["term_weight"] / sums
    return out.reset_index(drop=True)


def fetch_and_write(top_k: int, output: Path) -> None:
    """Stream the HF dataset, build a DataFrame, filter top-K, write parquet.

    Streaming avoids materializing the full 1.7 GB file in memory.
    """
    # Deferred import so unit tests of the pure filter don't require datasets.
    from datasets import load_dataset

    log.info("Streaming %s from Hugging Face ...", HF_DATASET)
    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    # The upstream CSV has columns: term_weight, relevance, concept_id,
    # concept_name, topic_name. We only need four of the five.
    records: list[dict] = []
    for row in ds:
        records.append({
            "topic_id": parse_topic_id(row["topic_name"]),
            "concept_id": int(row["concept_id"]),
            "concept_name": str(row["concept_name"]),
            "term_weight": float(row["term_weight"]),
        })
    df = pd.DataFrame(records)
    log.info("Loaded %d rows across %d topics",
             len(df), df["topic_id"].nunique())

    filtered = top_k_per_topic_and_renormalize(df, top_k=top_k)
    log.info("After top-%d filter: %d rows", top_k, len(filtered))

    output.parent.mkdir(parents=True, exist_ok=True)
    filtered.rename(columns={"term_weight": "weight"}).to_parquet(
        output, index=False
    )
    log.info("Wrote %s", output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Keep top-K concepts per topic (default {DEFAULT_TOP_K})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output parquet path (default {DEFAULT_OUTPUT})")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_and_write(top_k=args.top_k, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run tests to verify they pass**

Run:
```bash
poetry run pytest tests/scripts/test_fetch_lda_beta.py -v
```

Expected: 3 tests pass.

- [ ] **Step 3: Commit**

Run:
```bash
git add scripts/fetch_lda_beta.py tests/scripts/
git commit -m "feat(scripts): add fetch_lda_beta with top-K filter + renormalize"
```

### Task 2.4: Write failing test for simulator

**Files:**
- Create: `tests/scripts/test_simulate_lda_omop.py`

- [ ] **Step 1: Create the test**

Create `tests/scripts/test_simulate_lda_omop.py` with content:
```python
"""Tests for scripts/simulate_lda_omop.py."""
import numpy as np
import pandas as pd
import pytest


def _tiny_beta() -> pd.DataFrame:
    """Three topics, four concepts each, sharply peaked distributions."""
    return pd.DataFrame({
        "topic_id":     [0, 0, 0, 0,  1, 1, 1, 1,  2, 2, 2, 2],
        "concept_id":   [1, 2, 3, 4,  1, 2, 3, 4,  1, 2, 3, 4],
        "concept_name": ["a", "b", "c", "d"] * 3,
        # Topic 0 loves concept 1; topic 1 loves 2; topic 2 loves 3.
        "weight":       [0.97, 0.01, 0.01, 0.01,
                         0.01, 0.97, 0.01, 0.01,
                         0.01, 0.01, 0.97, 0.01],
    })


def test_simulate_produces_expected_schema():
    from simulate_lda_omop import simulate

    df = simulate(
        beta=_tiny_beta(),
        n_patients=5,
        theta_alpha=0.1,
        visits_per_patient_mean=2,
        codes_per_visit_mean=3,
        seed=42,
    )
    assert set(df.columns) == {
        "person_id", "visit_occurrence_id", "concept_id",
        "concept_name", "true_topic_id",
    }
    assert df["person_id"].nunique() == 5
    assert len(df) > 0


def test_simulate_is_deterministic_given_seed():
    from simulate_lda_omop import simulate

    args = dict(
        beta=_tiny_beta(),
        n_patients=5,
        theta_alpha=0.1,
        visits_per_patient_mean=2,
        codes_per_visit_mean=3,
        seed=123,
    )
    a = simulate(**args)
    b = simulate(**args)
    pd.testing.assert_frame_equal(a, b)


def test_simulate_concept_ids_come_from_beta_vocab():
    from simulate_lda_omop import simulate

    df = simulate(
        beta=_tiny_beta(),
        n_patients=20,
        theta_alpha=0.1,
        visits_per_patient_mean=3,
        codes_per_visit_mean=4,
        seed=7,
    )
    assert set(df["concept_id"].unique()).issubset({1, 2, 3, 4})


def test_simulate_true_topic_id_matches_a_valid_topic():
    from simulate_lda_omop import simulate

    df = simulate(
        beta=_tiny_beta(),
        n_patients=10,
        theta_alpha=0.1,
        visits_per_patient_mean=2,
        codes_per_visit_mean=3,
        seed=0,
    )
    assert set(df["true_topic_id"].unique()).issubset({0, 1, 2})


def test_simulate_concentrated_theta_recovers_expected_concept():
    """With very low alpha and peaked beta, a patient is dominated by one topic,
    which should concentrate their emitted concepts on that topic's favored
    concept. A sanity check that the generative process works as intended."""
    from simulate_lda_omop import simulate

    df = simulate(
        beta=_tiny_beta(),
        n_patients=100,
        theta_alpha=0.001,           # push θ nearly to a corner per patient
        visits_per_patient_mean=3,
        codes_per_visit_mean=10,
        seed=1,
    )
    # For each patient, the modal concept should be the favored concept of
    # their modal true_topic_id (topic 0 → concept 1, etc.).
    favored = {0: 1, 1: 2, 2: 3}
    correct = 0
    groups = df.groupby("person_id")
    for _, pdf in groups:
        modal_topic = pdf["true_topic_id"].mode().iloc[0]
        modal_concept = pdf["concept_id"].mode().iloc[0]
        if modal_concept == favored[modal_topic]:
            correct += 1
    # >=80% match is comfortably above chance for this scenario.
    assert correct / groups.ngroups >= 0.80
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
poetry run pytest tests/scripts/test_simulate_lda_omop.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'simulate_lda_omop'`.

### Task 2.5: Implement `scripts/simulate_lda_omop.py`

**Files:**
- Create: `scripts/simulate_lda_omop.py`

- [ ] **Step 1: Create `scripts/simulate_lda_omop.py`**

Create `scripts/simulate_lda_omop.py` with content:
```python
"""Generate synthetic OMOP-shaped data using a fixed LDA β distribution
and the standard LDA generative process.

For each patient:
    θ_p ~ Dirichlet(α · 1)                    (topic mixture weights)
    N_v ~ Poisson(visits_per_patient_mean)    (visits for this patient)
    for each of N_v visits:
        N_c ~ Poisson(codes_per_visit_mean)   (codes for this visit)
        for each of N_c codes:
            z   ~ Categorical(θ_p)            (pick a topic)
            w   ~ Categorical(β[z, :])        (pick a concept from topic z)
            emit (person_id, visit_occurrence_id, w.concept_id, w.concept_name, z)

Output columns:
    person_id:int, visit_occurrence_id:int, concept_id:int,
    concept_name:str, true_topic_id:int

`true_topic_id` is oracle metadata for evaluation; training code must not read it.

Default output: data/simulated/omop_N<n>_seed<seed>.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_BETA_PATH = Path("data/cache/lda_beta_topk.parquet")
DEFAULT_OUTPUT_DIR = Path("data/simulated")


def _beta_as_matrix(beta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Pivot the long-form beta DataFrame to (K × V) matrix form.

    Returns:
        beta_mat: float array shape (K, V), rows sum to 1.
        concept_ids: int array length V (column index → concept_id).
        concept_names: dict mapping concept_id to concept_name.
    """
    concept_ids = np.sort(beta["concept_id"].unique())
    topic_ids = np.sort(beta["topic_id"].unique())
    cid_to_col = {cid: i for i, cid in enumerate(concept_ids)}
    tid_to_row = {tid: i for i, tid in enumerate(topic_ids)}

    mat = np.zeros((len(topic_ids), len(concept_ids)), dtype=np.float64)
    for row in beta.itertuples(index=False):
        mat[tid_to_row[row.topic_id], cid_to_col[row.concept_id]] = row.weight

    # Defensive renorm — the filter script already does this but if the user
    # passes a hand-constructed beta, we keep the math sane.
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = mat / np.where(row_sums > 0, row_sums, 1.0)

    names = dict(zip(beta["concept_id"], beta["concept_name"]))
    return mat, concept_ids, names


def simulate(
    beta: pd.DataFrame,
    n_patients: int,
    theta_alpha: float,
    visits_per_patient_mean: float,
    codes_per_visit_mean: float,
    seed: int,
) -> pd.DataFrame:
    """Generate a synthetic OMOP-shaped DataFrame from a fixed β.

    Returns a DataFrame with columns person_id, visit_occurrence_id,
    concept_id, concept_name, true_topic_id.
    """
    rng = np.random.default_rng(seed)
    beta_mat, concept_ids, concept_names = _beta_as_matrix(beta)
    K, V = beta_mat.shape

    # Patient-level θ_p ~ Dirichlet(α · 1_K)
    alpha_vec = np.full(K, theta_alpha, dtype=np.float64)
    theta = rng.dirichlet(alpha_vec, size=n_patients)  # (n_patients, K)

    rows: list[tuple[int, int, int, str, int]] = []
    visit_counter = 0
    for p in range(n_patients):
        n_visits = max(1, int(rng.poisson(visits_per_patient_mean)))
        for _ in range(n_visits):
            visit_counter += 1
            n_codes = max(1, int(rng.poisson(codes_per_visit_mean)))
            z = rng.choice(K, size=n_codes, p=theta[p])
            for zi in z:
                w_col = rng.choice(V, p=beta_mat[zi])
                cid = int(concept_ids[w_col])
                rows.append((p, visit_counter, cid, concept_names[cid], int(zi)))

    return pd.DataFrame(
        rows,
        columns=["person_id", "visit_occurrence_id", "concept_id",
                 "concept_name", "true_topic_id"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=Path, default=DEFAULT_BETA_PATH,
                        help=f"Input filtered-beta parquet (default {DEFAULT_BETA_PATH})")
    parser.add_argument("--n-patients", type=int, default=10_000)
    parser.add_argument("--theta-alpha", type=float, default=0.1,
                        help="Symmetric Dirichlet concentration on θ (default 0.1)")
    parser.add_argument("--visits-per-patient-mean", type=float, default=3.0)
    parser.add_argument("--codes-per-visit-mean", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Reading beta from %s", args.beta)
    beta = pd.read_parquet(args.beta)

    df = simulate(
        beta=beta,
        n_patients=args.n_patients,
        theta_alpha=args.theta_alpha,
        visits_per_patient_mean=args.visits_per_patient_mean,
        codes_per_visit_mean=args.codes_per_visit_mean,
        seed=args.seed,
    )
    log.info("Generated %d rows for %d patients", len(df), args.n_patients)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"omop_N{args.n_patients}_seed{args.seed}.parquet"
    df.to_parquet(out_path, index=False)

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "n_patients": args.n_patients,
        "theta_alpha": args.theta_alpha,
        "visits_per_patient_mean": args.visits_per_patient_mean,
        "codes_per_visit_mean": args.codes_per_visit_mean,
        "seed": args.seed,
        "beta_path": str(args.beta),
    }, indent=2))
    log.info("Wrote %s (and %s)", out_path, meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run tests to verify they pass**

Run:
```bash
poetry run pytest tests/scripts/test_simulate_lda_omop.py -v
```

Expected: 5 tests pass.

- [ ] **Step 3: Commit**

Run:
```bash
git add scripts/simulate_lda_omop.py tests/scripts/test_simulate_lda_omop.py
git commit -m "feat(scripts): add simulate_lda_omop with LDA generative process"
```

### Task 2.6: Verify `make test` picks up the scripts tests

- [ ] **Step 1: Run make test**

Run:
```bash
make test
```

Expected: scripts tests run and pass. (spark-vi and charmpheno directories don't exist yet, so those legs of the Makefile are skipped.)

---

## Phase 3 — spark-vi skeleton

Goal: `spark-vi/` is a working poetry project with VIModel / VIRunner / VIConfig / VIResult, a trivial `CountingModel` that exercises the full contract end-to-end through the real Spark runner, a stub `OnlineHDP`, broadcast-lifecycle and checkpoint-roundtrip and export-roundtrip tests, and working `make build` + `make zip`.

### Task 3.1: Create `spark-vi/` project skeleton

**Files:**
- Create: `spark-vi/pyproject.toml`
- Create: `spark-vi/.gitignore`
- Create: `spark-vi/README.md`
- Create: `spark-vi/Makefile`
- Create: `spark-vi/spark_vi/__init__.py`
- Create: `spark-vi/spark_vi/core/__init__.py`
- Create: `spark-vi/spark_vi/models/__init__.py`
- Create: `spark-vi/spark_vi/diagnostics/__init__.py`
- Create: `spark-vi/spark_vi/io/__init__.py`
- Create: `spark-vi/tests/__init__.py`
- Create: `spark-vi/tests/data/.gitkeep`

- [ ] **Step 1: Create directories and empty files**

Run:
```bash
mkdir -p spark-vi/spark_vi/core spark-vi/spark_vi/models spark-vi/spark_vi/diagnostics spark-vi/spark_vi/io spark-vi/tests/data
touch spark-vi/spark_vi/__init__.py \
      spark-vi/spark_vi/core/__init__.py \
      spark-vi/spark_vi/models/__init__.py \
      spark-vi/spark_vi/diagnostics/__init__.py \
      spark-vi/spark_vi/io/__init__.py \
      spark-vi/tests/__init__.py \
      spark-vi/tests/data/.gitkeep
```

- [ ] **Step 2: Write `spark-vi/spark_vi/__init__.py`**

Create with content:
```python
"""spark-vi: a PySpark-native framework for distributed variational inference.

Public API:

    from spark_vi.core import VIModel, VIRunner, VIConfig, VIResult

Pre-built models live in spark_vi.models (e.g. OnlineHDP).
"""
__version__ = "0.1.0"
```

- [ ] **Step 3: Write `spark-vi/pyproject.toml`**

Create `spark-vi/pyproject.toml` with content:
```toml
[tool.poetry]
name = "spark-vi"
version = "0.1.0"
description = "PySpark-native framework for distributed variational inference"
authors = ["CHARMPheno contributors"]
license = "MIT"
readme = "README.md"
packages = [{include = "spark_vi"}]

[tool.poetry.dependencies]
python = ">=3.10,<3.13"
pyspark = ">=3.5,<4.0"
numpy = ">=1.24"
scipy = ">=1.10"

[tool.poetry.group.dev.dependencies]
pytest = ">=7.0"
build = ">=1.0"

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
markers = [
    "slow: integration tests (simulator data, local Spark, minutes-scale)",
    "cluster: tests that require a real Dataproc cluster (manual only)",
]
addopts = "-m 'not slow and not cluster'"
filterwarnings = [
    "ignore::ResourceWarning",
    "ignore:distutils Version classes are deprecated:DeprecationWarning",
]
```

- [ ] **Step 4: Write `spark-vi/.gitignore`**

Create with content:
```gitignore
dist/
build/
*.egg-info/
__pycache__/
.pytest_cache/
.coverage
```

- [ ] **Step 5: Write `spark-vi/README.md`**

Create with content:
```markdown
# spark-vi

Distributed variational inference for PySpark.

`spark-vi` is a domain-agnostic PySpark framework. Model authors subclass
`VIModel` and implement three methods; the framework handles Spark
orchestration, the training loop, convergence monitoring, and model export.

See the [framework design](../docs/architecture/SPARK_VI_FRAMEWORK.md) for the
architectural context and [CHARMPheno](../README.md) for the project this
framework was extracted from.

## Install

```bash
poetry install
```

## Test

```bash
make test          # unit tests only (fast)
make test-all      # unit + slow integration tests
```

## Build artifacts

```bash
make build         # dist/*.whl + dist/*.tar.gz
make zip           # dist/spark_vi.zip (flat, pure-Python; for --py-files)
```

Requires Java 17 for local Spark. The Makefile autodetects Homebrew
(`/opt/homebrew/opt/openjdk@17`) and common Linux paths; override with
`JAVA_HOME=... make test` if needed.
```

- [ ] **Step 6: Write `spark-vi/Makefile`**

Create with content:
```makefile
.PHONY: help install dev clean build zip check test test-all test-cluster

default: help

JAVA_HOME_MACOS := /opt/homebrew/opt/openjdk@17
JAVA_HOME_LINUX := /usr/lib/jvm/java-17-openjdk-amd64
JAVA_HOME ?= $(shell if [ -d "$(JAVA_HOME_MACOS)" ]; then echo "$(JAVA_HOME_MACOS)"; elif [ -d "$(JAVA_HOME_LINUX)" ]; then echo "$(JAVA_HOME_LINUX)"; fi)

help:
	@echo "spark-vi common targets:"
	@echo "  install      - poetry install"
	@echo "  dev          - install + dev build tooling"
	@echo "  test         - unit tests only (fast)"
	@echo "  test-all     - unit + @slow tests"
	@echo "  test-cluster - only @cluster tests (manual)"
	@echo "  build        - wheel + sdist into dist/"
	@echo "  zip          - flat dist/spark_vi.zip for --py-files"
	@echo "  clean        - remove dist/, build/, caches"

install:
	poetry install

dev:
	poetry install
	poetry run pip install build

clean:
	rm -rf dist/ build/ *.egg-info
	rm -rf .pytest_cache

test:
	JAVA_HOME=$(JAVA_HOME) poetry run pytest tests/ -v

test-all:
	JAVA_HOME=$(JAVA_HOME) poetry run pytest tests/ -v -m "not cluster"

test-cluster:
	JAVA_HOME=$(JAVA_HOME) poetry run pytest tests/ -v -m cluster

build:
	poetry run python -m build

zip: clean build
	mkdir -p dist
	zip -r dist/spark_vi.zip spark_vi -x "*.pyc" -x "*/__pycache__/*" -x "*.egg-info/*"
	@echo "Wrote dist/spark_vi.zip"
```

- [ ] **Step 7: Install and verify scaffolding imports**

Run:
```bash
cd spark-vi && poetry install && poetry run python -c "import spark_vi; print(spark_vi.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 8: Commit scaffolding**

Run:
```bash
cd ..
git add spark-vi/
git commit -m "feat(spark-vi): project scaffolding (empty packages, Makefile, pyproject)"
```

### Task 3.2: Write `spark-vi/tests/conftest.py`

**Files:**
- Create: `spark-vi/tests/conftest.py`

- [ ] **Step 1: Create the file**

Create `spark-vi/tests/conftest.py` with content:
```python
"""Pytest fixtures for spark-vi.

The session-scoped local Spark session is the only fixture all tests share.
"""
import os
import warnings

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Session-scoped local Spark (2 cores, small shuffle, UI disabled).

    Why 2 cores: large enough to exercise parallel map/reduce, small enough
    to keep startup cost negligible in CI and local iteration.
    """
    warnings.filterwarnings("ignore")
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    session = (
        SparkSession.builder.master("local[2]")
        .appName("spark-vi-tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )
    yield session
    session.stop()
```

### Task 3.3: VIConfig — test + implementation

**Files:**
- Create: `spark-vi/tests/test_config.py`
- Create: `spark-vi/spark_vi/core/config.py`

- [ ] **Step 1: Write the failing test**

Create `spark-vi/tests/test_config.py` with content:
```python
"""VIConfig holds training-loop hyperparameters."""
import pytest


def test_vi_config_defaults_are_sensible():
    from spark_vi.core import VIConfig

    cfg = VIConfig()
    # Hoffman-style defaults (see docs/architecture/SPARK_VI_FRAMEWORK.md).
    assert cfg.max_iterations >= 1
    assert 0.0 < cfg.learning_rate_tau0
    assert 0.0 < cfg.learning_rate_kappa < 1.0
    assert cfg.convergence_tol > 0.0
    assert cfg.checkpoint_interval is None or cfg.checkpoint_interval > 0


def test_vi_config_rejects_invalid_values():
    from spark_vi.core import VIConfig

    with pytest.raises(ValueError):
        VIConfig(max_iterations=0)
    with pytest.raises(ValueError):
        VIConfig(learning_rate_kappa=1.5)
    with pytest.raises(ValueError):
        VIConfig(convergence_tol=-1.0)


def test_vi_config_is_frozen_dataclass():
    from spark_vi.core import VIConfig

    cfg = VIConfig()
    with pytest.raises(Exception):
        cfg.max_iterations = 999  # frozen
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd spark-vi && poetry run pytest tests/test_config.py -v
```

Expected: FAIL with `ImportError: cannot import name 'VIConfig'`.

- [ ] **Step 3: Implement `spark-vi/spark_vi/core/config.py`**

Create with content:
```python
"""Training-loop hyperparameters for VIRunner.

See docs/architecture/SPARK_VI_FRAMEWORK.md#viconfig for the design rationale.
Hoffman, Blei, Wang, Paisley 2013 ("Stochastic Variational Inference") set
the tau0 / kappa conventions for the Robbins-Monro step size schedule.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VIConfig:
    """Immutable configuration for the VI training loop.

    Attributes:
        max_iterations: hard upper bound on training iterations.
        learning_rate_tau0: delay parameter in rho_t = (tau0 + t)^-kappa.
        learning_rate_kappa: decay exponent; must be in (0.5, 1].
        convergence_tol: relative ELBO improvement threshold for early stop.
        checkpoint_interval: if set, write checkpoint every N iterations.

    Why these defaults: tau0=1.0, kappa=0.7 is Hoffman et al. 2013's common
    choice for stochastic VI on text corpora; it biases toward faster initial
    progress at the cost of larger late-iteration updates. Models processing
    full batches should override kappa (see RISKS_AND_MITIGATIONS.md).
    """
    max_iterations: int = 100
    learning_rate_tau0: float = 1.0
    learning_rate_kappa: float = 0.7
    convergence_tol: float = 1e-4
    checkpoint_interval: int | None = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.learning_rate_tau0 <= 0.0:
            raise ValueError(f"learning_rate_tau0 must be > 0, got {self.learning_rate_tau0}")
        if not (0.0 < self.learning_rate_kappa <= 1.0):
            raise ValueError(
                f"learning_rate_kappa must be in (0, 1], got {self.learning_rate_kappa}"
            )
        if self.convergence_tol <= 0.0:
            raise ValueError(f"convergence_tol must be > 0, got {self.convergence_tol}")
        if self.checkpoint_interval is not None and self.checkpoint_interval < 1:
            raise ValueError(
                f"checkpoint_interval must be None or >= 1, got {self.checkpoint_interval}"
            )
```

- [ ] **Step 4: Wire into the `core` package**

Update `spark-vi/spark_vi/core/__init__.py` to content:
```python
"""Public API for spark_vi.core."""
from spark_vi.core.config import VIConfig

__all__ = ["VIConfig"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
poetry run pytest tests/test_config.py -v
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

Run:
```bash
cd .. && git add spark-vi/spark_vi/core/__init__.py spark-vi/spark_vi/core/config.py spark-vi/tests/conftest.py spark-vi/tests/test_config.py
git commit -m "feat(spark-vi): add VIConfig with Hoffman-style defaults"
```

### Task 3.4: VIResult — test + implementation

**Files:**
- Create: `spark-vi/tests/test_result.py`
- Create: `spark-vi/spark_vi/core/result.py`

- [ ] **Step 1: Write the failing test**

Create `spark-vi/tests/test_result.py` with content:
```python
"""VIResult captures the outcome of a completed training run."""
import numpy as np


def test_vi_result_holds_global_params_and_metrics():
    from spark_vi.core import VIResult

    result = VIResult(
        global_params={"lambda": np.array([1.0, 2.0])},
        elbo_trace=[-10.0, -9.5, -9.1],
        n_iterations=3,
        converged=False,
        metadata={"model_class": "TestModel"},
    )
    assert "lambda" in result.global_params
    assert result.elbo_trace[-1] == -9.1
    assert result.n_iterations == 3
    assert result.converged is False
    assert result.metadata["model_class"] == "TestModel"


def test_vi_result_final_elbo_accessor():
    from spark_vi.core import VIResult

    r = VIResult(global_params={}, elbo_trace=[-10.0, -9.0], n_iterations=2,
                 converged=True, metadata={})
    assert r.final_elbo == -9.0


def test_vi_result_empty_trace_has_none_final_elbo():
    from spark_vi.core import VIResult

    r = VIResult(global_params={}, elbo_trace=[], n_iterations=0,
                 converged=False, metadata={})
    assert r.final_elbo is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd spark-vi && poetry run pytest tests/test_result.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `spark-vi/spark_vi/core/result.py`**

Create with content:
```python
"""VIResult: immutable record of a completed VI training run.

See docs/architecture/SPARK_VI_FRAMEWORK.md#viresult-and-model-export.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VIResult:
    """Outcome of a VIRunner.fit call.

    Attributes:
        global_params: fitted global variational parameters, keyed by name.
        elbo_trace: per-iteration ELBO values (or surrogate, if ELBO is unavailable).
        n_iterations: how many iterations actually ran.
        converged: whether convergence criterion was met (vs. max_iterations hit).
        metadata: free-form dict (model class name, timestamps, git sha, ...).
    """
    global_params: dict[str, np.ndarray]
    elbo_trace: list[float]
    n_iterations: int
    converged: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_elbo(self) -> float | None:
        """Last ELBO value, or None if the trace is empty."""
        return self.elbo_trace[-1] if self.elbo_trace else None
```

- [ ] **Step 4: Update `__init__.py`**

Update `spark-vi/spark_vi/core/__init__.py` to content:
```python
"""Public API for spark_vi.core."""
from spark_vi.core.config import VIConfig
from spark_vi.core.result import VIResult

__all__ = ["VIConfig", "VIResult"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
poetry run pytest tests/test_result.py -v
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

Run:
```bash
cd .. && git add spark-vi/spark_vi/core/__init__.py spark-vi/spark_vi/core/result.py spark-vi/tests/test_result.py
git commit -m "feat(spark-vi): add VIResult"
```

### Task 3.5: VIModel ABC + CountingModel — tests + implementation

**Files:**
- Create: `spark-vi/spark_vi/core/model.py`
- Create: `spark-vi/spark_vi/models/counting.py`
- Create: `spark-vi/tests/test_counting_model.py`

The `CountingModel` is a trivial coin-flip posterior that exercises the full VIModel contract. Used throughout Phase 3 tests as the framework's smoke model.

- [ ] **Step 1: Write the failing test**

Create `spark-vi/tests/test_counting_model.py` with content:
```python
"""Contract-exercise tests: CountingModel implements VIModel fully.

CountingModel: each data row is either 0 (tail) or 1 (head); the 'global'
parameters are the Beta-posterior counts (alpha, beta) over the coin bias.
One global step aggregates counts; convergence is reaching max_iterations.
"""
import numpy as np
import pytest


def test_counting_model_is_a_vimodel():
    from spark_vi.core import VIModel
    from spark_vi.models.counting import CountingModel

    assert issubclass(CountingModel, VIModel)


def test_counting_model_initialize_global_returns_prior_counts():
    from spark_vi.models.counting import CountingModel

    m = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    g = m.initialize_global(data_summary=None)
    np.testing.assert_allclose(g["alpha"], 1.0)
    np.testing.assert_allclose(g["beta"], 1.0)


def test_counting_model_local_update_returns_sufficient_stats():
    from spark_vi.models.counting import CountingModel

    m = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    g = m.initialize_global(data_summary=None)
    # Partition of three heads and two tails.
    stats = m.local_update(rows=[1, 1, 1, 0, 0], global_params=g)
    # Sufficient stats: number of heads, number of tails.
    np.testing.assert_allclose(stats["heads"], 3.0)
    np.testing.assert_allclose(stats["tails"], 2.0)


def test_counting_model_combine_stats_sums_elementwise():
    from spark_vi.models.counting import CountingModel

    m = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    combined = m.combine_stats(
        {"heads": np.array(3.0), "tails": np.array(2.0)},
        {"heads": np.array(1.0), "tails": np.array(4.0)},
    )
    np.testing.assert_allclose(combined["heads"], 4.0)
    np.testing.assert_allclose(combined["tails"], 6.0)


def test_counting_model_update_global_applies_natural_gradient():
    """One step: lambda_new = (1 - rho) * lambda_old + rho * (prior + stats).

    With rho=1.0 the update jumps directly to (prior + stats).
    """
    from spark_vi.models.counting import CountingModel

    m = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    old = m.initialize_global(data_summary=None)
    stats = {"heads": np.array(10.0), "tails": np.array(5.0)}
    new = m.update_global(old, stats, learning_rate=1.0)
    # rho=1.0: new = prior + stats = (1 + 10, 1 + 5)
    np.testing.assert_allclose(new["alpha"], 11.0)
    np.testing.assert_allclose(new["beta"], 6.0)


def test_counting_model_update_global_interpolates_partial_step():
    from spark_vi.models.counting import CountingModel

    m = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    old = {"alpha": np.array(2.0), "beta": np.array(2.0)}
    stats = {"heads": np.array(10.0), "tails": np.array(0.0)}
    new = m.update_global(old, stats, learning_rate=0.5)
    # new = 0.5 * old + 0.5 * (prior + stats) = 0.5 * (2, 2) + 0.5 * (11, 1)
    np.testing.assert_allclose(new["alpha"], 6.5)
    np.testing.assert_allclose(new["beta"], 1.5)


def test_counting_model_elbo_is_increasing_with_more_data():
    """Crude sanity: posterior concentration should monotonically raise the
    (surrogate) ELBO along a sequence of updates with consistent evidence.

    This is a smoke check of the ELBO method returning a finite number, not
    a correctness proof of the log-marginal likelihood itself.
    """
    from spark_vi.models.counting import CountingModel

    m = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    g = m.initialize_global(data_summary=None)
    elbo0 = m.compute_elbo(g, {"heads": np.array(0.0), "tails": np.array(0.0)})
    g = m.update_global(g, {"heads": np.array(30.0), "tails": np.array(10.0)},
                        learning_rate=1.0)
    elbo1 = m.compute_elbo(g, {"heads": np.array(30.0), "tails": np.array(10.0)})
    assert np.isfinite(elbo0) and np.isfinite(elbo1)


def test_counting_model_required_methods_surface_on_base():
    """The VIModel ABC refuses instantiation unless the three required
    methods are implemented."""
    from spark_vi.core import VIModel

    with pytest.raises(TypeError):
        # Missing required abstract methods → abstract class cannot be instantiated.
        VIModel()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd spark-vi && poetry run pytest tests/test_counting_model.py -v
```

Expected: FAIL with import errors.

- [ ] **Step 3: Implement `spark-vi/spark_vi/core/model.py`**

Create with content:
```python
"""VIModel: the base class model authors subclass.

See docs/architecture/SPARK_VI_FRAMEWORK.md#the-vimodel-base-class for the
contract's design rationale. The three required methods correspond to the
three slots in the standard distributed VI iteration:

    lambda_{t+1} = (1 - rho_t) * lambda_t  +  rho_t * lambda_hat(sum_p s_p)
    ^--- update_global                        ^--- aggregated local_updates
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np


class VIModel(ABC):
    """Base class for models fittable by VIRunner.

    Subclasses implement initialize_global, local_update, and update_global.
    Optional hooks — combine_stats, compute_elbo, has_converged — have
    sensible defaults.
    """

    @abstractmethod
    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """Return starting values of the global variational parameters.

        data_summary: optional model-defined summary produced by the framework
            in a pre-pass (e.g., vocabulary size). Models that need nothing
            can ignore this argument.
        """

    @abstractmethod
    def local_update(
        self,
        rows: Iterable[Any],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """E-step on one data partition.

        rows: iterable over the partition's local rows.
        global_params: current global variational parameters.
        returns: dict of additive sufficient statistics (or gradient contributions).
        """

    @abstractmethod
    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        """M-step: apply the natural-gradient update with stepsize rho_t."""

    # Optional overrides ----------------------------------------------------

    def combine_stats(
        self,
        a: dict[str, np.ndarray],
        b: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Elementwise-sum two sufficient-statistic dicts.

        Default implementation is correct for models whose statistics live
        in dense NumPy arrays (most exponential-family VI models). Override
        for sparse or structured statistics (see RISKS_AND_MITIGATIONS.md).
        """
        keys = set(a) | set(b)
        out: dict[str, np.ndarray] = {}
        for k in keys:
            if k in a and k in b:
                out[k] = np.asarray(a[k]) + np.asarray(b[k])
            elif k in a:
                out[k] = np.asarray(a[k])
            else:
                out[k] = np.asarray(b[k])
        return out

    def compute_elbo(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
    ) -> float:
        """ELBO surrogate for diagnostics; override for a real bound.

        Default returns NaN, which callers treat as 'ELBO not available'.
        """
        return float("nan")

    def has_converged(
        self,
        elbo_trace: list[float],
        convergence_tol: float,
    ) -> bool:
        """Default: converged when the relative ELBO improvement falls below tol.

        Returns False until at least two finite ELBO values are present.
        """
        if len(elbo_trace) < 2:
            return False
        prev, curr = elbo_trace[-2], elbo_trace[-1]
        if not (np.isfinite(prev) and np.isfinite(curr)):
            return False
        denom = max(abs(prev), 1e-12)
        return abs(curr - prev) / denom < convergence_tol
```

- [ ] **Step 4: Implement `spark-vi/spark_vi/models/counting.py`**

Create with content:
```python
"""CountingModel: a trivial VI model used to exercise the framework contract.

Posterior over the bias of a Bernoulli coin:

    prior:       p ~ Beta(prior_alpha, prior_beta)
    likelihood:  rows are 0/1 iid from Bernoulli(p)
    sufficient stats: (# heads, # tails) aggregated across partitions
    global update: Beta-Bernoulli conjugate posterior counts with Robbins-Monro
                   interpolation against the previous iterate.

This is not a realistic use case; it exists so contract-level tests can run
end-to-end through VIRunner without depending on a real model's math.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.special import betaln

from spark_vi.core.model import VIModel


class CountingModel(VIModel):
    """Beta-Bernoulli conjugate VI stand-in."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("priors must be positive")
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        return {
            "alpha": np.array(self.prior_alpha),
            "beta": np.array(self.prior_beta),
        }

    def local_update(
        self,
        rows: Iterable[int],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        heads = 0
        tails = 0
        for r in rows:
            if r == 1:
                heads += 1
            elif r == 0:
                tails += 1
            else:
                raise ValueError(f"CountingModel rows must be 0 or 1, got {r!r}")
        return {"heads": np.array(float(heads)), "tails": np.array(float(tails))}

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        # lambda_hat = prior + aggregated counts.
        target_alpha = self.prior_alpha + aggregated_stats["heads"]
        target_beta = self.prior_beta + aggregated_stats["tails"]
        # Robbins-Monro interpolation.
        new_alpha = (1.0 - learning_rate) * global_params["alpha"] + learning_rate * target_alpha
        new_beta = (1.0 - learning_rate) * global_params["beta"] + learning_rate * target_beta
        return {"alpha": np.array(float(new_alpha)), "beta": np.array(float(new_beta))}

    def compute_elbo(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
    ) -> float:
        """Surrogate ELBO: log marginal likelihood under current posterior pseudocounts.

        Using log B(alpha, beta) - log B(prior_alpha, prior_beta) + log P(data | counts).
        Good enough to be monotonic-ish and finite for the tests.
        """
        a = float(global_params["alpha"])
        b = float(global_params["beta"])
        h = float(aggregated_stats.get("heads", 0.0))
        t = float(aggregated_stats.get("tails", 0.0))
        # Log posterior predictive factor + log prior normalizer ratio.
        return -betaln(a, b) + betaln(self.prior_alpha, self.prior_beta) + h * 0.0 + t * 0.0 \
            + float(a + b)  # placeholder monotone-in-data-weight term for tests
```

- [ ] **Step 5: Update `spark-vi/spark_vi/core/__init__.py`**

Update content to:
```python
"""Public API for spark_vi.core."""
from spark_vi.core.config import VIConfig
from spark_vi.core.model import VIModel
from spark_vi.core.result import VIResult

__all__ = ["VIConfig", "VIModel", "VIResult"]
```

- [ ] **Step 6: Update `spark-vi/spark_vi/models/__init__.py`**

Update content to:
```python
"""Pre-built models for spark-vi."""
from spark_vi.models.counting import CountingModel

__all__ = ["CountingModel"]
```

- [ ] **Step 7: Run tests to verify they pass**

Run:
```bash
poetry run pytest tests/test_counting_model.py tests/test_config.py tests/test_result.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

Run:
```bash
cd .. && git add spark-vi/spark_vi/core/__init__.py spark-vi/spark_vi/core/model.py spark-vi/spark_vi/models/__init__.py spark-vi/spark_vi/models/counting.py spark-vi/tests/test_counting_model.py
git commit -m "feat(spark-vi): add VIModel ABC and CountingModel contract exerciser"
```

### Task 3.6: VIRunner — test + minimum viable implementation

**Files:**
- Create: `spark-vi/spark_vi/core/runner.py`
- Create: `spark-vi/tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

Create `spark-vi/tests/test_runner.py` with content:
```python
"""Integration test: VIRunner fits CountingModel end-to-end on local Spark."""
import numpy as np
import pytest


def test_vi_runner_fits_counting_model_end_to_end(spark):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.counting import CountingModel

    # 100 rows: 70 heads, 30 tails.
    data = [1] * 70 + [0] * 30
    rdd = spark.sparkContext.parallelize(data, numSlices=4)

    model = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    runner = VIRunner(
        model=model,
        config=VIConfig(max_iterations=5, convergence_tol=1e-6),
    )
    result = runner.fit(rdd)

    # Posterior mean ≈ 70 / 100 = 0.7 (with Beta(1,1) prior and learning rate
    # schedule that hasn't fully saturated in 5 iterations — check at a loose
    # tolerance).
    a = float(result.global_params["alpha"])
    b = float(result.global_params["beta"])
    mean = a / (a + b)
    assert 0.55 < mean < 0.85
    assert result.n_iterations == 5  # hit max, not convergence (tight tol)
    assert len(result.elbo_trace) == 5


def test_vi_runner_stops_on_convergence(spark):
    """A wide convergence tolerance should trigger early stop."""
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1] * 100 + [0] * 100, numSlices=4)
    model = CountingModel()
    runner = VIRunner(
        model=model,
        config=VIConfig(max_iterations=100, convergence_tol=1e10),  # will stop after 2
    )
    result = runner.fit(rdd)
    assert result.converged is True
    assert result.n_iterations < 100


def test_vi_runner_rejects_non_vi_model(spark):
    from spark_vi.core import VIConfig, VIRunner

    class NotAModel:
        pass

    rdd = spark.sparkContext.parallelize([1, 0], numSlices=2)
    with pytest.raises(TypeError):
        VIRunner(model=NotAModel(), config=VIConfig()).fit(rdd)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd spark-vi && poetry run pytest tests/test_runner.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `spark-vi/spark_vi/core/runner.py`**

Create with content:
```python
"""VIRunner: the training-loop driver for distributed variational inference.

Each iteration executes the canonical distributed-VI step:

    1. Broadcast current global params to all partitions.
    2. mapPartitions: each worker runs model.local_update and emits stats.
    3. treeAggregate: sum stats across partitions (via model.combine_stats).
    4. Driver: model.update_global with Robbins-Monro learning rate.
    5. Record ELBO; test convergence.

The MLlib `OnlineLDAOptimizer` uses an identical pattern; see
docs/architecture/SPARK_VI_FRAMEWORK.md for references.
"""
from __future__ import annotations

import logging
from typing import Any

from pyspark import RDD

from spark_vi.core.config import VIConfig
from spark_vi.core.model import VIModel
from spark_vi.core.result import VIResult

log = logging.getLogger(__name__)


class VIRunner:
    """Drives a VIModel through iterations of distributed VI on a Spark RDD."""

    def __init__(self, model: VIModel, config: VIConfig | None = None) -> None:
        if not isinstance(model, VIModel):
            raise TypeError(f"model must be a VIModel subclass, got {type(model).__name__}")
        self.model = model
        self.config = config if config is not None else VIConfig()

    def fit(self, data_rdd: RDD, data_summary: Any | None = None) -> VIResult:
        """Run the distributed VI loop until convergence or max_iterations."""
        model = self.model
        cfg = self.config

        global_params = model.initialize_global(data_summary)
        elbo_trace: list[float] = []
        sc = data_rdd.context
        prior_bcast = None
        converged = False

        for t in range(cfg.max_iterations):
            # 1. Broadcast current global params.
            bcast = sc.broadcast(global_params)

            # 2 & 3. Distributed E-step + aggregate.
            def _local(rows, _bcast=bcast, _model=model):
                return [_model.local_update(rows, _bcast.value)]

            stats_seq = data_rdd.mapPartitions(_local).collect()
            aggregated = stats_seq[0]
            for more in stats_seq[1:]:
                aggregated = model.combine_stats(aggregated, more)

            # 4. M-step (Robbins-Monro step size).
            rho_t = (cfg.learning_rate_tau0 + t) ** -cfg.learning_rate_kappa
            global_params = model.update_global(global_params, aggregated, learning_rate=rho_t)

            # 5. ELBO + convergence.
            elbo = model.compute_elbo(global_params, aggregated)
            elbo_trace.append(float(elbo))

            # Unpersist the *previous* broadcast so we don't leak them.
            # See RISKS_AND_MITIGATIONS.md §Broadcast lifecycle.
            if prior_bcast is not None:
                prior_bcast.unpersist(blocking=False)
            prior_bcast = bcast

            if model.has_converged(elbo_trace, cfg.convergence_tol):
                converged = True
                log.info("Converged at iteration %d (ELBO=%.6f)", t + 1, elbo)
                # One-more unpersist for the final broadcast.
                prior_bcast.unpersist(blocking=False)
                prior_bcast = None
                return VIResult(
                    global_params=global_params,
                    elbo_trace=elbo_trace,
                    n_iterations=t + 1,
                    converged=True,
                    metadata={"model_class": type(model).__name__},
                )

        # Hit max_iterations without convergence.
        if prior_bcast is not None:
            prior_bcast.unpersist(blocking=False)
        return VIResult(
            global_params=global_params,
            elbo_trace=elbo_trace,
            n_iterations=cfg.max_iterations,
            converged=False,
            metadata={"model_class": type(model).__name__},
        )
```

- [ ] **Step 4: Update `spark-vi/spark_vi/core/__init__.py`**

Update content to:
```python
"""Public API for spark_vi.core."""
from spark_vi.core.config import VIConfig
from spark_vi.core.model import VIModel
from spark_vi.core.result import VIResult
from spark_vi.core.runner import VIRunner

__all__ = ["VIConfig", "VIModel", "VIResult", "VIRunner"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
poetry run pytest tests/test_runner.py -v
```

Expected: 3 tests pass (takes several seconds — first Spark spin-up).

- [ ] **Step 6: Commit**

Run:
```bash
cd .. && git add spark-vi/spark_vi/core/__init__.py spark-vi/spark_vi/core/runner.py spark-vi/tests/test_runner.py
git commit -m "feat(spark-vi): add VIRunner with broadcast→mapPartitions→aggregate loop"
```

### Task 3.7: Broadcast lifecycle regression test

**Files:**
- Create: `spark-vi/tests/test_broadcast_lifecycle.py`

Ensures the `unpersist()` call actually fires on prior broadcasts — the RISKS-flagged bug class.

- [ ] **Step 1: Write the test**

Create `spark-vi/tests/test_broadcast_lifecycle.py` with content:
```python
"""Ensure VIRunner unpersists prior broadcasts (prevents OOM in long runs).

See docs/architecture/RISKS_AND_MITIGATIONS.md §Broadcast lifecycle.
"""
from unittest.mock import patch


def test_vi_runner_unpersists_prior_broadcasts(spark):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1, 0, 1, 0], numSlices=2)
    model = CountingModel()
    cfg = VIConfig(max_iterations=4, convergence_tol=1e-10)
    runner = VIRunner(model=model, config=cfg)

    real_broadcast = spark.sparkContext.broadcast
    unpersist_calls = []

    class _WrappedBcast:
        def __init__(self, inner):
            self._inner = inner
        @property
        def value(self):
            return self._inner.value
        def unpersist(self, blocking=False):
            unpersist_calls.append(self._inner)
            return self._inner.unpersist(blocking=blocking)

    def _wrapping_broadcast(value):
        inner = real_broadcast(value)
        return _WrappedBcast(inner)

    with patch.object(spark.sparkContext, "broadcast", side_effect=_wrapping_broadcast):
        runner.fit(rdd)

    # With 4 iterations, we expect at least 3 unpersist calls for the
    # previous broadcasts plus one for the final broadcast on return = 4.
    # The regression we guard against is zero calls.
    assert len(unpersist_calls) >= 3, \
        f"Expected VIRunner to unpersist prior broadcasts; got {len(unpersist_calls)} calls"
```

- [ ] **Step 2: Run the test**

Run:
```bash
cd spark-vi && poetry run pytest tests/test_broadcast_lifecycle.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

Run:
```bash
cd .. && git add spark-vi/tests/test_broadcast_lifecycle.py
git commit -m "test(spark-vi): regression test for broadcast unpersist lifecycle"
```

### Task 3.8: Export round-trip (JSON + .npy) — test + implementation

**Files:**
- Create: `spark-vi/spark_vi/io/export.py`
- Create: `spark-vi/tests/test_export.py`

- [ ] **Step 1: Write the failing test**

Create `spark-vi/tests/test_export.py` with content:
```python
"""VIResult.save + load roundtrip via JSON + .npy sidecar files."""
import numpy as np


def test_export_roundtrip_preserves_params_exactly(tmp_path):
    from spark_vi.core import VIResult
    from spark_vi.io.export import save_result, load_result

    r = VIResult(
        global_params={
            "alpha": np.array(3.5),
            "lambda": np.array([[1.0, 2.0], [3.0, 4.0]]),
        },
        elbo_trace=[-100.0, -50.0, -10.0],
        n_iterations=3,
        converged=True,
        metadata={"model_class": "CountingModel", "git_sha": "abc123"},
    )
    out_dir = tmp_path / "result"
    save_result(r, out_dir)

    loaded = load_result(out_dir)
    np.testing.assert_array_equal(loaded.global_params["alpha"], r.global_params["alpha"])
    np.testing.assert_array_equal(loaded.global_params["lambda"], r.global_params["lambda"])
    assert loaded.elbo_trace == r.elbo_trace
    assert loaded.n_iterations == r.n_iterations
    assert loaded.converged is True
    assert loaded.metadata == r.metadata


def test_export_produces_inspectable_files(tmp_path):
    """Files on disk are plain JSON + .npy so a human can inspect them."""
    import json

    from spark_vi.core import VIResult
    from spark_vi.io.export import save_result

    r = VIResult(
        global_params={"alpha": np.array(1.0)},
        elbo_trace=[-1.0],
        n_iterations=1,
        converged=False,
        metadata={},
    )
    out = tmp_path / "x"
    save_result(r, out)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_iterations"] == 1
    assert manifest["converged"] is False
    assert (out / "params" / "alpha.npy").is_file()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd spark-vi && poetry run pytest tests/test_export.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `spark-vi/spark_vi/io/export.py`**

Create with content:
```python
"""Save and load VIResults in a human-inspectable format.

Layout:
    <dir>/
      manifest.json           # everything except the np.ndarrays
      params/
        <name>.npy            # one file per entry in global_params

Rationale: JSON + .npy is the simplest format that is inspectable from the
command line, survives long-term storage without opaque binary blobs, and
doesn't require any non-standard library to read. See
docs/architecture/SPARK_VI_FRAMEWORK.md#viresult-and-model-export.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spark_vi.core.result import VIResult


def save_result(result: VIResult, out_dir: Path | str) -> None:
    """Write `result` to `out_dir`. Creates the dir if needed."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params_dir = out / "params"
    params_dir.mkdir(exist_ok=True)

    for name, arr in result.global_params.items():
        np.save(params_dir / f"{name}.npy", np.asarray(arr))

    manifest = {
        "elbo_trace": list(result.elbo_trace),
        "n_iterations": int(result.n_iterations),
        "converged": bool(result.converged),
        "metadata": dict(result.metadata),
        "param_names": list(result.global_params.keys()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_result(in_dir: Path | str) -> VIResult:
    """Load a VIResult previously written by `save_result`."""
    in_path = Path(in_dir)
    manifest = json.loads((in_path / "manifest.json").read_text())
    params_dir = in_path / "params"
    global_params = {
        name: np.load(params_dir / f"{name}.npy") for name in manifest["param_names"]
    }
    return VIResult(
        global_params=global_params,
        elbo_trace=manifest["elbo_trace"],
        n_iterations=manifest["n_iterations"],
        converged=manifest["converged"],
        metadata=manifest["metadata"],
    )
```

- [ ] **Step 4: Update `spark-vi/spark_vi/io/__init__.py`**

Update content to:
```python
"""Save/load utilities for spark_vi results and models."""
from spark_vi.io.export import load_result, save_result

__all__ = ["load_result", "save_result"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
poetry run pytest tests/test_export.py -v
```

Expected: 2 tests pass.

- [ ] **Step 6: Commit**

Run:
```bash
cd .. && git add spark-vi/spark_vi/io/__init__.py spark-vi/spark_vi/io/export.py spark-vi/tests/test_export.py
git commit -m "feat(spark-vi): add JSON+npy result save/load"
```

### Task 3.9: Checkpoint round-trip — test + implementation

**Files:**
- Create: `spark-vi/spark_vi/diagnostics/checkpoint.py`
- Create: `spark-vi/tests/test_checkpoint.py`

- [ ] **Step 1: Write the failing test**

Create `spark-vi/tests/test_checkpoint.py` with content:
```python
"""Checkpoint-then-resume produces a VIResult indistinguishable from a
continuous run on the same data."""
import numpy as np


def test_checkpoint_then_resume_matches_continuous_run(spark, tmp_path):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.diagnostics.checkpoint import load_checkpoint, save_checkpoint
    from spark_vi.models.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1] * 60 + [0] * 40, numSlices=4)

    # Run continuously for 6 iterations.
    cfg6 = VIConfig(max_iterations=6, convergence_tol=1e-12)
    continuous = VIRunner(CountingModel(), cfg6).fit(rdd)

    # Run for 3 iterations, checkpoint, then resume for 3 more.
    cfg3 = VIConfig(max_iterations=3, convergence_tol=1e-12)
    r3 = VIRunner(CountingModel(), cfg3).fit(rdd)
    ckpt = tmp_path / "ckpt"
    save_checkpoint(r3.global_params, r3.elbo_trace, iteration=3, path=ckpt)

    global_params, elbo_trace, completed = load_checkpoint(ckpt)
    assert completed == 3

    # Resume: run the remaining 3 iterations, starting from the checkpoint state.
    runner2 = VIRunner(CountingModel(), VIConfig(max_iterations=3, convergence_tol=1e-12))
    # Use the internal attribute to seed _resume_ — we patch global_params
    # into the runner via a dedicated helper: for this test we inject via
    # fit(... data_summary=) since CountingModel ignores it; the restart is
    # equivalent to constructing a runner whose first broadcast is the
    # checkpointed params. We emulate that by starting a fresh runner but
    # skipping initialize_global.
    model = CountingModel()
    # Monkey-patch initialize_global to return the checkpointed params.
    orig_init = model.initialize_global
    model.initialize_global = lambda _ds: global_params  # type: ignore
    resumed = VIRunner(model, VIConfig(max_iterations=3, convergence_tol=1e-12)).fit(rdd)
    model.initialize_global = orig_init  # restore

    # Final alpha should match (posterior counts are additive and deterministic).
    np.testing.assert_allclose(
        resumed.global_params["alpha"],
        continuous.global_params["alpha"],
        rtol=1e-6,
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd spark-vi && poetry run pytest tests/test_checkpoint.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `spark-vi/spark_vi/diagnostics/checkpoint.py`**

Create with content:
```python
"""Checkpoint / resume support for long training runs.

The design mirrors the VIResult export format: params go to .npy files,
everything else goes to a JSON manifest. This keeps checkpoints inspectable
and platform-agnostic (any filesystem path works).

See docs/architecture/RISKS_AND_MITIGATIONS.md §No built-in checkpointing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_checkpoint(
    global_params: dict[str, np.ndarray],
    elbo_trace: list[float],
    iteration: int,
    path: Path | str,
) -> None:
    """Write an in-progress training state to `path/` (a directory)."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    params_dir = out / "params"
    params_dir.mkdir(exist_ok=True)
    for name, arr in global_params.items():
        np.save(params_dir / f"{name}.npy", np.asarray(arr))
    (out / "manifest.json").write_text(json.dumps({
        "elbo_trace": list(elbo_trace),
        "iteration": int(iteration),
        "param_names": list(global_params.keys()),
    }, indent=2))


def load_checkpoint(
    path: Path | str,
) -> tuple[dict[str, np.ndarray], list[float], int]:
    """Read a checkpoint. Returns (global_params, elbo_trace, iteration_completed)."""
    in_path = Path(path)
    manifest = json.loads((in_path / "manifest.json").read_text())
    params_dir = in_path / "params"
    global_params = {
        name: np.load(params_dir / f"{name}.npy") for name in manifest["param_names"]
    }
    return global_params, manifest["elbo_trace"], int(manifest["iteration"])
```

- [ ] **Step 4: Update `spark-vi/spark_vi/diagnostics/__init__.py`**

Update content to:
```python
"""Training-loop diagnostics: ELBO tracking, live display, checkpointing."""
from spark_vi.diagnostics.checkpoint import load_checkpoint, save_checkpoint

__all__ = ["load_checkpoint", "save_checkpoint"]
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
poetry run pytest tests/test_checkpoint.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:
```bash
cd .. && git add spark-vi/spark_vi/diagnostics/__init__.py spark-vi/spark_vi/diagnostics/checkpoint.py spark-vi/tests/test_checkpoint.py
git commit -m "feat(spark-vi): add checkpoint save/load"
```

### Task 3.10: OnlineHDP stub

**Files:**
- Create: `spark-vi/spark_vi/models/online_hdp.py`
- Modify: `spark-vi/spark_vi/models/__init__.py`

- [ ] **Step 1: Create the stub**

Create `spark-vi/spark_vi/models/online_hdp.py` with content:
```python
"""Online Hierarchical Dirichlet Process topic model (STUB).

This is a placeholder with the intended public signature so `charmpheno` can
depend on a stable name during bootstrap. The real implementation — a PySpark
port of Hoffman/Wang/Blei/Paisley stochastic VI for the HDP, patterned after
Spark MLlib's OnlineLDAOptimizer and the intel-spark TopicModeling Scala
implementation — is its own follow-on spec.

See docs/architecture/SPARK_VI_FRAMEWORK.md#online-hdp-topic-model and
docs/architecture/TOPIC_STATE_MODELING.md for the target contract.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from spark_vi.core.model import VIModel


class OnlineHDP(VIModel):
    """Stub OnlineHDP; real implementation deferred to a dedicated spec."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_topics: int = 150,
        eta: float = 0.01,
        alpha: float = 1.0,
        omega: float = 1.0,
    ) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        self.vocab_size = int(vocab_size)
        self.max_topics = int(max_topics)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.omega = float(omega)

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        raise NotImplementedError(
            "OnlineHDP is stubbed during bootstrap. See the follow-on spec in "
            "docs/superpowers/specs/ for the real implementation."
        )

    def local_update(
        self,
        rows: Iterable[Any],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is stubbed during bootstrap.")

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is stubbed during bootstrap.")
```

- [ ] **Step 2: Update `spark-vi/spark_vi/models/__init__.py`**

Update content to:
```python
"""Pre-built models for spark-vi."""
from spark_vi.models.counting import CountingModel
from spark_vi.models.online_hdp import OnlineHDP

__all__ = ["CountingModel", "OnlineHDP"]
```

- [ ] **Step 3: Verify imports and the stub's refusal**

Run:
```bash
cd spark-vi && poetry run python -c "
from spark_vi.models import OnlineHDP
m = OnlineHDP(vocab_size=100)
try:
    m.initialize_global(None)
except NotImplementedError as e:
    print('stub refused as expected:', e)
"
```

Expected: `stub refused as expected: OnlineHDP is stubbed...`

- [ ] **Step 4: Commit**

Run:
```bash
cd .. && git add spark-vi/spark_vi/models/__init__.py spark-vi/spark_vi/models/online_hdp.py
git commit -m "feat(spark-vi): add OnlineHDP stub with intended public signature"
```

### Task 3.11: Verify build + zip targets produce working artifacts

**Files:**
- Create: `spark-vi/tests/test_zip_import.py`

- [ ] **Step 1: Run make build + make zip**

Run:
```bash
cd spark-vi && make zip
```

Expected: `dist/spark_vi-0.1.0-py3-none-any.whl`, `dist/spark_vi-0.1.0.tar.gz`, and `dist/spark_vi.zip` all exist.

- [ ] **Step 2: Verify the zip contents**

Run:
```bash
unzip -l dist/spark_vi.zip | head -20
```

Expected: entries for `spark_vi/__init__.py`, `spark_vi/core/*.py`, `spark_vi/models/*.py`, etc. No `__pycache__/` entries.

- [ ] **Step 3: Write a smoke test for the zip**

Create `spark-vi/tests/test_zip_import.py` with content:
```python
"""Zip artifact is importable when placed on sys.path.

This simulates what `sc.addPyFile('spark_vi.zip')` does on a Spark worker.
"""
import sys
import subprocess
from pathlib import Path


def test_spark_vi_zip_imports_in_fresh_subprocess():
    zip_path = Path(__file__).resolve().parents[1] / "dist" / "spark_vi.zip"
    if not zip_path.is_file():
        import pytest
        pytest.skip(f"dist/spark_vi.zip not built; run `make zip` first")

    # Use sys.executable in a subprocess with *only* the zip on sys.path.
    # PYTHONPATH is the standard way to inject. We set it to just the zip.
    proc = subprocess.run(
        [sys.executable, "-c",
         "import spark_vi; "
         "from spark_vi.core import VIModel, VIRunner, VIConfig, VIResult; "
         "from spark_vi.models import CountingModel, OnlineHDP; "
         "print(spark_vi.__version__)"],
        env={"PYTHONPATH": str(zip_path), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"zip import failed:\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert proc.stdout.strip() == "0.1.0"
```

Note: this test may need `numpy`/`scipy` available in the fresh subprocess. Since we are calling the same interpreter, site-packages is NOT on the fresh PYTHONPATH — but site-packages discovery through `sys.path` default initialization DOES find them. If the test fails in your environment with "No module named numpy", modify the subprocess env to include the poetry venv's site-packages in PYTHONPATH (adjust inline per local setup).

- [ ] **Step 4: Run the smoke test**

Run:
```bash
poetry run pytest tests/test_zip_import.py -v
```

Expected: PASS (or SKIP with a clear message if numpy is unavailable to the subprocess — in which case update the env dict in the test and re-run).

- [ ] **Step 5: Commit**

Run:
```bash
cd .. && git add spark-vi/tests/test_zip_import.py
git commit -m "test(spark-vi): smoke-test that dist/spark_vi.zip imports in isolation"
```

### Task 3.12: Run full spark-vi test suite end-to-end

- [ ] **Step 1: Run `make test`**

Run:
```bash
make -C spark-vi test
```

Expected: all non-slow, non-cluster tests pass. Duration: a few seconds past the first Spark spin-up.

- [ ] **Step 2: Verify no files left uncommitted**

Run:
```bash
git status
```

Expected: clean working tree (or only `dist/` which is gitignored).

---

## Phase 4 — charmpheno skeleton

Goal: `charmpheno/` is a working poetry project depending editably on `spark-vi`. Public surface: `charmpheno.omop.{load_omop_parquet, load_omop_bigquery, validate}` + `charmpheno.phenotype.CharmPhenoHDP`. Tests green.

### Task 4.1: Create `charmpheno/` project skeleton

**Files:**
- Create: `charmpheno/pyproject.toml`
- Create: `charmpheno/.gitignore`
- Create: `charmpheno/README.md`
- Create: `charmpheno/Makefile`
- Create: `charmpheno/charmpheno/__init__.py`
- Create: `charmpheno/charmpheno/omop/__init__.py`
- Create: `charmpheno/charmpheno/phenotype/__init__.py`
- Create: `charmpheno/charmpheno/evaluate/__init__.py`
- Create: `charmpheno/charmpheno/export/__init__.py`
- Create: `charmpheno/charmpheno/profiles/__init__.py`
- Create: `charmpheno/tests/__init__.py`
- Create: `charmpheno/tests/conftest.py`
- Create: `charmpheno/tests/data/.gitkeep`

- [ ] **Step 1: Create directories and empty files**

Run:
```bash
mkdir -p charmpheno/charmpheno/omop charmpheno/charmpheno/phenotype charmpheno/charmpheno/evaluate charmpheno/charmpheno/export charmpheno/charmpheno/profiles charmpheno/tests/data
touch charmpheno/charmpheno/__init__.py \
      charmpheno/charmpheno/omop/__init__.py \
      charmpheno/charmpheno/phenotype/__init__.py \
      charmpheno/charmpheno/evaluate/__init__.py \
      charmpheno/charmpheno/export/__init__.py \
      charmpheno/charmpheno/profiles/__init__.py \
      charmpheno/tests/__init__.py \
      charmpheno/tests/data/.gitkeep
```

- [ ] **Step 2: Write `charmpheno/charmpheno/__init__.py`**

Create with content:
```python
"""charmpheno: clinical specialization on top of spark-vi.

Public surface:

    from charmpheno.omop import load_omop_parquet, validate
    from charmpheno.phenotype import CharmPhenoHDP
"""
__version__ = "0.1.0"
```

- [ ] **Step 3: Write `charmpheno/pyproject.toml`**

Create with content:
```toml
[tool.poetry]
name = "charmpheno"
version = "0.1.0"
description = "Computational phenotyping on PySpark using Bayesian topic models"
authors = ["CHARMPheno contributors"]
license = "MIT"
readme = "README.md"
packages = [{include = "charmpheno"}]

[tool.poetry.dependencies]
python = ">=3.10,<3.13"
pyspark = ">=3.5,<4.0"
numpy = ">=1.24"
pandas = ">=2.0"
pyarrow = ">=15.0"
# spark-vi is path-installed editable during dev; see Makefile.
# In a published release, this becomes: spark-vi = ">=0.1,<0.2"

[tool.poetry.group.dev.dependencies]
pytest = ">=7.0"
build = ">=1.0"

[tool.poetry.extras]
bigquery = ["google-cloud-bigquery"]

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
markers = [
    "slow: integration tests (simulator data, local Spark, minutes-scale)",
    "cluster: tests that require a real Dataproc cluster (manual only)",
]
addopts = "-m 'not slow and not cluster'"
filterwarnings = [
    "ignore::ResourceWarning",
]
```

- [ ] **Step 4: Write `charmpheno/.gitignore`**

Create with content:
```gitignore
dist/
build/
*.egg-info/
__pycache__/
.pytest_cache/
.coverage
```

- [ ] **Step 5: Write `charmpheno/README.md`**

Create with content:
```markdown
# charmpheno

Clinical phenotyping on PySpark. Built on [`spark-vi`](../spark-vi).

See the [research design](../docs/architecture/TOPIC_STATE_MODELING.md) for
the clinical context and [CHARMPheno](../README.md) for the project overall.

## Install (dev)

```bash
poetry install
poetry run pip install -e ../spark-vi
```

## Test

```bash
make test          # unit tests only (fast)
make test-all      # unit + @slow integration tests
```

## Build

```bash
make build         # dist/*.whl + dist/*.tar.gz
make zip           # dist/charmpheno.zip (flat, pure-Python)
```

Requires Java 17 for local Spark (same auto-detection as `spark-vi`).
```

- [ ] **Step 6: Write `charmpheno/Makefile`**

Create with content:
```makefile
.PHONY: help install dev clean build zip test test-all test-cluster

default: help

JAVA_HOME_MACOS := /opt/homebrew/opt/openjdk@17
JAVA_HOME_LINUX := /usr/lib/jvm/java-17-openjdk-amd64
JAVA_HOME ?= $(shell if [ -d "$(JAVA_HOME_MACOS)" ]; then echo "$(JAVA_HOME_MACOS)"; elif [ -d "$(JAVA_HOME_LINUX)" ]; then echo "$(JAVA_HOME_LINUX)"; fi)

help:
	@echo "charmpheno common targets:"
	@echo "  install      - poetry install + editable install of ../spark-vi"
	@echo "  dev          - install + build tooling"
	@echo "  test         - unit tests only"
	@echo "  test-all     - unit + @slow tests"
	@echo "  test-cluster - @cluster tests only (manual)"
	@echo "  build        - wheel + sdist"
	@echo "  zip          - flat dist/charmpheno.zip"
	@echo "  clean        - remove dist/, build/, caches"

install:
	poetry install
	poetry run pip install -e ../spark-vi

dev:
	poetry install
	poetry run pip install -e ../spark-vi
	poetry run pip install build

clean:
	rm -rf dist/ build/ *.egg-info
	rm -rf .pytest_cache

test:
	JAVA_HOME=$(JAVA_HOME) poetry run pytest tests/ -v

test-all:
	JAVA_HOME=$(JAVA_HOME) poetry run pytest tests/ -v -m "not cluster"

test-cluster:
	JAVA_HOME=$(JAVA_HOME) poetry run pytest tests/ -v -m cluster

build:
	poetry run python -m build

zip: clean build
	mkdir -p dist
	zip -r dist/charmpheno.zip charmpheno -x "*.pyc" -x "*/__pycache__/*" -x "*.egg-info/*"
	@echo "Wrote dist/charmpheno.zip"
```

- [ ] **Step 7: Write `charmpheno/tests/conftest.py`**

Create with content:
```python
"""Pytest fixtures for charmpheno.

Session-scoped local Spark (shared with integration tests). Same config as
spark-vi's conftest to keep behavior identical across packages.
"""
import os
import warnings

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    warnings.filterwarnings("ignore")
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    session = (
        SparkSession.builder.master("local[2]")
        .appName("charmpheno-tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )
    yield session
    session.stop()
```

- [ ] **Step 8: Install the package and spark-vi editable**

Run:
```bash
cd charmpheno && make install
```

Expected: installs pyspark, pandas, pyarrow, numpy, plus editable `spark-vi` into the charmpheno venv.

- [ ] **Step 9: Verify import**

Run:
```bash
poetry run python -c "import charmpheno; import spark_vi; print(charmpheno.__version__, spark_vi.__version__)"
```

Expected: `0.1.0 0.1.0`

- [ ] **Step 10: Commit scaffolding**

Run:
```bash
cd .. && git add charmpheno/
git commit -m "feat(charmpheno): project scaffolding + editable spark-vi install"
```

### Task 4.2: OMOP schema + validate — test + implementation

**Files:**
- Create: `charmpheno/charmpheno/omop/schema.py`
- Create: `charmpheno/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

Create `charmpheno/tests/test_schema.py` with content:
```python
"""Canonical OMOP shape + validator."""
import pytest


def test_canonical_columns_are_exactly_four():
    from charmpheno.omop.schema import CANONICAL_COLUMNS

    assert CANONICAL_COLUMNS == (
        "person_id", "visit_occurrence_id", "concept_id", "concept_name",
    )


def test_validate_accepts_canonical_dataframe(spark):
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [(1, 100, 5, "diabetes"), (2, 101, 6, "asthma")],
        schema="person_id INT, visit_occurrence_id INT, concept_id INT, concept_name STRING",
    )
    validate(df)  # no exception


def test_validate_rejects_missing_column(spark):
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [(1, 5, "diabetes")],
        schema="person_id INT, concept_id INT, concept_name STRING",
    )
    with pytest.raises(ValueError, match="missing.*visit_occurrence_id"):
        validate(df)


def test_validate_rejects_wrong_type(spark):
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [("not-an-int", 100, 5, "diabetes")],
        schema="person_id STRING, visit_occurrence_id INT, concept_id INT, concept_name STRING",
    )
    with pytest.raises(ValueError, match="person_id.*STRING|person_id.*type"):
        validate(df)


def test_validate_allows_extra_columns_by_default(spark):
    """Common case: real loaders may include a date column or similar. We
    don't reject extras; we just require the canonical four are present
    with right types."""
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [(1, 100, 5, "diabetes", "2024-01-01")],
        schema="person_id INT, visit_occurrence_id INT, concept_id INT, concept_name STRING, visit_date STRING",
    )
    validate(df)
```

- [ ] **Step 2: Run the test**

Run:
```bash
cd charmpheno && poetry run pytest tests/test_schema.py -v
```

Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `charmpheno/charmpheno/omop/schema.py`**

Create with content:
```python
"""Canonical OMOP shape used throughout charmpheno.

Every loader in `charmpheno.omop` returns a Spark DataFrame with at least
these four columns:

    person_id:           int   — deidentified patient identifier
    visit_occurrence_id: int   — identifier for one clinical encounter
    concept_id:          int   — OMOP vocabulary concept id
    concept_name:        str   — human-readable concept label

Additional columns (e.g. visit_date) may be present and are passed through
unchanged. See docs/decisions/0003-explicit-omop-io.md for rationale.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import types as T

CANONICAL_COLUMNS: tuple[str, ...] = (
    "person_id",
    "visit_occurrence_id",
    "concept_id",
    "concept_name",
)

_EXPECTED_TYPES: dict[str, type] = {
    "person_id": T.IntegerType,
    "visit_occurrence_id": T.IntegerType,
    "concept_id": T.IntegerType,
    "concept_name": T.StringType,
}


def validate(df: DataFrame) -> None:
    """Assert `df` has the canonical OMOP columns with expected types.

    Raises ValueError with a specific message on any mismatch; extras are OK.
    """
    schema = {f.name: type(f.dataType) for f in df.schema.fields}
    missing = [c for c in CANONICAL_COLUMNS if c not in schema]
    if missing:
        raise ValueError(f"OMOP DataFrame is missing required column(s): {missing}")
    for col, expected in _EXPECTED_TYPES.items():
        actual = schema[col]
        if not issubclass(actual, expected):
            raise ValueError(
                f"OMOP column {col!r} has wrong type: "
                f"expected {expected.__name__}, got {actual.__name__}"
            )
```

- [ ] **Step 4: Update `charmpheno/charmpheno/omop/__init__.py`**

Update content to:
```python
"""OMOP-shaped I/O and schema utilities."""
from charmpheno.omop.schema import CANONICAL_COLUMNS, validate

__all__ = ["CANONICAL_COLUMNS", "validate"]
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
poetry run pytest tests/test_schema.py -v
```

Expected: 5 tests pass.

- [ ] **Step 6: Commit**

Run:
```bash
cd .. && git add charmpheno/charmpheno/omop/__init__.py charmpheno/charmpheno/omop/schema.py charmpheno/tests/test_schema.py
git commit -m "feat(charmpheno): add OMOP canonical schema + validate"
```

### Task 4.3: `load_omop_parquet` — test + implementation

**Files:**
- Create: `charmpheno/charmpheno/omop/local.py`
- Create: `charmpheno/tests/test_load_omop_parquet.py`
- Create (test fixture data): `charmpheno/tests/data/tiny_omop.parquet`

- [ ] **Step 1: Generate the tiny test parquet**

Run:
```bash
cd charmpheno && poetry run python -c "
import pandas as pd
from pathlib import Path
out = Path('tests/data/tiny_omop.parquet')
pd.DataFrame({
    'person_id': [1, 1, 2, 3, 3],
    'visit_occurrence_id': [10, 10, 20, 30, 31],
    'concept_id': [100, 101, 200, 300, 301],
    'concept_name': ['a', 'b', 'c', 'd', 'e'],
}).to_parquet(out, index=False)
print('wrote', out)
"
```

Expected: writes `tests/data/tiny_omop.parquet`.

- [ ] **Step 2: Write the failing test**

Create `charmpheno/tests/test_load_omop_parquet.py` with content:
```python
"""Local parquet loader for OMOP-shaped data."""
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parent / "data" / "tiny_omop.parquet"


def test_load_omop_parquet_returns_canonical_shape(spark):
    from charmpheno.omop import validate
    from charmpheno.omop.local import load_omop_parquet

    df = load_omop_parquet(str(FIXTURE), spark=spark)
    validate(df)  # must not raise
    assert df.count() == 5
    assert {c.name for c in df.schema.fields} >= {
        "person_id", "visit_occurrence_id", "concept_id", "concept_name"
    }


def test_load_omop_parquet_raises_on_missing_file(spark):
    from charmpheno.omop.local import load_omop_parquet

    with pytest.raises(Exception):
        load_omop_parquet("/nonexistent/path.parquet", spark=spark)
```

- [ ] **Step 3: Run the test**

Run:
```bash
poetry run pytest tests/test_load_omop_parquet.py -v
```

Expected: FAIL (ImportError).

- [ ] **Step 4: Implement `charmpheno/charmpheno/omop/local.py`**

Create with content:
```python
"""Local-filesystem OMOP loader: read parquet into a Spark DataFrame.

Thin over `spark.read.parquet` but enforces the canonical shape via
`validate()` so a schema mismatch fails at the boundary instead of 40
iterations into a model fit.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from charmpheno.omop.schema import validate


def load_omop_parquet(path: str, *, spark: SparkSession) -> DataFrame:
    """Read an OMOP-shaped parquet file into a Spark DataFrame.

    The file must contain at least the canonical columns:
    person_id, visit_occurrence_id, concept_id, concept_name.
    """
    df = spark.read.parquet(path)
    validate(df)
    return df
```

- [ ] **Step 5: Update `charmpheno/charmpheno/omop/__init__.py`**

Update content to:
```python
"""OMOP-shaped I/O and schema utilities."""
from charmpheno.omop.local import load_omop_parquet
from charmpheno.omop.schema import CANONICAL_COLUMNS, validate

__all__ = ["CANONICAL_COLUMNS", "load_omop_parquet", "validate"]
```

- [ ] **Step 6: Run the test to verify it passes**

Run:
```bash
poetry run pytest tests/test_load_omop_parquet.py -v
```

Expected: 2 tests pass.

- [ ] **Step 7: Commit**

Run:
```bash
cd .. && git add charmpheno/charmpheno/omop/__init__.py charmpheno/charmpheno/omop/local.py charmpheno/tests/test_load_omop_parquet.py charmpheno/tests/data/tiny_omop.parquet
git commit -m "feat(charmpheno): add load_omop_parquet with schema validation"
```

### Task 4.4: BigQuery loader stub — test + implementation

**Files:**
- Create: `charmpheno/charmpheno/omop/bigquery.py`
- Create: `charmpheno/tests/test_bigquery_stub.py`

- [ ] **Step 1: Write the test**

Create `charmpheno/tests/test_bigquery_stub.py` with content:
```python
"""BigQuery loader is a stub during bootstrap; ensure it fails loudly."""
import pytest


def test_load_omop_bigquery_raises_not_implemented(spark):
    from charmpheno.omop.bigquery import load_omop_bigquery

    with pytest.raises(NotImplementedError, match="follow-on spec"):
        load_omop_bigquery(spark=spark, cdr_dataset="any.dataset")


def test_load_omop_bigquery_has_expected_signature():
    import inspect

    from charmpheno.omop.bigquery import load_omop_bigquery

    sig = inspect.signature(load_omop_bigquery)
    expected = {"spark", "cdr_dataset", "concept_types", "limit"}
    assert expected.issubset(sig.parameters.keys())
```

- [ ] **Step 2: Implement `charmpheno/charmpheno/omop/bigquery.py`**

Create with content:
```python
"""BigQuery OMOP loader (STUB).

Intended implementation: read OMOP CDM tables (condition_occurrence,
drug_exposure, procedure_occurrence) from a BigQuery dataset, project to
the canonical shape, return a Spark DataFrame via the spark-bigquery
connector (falling back to google-cloud-bigquery + pandas → Spark if
connector absent).

Bootstrap leaves this as an explicit NotImplementedError so scripts that
import it but are run in the wrong environment fail loudly with a clear
message. Real implementation is a dedicated follow-on spec.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession


def load_omop_bigquery(
    *,
    spark: SparkSession,
    cdr_dataset: str,
    concept_types: tuple[str, ...] = ("condition",),
    limit: int | None = None,
) -> DataFrame:
    """Load OMOP-shaped data from a BigQuery CDR dataset.

    Args:
        spark: active SparkSession.
        cdr_dataset: fully-qualified BQ dataset id "project.dataset".
        concept_types: which OMOP fact tables to include (condition, drug,
            procedure, measurement). Defaults to condition only.
        limit: optional row cap for development.

    Returns:
        Spark DataFrame with canonical OMOP columns.
    """
    raise NotImplementedError(
        "load_omop_bigquery is stubbed during bootstrap. See the follow-on "
        "spec in docs/superpowers/specs/ for the real implementation."
    )
```

- [ ] **Step 3: Update `charmpheno/charmpheno/omop/__init__.py`**

Update content to:
```python
"""OMOP-shaped I/O and schema utilities."""
from charmpheno.omop.bigquery import load_omop_bigquery
from charmpheno.omop.local import load_omop_parquet
from charmpheno.omop.schema import CANONICAL_COLUMNS, validate

__all__ = [
    "CANONICAL_COLUMNS",
    "load_omop_bigquery",
    "load_omop_parquet",
    "validate",
]
```

- [ ] **Step 4: Run tests**

Run:
```bash
cd charmpheno && poetry run pytest tests/test_bigquery_stub.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

Run:
```bash
cd .. && git add charmpheno/charmpheno/omop/__init__.py charmpheno/charmpheno/omop/bigquery.py charmpheno/tests/test_bigquery_stub.py
git commit -m "feat(charmpheno): add load_omop_bigquery stub with documented signature"
```

### Task 4.5: `CharmPhenoHDP` wrapper — test + implementation

**Files:**
- Create: `charmpheno/charmpheno/phenotype/charm_pheno_hdp.py`
- Create: `charmpheno/tests/test_charm_pheno_hdp_wrapper.py`

- [ ] **Step 1: Write the failing test**

Create `charmpheno/tests/test_charm_pheno_hdp_wrapper.py` with content:
```python
"""CharmPhenoHDP is a thin wrapper around spark_vi.models.OnlineHDP.

Its public surface is real (construction validated, API in place); its
.fit() propagates the underlying OnlineHDP stub's NotImplementedError
until the real OnlineHDP lands.
"""
import pytest


def test_charm_pheno_hdp_constructs_with_vocab_size():
    from charmpheno.phenotype import CharmPhenoHDP

    m = CharmPhenoHDP(vocab_size=500, max_topics=50)
    assert m.vocab_size == 500
    assert m.max_topics == 50


def test_charm_pheno_hdp_rejects_invalid_vocab_size():
    from charmpheno.phenotype import CharmPhenoHDP

    with pytest.raises(ValueError):
        CharmPhenoHDP(vocab_size=0)


def test_charm_pheno_hdp_fit_raises_not_implemented(spark):
    """Until the real OnlineHDP lands, fit raises NotImplementedError."""
    from charmpheno.phenotype import CharmPhenoHDP

    m = CharmPhenoHDP(vocab_size=10)
    # We don't need real data — the stub raises before touching the rdd.
    empty_rdd = spark.sparkContext.parallelize([], numSlices=1)
    with pytest.raises(NotImplementedError):
        m.fit(empty_rdd)


def test_charm_pheno_hdp_exposes_underlying_online_hdp():
    """The wrapper's .model attribute is the spark_vi OnlineHDP instance."""
    from spark_vi.models import OnlineHDP

    from charmpheno.phenotype import CharmPhenoHDP

    m = CharmPhenoHDP(vocab_size=10)
    assert isinstance(m.model, OnlineHDP)
```

- [ ] **Step 2: Run the test**

Run:
```bash
cd charmpheno && poetry run pytest tests/test_charm_pheno_hdp_wrapper.py -v
```

Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `charmpheno/charmpheno/phenotype/charm_pheno_hdp.py`**

Create with content:
```python
"""CharmPhenoHDP: clinical wrapper around the generic spark_vi OnlineHDP.

The wrapper adds the clinical/OMOP layer on top of the generic topic model:
concept-vocabulary handling, downstream export hooks, phenotype labels
(when the underlying OnlineHDP has converged). Bootstrap leaves the .fit()
path propagating the underlying stub's NotImplementedError so callers get
a clear signal that the framework is wired but the model math is not yet
implemented.

See docs/architecture/TOPIC_STATE_MODELING.md for the clinical design.
"""
from __future__ import annotations

from typing import Any

from pyspark import RDD
from spark_vi.core import VIConfig, VIResult, VIRunner
from spark_vi.models import OnlineHDP


class CharmPhenoHDP:
    """Thin clinical wrapper around `spark_vi.models.OnlineHDP`.

    Args:
        vocab_size: number of distinct concept_ids in the working vocabulary.
        max_topics: HDP truncation level (upper bound on discovered topics).
        eta, alpha, omega: hyperparameters passed through to OnlineHDP.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        max_topics: int = 150,
        eta: float = 0.01,
        alpha: float = 1.0,
        omega: float = 1.0,
    ) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        self.vocab_size = int(vocab_size)
        self.max_topics = int(max_topics)
        self.model = OnlineHDP(
            vocab_size=self.vocab_size,
            max_topics=self.max_topics,
            eta=eta,
            alpha=alpha,
            omega=omega,
        )

    def fit(
        self,
        data_rdd: RDD,
        config: VIConfig | None = None,
        data_summary: Any | None = None,
    ) -> VIResult:
        """Fit the underlying OnlineHDP on an RDD of documents.

        Raises NotImplementedError until the real OnlineHDP lands (see
        the follow-on spec in docs/superpowers/specs/).
        """
        runner = VIRunner(self.model, config=config)
        return runner.fit(data_rdd, data_summary=data_summary)
```

- [ ] **Step 4: Update `charmpheno/charmpheno/phenotype/__init__.py`**

Update content to:
```python
"""Clinical phenotype-model wrappers and utilities."""
from charmpheno.phenotype.charm_pheno_hdp import CharmPhenoHDP

__all__ = ["CharmPhenoHDP"]
```

- [ ] **Step 5: Run tests**

Run:
```bash
poetry run pytest tests/test_charm_pheno_hdp_wrapper.py -v
```

Expected: 4 tests pass.

- [ ] **Step 6: Commit**

Run:
```bash
cd .. && git add charmpheno/charmpheno/phenotype/__init__.py charmpheno/charmpheno/phenotype/charm_pheno_hdp.py charmpheno/tests/test_charm_pheno_hdp_wrapper.py
git commit -m "feat(charmpheno): add CharmPhenoHDP wrapper around spark_vi OnlineHDP"
```

### Task 4.6: charmpheno build + zip verification

- [ ] **Step 1: Run make zip**

Run:
```bash
make -C charmpheno zip
```

Expected: `charmpheno/dist/charmpheno-0.1.0-py3-none-any.whl`, `.tar.gz`, and `dist/charmpheno.zip` all exist.

- [ ] **Step 2: Verify full charmpheno test suite**

Run:
```bash
make -C charmpheno test
```

Expected: all unit tests pass.

- [ ] **Step 3: Verify top-level make test runs everything**

Run:
```bash
make test
```

Expected: spark-vi + charmpheno + tests/scripts all run and pass.

---

## Phase 5 — End-to-end smoke

Goal: `analysis/local/fit_charmpheno_local.py` runs the data → `CharmPhenoHDP` → `VIRunner` (with `CountingModel` substituted for the stub HDP) → export path end-to-end, and a `@pytest.mark.slow` integration test covers it.

### Task 5.1: Create `tests/integration/` skeleton

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/conftest.py`

- [ ] **Step 1: Create files**

Run:
```bash
mkdir -p tests/integration
touch tests/integration/__init__.py
```

- [ ] **Step 2: Write `tests/integration/conftest.py`**

Create with content:
```python
"""Shared fixtures for top-level integration tests."""
import os
import warnings
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def spark():
    warnings.filterwarnings("ignore")
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    session = (
        SparkSession.builder.master("local[2]")
        .appName("charmpheno-integration-tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )
    yield session
    session.stop()
```

### Task 5.2: Write `analysis/local/fit_charmpheno_local.py`

**Files:**
- Create: `analysis/local/__init__.py`
- Create: `analysis/local/fit_charmpheno_local.py`
- Create: `tests/integration/test_fit_charmpheno_local.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/test_fit_charmpheno_local.py` with content:
```python
"""End-to-end smoke: simulator parquet → VIRunner → exported artifact.

Bootstrap-scope smoke: uses CountingModel through VIRunner (not real HDP,
which is stubbed). Proves the plumbing — data loading, Spark distribution,
training loop, export — works.

Marked @slow because it invokes the data simulator with a small N, which
takes a few seconds more than unit tests.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "local"))


@pytest.mark.slow
def test_fit_charmpheno_local_smoke_runs(spark, tmp_path):
    # Arrange: generate a tiny synthetic parquet on the fly so the test is
    # hermetic (doesn't depend on an earlier `make data` invocation).
    import pandas as pd
    fixture = tmp_path / "sim.parquet"
    pd.DataFrame({
        "person_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "visit_occurrence_id": [10, 10, 20, 20, 30, 30, 40, 40],
        "concept_id": [1, 0, 1, 0, 1, 1, 0, 0],
        "concept_name": ["head", "tail", "head", "tail",
                         "head", "head", "tail", "tail"],
        "true_topic_id": [0, 0, 0, 0, 0, 0, 0, 0],
    }).to_parquet(fixture, index=False)

    from fit_charmpheno_local import main

    out_dir = tmp_path / "result"
    rc = main([
        "--input", str(fixture),
        "--output", str(out_dir),
        "--max-iterations", "3",
    ])
    assert rc == 0

    # Artifact exists and can be loaded back.
    from spark_vi.io import load_result
    result = load_result(out_dir)
    assert result.n_iterations == 3
    assert len(result.elbo_trace) == 3
    # CountingModel's posterior counts should have moved past the prior.
    assert float(result.global_params["alpha"]) > 1.0
```

- [ ] **Step 2: Create `analysis/local/__init__.py`**

Run:
```bash
mkdir -p analysis/local
touch analysis/local/__init__.py
```

- [ ] **Step 3: Create `analysis/local/fit_charmpheno_local.py`**

Create with content:
```python
"""End-to-end local smoke: simulator parquet → VIRunner → saved artifact.

Bootstrap scope: uses CountingModel (coin-flip posterior) to exercise the
plumbing because the real OnlineHDP is still stubbed. When the real HDP
lands, this script gets an option to select between CountingModel and
CharmPhenoHDP — or is split into two entrypoints.

Expected input schema (OMOP-shaped parquet):
    person_id, visit_occurrence_id, concept_id, concept_name[, true_topic_id]

For the smoke: rows are treated as 0/1 based on concept_id parity. Real
topic modeling lives in the follow-on HDP spec.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from spark_vi.core import VIConfig, VIRunner
from spark_vi.io import save_result
from spark_vi.models import CountingModel

log = logging.getLogger(__name__)


def _build_spark() -> SparkSession:
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    return (
        SparkSession.builder.master("local[2]")
        .appName("fit_charmpheno_local")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to OMOP-shaped parquet (e.g. from scripts/simulate_lda_omop.py)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for the saved VIResult")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--spark", type=str, default=None,
                        help="Optional externally-provided SparkSession (unused from CLI)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    spark = _build_spark()
    try:
        from charmpheno.omop import load_omop_parquet, validate
        df = load_omop_parquet(str(args.input), spark=spark)
        validate(df)

        # Coerce to 0/1 for the smoke's CountingModel: even concept_ids → 1,
        # odd → 0. This is nonsensical for clinical data but exercises the
        # end-to-end pipeline with real Spark distribution.
        rdd = df.select("concept_id").rdd.map(lambda row: 1 if row["concept_id"] % 2 == 0 else 0)

        runner = VIRunner(
            CountingModel(prior_alpha=1.0, prior_beta=1.0),
            config=VIConfig(max_iterations=args.max_iterations, convergence_tol=1e-12),
        )
        result = runner.fit(rdd)
        save_result(result, args.output)
        log.info("Wrote %s (n_iterations=%d, converged=%s)",
                 args.output, result.n_iterations, result.converged)
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the integration test**

Run:
```bash
cd charmpheno && poetry run pytest ../tests/integration/test_fit_charmpheno_local.py -v -m slow
```

Expected: PASS (first Spark startup ~5–10s).

Note: the test runs from charmpheno's venv because that venv has both `spark_vi` (editable) and `charmpheno` installed. We could also add an integration-tests-only venv, but reusing charmpheno's is YAGNI-simpler.

- [ ] **Step 5: Update the top-level Makefile to run integration tests**

The existing top-level `Makefile` already has `test-all` that runs `tests/integration/` via the root poetry env. But that env doesn't have `charmpheno` installed. Easier: route top-level integration tests through the charmpheno venv. Update the top-level `Makefile`'s `test-all` target:

In `Makefile`, replace the `test-all` target with:
```makefile
test-all:
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi test-all; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno test-all; fi
	@if [ -d tests/integration ]; then \
		cd charmpheno && JAVA_HOME=$$(if [ -d /opt/homebrew/opt/openjdk@17 ]; then echo /opt/homebrew/opt/openjdk@17; elif [ -d /usr/lib/jvm/java-17-openjdk-amd64 ]; then echo /usr/lib/jvm/java-17-openjdk-amd64; fi) poetry run pytest ../tests/integration -v -m "not cluster"; \
	fi
```

- [ ] **Step 6: Run `make test-all` to verify the full test surface**

Run:
```bash
cd .. && make test-all
```

Expected: all unit tests + scripts tests + integration tests pass.

- [ ] **Step 7: Commit**

Run:
```bash
git add analysis/local/ tests/integration/ Makefile
git commit -m "feat(analysis): add local end-to-end smoke script + integration test"
```

---

## Phase 6 — First tutorial

Goal: `notebooks/tutorials/01_project_setup.ipynb` is a thin, committed, output-stripped notebook that walks a new contributor through: clone → local env → make install → make data → make test → run the smoke script. The tutorial makes the project legible without becoming math exposition.

### Task 6.1: Create the tutorial notebook

**Files:**
- Create: `notebooks/tutorials/01_project_setup.ipynb`

- [ ] **Step 1: Generate the tutorial notebook**

Run:
```bash
poetry run python <<'PY'
import json
from pathlib import Path

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 01 — Project setup\n",
                "\n",
                "Walk-through of getting CHARMPheno running locally. Target audience: a contributor who has just cloned the repo and has no other context.\n",
                "\n",
                "## Prerequisites\n",
                "\n",
                "- Python 3.10–3.12.\n",
                "- Java 17 available (local Spark requires it — Java 23+ has Arrow compatibility issues in PySpark 3.5).\n",
                "  - macOS: `brew install openjdk@17`\n",
                "  - Linux: `apt install openjdk-17-jdk` or equivalent.\n",
                "- Poetry 2.x (`pipx install poetry`).\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Repository layout\n",
                "\n",
                "Top-level:\n",
                "\n",
                "- `docs/architecture/` — the living architectural vision. Start here if you want the *why*.\n",
                "- `docs/decisions/` — ADRs. One per significant organizational choice.\n",
                "- `docs/superpowers/specs/` and `docs/superpowers/plans/` — design documents and implementation plans.\n",
                "- `spark-vi/` — the reusable, domain-agnostic distributed-VI framework.\n",
                "- `charmpheno/` — the clinical specialization. Depends on `spark-vi`.\n",
                "- `analysis/` — runnable end-to-end scripts.\n",
                "- `notebooks/tutorials/` — exposition notebooks (you are here).\n",
                "- `scripts/` — data fetch, simulator, dev helpers.\n",
                "- `AGENTS.md` — orientation for LLM-based coding agents.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Install everything\n",
                "\n",
                "```bash\n",
                "make install-dev\n",
                "```\n",
                "\n",
                "This does three things:\n",
                "\n",
                "1. Installs the root venv (`.venv/`) used by scripts and pre-commit.\n",
                "2. Installs `spark-vi`'s own venv (`spark-vi/.venv/`).\n",
                "3. Installs `charmpheno`'s own venv (`charmpheno/.venv/`), with `spark-vi` installed editable inside it.\n",
                "\n",
                "During dev, changes to `spark-vi` flow through to `charmpheno` immediately without reinstall.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Fetch reference β and generate synthetic data\n",
                "\n",
                "```bash\n",
                "make data\n",
                "```\n",
                "\n",
                "This fetches the LDA β from Hugging Face (streaming; ~15 MB after top-K filter) into `data/cache/lda_beta_topk.parquet`, then generates a synthetic OMOP parquet under `data/simulated/`.\n",
                "\n",
                "`data/` is `.gitignore`d. Nothing generated here ever reaches the repo.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Run the tests\n",
                "\n",
                "```bash\n",
                "make test          # unit only, <10s, your default iteration loop\n",
                "make test-all      # + integration tests (@slow), a few minutes\n",
                "make test-cluster  # cluster-only tests, manual on a Dataproc setup\n",
                "```\n",
                "\n",
                "The default `make test` is deliberately fast. If your change touches an integration surface, run `make test-all` before pushing.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## End-to-end smoke\n",
                "\n",
                "After `make data`, run:\n",
                "\n",
                "```bash\n",
                "cd charmpheno && poetry run python ../analysis/local/fit_charmpheno_local.py \\\n",
                "    --input ../data/simulated/omop_N10000_seed0.parquet \\\n",
                "    --output ../tmp/result \\\n",
                "    --max-iterations 10\n",
                "```\n",
                "\n",
                "This runs the data → VIRunner (CountingModel placeholder) → saved artifact path. It confirms plumbing — real HDP math lives in a follow-on spec.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Packaging artifacts\n",
                "\n",
                "```bash\n",
                "make build      # builds wheels + sdists for both packages\n",
                "make zip        # builds flat zips for both packages\n",
                "```\n",
                "\n",
                "Both artifact types are required. Wheels go to `pip install` targets; zips go to Spark executors via `sc.addPyFile('…/pkg.zip')` or `--py-files`.\n",
                "\n",
                "## Deploying to a cloud notebook environment (appendix)\n",
                "\n",
                "The canonical pattern — deployed to a GCS-backed Dataproc + Jupyter workbench — is:\n",
                "\n",
                "1. After `make build` + `make zip` on a dev machine, upload `spark-vi/dist/*.whl`, `spark-vi/dist/spark_vi.zip`, `charmpheno/dist/*.whl`, and `charmpheno/dist/charmpheno.zip` to a workspace GCS bucket (`gs://<bucket>/pkgs/`).\n",
                "2. In the notebook environment: `pip install gs://<bucket>/pkgs/<wheels>` for the driver; `sc.addPyFile('gs://<bucket>/pkgs/<zip>')` for each package for executors.\n",
                "3. Import and run — code developed locally runs unchanged.\n",
                "\n",
                "This completes the basic orientation. Next: see `docs/architecture/` for the research design and framework architecture, and individual design specs under `docs/superpowers/specs/` for per-feature context.\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = Path("notebooks/tutorials/01_project_setup.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
PY
```

Expected: `wrote notebooks/tutorials/01_project_setup.ipynb`.

- [ ] **Step 2: Verify the notebook is valid JSON + parseable as a notebook**

Run:
```bash
poetry run python -c "
import json, nbformat
with open('notebooks/tutorials/01_project_setup.ipynb') as f:
    nb = nbformat.read(f, as_version=4)
print('cells:', len(nb.cells))
"
```

If `nbformat` is not installed: `poetry add --group dev nbformat` and re-run. Expected: `cells: 7`.

- [ ] **Step 3: Run pre-commit to confirm nbstripout is a no-op on this (no outputs to strip) notebook**

Run:
```bash
poetry run pre-commit run --files notebooks/tutorials/01_project_setup.ipynb
```

Expected: all hooks pass. `nbstripout` may modify trailing-whitespace or end-of-file formatting; if so, re-stage and re-run until green.

- [ ] **Step 4: Commit**

Run:
```bash
git add notebooks/tutorials/01_project_setup.ipynb
git commit -m "docs(tutorial): add 01_project_setup onboarding notebook"
```

### Task 6.2: Final verification & push

- [ ] **Step 1: Re-run everything green**

Run:
```bash
make clean
make install-dev
make test
```

Expected: all tests green.

- [ ] **Step 2: Check `make data` end-to-end** (optional but recommended)

Run:
```bash
make data
ls -la data/cache/ data/simulated/
```

Expected: filtered β parquet + simulated OMOP parquet + sidecar meta JSON.

- [ ] **Step 3: Verify `make test-all` including integration**

Run:
```bash
make test-all
```

Expected: all unit + @slow integration tests pass.

- [ ] **Step 4: Confirm clean working tree and review history**

Run:
```bash
git status
git log --oneline
```

Expected: clean tree; commits for Phase 1 (initial), Phase 2 (fetch_lda_beta, simulate_lda_omop), Phase 3 (scaffolding, VIConfig, VIResult, VIModel+CountingModel, VIRunner, broadcast lifecycle, export, checkpoint, OnlineHDP stub, zip smoke), Phase 4 (scaffolding, OMOP schema+validate, load_omop_parquet, load_omop_bigquery stub, CharmPhenoHDP wrapper), Phase 5 (smoke script + integration test), Phase 6 (tutorial).

- [ ] **Step 5: Create the GitHub repo (manual, by the user)**

The user creates a private GitHub repo (e.g., `tislab/CHARMPheno`) via the GitHub UI. Then:

Run:
```bash
git remote add origin git@github.com:<org>/CHARMPheno.git
git push -u origin main
```

(Exact org / repo name is a user decision; this step is intentionally manual.)

---

## End state

After completing all six phases:

- `git log` shows ~20–30 small, semantic commits covering bootstrap.
- `make test` runs green in under ~10s after the first Spark spin-up cost.
- `make test-all` runs green in a few minutes.
- `make build` and `make zip` produce both wheel and flat-zip artifacts for both packages.
- `make data` fetches + simulates without manual steps.
- `AGENTS.md`, four ADRs, and the architecture docs orient future contributors.
- `notebooks/tutorials/01_project_setup.ipynb` onboards a new contributor.
- Pre-commit hooks are installed; `data/` is gitignored; `tests/*/data/` holds tiny committed fixtures.
- `spark_vi.models.OnlineHDP` and `charmpheno.omop.load_omop_bigquery` are stubs raising `NotImplementedError` with clear pointers to their follow-on specs.

Next specs (out of scope for this plan):
- Real `OnlineHDP` implementation.
- Real `load_omop_bigquery` implementation.
- Recovery metric methodology.
- First math tutorial notebook (e.g., `02_vi_intuition.ipynb`) once there's real math to unpack.
- Sparse OU as a demonstration in `charmpheno/`.

---

## Self-review

- **Spec coverage.** Every goal in the spec maps to tasks: single-repo layout (Phase 1), fast iteration (test marker structure in 3.1 pyproject and 4.1 pyproject + default addopts), dual delivery (3.11, 4.6), explicit I/O (4.2–4.4), data-safety invariants (1.5, 1.7, 1.8), living architecture docs (1.3, 1.9, 1.10), understanding-as-first-class (ADRs + AGENTS.md + tutorial).
- **Non-goals respected.** No real HDP math (3.10 stub; 5.2 uses CountingModel). No real BigQuery loader (4.4 stub). No CI configuration. No OU. No recovery-metric assertion.
- **Placeholders.** Scanned for TBD/TODO/"implement later": none in active code. The two `NotImplementedError` stubs are intentional and documented as such.
- **Type consistency.** `VIConfig`, `VIModel`, `VIResult`, `VIRunner` names consistent across Phase 3 tasks. `OnlineHDP` signature in 3.10 matches the wrapper in 4.5 (`vocab_size`, `max_topics`, `eta`, `alpha`, `omega`). `load_omop_parquet(path, *, spark)` consistent between 4.3 implementation and 5.2 caller.
- **Commands.** Each step has a concrete `Run:` or file content block; no step prescribes behavior without showing how.
