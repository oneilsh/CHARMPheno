# AGENTS.md

Orientation for LLM-based coding agents working on CHARMPheno, and the
**overview of record** for the repo's conventions. Humans can read it too; it is
not a user-facing README. It is loaded into every session via `CLAUDE.md`
(`@AGENTS.md`), so the top sections are the fast on-ramp and the rest is
pointer-heavy for as-needed lookup.

---

## Start here

CHARMPheno is interpretable computational phenotyping over All of Us OMOP EHR:
gated LDA / prediction-constrained (PC) topic models fit with Bayesian
variational inference, distributed on Spark/Dataproc. Three layers:

- `spark-vi/` — the domain-agnostic VI framework (`VIModel`, `VIRunner`,
  `OnlineHDP`). Never imports `charmpheno`.
- `charmpheno/` — the clinical specialization (OMOP semantics, concept vocab,
  export, metrics). May depend on `spark-vi`; the reverse is forbidden.
- `analysis/cloud/` — the runnable Dataproc drivers (fit, readout, diagnostics)
  and their `Makefile`.

**Where the active line of work lives — read this before reading code.** The
current experiment, its design, and its results are the newest entries in:

- `docs/experiments/NNNN-*.md` — the experiment log (what was run, why, results).
- `docs/superpowers/specs/` and `docs/superpowers/plans/` — the normative spec
  and build plan for in-flight work.
- `docs/reports/` — analyses and scouting notes feeding decisions.

Orient from those, then the code. The active development branch is named in
**Cluster practices** below.

## Operational invariants you can break *silently*

These are the mistakes that don't fail loudly — they corrupt caches, leak data,
or ship wrong numbers. Read before editing anything under `charmpheno/omop/` or
the label-DAG / assembler modules.

### The cache-key landmine (source-hashed modules)

Bundle/corpus/sidecar cache keys fold in the **source hash of whole modules**
(`cohort_defs_version()` over `cohorts.py`; `_module_source_hash` over the
assembler and label-DAG modules). Editing one **silently invalidates every cache
key in the repo, or poisons a cache under a byte-identical key** — a
wrong-results hazard, not merely a rebuild cost. Several also carry **byte-pinned
tripwire hashes** in `tests/scripts/test_case_finding_cache_mondo.py` (60 tests).

**Treat as source-hashed / do-not-edit-casually:**
`charmpheno/charmpheno/omop/{cohorts,multi_domain,case_finding_assembly}.py`;
`analysis/cloud/{mondo_dag,mondo_native_dag,mondo_usage_core,mondo_collapse,
mondo_to_omop_mapping,condition_dag,preindex_closure}.py`. The **authoritative
check** is that tripwire suite passing byte-identical.

- **Driver-owned files are free to edit:** `analysis/cloud/{gated_pc_cloud,
  gated_pc_readout,distributed_readout,conversion_*,diag_*,episode_index}.py`,
  `scripts/run_experiment.py`, the `Makefile`, tests. Prefer solving at a driver
  **seam** (the `attested_provider` / injection-parameter pattern) over editing
  a hashed module.
- Editing a hashed module is a **deliberate, announced, full cache-drop** in ONE
  commit that re-pins the tripwire hashes and names the drop. Never incidental.
- **`--py-files` rule:** any module whose functions run executor-side (UDFs,
  `mapPartitions`) must be on `--py-files` in **every** submit path that can
  trigger it, or executors hit `ModuleNotFoundError`.
- **ADR 0047 closure doctrine:** nothing array-shaped rides a task closure;
  `treeAggregate` zeros are None-sentinel. Study `distributed_readout.py`'s
  `SparkStatsFn` before touching the distributed paths.

### Egress floor (All of Us)

Aggregate outputs respect the disclosure floor: **any cell < 20 is not
disclosable**. Per-node / per-cell tables stay workspace-internal; committed
reports and log banners carry pooled figures and counts-of-nodes only. **Never
commit patient-level data** to the tree under any circumstances (see Data and
repo hygiene).

## Cluster practices

Cluster runs happen on the user's Dataproc cluster, checked out at
`~/repos/CHARMPheno`. Two facts govern every cluster command:

1. **A fresh cluster clones on `main`.** A bare `make …` there runs stale code or
   fails "No rule to make target", and a plain `git pull origin <branch>` from
   `main` fails on divergent branches. So **every command handed to the user for
   the cluster MUST carry this preamble** (it creates a tracking branch on a
   fresh clone and fast-forwards an existing one):

   ```bash
   cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
   ```

   Then the real command on the next line(s), e.g.:

   ```bash
   cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
   make -C analysis/cloud diag-episode-probe ID=110 GPR_ARGS="--gap-days 90"
   ```

   - The active development branch is currently **`claude/gated-conditional-voi`**.
     Update the name here when it changes.
   - `git pull --ff-only` fast-forwards or refuses — never merges or destroys
     local state. The cluster is a pure runner; a refusal means something
     unexpected, so surface it rather than papering over with `reset --hard`.
   - This applies ONLY to commands the user runs on the cluster. Commands run in
     the session's own working copy do not need it.

2. **Caches on HDFS are cluster-local and ephemeral.** Bundle, corpus, and
   sidecar caches default to `hdfs://…`, which dies with the cluster — a fresh
   cluster has empty HDFS and must rebuild. Expensive artifacts (the sidecar,
   the bundle) therefore rebuild per cluster unless pointed at a **persistent,
   in-boundary GCS bucket** (`GPR_SIDECAR_URI` / `GPR_CACHE_URI` / `LR_CACHE_URI`).
   Do not guess bucket names — the Makefile defaults may be stale; ask the user
   for the current in-boundary staging bucket before writing to GCS.
   Right-size the master for the driver-collect wall of record runs, not just
   the smoke.

## Read these before suggesting architectural changes

The architectural vision lives in [`docs/architecture/`](docs/architecture/):
- `TOPIC_STATE_MODELING.md` — research design for phenotype discovery.
- `SPARK_VI_FRAMEWORK.md` — framework design (VIRunner, VIModel contract).
- `RISKS_AND_MITIGATIONS.md` — known risks, constraints, deployment notes.

External references for the Online HDP port:
- Wang, Paisley & Blei 2011, "Online Variational Inference for the
  Hierarchical Dirichlet Process" (AISTATS) — primary algorithmic reference.
- Teh, Jordan, Beal & Blei 2006, "Hierarchical Dirichlet Processes" (JASA) —
  the underlying nonparametric model.
- Hoffman, Blei, Wang & Paisley 2013, "Stochastic Variational Inference"
  (JMLR) — general SVI framework these models live in.
- Spark MLlib `OnlineLDAOptimizer` — the distribute-the-E-step pattern to
  emulate.
- intel-spark TopicModeling, https://github.com/intel-spark/TopicModeling —
  existing Scala port of Wang/Paisley/Blei 2011 to Spark; consult for layout
  and reference outputs, but its `chunk.collect()` driver-side E-step is the
  anti-pattern we explicitly diverge from.

## Rendering: agent chat output is plain text

Agent responses render as plain text in this user's IDE. **LaTeX does not
render** — do not write `$\lambda$`, `$\sum_k$`, `\begin{equation}`, etc. in
chat. Use plain Greek letters (λ, γ, α, θ, β) and ASCII math (`E[log β]`,
`sum_k`, `phi_dnk`) instead. Markdown, code fences, and clickable
`[text](path)` links DO render and should be used. (LaTeX inside
`docs/architecture/*.md` files is fine — it renders when those files are
viewed in a Markdown previewer.)

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
- `analysis/` holds runnable end-to-end scripts (thin). `analysis/cloud/` is the
  Dataproc drivers + `Makefile`; `scripts/run_experiment.py` is the fit entry.
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
  `tests/*/data/` or `docs/`. Run `make precommit-install` once per clone
  to install the hooks and the nbstripout git clean filter (so notebook
  outputs strip at `git add` time, not at commit time).
- Work with clinical data only in its approved environment; do not check
  patient-level data into the working tree under any circumstances. Aggregate
  outputs respect the egress floor (Operational invariants).

## The `docs/` map

Every non-obvious choice or observation leaves a durable, numbered artifact in
`docs/` — the discipline is spelled out in
[`docs/META_process.md`](docs/META_process.md). The systems, and their indexes:

- [`docs/decisions/`](docs/decisions/) — **ADRs**: forward-looking architectural
  *decisions* (why X over Y). ~200 words each; supersession named, never
  overwritten. Index + skeleton in its README.
- [`docs/insights/`](docs/insights/) — **Insights**: backward-looking empirical
  *observations* from runs (failure modes, which diagnostics discriminate,
  hypotheses that did/didn't survive). Name the setting that produced each.
  Index + format in its README.
- [`docs/experiments/`](docs/experiments/) — **Experiment log**: one dated,
  numbered doc per experiment (front matter + run log + results). Status
  lifecycle and index in its README.
- [`docs/superpowers/`](docs/superpowers/) — **Specs & plans**: `specs/` hold
  the normative design (definitions, requirements) for a body of work; `plans/`
  hold the work-package build plan that implements a spec. The
  audit → spec → plan → experiment → report flow is described in its README.
- [`docs/reports/`](docs/reports/) — analyses, scouting notes, and audits that
  feed decisions but are not themselves experiments (dated, not numbered).
- [`docs/architecture/`](docs/architecture/) — the living architectural vision
  (above). If a change contradicts these, update the section in the same commit
  or write an ADR recording the departure. Never silently diverge.
- [`docs/REVIEW_LOG.md`](docs/REVIEW_LOG.md) — a running log of code-walkthrough
  and refactor sessions. After a substantive review or refactor, append a dated
  entry at the top: areas reviewed, what shipped, pre-existing issues caught,
  open threads parked. Impersonal and project-scoped.

## Testing expectations

- **Package unit tests:** `make test` — unit only, must finish under ~10s.
- **Package integration:** `make test-all` (`@pytest.mark.slow`: simulator data,
  local Spark, minutes-scale).
- **Cluster-driver tests** (`analysis/cloud/`, under `tests/scripts/`) need the
  framework on the path, and `-m ""` to include the slow local-Spark cases:
  ```bash
  PYTHONPATH=spark-vi poetry run pytest tests/scripts/<file> -p no:randomly -m ""
  ```
  Some failures are pre-existing/environmental (e.g. `charmpheno.omop` import
  errors inside PySpark UDF workers) — baseline with `git stash` first; the
  obligation is no NEW failures, and the tripwire suite stays byte-identical.
- **Cluster tests:** `@pytest.mark.cluster`, manual only via `make test-cluster`.

## When you finish a change

Before declaring work complete:
- Tests pass (`make test` minimum; `make test-all` if integration surfaced
  changed; the relevant `tests/scripts/` suite if a driver changed).
- Relevant `docs/architecture/*.md` section updated if the change was
  architectural, or an ADR records the exception. A run that surfaced something
  gets an experiment/insight/report entry.
- Docstrings for new math functions explain *why*, not just *what*.
- No data files, secrets, or large binaries staged; no patient-level data;
  aggregate outputs within the egress floor.
