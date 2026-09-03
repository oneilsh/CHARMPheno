# CHARMPheno — working notes for Claude

Auto-loaded every session. Keep it TIGHT and pointer-heavy — it costs context on
every turn. Depth lives in [`AGENTS.md`](AGENTS.md) (architecture, boundaries,
packaging, testing philosophy) and [`docs/META_process.md`](docs/META_process.md)
(how the durable-docs systems work). Read those before anything architectural.

## What this is, and where the active work lives

Gated LDA / PC phenotyping over All of Us OMOP EHR, fit with Bayesian VI on
Spark/Dataproc. `spark-vi/` is the domain-agnostic framework; `charmpheno/` the
clinical specialization; `analysis/cloud/` the runnable cluster drivers.

The **current line of work** is whatever the newest entries in
`docs/experiments/`, `docs/superpowers/specs/`, `docs/superpowers/plans/`, and
`docs/reports/` describe — start there to orient, not from the code. The active
development branch is named in the cluster preamble below.

## THE cache-key landmine — read before editing any assembler/label module

Bundle/corpus/sidecar cache keys fold in the **source hash of whole modules**
(`cohort_defs_version()` over `cohorts.py`; `_module_source_hash` over the
assembler and label-DAG modules). So editing one of these **silently
invalidates every cache key in the repo, or poisons a cache under a
byte-identical key** — a wrong-results hazard, not just a rebuild cost. Several
also have **byte-pinned tripwire hashes** in
`tests/scripts/test_case_finding_cache_mondo.py` (60 tests) that break on any
edit.

**Treat these as source-hashed / do-not-edit casually:**
`charmpheno/charmpheno/omop/{cohorts,multi_domain,case_finding_assembly}.py`,
`analysis/cloud/{mondo_dag,mondo_native_dag,mondo_usage_core,mondo_collapse,
mondo_to_omop_mapping,condition_dag,preindex_closure}.py`. The **authoritative
check** is that tripwire test passing byte-identical.

- Driver-owned files are free to edit: `analysis/cloud/{gated_pc_cloud,
  gated_pc_readout,distributed_readout,conversion_*,diag_*,episode_index}.py`,
  `run_experiment.py`, the Makefile, tests. Prefer solving at a driver seam
  (the `attested_provider` / injection-param pattern) over editing a hashed
  module.
- Editing a hashed module is a **deliberate, announced, full cache-drop**, done
  in ONE commit that re-pins the tripwire hashes and names the drop (see the
  active plan's "one-blast" commit for the sanctioned procedure). Never do it
  incidentally.
- **`--py-files` rule:** any module whose functions run executor-side (UDFs,
  mapPartitions) must be on `--py-files` in EVERY submit path that can trigger
  it, or executors hit `ModuleNotFoundError`.
- **ADR 0047 closure doctrine:** nothing array-shaped rides a task closure;
  `treeAggregate` zeros are None-sentinel. Study `distributed_readout.py`'s
  `SparkStatsFn` before touching the distributed paths.

## Egress floor (All of Us)

Aggregate outputs must respect the disclosure floor: **any cell < 20 is not
disclosable**. Per-node / per-cell tables stay workspace-internal; committed
reports and log banners carry pooled figures and counts-of-nodes only. Never
commit patient-level data to the tree under any circumstances.

## Running tests

- **Package unit tests:** `make test` (must stay < ~10s). Integration:
  `make test-all` (`@pytest.mark.slow`, local Spark, minutes).
- **Cluster-driver tests** (`analysis/cloud/`, under `tests/scripts/`) need the
  framework on the path and `-m ""` to include the slow Spark cases:
  ```bash
  PYTHONPATH=spark-vi poetry run pytest tests/scripts/<file> -p no:randomly -m ""
  ```
  Some failures are pre-existing/environmental (e.g. `charmpheno.omop` import
  errors inside PySpark UDF workers) — baseline first with `git stash`; the
  obligation is no NEW failures. The tripwire suite above must stay green.

## Cluster commands must be self-contained (runnable on a fresh cluster)

Every command handed to the user to run **on the Dataproc cluster** (any `make
-C analysis/cloud …`, `spark-submit`, or other cluster-side command) MUST be
prefixed with a preamble that changes into the repo and puts the checkout on
the current development branch at the latest commit. A fresh cluster clones the
repo on `main`, so a bare `make …` runs stale code or fails with "No rule to
make target"; and a plain `git pull origin <branch>` from `main` fails on
divergent branches. The preamble below handles both the fresh-clone case (the
branch does not exist locally — `git checkout` creates a tracking branch) and
the already-checked-out case (fast-forward to origin):

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
```

Then the actual command on the next line(s). So a cluster command is always
delivered as, e.g.:

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
make -C analysis/cloud diag-episode-probe ID=110 GPR_ARGS="--gap-days 90"
```

- The development branch is currently **`claude/gated-conditional-voi`**. When
  the active branch changes, update the name in the preamble above and in this
  note.
- `git pull --ff-only` is deliberate: it fast-forwards or refuses, never
  merges or destroys local state. The cluster is a pure runner with no local
  commits, so a refusal means something unexpected — surface it, do not paper
  over it with `reset --hard`.
- This applies ONLY to commands the user runs on the cluster. Commands I run
  here in the session's own working copy do not need it.

## Chat rendering

Responses render as plain text (per `AGENTS.md`): no LaTeX — use plain Greek
(λ, γ, θ, β) and ASCII math (`E[log β]`, `sum_k`). Markdown, code fences, and
`[text](path)` links do render.
