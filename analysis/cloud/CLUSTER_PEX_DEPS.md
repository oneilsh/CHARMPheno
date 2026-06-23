# Cluster Python deps the AoU Dataproc image lacks (PEX delivery)

**Status:** implemented, uncommitted (working tree) as of this note.
**Scope:** `analysis/cloud/` + `scripts/run_experiment.py`. No behavior change for LDA/HDP.

## Problem

`make exp ID=3` (the first **STM** experiment — `stm_bigquery_cloud.py` with
`--covariate-formula`) failed on the cluster with:

```
ModuleNotFoundError: No module named 'formulaic'
... ImportError: Formula path requires the optional 'formula' extra: pip install spark-vi[formula]
```

This is **not** a regression. Earlier cloud runs were LDA/HDP, which never touch
the formula path. `formulaic` is an intentionally-optional extra of spark-vi
(`spark-vi[formula]`, lazily imported in `spark_vi/mllib/topic/_formula.py`), so
LDA/HDP stay lean. STM is the first experiment that imports it. The AoU-managed
Dataproc image ships numpy/pandas/scipy/pyarrow but **not** formulaic, and
`make setup` installs no Python packages — so the cluster never received it.
(Local STM tests pass only because the dev's laptop env has the extra.)

`formulaic` is needed in **two** places:
- **Driver** (master, client mode): `validate_formula` / `fit_model_spec_from_spark`
  in `build_patient_covariate_df`.
- **Executors**: `_flush() → apply_model_spec` runs inside `mapPartitions` in
  `charmpheno/omop/covariates.py`.

## Why not the obvious fixes

- **`pip install` on the cluster** — AoU clusters are managed/ephemeral; no
  one-command all-node install, lost on recreate, not reproducible.
- **Cluster-create init action / `dataproc:pip.packages`** — the canonical
  Dataproc answer, but the AoU Workbench exposes no creation hook here (confirmed
  UI-only).
- **Bundle deps into `--py-files`** — fragile: `--py-files` is for *our own*
  modules, not third-party dep closures; hand-pruning transitive deps and risking
  shadowing the image's numpy/pandas/scipy. Rejected.
- **`venv-pack` of a `--system-site-packages` venv** — would be ideal (ship only
  formulaic, reuse the image's numpy/pandas/scipy) but **venv-pack cannot pack a
  `--system-site-packages` venv** — it follows back to the base interpreter and
  errors `'…' is not a valid virtual environment`. Verified, dead end.
- **`conda-pack` a self-contained clone of base** — works but multi-GB to build
  and unpack on every executor of an ephemeral cluster. Too heavy.

## Solution: PEX with `--inherit-path=fallback`

A **PEX** carries only the deps the image lacks and falls back to the image's
numpy/pandas/scipy/pyarrow at runtime — small (~38 MB), reproducible, and driven
by a single declared requirements file. This is the lightweight member of the
Spark "Python Package Management" family and fits an ephemeral, managed cluster.

Delivery splits by role:
- **Executors** get the PEX via `spark-submit --files cluster_env.pex` (lands as
  `./cluster_env.pex` in each executor cwd) + `--conf
  spark.pyspark.python=./cluster_env.pex`. (The `spark.pyspark.python` conf is
  the standard, honored executor mechanism.)
- **Driver** (client mode, on the master, where `--files` is *not* localized
  into cwd) is steered by the **`PYSPARK_DRIVER_PYTHON` environment variable**
  pointed at the PEX's absolute path. The `spark.pyspark.driver.python` conf is
  **not reliably honored** for the client-mode driver, and a Dataproc-provided
  `PYSPARK_DRIVER_PYTHON` (plain python, no formulaic) would otherwise win — so
  we override the env var.

Our own source (`spark_vi`, `charmpheno`) keeps riding as `--py-files` zips for
fast dev iteration; only third-party deps go in the PEX.

### Verified locally
- PEX runs as an interpreter, imports `formulaic` from inside itself.
- With `--inherit-path=fallback`, a module on `PYTHONPATH` (simulating how
  spark-submit injects pyspark) still imports — i.e. the driver gets *both*
  formulaic (from PEX) and pyspark (from PYTHONPATH). This is the exact driver
  scenario.
- PEX output is 0755 (executable), required for `--files` → executor interpreter.

## Files changed

### GOTCHA — the PEX must EXCLUDE image-provided compiled libs

`--inherit-path=fallback` only falls back for packages **not in the PEX**. But
`pex -r cluster-requirements.txt` resolves formulaic's full closure, which pulls
**pandas + numpy** (and their newest versions) *into* the PEX — so they shadow
the image's copies rather than falling back. A bundled numpy 2.x under the
image's numpy-1.x-built `pyarrow` then fails at import:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.6 ...
AttributeError: _ARRAY_API not found   (pyarrow/__init__.py importing pyarrow.lib)
```

(In practice this was **non-fatal** — pandas catches the pyarrow import and runs
without Arrow — but it is wrong and noisy.) Fix: the `cluster-env` Makefile target
passes `--exclude numpy --exclude pandas --exclude scipy --exclude pyarrow`
(`CLUSTER_PEX_EXCLUDE`), so those are left to the image and the PEX carries only
formulaic + its pure-Python deps. The PEX target also depends on the Makefile so
a recipe change forces a rebuild.

### NEW — `analysis/cloud/cluster-requirements.txt`
Declared source of truth for cluster-only deps (pip resolves the closure):
```
formulaic>=1.0
```
Migration note in the file: when spark-vi/charmpheno become standalone pip
packages, add them here and drop their `--py-files` zip overlay.

### `analysis/cloud/Makefile`
- Vars: `CLUSTER_REQS`, `CLUSTER_ENV_PEX` (= `analysis/cloud/dist/cluster_env.pex`;
  `dist/` is already gitignored).
- New `cluster-env` target builds the PEX with the **cluster's own `python3`**
  (so it targets that interpreter) + `--inherit-path=fallback`:
  ```make
  cluster-env: $(CLUSTER_ENV_PEX)
  $(CLUSTER_ENV_PEX): $(CLUSTER_REQS)
  	mkdir -p $(dir $(CLUSTER_ENV_PEX))
  	python3 -m pip install --quiet --user --upgrade pex
  	python3 -m pex -r $(CLUSTER_REQS) --inherit-path=fallback -o $(CLUSTER_ENV_PEX)
  ```
- Wired `cluster-env` as a prereq of `setup` (one-time provisioning) **and** of
  `exp` / `eval-exp` / `build-dashboard-exp` / `build-covariates` (cheap — it's a
  file-target keyed on `cluster-requirements.txt`, so it only rebuilds when deps
  change). So `make setup` then `make exp ID=N` is enough on a fresh cluster.
- `setup` help text updated.

### `scripts/run_experiment.py`
- `cluster_pex_path(repo_root)` — single source for the PEX path.
- `configure_driver_interpreter(repo_root, env)` — sets
  `env["PYSPARK_DRIVER_PYTHON"] = <abs pex>` when the PEX exists (overriding any
  inherited value); no-op otherwise. Called once at the top of `main()` so every
  spark-submit child inherits it.
- `build_spark_submit_cmd(...)` — when the PEX exists, appends:
  `--files <abs pex>`, `--conf spark.pyspark.python=./cluster_env.pex`,
  `--conf spark.pyspark.driver.python=<abs pex>` (belt-and-suspenders),
  `--conf spark.executorEnv.PEX_ROOT=/tmp/pex_root` (default `~/.pex` may be
  unwritable on YARN). Absent PEX ⇒ nothing added (LDA/HDP unaffected).

### `scripts/tests/test_run_experiment.py`
- PEX present ⇒ `--files` + the three confs emitted; absent ⇒ untouched.
- `configure_driver_interpreter` overrides an existing `PYSPARK_DRIVER_PYTHON`
  when PEX present; no-op when absent.
- Full suite: 97 passing.

## How to run / test on the cluster
```bash
make setup            # discover workspace env + build cluster_env.pex
make exp ID=3         # zip + fit (PEX already built; rebuilt only if reqs change)
# or just: make exp ID=3   (its cluster-env prereq builds the PEX if missing)
```
Tests: `python -m pytest scripts/tests -q`

## Unverified cluster-only risks (watch in this order if it still fails)
1. **Executors still `ModuleNotFoundError: formulaic`** — points at `--files`
   exec-bit preservation under YARN, or PEX interpreter discovery on executors.
   (Driver is fixed by the env var; executors rely on `--files` + the
   `spark.pyspark.python` conf.)
2. **`./cluster_env.pex: permission denied`** — YARN didn't preserve the 0755
   bit on the localized file.
3. **`No interpreter compatible …`** — PEX (built with cluster `python3`) can't
   find a matching interpreter on a node; check the node python.
4. **`PEX_ROOT` not writable** — adjust the `spark.executorEnv.PEX_ROOT` path.

## Note for the agent in this repo
These changes are in the working tree, **uncommitted**. The original error logs
were initially pasted into a *different* session (an unrelated repo) — this repo
is the correct home. Nothing has been committed; review and commit as you see fit.
