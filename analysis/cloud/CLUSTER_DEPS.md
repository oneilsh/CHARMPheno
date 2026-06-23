# Cluster Python deps the AoU Dataproc image lacks (dependency overlay)

**Scope:** `analysis/cloud/` + `scripts/run_experiment.py`. No behavior change
for LDA/HDP runs.

## Problem

The first **STM** experiment (`stm_bigquery_cloud.py` with `--covariate-formula`)
needs `formulaic`, which the AoU-managed Dataproc image does not ship. LDA/HDP
never import it — `formulaic` is an optional extra of spark-vi
(`spark-vi[formula]`, lazily imported in `spark_vi/mllib/topic/_formula.py`), so
those models stay lean. `make setup` installs no Python packages, and the
managed cluster exposes no all-node `pip install` hook, so the cluster never
received it.

`formulaic` is needed on both the **driver** (`validate_formula` /
`fit_model_spec_from_spark` in `build_patient_covariate_df`) and the
**executors** (`apply_model_spec` runs inside `mapPartitions` in
`charmpheno/omop/covariates.py`).

## What the image already provides

The image ships the heavy compiled scientific stack — `numpy`, `pandas`,
`scipy`, `pyarrow` — and `spark-vi` itself relies on the image's `scipy`
(e.g. `spark_vi/models/topic/counting.py` imports `scipy.special`). The only
things genuinely missing are `formulaic` and its **pure-Python** leaves:
`interface-meta`, `narwhals`, `wrapt`, `typing-extensions`.

## Solution: a pure-Python `--py-files` overlay on the image's interpreter

Ship only the missing pure-Python leaves as a zip on `--py-files`, exactly the
way our own `spark_vi.zip` / `charmpheno.zip` source overlays ride. The job
runs on the **image's own Python**, which supplies `numpy/pandas/scipy/pyarrow`
natively — so there is no interpreter override, no ABI shadowing, and no
version matching against the image.

`build_spark_submit_cmd` (`scripts/run_experiment.py`) appends the overlay to
`--py-files` when it exists; absent overlay (LDA/HDP) ⇒ only the source zips
ride. The driver and every executor see the same `PYTHONPATH` overlay.

### Why not the alternatives

- **`pip install` on the cluster** — managed/ephemeral; no all-node install,
  lost on recreate, not reproducible.
- **Cluster-create init action / `dataproc:pip.packages`** — the canonical
  Dataproc answer, but the AoU Workbench exposes no creation hook here.
- **A PEX as a replacement interpreter** — a PEX run via
  `spark.pyspark.python=./x.pex` scrubs `sys.path`, so it must either bundle
  the whole scientific closure (heavy, and a bundled `numpy 2.x` shadows the
  image's `numpy-1.x`-built `pyarrow` → `_ARRAY_API not found`) or rely on
  `--inherit-path=fallback` to recover the image's libs — which does not reach
  the executor site-packages, so `import scipy` fails on executors. Both modes
  fight the image. Since the image already has a consistent scientific stack
  and the only gap is a few pure-Python modules, a `--py-files` overlay on the
  image's interpreter is simpler and avoids the interpreter swap entirely.

## How the overlay is built (Makefile `cluster-overlay`)

Built on the master so pip resolves Linux wheels for the cluster's `python3`:

1. `pip install --target dist/overlay-build -r cluster-requirements.txt` —
   resolves `formulaic`'s full closure (pulls in `numpy/pandas/scipy/pyarrow`
   too).
2. **Prune** the compiled libs the image already ships
   (`CLUSTER_IMAGE_LIBS := numpy pandas scipy pyarrow`, plus their `.libs` and
   `.dist-info`), leaving only the pure-Python leaves. This is what keeps the
   overlay from shadowing the image's compiled libs.
3. `zip` the remainder to `dist/formulaic_overlay.zip` (~1 MB).

The target is keyed on `cluster-requirements.txt` and the Makefile, so it only
rebuilds when deps (or the recipe) change. `make setup` builds it; `make exp
ID=N` rebuilds it if missing.

## How to run on the cluster

```bash
make setup            # discover workspace env + build the overlay
make exp ID=N         # zip + fit (overlay already built; rebuilt only if reqs change)
```

Tests: `python -m pytest scripts/tests -q`

## Migration note

When `spark-vi` / `charmpheno` become standalone pip packages, declare
`formulaic` as a real dependency and drop this overlay along with the
`spark_vi.zip` / `charmpheno.zip` `--py-files` overlays — in a non-air-gapped
environment the deps are simply `pip install`ed and nothing is shipped.
