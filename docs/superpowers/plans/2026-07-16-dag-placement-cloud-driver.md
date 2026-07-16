# DAG-Placement Cloud Driver + run_experiment Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the gated-SVI hierarchical case-finding engine a first-class tracked experiment: a thin cloud driver (`dag_placement_cloud.py`) that assembles the diabetes corpus, fits the gated shim, scores held-out placement, and saves an artifact; the `run_experiment.py` wiring so `make exp ID=N` drives it; a write-through bundle cache; and the `init` shim change that makes `random` vs `spectral` a config flip for the pre-registered A/B.

**Architecture:** Five pieces — (A) expose `init` on `GatedLDAEstimator` + a dense-spectral `data_summary` path mirroring the STM shim; (B) a write-through `CaseFindingBundle` cache in the driver layer; (C) `dag_placement_cloud.py` (assemble→fit→transform→inline `evaluate`→save, pg_stm-style artifact); (D) the four `run_experiment` chains + `build_dag_placement_args`; (E) config (`_base.yaml` block + a diabetes cohort YAML + two A/B experiment files).

**Tech Stack:** Python 3.12, PySpark, numpy; `spark_vi` (GatedLDAEstimator/Model, DagLayout, evaluate, gated_init); the piece-2 `charmpheno.omop.case_finding_assembly`. Tests: pytest with the repos' session-scoped local Spark fixtures.

## Global Constraints

- **Branch:** `case-finding`. Do NOT merge to `main` (experimental). Verify + push to `origin/case-finding` after committing (this branch does not auto-push).
- **Commit trailer, EXACT** (blank line before it):
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **No LaTeX** anywhere. Plain text + Unicode Greek (α, β, η, λ, Σ).
- **Cite literature** for any method/default/constant from a paper, in the docstring.
- **Hash IDs in row-level log output:** any `.show()`/`print()` of doc/patient-level rows SHA-256-truncates id columns first. Metrics, counts, ledger fields print raw.
- **Domain vs engine separation:** concept-ids/diabetes/the three id spaces live in the driver + config + `case_finding_assembly` (piece 2). The engine (`spark_vi`) stays integer-id agnostic — the shim change in Task 1 must NOT introduce concept-id knowledge.
- **`nBg`/`tpn` single-sourced:** the driver passes the SAME `n_bg`/`tpn` to the assembly, the estimator, AND the scoring `DagLayout`. A mismatch makes the gate wrong. Asserted in Task 3.
- **Test honesty:** never loosen a threshold to pass. `xfail` with a reason if an assertion cannot hold; do not weaken it.
- **Resume is NOT supported** (GatedLDAModel is not persistable in v1); the driver always fits fresh and `build_dag_placement_args` ignores `resume_from`.

## Reference paths (read-only, for mirroring)

- Cache template: `analysis/cloud/_corpus_cache.py`; corpus-load wrapper: `analysis/cloud/_corpus_load.py`.
- Driver skeleton + save: `analysis/cloud/pg_stm_bigquery_cloud.py`; driver helpers: `analysis/cloud/_driver_common.py` (`_phase`, `configure_logging`, `make_spark_session(app_name)`).
- run_experiment: `scripts/run_experiment.py` (`validate_frontmatter`, `build_fit_driver_path`, `build_fit_args`, `build_pg_stm_args`, `_require_workspace_env`, eval-dispatch in `main`).
- Shim: `spark-vi/spark_vi/mllib/topic/gated_lda.py`; engine init: `spark-vi/spark_vi/models/topic/gated_lda.py` (`initialize_global`), `gated_init.INIT_STRATEGIES`; runner: `spark-vi/spark_vi/core/runner.py` (`VIRunner.fit(data_rdd, data_summary=None, ...)`); BOW helper: `spark-vi/spark_vi/mllib/topic/_common.py` (`_vector_to_bow_document`).
- Assembly (piece 2): `charmpheno/charmpheno/omop/case_finding_assembly.py` (`assemble_case_finding_corpus`, `CaseFindingBundle`).
- STM dense-spectral precedent: `spark-vi/spark_vi/mllib/topic/stm.py:1524-1541`.

---

### Task 1: Expose `init` on GatedLDAEstimator + dense-spectral data_summary path

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/gated_lda.py` (`_GatedLDAParams`, `GatedLDAEstimator.__init__`, `_fit`)
- Test: `spark-vi/tests/test_gated_lda_shim.py` (extend)

**Interfaces:**
- Consumes: `gated_init.INIT_STRATEGIES` (currently `{"spectral": spectral_block_aligned_lambda}`); `GatedOnlineLDA(lay, V, init=…)`; `VIRunner.fit(rdd, data_summary=…)`; `_vector_to_bow_document`.
- Produces: `GatedLDAEstimator(..., init="random"|"spectral", spectralMaxVocab=8000)`. When `init="spectral"`, `_fit` collects docs into `data_summary={"train_docs":[token-id arrays], "train_labels":[frontier frozensets]}` and passes it to `VIRunner.fit`; a dense-vocab guard raises `NotImplementedError` at large V.

Run tests: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda_shim.py -v`

- [ ] **Step 1: Write the failing tests**

Append to `spark-vi/tests/test_gated_lda_shim.py`:

```python
def test_gated_shim_init_param_defaults_random():
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    est = GatedLDAEstimator(parent={1: 0, 2: 0})
    assert est.getOrDefault("init") == "random"


def test_gated_shim_unknown_init_raises(spark):
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from pyspark.ml.linalg import SparseVector
    import pytest
    df = spark.createDataFrame(
        [(SparseVector(5, [0, 1], [1.0, 1.0]), [1])],
        ["features", "frontier"],
    )
    est = GatedLDAEstimator(parent={1: 0, 2: 0}, init="banana", maxIter=1)
    with pytest.raises(ValueError, match="init"):
        est.fit(df)


def test_gated_shim_spectral_vocab_guard_raises(spark):
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from pyspark.ml.linalg import SparseVector
    import pytest
    # V = 6 features but spectralMaxVocab = 4 -> dense V x V guard trips.
    df = spark.createDataFrame(
        [(SparseVector(6, [0, 1], [1.0, 1.0]), [1])],
        ["features", "frontier"],
    )
    est = GatedLDAEstimator(parent={1: 0, 2: 0}, init="spectral",
                            spectralMaxVocab=4, maxIter=1)
    with pytest.raises(NotImplementedError, match="scalable"):
        est.fit(df)


def test_gated_shim_spectral_fits_and_seeds_lambda(spark):
    """init='spectral' collects docs, runs the block-aligned spectral seed via
    data_summary, and fits. The resulting lambda must differ from a random-init
    fit on the same corpus (the spectral seed took effect)."""
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from pyspark.ml.linalg import SparseVector
    import numpy as np
    # Two nodes under root; docs attest node 1 (tokens 0,1) or node 2 (tokens 2,3).
    rows = []
    for _ in range(20):
        rows.append((SparseVector(6, [0, 1], [3.0, 2.0]), [1]))
        rows.append((SparseVector(6, [2, 3], [3.0, 2.0]), [2]))
    df = spark.createDataFrame(rows, ["features", "frontier"])
    parent = {1: 0, 2: 0}
    m_rand = GatedLDAEstimator(parent=parent, nBg=2, tpn=1, init="random",
                               maxIter=2, seed=0).fit(df)
    m_spec = GatedLDAEstimator(parent=parent, nBg=2, tpn=1, init="spectral",
                               spectralMaxVocab=1000, maxIter=2, seed=0).fit(df)
    lam_r = m_rand.result.global_params["lambda"]
    lam_s = m_spec.result.global_params["lambda"]
    assert lam_r.shape == lam_s.shape
    assert not np.allclose(lam_r, lam_s)   # spectral seed changed the trajectory
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda_shim.py -k "init_param or unknown_init or vocab_guard or spectral_fits" -v`
Expected: FAIL (no `init` param; `getOrDefault("init")` raises or default missing).

- [ ] **Step 3: Add the `init` + `spectralMaxVocab` Params**

In `_GatedLDAParams` (after `gammaShape`), add:

```python
    init = Param(Params._dummy(), "init",
                 "lambda init strategy: 'random' (default) or 'spectral' "
                 "(dense block-aligned anchor-word seed, Arora et al. 2013)",
                 typeConverter=TypeConverters.toString)
    spectralMaxVocab = Param(Params._dummy(), "spectralMaxVocab",
                             "max vocab for the DENSE spectral init (V x V driver "
                             "co-occurrence); at/above this, spectral raises "
                             "(scalable projected variant deferred)",
                             typeConverter=TypeConverters.toInt)
```

- [ ] **Step 4: Thread `init`/`spectralMaxVocab` through `__init__`**

Replace `GatedLDAEstimator.__init__` (keep the existing body, add the two kwargs + defaults):

```python
    @keyword_only
    def __init__(self, *, featuresCol="features", labelCol="frontier", parent=None,
                 nBg=2, tpn=1, maxIter=20, seed=None, caviMaxIter=100, caviTol=1e-3,
                 gammaShape=100.0, init="random", spectralMaxVocab=8000):
        super().__init__()
        self._setDefault(featuresCol="features", labelCol="frontier", nBg=2, tpn=1,
                         maxIter=20, nodeAffinityCol="nodeAffinity",
                         caviMaxIter=100, caviTol=1e-3, gammaShape=100.0,
                         init="random", spectralMaxVocab=8000)
        self.setParams(**self._input_kwargs)
```

- [ ] **Step 5: Rewrite `_fit` to honor `init` + build the spectral data_summary**

Replace the `model_obj = GatedOnlineLDA(...)` construction and the `VIRunner(...).fit(rdd)` call. The new `_fit` body (from the `V = first[0][0].size` line onward, keeping the rest):

```python
        V = first[0][0].size
        seed = self.getOrDefault("seed") if self.isSet("seed") else None
        init = self.getOrDefault("init")

        # Validate init early (fail fast on the driver, not deep in a Spark task).
        from spark_vi.models.topic.gated_init import INIT_STRATEGIES
        if init != "random" and init not in INIT_STRATEGIES:
            raise ValueError(
                f"unknown init strategy {init!r}; "
                f"known: {['random'] + sorted(INIT_STRATEGIES)}"
            )
        # Dense spectral init builds a driver-side V x V co-occurrence; guard it.
        # The scalable projected block-aligned variant (distributed co-occurrence +
        # random projection, the gated analogue of STM ADR 0032) is deferred.
        if init == "spectral" and V >= self.getOrDefault("spectralMaxVocab"):
            raise NotImplementedError(
                f"dense spectral init needs a {V}x{V} driver co-occurrence "
                f"(vocab {V} >= spectralMaxVocab {self.getOrDefault('spectralMaxVocab')}); "
                "the scalable projected block-aligned init is not built yet. "
                "Use init='random' or reduce vocab_size."
            )

        model_obj = GatedOnlineLDA(
            lay, V, init=init,
            alpha=1.0 / lay.K, eta=1.0 / lay.K,
            gamma_shape=self.getOrDefault("gammaShape"),
            cavi_max_iter=self.getOrDefault("caviMaxIter"),
            cavi_tol=self.getOrDefault("caviTol"),
            random_seed=seed,
        )
        config = VIConfig(max_iterations=self.getOrDefault("maxIter"), random_seed=seed)

        def _to_gated(row):
            bow = _vector_to_bow_document(row[0])
            frontier = frozenset(int(x) for x in (row[1] or []))
            return GatedBOWDocument(indices=bow.indices, counts=bow.counts,
                                    length=bow.length, frontier=frontier)

        rdd = (dataset.select(features_col, label_col).rdd.map(_to_gated)
               .persist(StorageLevel.MEMORY_AND_DISK))
        rdd.count()

        # Non-random init needs the training corpus in the driver (the dense,
        # collect-to-driver path, mirroring the STM shim's dense spectral seed,
        # mllib/topic/stm.py). The block-aligned strategy expects token-id arrays
        # + per-doc frontier labels via data_summary; initialize_global runs it.
        data_summary = None
        if init != "random":
            collected = dataset.select(features_col, label_col).collect()
            train_docs, train_labels = [], []
            for r in collected:
                bow = _vector_to_bow_document(r[0])
                train_docs.append(np.repeat(bow.indices, bow.counts.astype(int)))
                train_labels.append(frozenset(int(x) for x in (r[1] or [])))
            data_summary = {"train_docs": train_docs, "train_labels": train_labels}

        try:
            result = VIRunner(model_obj, config=config).fit(
                rdd, data_summary=data_summary)
        finally:
            rdd.unpersist(blocking=False)
```

(The trailing `out = GatedLDAModel(...)` + param-copy loop is unchanged; the loop already propagates `init`/`spectralMaxVocab` to the model since they are params.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda_shim.py -k "init_param or unknown_init or vocab_guard or spectral_fits" -v`
Expected: PASS (4 tests). Then run the whole shim file to confirm no regression: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda_shim.py -v` (the slow placement-equivalence gate is NOT in this file; if any test here is slow, that's expected).

- [ ] **Step 7: Commit**

```bash
git add spark-vi/spark_vi/mllib/topic/gated_lda.py spark-vi/tests/test_gated_lda_shim.py
git commit -m "feat(gated-shim): expose init param + dense-spectral data_summary path

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: CaseFindingBundle write-through cache + load_or_build wrapper

**Files:**
- Create: `analysis/cloud/_case_finding_cache.py`
- Test: `analysis/cloud/tests/test_case_finding_cache.py` (create; add `conftest.py` there only if `analysis/cloud/tests/` has no Spark fixture — check first, reuse if present)

**Interfaces:**
- Consumes: `charmpheno.omop.case_finding_assembly.CaseFindingBundle` / `assemble_case_finding_corpus`; `charmpheno.omop.cohorts.cohort_defs_version`; the piece-2 modules' source (for content-hash invalidation).
- Produces:
  - `compute_bundle_cache_key(**params) -> str` (16-hex).
  - `save(spark, bundle, cache_uri, key) -> None` — parquet train/test + a text-serialized `meta.json` of the python fields.
  - `try_load(spark, cache_uri, key) -> CaseFindingBundle | None`.
  - `load_or_build_case_finding_bundle(spark, *, cache_uri, _assemble_fn=None, **assembly_params) -> CaseFindingBundle`.

Run tests (check the analysis/cloud test path first): `.venv/bin/python -m pytest analysis/cloud/tests/test_case_finding_cache.py -v`

- [ ] **Step 1: Create the analysis/cloud/tests conftest (sys.path + spark fixture)**

`analysis/cloud/tests/` has NO `conftest.py` and none of its tests use Spark. `charmpheno`/`spark_vi` are editable-installed (import anywhere), so the only path need is importing the driver basenames (`_case_finding_cache`, `dag_placement_cloud`). Create `analysis/cloud/tests/conftest.py` with (a) the `sys.path` insert for the `analysis/cloud` dir (mirroring the per-file pattern in `test_covariate_cache_key.py`), and (b) the session-scoped `spark` fixture copied VERBATIM from `charmpheno/tests/conftest.py` (same config, including `-Djava.security.manager=allow`):

```python
"""Shared fixtures for analysis/cloud driver tests: import path + local Spark."""
import os
import sys
import warnings
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

_CLOUD = str(Path(__file__).resolve().parent.parent)   # analysis/cloud
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)


@pytest.fixture(scope="session")
def spark():
    warnings.filterwarnings("ignore")
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    session = (
        SparkSession.builder.master("local[2]")
        .appName("cloud-tests")
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

(This conftest also serves Task 3's `test_dag_placement_cloud.py`, which imports `dag_placement_cloud` by basename.)

- [ ] **Step 2: Write the failing tests**

Create `analysis/cloud/tests/test_case_finding_cache.py`. It builds a real `CaseFindingBundle` from tiny synthetic frames via the piece-2 `assemble_from_events` (no BQ), then exercises the cache:

```python
"""Tests for the CaseFindingBundle write-through cache (piece 3)."""
import datetime as dt


def _tiny_bundle(spark):
    from charmpheno.omop.case_finding_assembly import assemble_from_events
    from charmpheno.omop.condition_dag import build_condition_dag
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    edges = [(100, 200), (100, 300), (200, 400)]
    before = build_condition_dag(edges, anchor=100, node_ids=[200, 300, 400],
                                 names={100: "dm", 200: "T2", 300: "T1", 400: "T2r"})
    rows = []
    for pid in range(20):
        rows.append((pid, 200, "diabetes", dt.date(2015, 1, 1)))
        rows.append((pid, 999, "diabetes", dt.date(2015, 2, 1)))
    for pid in range(100, 120):
        rows.append((pid, 888, "general", dt.date(2016, 1, 1)))
        rows.append((pid, 777, "general", dt.date(2016, 2, 1)))
    ev = spark.createDataFrame(
        rows, ["person_id", "concept_id", "source_cohort",
               "condition_era_start_date"])
    return assemble_from_events(
        ev, before, doc_spec=PatientCohortDocSpec(min_doc_length=0),
        min_n=1, holdout_frac=0.3, split_salt=20260716,
        vocab_size=50, min_df=1, min_patient_count=1, n_bg=2, tpn=1)


def test_bundle_cache_key_sensitive_and_stable():
    from _case_finding_cache import compute_bundle_cache_key
    base = dict(source_table="condition_era", person_mod=10, vocab_size=5000,
                min_df=20, min_patient_count=20, doc_min_length=0,
                prior_obs_days=365, window_days=365, anchor=201820, min_n=50,
                holdout_frac=0.2, split_salt=20260716, n_bg=2, tpn=1, cdr="p.d")
    k0 = compute_bundle_cache_key(**base)
    assert k0 == compute_bundle_cache_key(**base)              # stable
    for field, val in [("anchor", 442793), ("min_n", 25), ("holdout_frac", 0.3),
                       ("person_mod", 20), ("vocab_size", 3000), ("n_bg", 3),
                       ("tpn", 2)]:
        assert compute_bundle_cache_key(**{**base, field: val}) != k0


def test_bundle_cache_save_load_round_trip(spark, tmp_path):
    from _case_finding_cache import save, try_load
    bundle = _tiny_bundle(spark)
    uri = f"file://{tmp_path}/cache"
    save(spark, bundle, uri, "k1")
    loaded = try_load(spark, uri, "k1")
    assert loaded is not None
    # python fields restored with int keys
    assert loaded.parent_int == bundle.parent_int
    assert loaded.int2cid == bundle.int2cid
    assert loaded.cid2int == bundle.cid2int
    assert loaded.vocab_map == bundle.vocab_map
    assert loaded.name_by_id == bundle.name_by_id
    assert loaded.ledger["K_nodes"] == bundle.ledger["K_nodes"]
    # DataFrame contents preserved (compare as sets of person_ids per split)
    assert ({r["person_id"] for r in loaded.train_df.collect()}
            == {r["person_id"] for r in bundle.train_df.collect()})
    assert ({r["person_id"] for r in loaded.test_df.collect()}
            == {r["person_id"] for r in bundle.test_df.collect()})


def test_bundle_cache_miss_then_hit(spark, tmp_path):
    from _case_finding_cache import load_or_build_case_finding_bundle
    built = _tiny_bundle(spark)
    calls = {"n": 0}

    def _stub_assemble(spark_, **kw):
        calls["n"] += 1
        return built

    uri = f"file://{tmp_path}/cache2"
    params = dict(source_table="condition_era", person_mod=10, vocab_size=5000,
                  min_df=20, min_patient_count=20, doc_min_length=0,
                  prior_obs_days=365, window_days=365, anchor=201820, min_n=50,
                  holdout_frac=0.2, split_salt=20260716, n_bg=2, tpn=1, cdr="p.d",
                  billing="bp")
    b1 = load_or_build_case_finding_bundle(
        spark, cache_uri=uri, _assemble_fn=_stub_assemble, **params)
    b2 = load_or_build_case_finding_bundle(
        spark, cache_uri=uri, _assemble_fn=_stub_assemble, **params)
    assert calls["n"] == 1                       # built once, second call is a HIT
    assert b1.parent_int == b2.parent_int == built.parent_int
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest analysis/cloud/tests/test_case_finding_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named '_case_finding_cache'`. (If import-path issues arise, note that the analysis/cloud drivers are imported by basename — the existing `_corpus_cache` tests show the pattern; mirror their `sys.path`/conftest setup.)

- [ ] **Step 4: Implement `_case_finding_cache.py`**

Create `analysis/cloud/_case_finding_cache.py`:

```python
"""Write-through cache for the CaseFindingBundle (piece-3 driver).

assemble_case_finding_corpus re-extracts from BigQuery, refits CountVectorizer,
and rebuilds/prunes the DAG on every call. This module caches the resulting
bundle under a content-hash key so repeat runs (same assembly inputs) reload
parquet + a small JSON instead of rebuilding. Mirrors analysis/cloud/
_corpus_cache.py, but the bundle is richer than (bow, vocab, names): it also
carries the DAG maps + ledger, stored as a text-serialized meta.json.

The domain module (charmpheno.omop.case_finding_assembly) stays cache-free; the
driver layer owns caching, exactly as _corpus_load wraps to_bow_dataframe.

Cache layout under {cache_uri}/{key}/:
    train.parquet/   test.parquet/    the split DataFrames
    meta/            a one-column text file holding json.dumps(python fields)
"""
from __future__ import annotations

import hashlib
import inspect
import json
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from charmpheno.omop.case_finding_assembly import CaseFindingBundle


def _module_source_hash(module) -> str:
    try:
        return hashlib.sha256(inspect.getsource(module).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return "src-unavailable"


def compute_bundle_cache_key(*, source_table, person_mod, vocab_size, min_df,
                             min_patient_count, doc_min_length, prior_obs_days,
                             window_days, anchor, min_n, holdout_frac, split_salt,
                             n_bg, tpn, cdr=None) -> str:
    """Stable 16-hex hash of the inputs that determine the assembled bundle.

    Folds cohort_defs_version() plus content hashes of condition_dag +
    case_finding_assembly, so any assembly-logic edit auto-invalidates the cache
    (same discipline as _corpus_cache's cohort_defs). `v` is the manual shape
    version for layout changes unrelated to that source.
    """
    from charmpheno.omop import condition_dag, case_finding_assembly
    from charmpheno.omop.cohorts import cohort_defs_version
    payload = {
        "source_table": source_table, "person_mod": int(person_mod),
        "vocab_size": vocab_size, "min_df": float(min_df),
        "min_patient_count": int(min_patient_count),
        "doc_min_length": int(doc_min_length), "prior_obs_days": int(prior_obs_days),
        "window_days": int(window_days), "anchor": int(anchor), "min_n": int(min_n),
        "holdout_frac": float(holdout_frac), "split_salt": int(split_salt),
        "n_bg": int(n_bg), "tpn": int(tpn), "cdr": cdr,
        "cohort_defs": cohort_defs_version(),
        "dag_src": _module_source_hash(condition_dag),
        "assembly_src": _module_source_hash(case_finding_assembly),
        "v": 1,
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _meta_dict(bundle) -> dict:
    # int keys are JSON-stringified; restored on load.
    return {
        "parent_int": {str(c): list(ps) for c, ps in bundle.parent_int.items()},
        "int2cid": {str(i): c for i, c in bundle.int2cid.items()},
        "cid2int": {str(c): i for c, i in bundle.cid2int.items()},
        "vocab_map": {str(c): i for c, i in bundle.vocab_map.items()},
        "name_by_id": {str(c): n for c, n in bundle.name_by_id.items()},
        "ledger": bundle.ledger,
    }


def _restore_meta(meta: dict) -> dict:
    return {
        "parent_int": {int(c): [int(p) for p in ps]
                       for c, ps in meta["parent_int"].items()},
        "int2cid": {int(i): int(c) for i, c in meta["int2cid"].items()},
        "cid2int": {int(c): int(i) for c, i in meta["cid2int"].items()},
        "vocab_map": {int(c): int(i) for c, i in meta["vocab_map"].items()},
        "name_by_id": {int(c): n for c, n in meta["name_by_id"].items()},
        "ledger": meta["ledger"],
    }


def save(spark, bundle, cache_uri, key) -> None:
    """Persist a CaseFindingBundle under {cache_uri}/{key}/ (overwrite mode)."""
    base = f"{cache_uri.rstrip('/')}/{key}"
    bundle.train_df.write.mode("overwrite").parquet(f"{base}/train.parquet")
    bundle.test_df.write.mode("overwrite").parquet(f"{base}/test.parquet")
    meta_json = json.dumps(_meta_dict(bundle))
    (spark.createDataFrame([(meta_json,)], "value STRING")
         .coalesce(1).write.mode("overwrite").text(f"{base}/meta"))


def try_load(spark, cache_uri, key) -> Optional["CaseFindingBundle"]:
    """Return the cached CaseFindingBundle on hit, None on any miss/read failure."""
    from charmpheno.omop.case_finding_assembly import CaseFindingBundle
    base = f"{cache_uri.rstrip('/')}/{key}"
    try:
        train_df = spark.read.parquet(f"{base}/train.parquet")
        test_df = spark.read.parquet(f"{base}/test.parquet")
        meta_rows = spark.read.text(f"{base}/meta").collect()
    except Exception:
        return None
    meta = _restore_meta(json.loads(meta_rows[0]["value"]))
    return CaseFindingBundle(
        train_df=train_df, test_df=test_df, parent_int=meta["parent_int"],
        int2cid=meta["int2cid"], cid2int=meta["cid2int"],
        vocab_map=meta["vocab_map"], name_by_id=meta["name_by_id"],
        ledger=meta["ledger"])


def load_or_build_case_finding_bundle(spark, *, cache_uri=None, _assemble_fn=None,
                                      **assembly_params) -> "CaseFindingBundle":
    """Return the cached bundle on hit; otherwise assemble + write through.

    `assembly_params` are the assemble_case_finding_corpus kwargs (cdr, billing,
    anchor, person_mod, min_n, n_bg, tpn, vocab_size, ...). `_assemble_fn` is a
    seam for testing (defaults to assemble_case_finding_corpus).
    """
    from charmpheno.omop.case_finding_assembly import assemble_case_finding_corpus
    assemble = _assemble_fn or assemble_case_finding_corpus

    key = None
    if cache_uri:
        key_params = {k: assembly_params[k] for k in (
            "source_table", "person_mod", "vocab_size", "min_df",
            "min_patient_count", "doc_min_length", "prior_obs_days", "window_days",
            "anchor", "min_n", "holdout_frac", "split_salt", "n_bg", "tpn", "cdr",
        ) if k in assembly_params}
        key = compute_bundle_cache_key(**key_params)
        cached = try_load(spark, cache_uri, key)
        if cached is not None:
            print("[driver]   case-finding-cache HIT", flush=True)
            return cached
        print("[driver]   case-finding-cache MISS, building...", flush=True)

    bundle = assemble(spark, **assembly_params)
    if cache_uri:
        save(spark, bundle, cache_uri, key)
    return bundle
```

Note: `assemble_case_finding_corpus` takes `split_salt` with a default; the wrapper's `key_params` reads `split_salt` from `assembly_params` only if present — pass it explicitly from the driver so the key is complete, or fold the default in `compute_bundle_cache_key` call. The driver (Task 3) passes all key fields explicitly.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest analysis/cloud/tests/test_case_finding_cache.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add analysis/cloud/_case_finding_cache.py analysis/cloud/tests/
git commit -m "feat(dag-placement): CaseFindingBundle write-through cache + load_or_build

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: dag_placement_cloud.py driver (adapter + arg parse + main)

**Files:**
- Create: `analysis/cloud/dag_placement_cloud.py`
- Test: `analysis/cloud/tests/test_dag_placement_cloud.py`

**Interfaces:**
- Consumes: `load_or_build_case_finding_bundle` (Task 2); `GatedLDAEstimator` (Task 1); `spark_vi.models.topic.dag_placement.{DagLayout, evaluate, render_profile}`; `_driver_common.{_phase, configure_logging, make_spark_session}`.
- Produces:
  - `profiles_from_scored_rows(rows, lay) -> (profiles: list[dict], test_labels: list[set])` — pure adapter (nodeAffinity DenseVector → `dict(zip(lay.nodes, ...))`, frontier array → set).
  - `parse_args(argv=None)` and `main() -> int`.

Run tests: `.venv/bin/python -m pytest analysis/cloud/tests/test_dag_placement_cloud.py -v`

- [ ] **Step 1: Write the failing tests**

Create `analysis/cloud/tests/test_dag_placement_cloud.py`:

```python
"""Tests for the dag_placement cloud driver (piece 3): the pure scoring adapter
and the arg surface. The end-to-end BQ+fit run is the cluster smoke."""


def test_profiles_from_scored_rows_maps_affinity_and_frontier():
    from pyspark.ml.linalg import DenseVector
    from spark_vi.models.topic.dag_placement import DagLayout
    from dag_placement_cloud import profiles_from_scored_rows
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=2, tpn=1)   # nodes = [1,2,3]
    # a "row" needs __getitem__ by name; use dicts (the driver indexes by name).
    rows = [
        {"nodeAffinity": DenseVector([0.5, 0.3, 0.2]), "frontier": [3]},
        {"nodeAffinity": DenseVector([0.1, 0.8, 0.1]), "frontier": [2, 1]},
    ]
    profiles, labels = profiles_from_scored_rows(rows, lay)
    assert profiles[0] == {1: 0.5, 2: 0.3, 3: 0.2}
    assert labels[0] == {3}
    assert labels[1] == {1, 2}
    # profiles feed evaluate cleanly
    from spark_vi.models.topic.dag_placement import evaluate
    ev = evaluate(profiles, labels, lay)
    assert "auc_by_depth" in ev and "mrr" in ev


def test_parse_args_surface():
    from dag_placement_cloud import parse_args
    a = parse_args([
        "--cdr", "p.d", "--billing", "bp", "--anchor", "201820",
        "--min-n", "50", "--n-bg", "2", "--tpn", "1", "--person-mod", "10",
        "--vocab-size", "5000", "--init", "spectral", "--out-dir", "/tmp/x",
    ])
    assert a.anchor == 201820 and a.min_n == 50 and a.n_bg == 2 and a.tpn == 1
    assert a.init == "spectral" and a.out_dir == "/tmp/x"
    # K is emergent: there must be NO --K arg.
    assert not hasattr(a, "K")


def test_main_importable():
    import dag_placement_cloud
    assert hasattr(dag_placement_cloud, "main")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest analysis/cloud/tests/test_dag_placement_cloud.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dag_placement_cloud'`.

- [ ] **Step 3: Implement `dag_placement_cloud.py`**

Create `analysis/cloud/dag_placement_cloud.py`:

```python
"""Cloud fit+eval driver for the gated-SVI hierarchical case-finding engine.

Assembles the diabetes case-finding corpus (piece-2 assemble_case_finding_corpus,
cached), fits the gated MLlib shim (GatedLDAEstimator), scores held-out placement
inline (dag_placement.evaluate), and saves an npz + manifest.json artifact (the
pg_stm methods-experiment pattern; the NPMI coherence eval cannot score a
placement model). K is EMERGENT (n_bg + surviving-DAG-nodes * tpn), so there is
no --K. Resume is unsupported (GatedLDAModel is not persistable in v1).

The init flag (random | spectral) is the pre-registered A/B: spectral uses the
dense block-aligned anchor-word seed (Arora et al. 2013) collected to the driver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _driver_common import _phase, configure_logging, make_spark_session


def profiles_from_scored_rows(rows, lay):
    """Adapt transform() output rows to dag_placement.evaluate inputs.

    Each row's `nodeAffinity` is a DenseVector ordered by lay.nodes; the profile
    is dict(zip(lay.nodes, affinity)). `frontier` (engine-ids) becomes the truth
    set. Pure; the driver collects the test set (held-out scale) before calling."""
    profiles, test_labels = [], []
    for r in rows:
        aff = r["nodeAffinity"].toArray()
        profiles.append({u: float(aff[i]) for i, u in enumerate(lay.nodes)})
        test_labels.append({int(x) for x in r["frontier"]})
    return profiles, test_labels


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Gated-SVI hierarchical case-finding fit + inline placement eval.")
    p.add_argument("--cdr", required=True)
    p.add_argument("--billing", required=True)
    p.add_argument("--source-table", default="condition_era")
    p.add_argument("--person-mod", type=int, default=10)
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument("--min-df", type=int, default=20)
    p.add_argument("--min-patient-count", type=int, default=20)
    p.add_argument("--doc-min-length", type=int, default=0)
    p.add_argument("--prior-obs-days", type=int, default=365)
    p.add_argument("--window-days", type=int, default=365)
    # assembly / DAG
    p.add_argument("--anchor", type=int, default=201820)
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    # gating
    p.add_argument("--n-bg", type=int, default=2)
    p.add_argument("--tpn", type=int, default=1)
    # SVI
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cavi-max-iter", type=int, default=100)
    p.add_argument("--cavi-tol", type=float, default=1e-3)
    p.add_argument("--init", choices=["random", "spectral"], default="random")
    p.add_argument("--spectral-max-vocab", type=int, default=8000)
    p.add_argument("--cache-uri", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume-from", default="",
                   help="Unused (GatedLDAModel is not persistable in v1); "
                        "accepted for run_experiment parity.")
    return p.parse_args(argv)


def main() -> int:
    from _case_finding_cache import load_or_build_case_finding_bundle
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from spark_vi.models.topic.dag_placement import DagLayout, evaluate, render_profile

    args = parse_args()
    configure_logging()
    with make_spark_session(app_name="dag-placement-fit") as spark:
        with _phase("assemble corpus (cached)"):
            bundle = load_or_build_case_finding_bundle(
                spark, cache_uri=args.cache_uri,
                cdr=args.cdr, billing=args.billing, source_table=args.source_table,
                person_mod=args.person_mod, vocab_size=args.vocab_size,
                min_df=args.min_df, min_patient_count=args.min_patient_count,
                doc_min_length=args.doc_min_length, prior_obs_days=args.prior_obs_days,
                window_days=args.window_days, anchor=args.anchor, min_n=args.min_n,
                holdout_frac=args.holdout_frac, n_bg=args.n_bg, tpn=args.tpn)
            print(f"[driver]   ledger: {json.dumps(bundle.ledger)}", flush=True)

        lay = DagLayout(bundle.parent_int, n_bg=args.n_bg, tpn=args.tpn)

        with _phase(f"gated-svi fit (init={args.init}, K={lay.K})"):
            est = GatedLDAEstimator(
                featuresCol="features", labelCol="frontier",
                parent=bundle.parent_int, nBg=args.n_bg, tpn=args.tpn,
                maxIter=args.max_iter, seed=args.seed,
                caviMaxIter=args.cavi_max_iter, caviTol=args.cavi_tol,
                init=args.init, spectralMaxVocab=args.spectral_max_vocab)
            model = est.fit(bundle.train_df)

        with _phase("transform + inline placement eval"):
            scored = model.transform(bundle.test_df).select("nodeAffinity", "frontier")
            rows = scored.collect()
            profiles, test_labels = profiles_from_scored_rows(rows, lay)
            metrics = evaluate(profiles, test_labels, lay)
            print(f"[driver]   placement metrics: "
                  f"auc_by_depth={metrics['auc_by_depth']} mrr={metrics['mrr']:.3f} "
                  f"top2={metrics['top2']:.3f} mean_hops={metrics['mean_hops']:.2f} "
                  f"frontier_size_mean={metrics['frontier_size_mean']:.2f} "
                  f"multi_frontier_rate={metrics['multi_frontier_rate']:.3f}",
                  flush=True)
            # Spot-check render for a few foreground held-out docs. names must be
            # ENGINE-id-keyed (remap concept-id name_by_id via int2cid).
            names = {i: bundle.name_by_id[c] for i, c in bundle.int2cid.items()
                     if c in bundle.name_by_id}
            for pr, lab in list(zip(profiles, test_labels))[:5]:
                if lab:
                    print(render_profile(pr, lay, names=names, true_node=lab),
                          flush=True)

        with _phase("save"):
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            gp = model.result.global_params
            np.savez(out / "dag_placement_result.npz",
                     **{"lambda": gp["lambda"], "alpha": gp["alpha"]})
            manifest = {
                "model_class": "dag_placement",
                "init": args.init, "K": lay.K, "n_bg": args.n_bg, "tpn": args.tpn,
                "anchor": args.anchor, "min_n": args.min_n,
                "max_iter": args.max_iter, "metrics": metrics, "ledger": bundle.ledger,
                "corpus_manifest": {
                    "cdr": args.cdr, "source_table": args.source_table,
                    "person_mod": args.person_mod, "vocab_size": args.vocab_size,
                    "min_df": args.min_df, "min_patient_count": args.min_patient_count,
                    "prior_obs_days": args.prior_obs_days, "window_days": args.window_days,
                    "holdout_frac": args.holdout_frac,
                    "int2cid": {str(i): c for i, c in bundle.int2cid.items()},
                    "name_by_id": {str(c): n for c, n in bundle.name_by_id.items()}},
            }
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"[driver]   saved dag_placement result to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest analysis/cloud/tests/test_dag_placement_cloud.py -v`
Expected: PASS (3 tests). `evaluate` on the tiny synthetic profiles may emit `nan` for some depths (single doc) — the adapter test only asserts the keys exist + labels map, so this is fine.

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/dag_placement_cloud.py analysis/cloud/tests/test_dag_placement_cloud.py
git commit -m "feat(dag-placement): cloud fit+eval driver (assemble->fit->evaluate->save)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: run_experiment wiring (four chains + build_dag_placement_args)

**Files:**
- Modify: `scripts/run_experiment.py` (`validate_frontmatter`, `build_fit_driver_path`, `build_fit_args`, the eval-dispatch elif in `main`, + new `build_dag_placement_args`)
- Test: `scripts/tests/test_run_experiment_dag_placement.py` (create; check `scripts/tests/` conftest for how `run_experiment` is imported)

**Interfaces:**
- Consumes: `_require_workspace_env() -> (cdr, billing)`; the `effective` merged-config dict.
- Produces: `build_dag_placement_args(effective, out_dir, resume_from=None) -> list[str]`; `dag_placement` accepted by the four chains.

Run tests: `.venv/bin/python -m pytest scripts/tests/test_run_experiment_dag_placement.py -v`

- [ ] **Step 1: Write the failing tests**

Create `scripts/tests/test_run_experiment_dag_placement.py` (mirror how existing `scripts/tests/` import `run_experiment` — check a neighboring test for the import/monkeypatch of `_require_workspace_env`):

```python
"""run_experiment wiring for model_class=dag_placement (piece 3)."""
import importlib


def _run_exp(monkeypatch):
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("proj.ds", "bill"))
    return mod


def test_validate_frontmatter_accepts_dag_placement(monkeypatch):
    mod = _run_exp(monkeypatch)
    # minimal valid frontmatter; must not sys.exit
    mod.validate_frontmatter({
        "id": 52, "slug": "x", "cohort": "population_diabetes",
        "model_class": "dag_placement"})


def test_driver_path_for_dag_placement(monkeypatch):
    mod = _run_exp(monkeypatch)
    assert mod.build_fit_driver_path({"model_class": "dag_placement"}) \
        == "analysis/cloud/dag_placement_cloud.py"


def test_build_dag_placement_args_shape(monkeypatch):
    mod = _run_exp(monkeypatch)
    eff = {"model_class": "dag_placement", "source_table": "condition_era",
           "person_mod": 10, "vocab_size": 5000, "min_df": 20,
           "min_patient_count": 20, "doc_min_length": 0, "prior_obs_days": 365,
           "window_days": 365, "anchor": 201820, "min_n": 50, "holdout_frac": 0.2,
           "n_bg": 2, "tpn": 1, "max_iter": 100, "seed": 42, "init": "spectral",
           "spectral_max_vocab": 8000, "cache_uri": "hdfs:///c"}
    args = mod.build_dag_placement_args(eff, "/out")
    assert "--init" in args and args[args.index("--init") + 1] == "spectral"
    assert "--anchor" in args and args[args.index("--anchor") + 1] == "201820"
    assert "--out-dir" in args and args[args.index("--out-dir") + 1] == "/out"
    assert "--cache-uri" in args and args[args.index("--cache-uri") + 1] == "hdfs:///c"
    assert "--K" not in args                       # K is emergent
    assert "--resume-from" not in args             # resume unsupported


def test_build_fit_args_routes_dag_placement(monkeypatch):
    mod = _run_exp(monkeypatch)
    eff = {"model_class": "dag_placement", "source_table": "condition_era",
           "person_mod": 10, "vocab_size": 5000, "min_df": 20,
           "min_patient_count": 20, "doc_min_length": 0, "prior_obs_days": 365,
           "window_days": 365, "anchor": 201820, "min_n": 50, "holdout_frac": 0.2,
           "n_bg": 2, "tpn": 1, "max_iter": 100, "seed": 42, "init": "random",
           "spectral_max_vocab": 8000}
    args = mod.build_fit_args(eff, "/out")
    assert "--anchor" in args     # routed to build_dag_placement_args
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest scripts/tests/test_run_experiment_dag_placement.py -v`
Expected: FAIL (validate rejects `dag_placement`; no driver path; no `build_dag_placement_args`).

- [ ] **Step 3: Wire chain 1 (validate_frontmatter)**

In `validate_frontmatter`, change the allow-list line:

```python
    if model_class not in ("lda", "stm", "pg_stm", "dag_placement"):
        print(f"[run-exp] ERROR: model_class {model_class!r} not supported "
              f"(currently: lda, stm, pg_stm, dag_placement; hdp planned)", flush=True)
        sys.exit(2)
```

(Leave the `stm`/`pg_stm` covariate-required block unchanged — `dag_placement` needs no covariate fields; its knobs all have `_base.yaml` defaults.)

- [ ] **Step 4: Wire chain 2 (build_fit_driver_path)**

Add before the final `raise`:

```python
    if model_class == "dag_placement":
        return f"{base}/dag_placement_cloud.py"
```

- [ ] **Step 5: Wire chain 3 (build_fit_args) + add build_dag_placement_args**

In `build_fit_args`, add before the final `raise`:

```python
    if model_class == "dag_placement":
        return build_dag_placement_args(effective, out_dir, resume_from)
```

Add the new builder (next to `build_pg_stm_args`):

```python
def build_dag_placement_args(
    effective: dict, out_dir: str, resume_from: Path | None = None,
) -> list[str]:
    """Build argv for analysis/cloud/dag_placement_cloud.py (gated-SVI case-finding).

    K is emergent (n_bg + surviving-DAG-nodes * tpn), so there is NO --K. Resume is
    unsupported (GatedLDAModel is not persistable in v1); resume_from is ignored.
    """
    cdr, billing = _require_workspace_env()
    args = [
        "--cdr", cdr,
        "--billing", billing,
        "--source-table", str(effective["source_table"]),
        "--person-mod", str(effective["person_mod"]),
        "--vocab-size", str(effective["vocab_size"]),
        "--min-df", str(effective["min_df"]),
        "--min-patient-count", str(effective["min_patient_count"]),
        "--doc-min-length", str(effective["doc_min_length"]),
        "--prior-obs-days", str(effective.get("prior_obs_days", 365)),
        "--window-days", str(effective.get("window_days", 365)),
        "--anchor", str(effective.get("anchor", 201820)),
        "--min-n", str(effective["min_n"]),
        "--holdout-frac", str(effective.get("holdout_frac", 0.2)),
        "--n-bg", str(effective["n_bg"]),
        "--tpn", str(effective["tpn"]),
        "--max-iter", str(effective["max_iter"]),
        "--cavi-max-iter", str(effective.get("cavi_max_iter", 100)),
        "--cavi-tol", str(effective.get("cavi_tol", 1e-3)),
        "--init", str(effective.get("init", "random")),
        "--spectral-max-vocab", str(effective.get("spectral_max_vocab", 8000)),
        "--out-dir", str(out_dir),
    ]
    if effective.get("seed") is not None:
        args.extend(["--seed", str(effective["seed"])])
    if effective.get("cache_uri"):
        args.extend(["--cache-uri", str(effective["cache_uri"])])
    return args
```

- [ ] **Step 6: Wire chain 4 (eval-dispatch skip in main)**

Change the `pg_stm` opt-out `elif` to include `dag_placement`:

```python
    elif effective.get("model_class") in ("pg_stm", "dag_placement"):
        # pg_stm and dag_placement save an npz + manifest (a methods-experiment
        # artifact), not a topic-word bundle the NPMI eval driver can read. Their
        # metrics live in manifest.json (dag_placement: placement AUC/MRR; pg_stm:
        # Sigma diagnostics) + the fit log.
        mc = effective.get("model_class")
        print(f"[run-exp] model_class={mc}: NPMI eval not wired for the npz result; "
              "skipping eval (see manifest.json + fit log).", flush=True)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest scripts/tests/test_run_experiment_dag_placement.py -v`
Expected: PASS (4 tests). Also run the existing run_experiment tests to confirm no regression: `.venv/bin/python -m pytest scripts/tests/ -q -k run_experiment`.

- [ ] **Step 8: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment_dag_placement.py
git commit -m "feat(dag-placement): wire model_class=dag_placement into run_experiment

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Config — _base.yaml defaults + diabetes cohort YAML + two A/B experiment files

**Files:**
- Modify: `experiments/defaults/_base.yaml` (append a dag_placement block)
- Create: `experiments/defaults/population_diabetes.yaml`
- Create: `docs/experiments/0052-dag-placement-diabetes-random.md`
- Create: `docs/experiments/0053-dag-placement-diabetes-spectral.md`
- Test: `scripts/tests/test_dag_placement_config.py` (create)

**Interfaces:**
- Consumes: the `run_experiment` config layering (`_base.yaml` → `<cohort>.yaml` → frontmatter) and `build_dag_placement_args` (Task 4).
- Produces: the two runnable A/B experiments; `make exp ID=52` / `ID=53`.

Run tests: `.venv/bin/python -m pytest scripts/tests/test_dag_placement_config.py -v`

- [ ] **Step 1: Write the failing test**

Create `scripts/tests/test_dag_placement_config.py`:

```python
"""The dag_placement A/B experiment files parse, merge, and build valid argv."""
import importlib
from pathlib import Path

# Verified run_experiment APIs (scripts/run_experiment.py):
#   read_frontmatter(path: Path) -> dict
#   load_defaults(cohort: str, defaults_dir: Path) -> dict   (merges _base + <cohort>.yaml)
#   merge_config(base: dict, override: dict) -> dict
_REPO = Path(__file__).resolve().parents[2]        # scripts/tests/ -> repo root
_DEFAULTS = _REPO / "experiments" / "defaults"


def _load_effective(mod, exp_path):
    fm = mod.read_frontmatter(_REPO / exp_path)
    defaults = mod.load_defaults(fm["cohort"], _DEFAULTS)
    return mod.merge_config(defaults, fm)


def test_diabetes_experiments_parse_and_build(monkeypatch):
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("p.d", "bill"))
    for exp, init in [("docs/experiments/0052-dag-placement-diabetes-random.md", "random"),
                      ("docs/experiments/0053-dag-placement-diabetes-spectral.md", "spectral")]:
        eff = _load_effective(mod, exp)
        assert eff["model_class"] == "dag_placement"
        assert eff["init"] == init
        assert eff["anchor"] == 201820
        mod.validate_frontmatter(eff)                 # must not exit
        args = mod.build_dag_placement_args(eff, "/out")
        assert args[args.index("--init") + 1] == init
```

(`read_frontmatter`/`load_defaults`/`merge_config` are the real, verified APIs — `scripts/tests/conftest.py` already puts `scripts/` on `sys.path`, so `import run_experiment` works.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest scripts/tests/test_dag_placement_config.py -v`
Expected: FAIL (experiment files + population_diabetes.yaml + `init` default missing).

- [ ] **Step 3: Append the dag_placement block to `_base.yaml`**

At the end of `experiments/defaults/_base.yaml`:

```yaml

# --- dag_placement (hierarchical case-finding) defaults ---
# model_class: dag_placement triggers the gated-SVI placement engine. K is
# EMERGENT (n_bg + surviving-DAG-nodes * tpn), NOT set here. init=random is the
# validated default; init=spectral is the pre-registered A/B arm (dense
# block-aligned anchor-word seed, Arora et al. 2013). anchor 201820 = diabetes.
anchor: 201820
min_n: 50
n_bg: 2
tpn: 1
holdout_frac: 0.2
window_days: 365
init: random
spectral_max_vocab: 8000
cavi_max_iter: 100
cavi_tol: 1.0e-3
```

- [ ] **Step 4: Create `experiments/defaults/population_diabetes.yaml`**

```yaml
# Diabetes case-finding cohort defaults. The corpus identity actually comes from
# `anchor` (201820) + the diabetes+background population hard-coded inside
# assemble_case_finding_corpus; this file exists so load_defaults finds a
# <cohort>.yaml to merge over _base.yaml (--cohort is vestigial for this model,
# as with the gated STM cohorts).
cohort: population_diabetes
cohort_def: population_diabetes
```

- [ ] **Step 5: Create the two experiment files**

`docs/experiments/0052-dag-placement-diabetes-random.md`:

```markdown
---
id: 52
slug: dag-placement-diabetes-random
status: pending
model_class: dag_placement
cohort: population_diabetes
cohort_def: population_diabetes
person_mod: 10
prior_obs_days: 365
anchor: 201820
min_n: 50
n_bg: 2
tpn: 1
holdout_frac: 0.2
init: random
max_iter: 100
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0052 — DAG-placement diabetes case-finding (random init)

Baseline arm of the pre-registered init A/B: fit the gated-SVI hierarchical
case-finding engine on the diabetes type taxonomy (anchor 201820) + background
population, with random lambda init. Reports held-out placement AUC-by-depth,
MRR, top2 (see manifest.json). Pair: exp 0053 (spectral init).
```

`docs/experiments/0053-dag-placement-diabetes-spectral.md` — identical except:

```markdown
---
id: 53
slug: dag-placement-diabetes-spectral
status: pending
model_class: dag_placement
cohort: population_diabetes
cohort_def: population_diabetes
person_mod: 10
prior_obs_days: 365
anchor: 201820
min_n: 50
n_bg: 2
tpn: 1
holdout_frac: 0.2
init: spectral
max_iter: 100
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0053 — DAG-placement diabetes case-finding (spectral init)

Spectral arm of the pre-registered init A/B: same corpus + engine as exp 0052
but with the dense block-aligned anchor-word spectral seed (Arora et al. 2013).
Tests the user's gated-STM observation that spectral init helps on real data —
on the synthetic plants it was validated-negative (the gate already breaks
symmetry). Shares the case_finding_cache with exp 0052 (identical corpus).
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest scripts/tests/test_dag_placement_config.py -v`
Expected: PASS. If the frontmatter/merge helper names differ, fix `_load_effective` to the real API (Step 1 NOTE), not the config files.

- [ ] **Step 7: Commit**

```bash
git add experiments/defaults/_base.yaml experiments/defaults/population_diabetes.yaml docs/experiments/0052-dag-placement-diabetes-random.md docs/experiments/0053-dag-placement-diabetes-spectral.md scripts/tests/test_dag_placement_config.py
git commit -m "feat(dag-placement): _base.yaml defaults + diabetes cohort + init A/B experiments

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (against the spec)

**Spec coverage:** (A) init shim wiring = Task 1; (B) bundle cache + load_or_build = Task 2; (C) dag_placement_cloud.py assemble→fit→transform→inline evaluate→save = Task 3; (D) four run_experiment chains + build_dag_placement_args = Task 4; (E) config (_base.yaml + cohort YAML + two A/B experiments) = Task 5. The `nBg`/`tpn` single-source invariant is enforced in Task 3 (one arg pair fed to assembly-via-cache, estimator, and scoring DagLayout). Deferred items (scalable projected init, Makefile target, resume, dashboard) are honored as guards/omissions, not built.

**Placeholder scan:** none — every step carries runnable code. The two API-name caveats (analysis/cloud test import path in Task 2 Step 1; run_experiment frontmatter-reader name in Task 5 Step 1) are explicit "confirm-the-real-name" notes with a concrete fallback, not blanks.

**Type consistency:** `parent_int` = `{int: [int]}` (restored int keys, Task 2); `frontier` = engine-id `array<bigint>` → `set[int]` (Task 3 adapter); `nodeAffinity` = DenseVector ordered by `lay.nodes` (Task 3); estimator params `init`/`spectralMaxVocab` are `TypeConverters.toString`/`toInt` (Task 1) and read back the same way in Task 3/4. `build_dag_placement_args` emits no `--K` and no `--resume-from`, matching the driver's arg surface (Task 3) and the emergent-K / no-resume decisions.
