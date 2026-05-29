# STM (prevalence-only) — charmpheno integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Prerequisite:** [the spark-vi STM plan](2026-05-29-stm-spark-vi.md) must land first — this plan depends on `spark_vi.mllib.topic.stm.StreamingSTM`, `STMModel`, `_vector_to_stm_document`, and `spark_vi.mllib.topic._formula.fit_model_spec_from_spark`.

**Goal:** Integrate STM (prevalence-only) into charmpheno. Adds patient-covariate materialization from BigQuery, a covariates cache mirroring the existing corpus cache, a new `stm_bigquery_cloud.py` fit driver, experiment-tracking wrapper dispatch, and a dashboard adapter. Phases 4–6 of the [2026-05-29 STM design spec](../specs/2026-05-29-stm-prevalence-design.md).

**Architecture:** Covariates flow through a parallel pipeline to the BOW corpus — BigQuery query over the OMOP person table (or per-cohort lookup tables for condition/observation covariates) → formulaic `ModelSpec` discovery via the schema-frame trick → cached parquet of `(person_id, x as DenseVector)` keyed on formula + source params. The STM fit driver loads both the corpus DataFrame (via the existing `_corpus_load.load_or_build_corpus`) and the covariates DataFrame (via the new `_covariates_load.load_or_build_covariates`), broadcast-joins by `person_id`, and runs `StreamingSTM` Path A. Dashboard bundle gains an `adapt_stm` adapter writing Γ̂ alongside the existing per-topic outputs. See [ADR 0025](../../decisions/0025-charmpheno-covariate-sidecar-parquet.md).

**Tech Stack:** Python 3.10+, PySpark, `spark-vi[formula]` (from plan 1), `formulaic`, `numpy`, BigQuery via existing `charmpheno.omop` machinery.

---

## Context

This plan consumes the spark-vi STM API delivered by [plan 1](2026-05-29-stm-spark-vi.md). The relevant entry points the charmpheno side calls:

- `spark_vi.mllib.topic.stm.StreamingSTM(K, features_col, covariates_col, covariate_names, ...)` — Path A estimator. We use Path A (not Path B) in the charmpheno driver because the covariate sidecar build separates formula-resolution from fit: the sidecar produces a pre-built `covariates` Vector column and the driver hands it to StreamingSTM Path A.
- `spark_vi.mllib.topic.stm.STMModel.save(out_dir)` / `STMModel.load(out_dir)` — fitted-model persistence.
- `spark_vi.mllib.topic._formula.fit_model_spec_from_spark(formula, spark_df, categorical_cols, continuous_cols, max_levels)` — schema-frame ModelSpec discovery against a Spark DataFrame.
- `spark_vi.mllib.topic._formula.apply_model_spec(spec, pandas_df)` — applies a fitted ModelSpec to produce a numpy design matrix.
- `spark_vi.mllib.topic._formula.validate_formula(formula_str)` — rejects spline / standardization terms at parse time.

The charmpheno driver pattern is well-established. Compare:

- [`analysis/cloud/lda_bigquery_cloud.py`](../../analysis/cloud/lda_bigquery_cloud.py) — LDA fit driver. STM driver mirrors its structure.
- [`analysis/cloud/hdp_bigquery_cloud.py`](../../analysis/cloud/hdp_bigquery_cloud.py) — HDP fit driver. Similar shape with HDP-specific config.
- [`analysis/cloud/_corpus_load.py`](../../analysis/cloud/_corpus_load.py) — the load-or-build pattern STM covariates copy.
- [`analysis/cloud/_corpus_cache.py`](../../analysis/cloud/_corpus_cache.py) — the cache pattern STM covariates copy.

Two facts surfaced from the existing code base that shape this plan:

1. **The corpus parquet lives in a cache, not in the run directory.** [`load_or_build_corpus`](../../analysis/cloud/_corpus_load.py#L26) reads `(bow_df, vocab_map, name_by_id)` from a cached parquet (if cache hit) or rebuilds from BigQuery and writes through. The "manifest" data flows downstream via `VIResult.metadata['corpus_manifest']`. STM covariates follow the same pattern — a separate cache parquet, with the manifest pointer added to VIResult.metadata. (ADR 0025's "sidecar parquet" framing is correct conceptually; the implementation is a parallel cache layer, not a colocated file.)

2. **`scripts/run_experiment.py` currently gates on `model_class == "lda"`** (line 562-564) and exits with code 2 otherwise. STM dispatch is a structural extension of that gate, not a side path.

---

## File Structure

**New files:**
- `charmpheno/charmpheno/omop/covariates.py` — `build_patient_covariate_df` helper consuming spark-vi's formula machinery
- `analysis/cloud/_covariates_cache.py` — cache key + try_load + save (mirrors `_corpus_cache.py`)
- `analysis/cloud/_covariates_load.py` — `load_or_build_covariates` (mirrors `_corpus_load.py`)
- `analysis/cloud/stm_bigquery_cloud.py` — STM fit driver
- `charmpheno/tests/test_covariates_build.py` — covariates module tests
- `charmpheno/tests/test_export_dashboard_stm.py` — adapt_stm bundle tests
- `analysis/cloud/tests/test_stm_smoke.py` — synthetic end-to-end smoke (if `analysis/cloud/tests/` doesn't exist yet, this lives at `analysis/cloud/tests/test_stm_smoke.py` or `charmpheno/tests/test_stm_smoke.py` — implementer picks the existing convention)

**Modified files:**
- `charmpheno/charmpheno/export/dashboard.py` — add `adapt_stm` function alongside the existing LDA adapter calls
- `analysis/cloud/build_dashboard_cloud.py` — dispatch `adapt_stm` vs `adapt_lda` based on `model_class` in VIResult.metadata
- `scripts/run_experiment.py` — extend model_class gate to accept "stm"; thread `covariate_formula` from defaults YAML into fit args
- `experiments/defaults/_base.yaml` — add `covariate_formula: null` (default off) and STM-specific hyperparams as commented hints
- `analysis/cloud/Makefile` — `make build-covariates EXP=N` target; existing `next-exp`, `exp ID=`, `eval-exp`, `build-dashboard-exp` targets stay unchanged (they already dispatch on model_class via the wrapper)

**Not modified:**
- `analysis/cloud/lda_bigquery_cloud.py`, `analysis/cloud/hdp_bigquery_cloud.py` — LDA/HDP drivers untouched
- `analysis/cloud/_corpus_load.py`, `analysis/cloud/_corpus_cache.py` — corpus loading pattern reused as-is; STM covariates parallel them without modifying them
- `charmpheno/charmpheno/omop/topic_prep.py` — BOW corpus prep unchanged; covariates are a separate pipeline keyed on `person_id`
- `charmpheno/charmpheno/export/model_adapter.py` — keep LDA adapter as-is; STM adapter is a new sibling

---

## Phase 4 — Covariate materialization + caching

### Task 1: `charmpheno.omop.covariates` — `build_patient_covariate_df`

**Files:**
- Create: `charmpheno/charmpheno/omop/covariates.py`
- Create: `charmpheno/tests/test_covariates_build.py`

- [ ] **Step 1: Write the helper test**

Create `charmpheno/tests/test_covariates_build.py`:

```python
"""Tests for charmpheno.omop.covariates.build_patient_covariate_df.

The helper takes a person-level Spark DataFrame and a formula string,
fits a formulaic ModelSpec via the spark-vi shim, applies it per row,
and returns (covariates_df, model_spec, covariate_names).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
formulaic = pytest.importorskip("formulaic")

from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def spark():
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("test-covariates")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


class TestBuildPatientCovariateDF:
    def test_categorical_and_continuous(self, spark):
        from charmpheno.omop.covariates import build_patient_covariate_df
        pdf = pd.DataFrame({
            "person_id": [1, 2, 3, 4, 5],
            "cohort":    ["control", "case", "case", "control", "case"],
            "age":       [25.0, 40.0, 55.0, 30.0, 45.0],
        })
        person_df = spark.createDataFrame(pdf)
        cov_df, spec, names = build_patient_covariate_df(
            person_df,
            covariate_formula="~ C(cohort) + age",
            categorical_cols=["cohort"],
            continuous_cols=["age"],
        )
        rows = cov_df.orderBy("person_id").collect()
        assert len(rows) == 5
        # Each row should have person_id + covariates (DenseVector).
        for row in rows:
            assert "person_id" in row.asDict()
            assert "covariates" in row.asDict()
            assert len(row["covariates"]) == len(names)

    def test_rejects_stateful_transforms_at_build_time(self, spark):
        from charmpheno.omop.covariates import build_patient_covariate_df
        pdf = pd.DataFrame({
            "person_id": [1, 2, 3],
            "age": [25.0, 40.0, 55.0],
        })
        person_df = spark.createDataFrame(pdf)
        with pytest.raises(ValueError, match="bs|spline|stateful"):
            build_patient_covariate_df(
                person_df,
                covariate_formula="~ bs(age, df=4)",
                categorical_cols=[],
                continuous_cols=["age"],
            )

    def test_unseen_level_handling(self, spark):
        """build only sees the person_df it's given; if a downstream join
        reveals new levels later, that's a runtime error at apply time
        — not this helper's concern. Documented contract."""
        from charmpheno.omop.covariates import build_patient_covariate_df
        pdf = pd.DataFrame({
            "person_id": [1, 2, 3],
            "cohort":    ["control", "case", "case"],
        })
        person_df = spark.createDataFrame(pdf)
        _, spec, names = build_patient_covariate_df(
            person_df,
            covariate_formula="~ C(cohort)",
            categorical_cols=["cohort"],
            continuous_cols=[],
        )
        # Spec captured both levels.
        assert any("control" in str(n) or "case" in str(n) for n in names)
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
poetry run pytest charmpheno/tests/test_covariates_build.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `build_patient_covariate_df`**

Create `charmpheno/charmpheno/omop/covariates.py`:

```python
"""STM patient-covariate materialization.

build_patient_covariate_df takes a person-level Spark DataFrame and a
formula string, uses spark-vi's formula machinery (schema-frame
discovery via Spark `select distinct`) to fit a formulaic ModelSpec,
and applies the spec per row to produce a per-person `(person_id,
covariates)` DataFrame with a DenseVector column.

Decision context: docs/decisions/0025-charmpheno-covariate-sidecar-parquet.md
                  docs/decisions/0024-formulaic-in-mllib-shim-with-schema-frame-discovery.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, LongType


def build_patient_covariate_df(
    person_df: DataFrame,
    *,
    covariate_formula: str,
    categorical_cols: list[str],
    continuous_cols: list[str],
    max_levels: int = 10_000,
) -> tuple[DataFrame, Any, list[str]]:
    """Materialize per-person covariates from a Spark DataFrame + formula.

    Returns:
        cov_df:      (person_id: long, covariates: DenseVector) Spark DataFrame.
                     One row per distinct person_id in person_df.
        model_spec:  formulaic ModelSpec, ready to persist + apply at
                     transform / scoring time.
        names:       list of covariate column names (length P).

    Raises:
        ValueError if covariate_formula contains a stateful transform
        (spline, standardization). See ADR 0022 / 0024.
    """
    from spark_vi.mllib.topic._formula import (
        validate_formula, fit_model_spec_from_spark, apply_model_spec,
    )

    validate_formula(covariate_formula)

    # Project only the columns the formula references + person_id.
    needed_cols = ["person_id"] + categorical_cols + continuous_cols
    projected = person_df.select(*needed_cols).dropDuplicates(["person_id"])

    model_spec, names = fit_model_spec_from_spark(
        formula=covariate_formula,
        spark_df=projected,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        max_levels=max_levels,
    )

    # Apply the spec per partition. ModelSpec is small (factor mappings);
    # serializing it into the closure is fine for a few-KB object.
    spec_broadcast = projected.sparkSession.sparkContext.broadcast(model_spec)
    cat_set = list(categorical_cols)
    cont_set = list(continuous_cols)

    def _apply_partition(rows_iter):
        import pandas as pd
        import numpy as _np
        spec = spec_broadcast.value
        # Buffer the partition's rows into a small pandas frame for vectorized
        # apply_model_spec — chunk size keeps memory bounded if partitions are large.
        CHUNK = 10_000
        buf = []
        for r in rows_iter:
            buf.append(r.asDict())
            if len(buf) >= CHUNK:
                yield from _flush(buf, spec, cat_set, cont_set)
                buf = []
        if buf:
            yield from _flush(buf, spec, cat_set, cont_set)

    def _flush(buf, spec, cat_set, cont_set):
        import pandas as pd
        from spark_vi.mllib.topic._formula import apply_model_spec
        from pyspark.ml.linalg import DenseVector
        pdf = pd.DataFrame(buf)
        X = apply_model_spec(spec, pdf)   # (chunk_size, P)
        for pid, row in zip(pdf["person_id"].tolist(), X):
            yield (int(pid), DenseVector(list(row)))

    schema = StructType([
        StructField("person_id", LongType(), False),
        StructField("covariates", VectorUDT(), False),
    ])

    cov_df = projected.rdd.mapPartitions(_apply_partition).toDF(schema)
    return cov_df, model_spec, names
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest charmpheno/tests/test_covariates_build.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/covariates.py charmpheno/tests/test_covariates_build.py
git commit -m "feat(charmpheno/omop): build_patient_covariate_df materializes per-person STM covariates"
```

---

### Task 2: `_covariates_cache.py` — cache key + try_load + save

**Files:**
- Create: `analysis/cloud/_covariates_cache.py`
- Test: integrated into Task 3's `_covariates_load.py` tests (covariate cache has no behavior outside its load wrapper)

- [ ] **Step 1: Implement the cache module**

Create `analysis/cloud/_covariates_cache.py`:

```python
"""Covariate cache: parallel to _corpus_cache.py for STM patient covariates.

Cache key derives from formula text + person table source + person_mod
so a covariates cache hit guarantees identical (person_id, covariates)
output. ModelSpec is persisted alongside the parquet so the cache
hit path can return the spec without re-fitting it from BigQuery.

Layout under <cache_uri>/<key>/:
    covariates.parquet     # (person_id, covariates as DenseVector)
    model_spec.pkl         # pickled formulaic ModelSpec
    covariate_names.json   # list of P term names

Format mirrors _corpus_cache.py; see that module for the read/write helpers
this one parallels.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from typing import Any

from pyspark.sql import DataFrame, SparkSession


def compute_cache_key(
    *,
    covariate_formula: str,
    person_mod: int,
    cdr: str,
    source_table: str,
    cohort: str | None,
) -> str:
    """Stable hex digest of the inputs that determine the covariate output."""
    payload = json.dumps({
        "covariate_formula": covariate_formula,
        "person_mod": person_mod,
        "cdr": cdr,
        "source_table": source_table,
        "cohort": cohort,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def try_load(
    spark: SparkSession, cache_uri: str, cache_key: str,
) -> tuple[DataFrame, Any, list[str]] | None:
    """Return (cov_df, model_spec, names) on cache hit, None on miss."""
    base = f"{cache_uri.rstrip('/')}/{cache_key}"
    try:
        cov_df = spark.read.parquet(f"{base}/covariates.parquet")
    except Exception:
        return None
    # Spec + names are driver-side reads via Spark's Hadoop FS.
    try:
        spec_path = f"{base}/model_spec.pkl"
        names_path = f"{base}/covariate_names.json"
        spec_bytes = _read_bytes(spark, spec_path)
        names_bytes = _read_bytes(spark, names_path)
    except Exception:
        return None
    spec = pickle.loads(spec_bytes)
    names = json.loads(names_bytes.decode("utf-8"))
    return cov_df, spec, names


def save(
    spark: SparkSession,
    cache_uri: str,
    cache_key: str,
    *,
    cov_df: DataFrame,
    model_spec: Any,
    covariate_names: list[str],
) -> None:
    """Write through to <cache_uri>/<key>/."""
    base = f"{cache_uri.rstrip('/')}/{cache_key}"
    cov_df.write.mode("overwrite").parquet(f"{base}/covariates.parquet")
    _write_bytes(spark, f"{base}/model_spec.pkl", pickle.dumps(model_spec))
    _write_bytes(
        spark, f"{base}/covariate_names.json",
        json.dumps(covariate_names).encode("utf-8"),
    )


def _read_bytes(spark: SparkSession, path: str) -> bytes:
    """Read a small file via Spark's Hadoop FileSystem (handles GCS/HDFS/local)."""
    sc = spark.sparkContext
    jvm = sc._jvm
    fs_path = jvm.org.apache.hadoop.fs.Path(path)
    fs = fs_path.getFileSystem(sc._jsc.hadoopConfiguration())
    in_stream = fs.open(fs_path)
    try:
        out = jvm.java.io.ByteArrayOutputStream()
        jvm.org.apache.hadoop.io.IOUtils.copyBytes(in_stream, out, sc._jsc.hadoopConfiguration())
        return bytes(out.toByteArray())
    finally:
        in_stream.close()


def _write_bytes(spark: SparkSession, path: str, data: bytes) -> None:
    sc = spark.sparkContext
    jvm = sc._jvm
    fs_path = jvm.org.apache.hadoop.fs.Path(path)
    fs = fs_path.getFileSystem(sc._jsc.hadoopConfiguration())
    out_stream = fs.create(fs_path, True)
    try:
        out_stream.write(data)
    finally:
        out_stream.close()
```

Note: the `_read_bytes` / `_write_bytes` helpers mirror what `_corpus_cache.py` does for its `vocab.json` and `names.json` sidecar files (run `grep -n "_read_bytes\|_write_bytes\|Path(" analysis/cloud/_corpus_cache.py` to confirm the existing patterns; if `_corpus_cache.py` exposes them as module functions, import them instead of duplicating).

- [ ] **Step 2: Quick smoke check that the module imports**

```bash
poetry run python -c "from analysis.cloud import _covariates_cache; print(_covariates_cache.compute_cache_key(covariate_formula='~ x', person_mod=10, cdr='cdr1', source_table='condition_era', cohort=None))"
```

Expected: prints a 16-char hex string.

- [ ] **Step 3: Commit**

```bash
git add analysis/cloud/_covariates_cache.py
git commit -m "feat(cloud): _covariates_cache mirrors _corpus_cache for STM covariate parquets"
```

---

### Task 3: `_covariates_load.py` — `load_or_build_covariates`

**Files:**
- Create: `analysis/cloud/_covariates_load.py`
- Create: `analysis/cloud/tests/test_covariates_load.py` (if no `analysis/cloud/tests/` dir exists, use `charmpheno/tests/test_covariates_load.py`)

- [ ] **Step 1: Write the load-or-build test**

Create the test (path per existing convention — discover with `ls analysis/cloud/tests/ 2>/dev/null || ls charmpheno/tests/`):

```python
"""Tests for load_or_build_covariates: cache-hit fast path + cache-miss build-through."""
from __future__ import annotations

import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
formulaic = pytest.importorskip("formulaic")

from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def spark():
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("test-cov-load")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


class TestLoadOrBuildCovariates:
    def test_miss_then_hit_roundtrips(self, spark, tmp_path):
        from analysis.cloud._covariates_load import load_or_build_covariates

        # Build a synthetic person_df.
        pdf = pd.DataFrame({
            "person_id": [1, 2, 3, 4],
            "cohort":    ["case", "control", "case", "control"],
            "age":       [25.0, 40.0, 55.0, 30.0],
        })
        person_df = spark.createDataFrame(pdf)

        cache_uri = str(tmp_path)

        # First call: miss, builds, writes through.
        cov1, spec1, names1 = load_or_build_covariates(
            spark, person_df=person_df,
            covariate_formula="~ C(cohort) + age",
            categorical_cols=["cohort"], continuous_cols=["age"],
            cache_uri=cache_uri,
            cdr="test-cdr", source_table="condition_era",
            cohort=None, person_mod=10,
        )
        rows1 = sorted(cov1.collect(), key=lambda r: r.person_id)

        # Second call: cache hit, same names + spec, same row contents.
        cov2, spec2, names2 = load_or_build_covariates(
            spark, person_df=person_df,
            covariate_formula="~ C(cohort) + age",
            categorical_cols=["cohort"], continuous_cols=["age"],
            cache_uri=cache_uri,
            cdr="test-cdr", source_table="condition_era",
            cohort=None, person_mod=10,
        )
        rows2 = sorted(cov2.collect(), key=lambda r: r.person_id)
        assert names1 == names2
        for r1, r2 in zip(rows1, rows2):
            assert r1.person_id == r2.person_id
            assert list(r1.covariates) == list(r2.covariates)

    def test_cache_uri_none_skips_cache(self, spark):
        from analysis.cloud._covariates_load import load_or_build_covariates
        pdf = pd.DataFrame({
            "person_id": [1, 2], "cohort": ["a", "b"], "age": [25.0, 40.0],
        })
        person_df = spark.createDataFrame(pdf)
        cov, spec, names = load_or_build_covariates(
            spark, person_df=person_df,
            covariate_formula="~ C(cohort) + age",
            categorical_cols=["cohort"], continuous_cols=["age"],
            cache_uri=None,
            cdr="test", source_table="condition_era",
            cohort=None, person_mod=10,
        )
        assert cov.count() == 2
```

- [ ] **Step 2: Run to verify failure**

```bash
poetry run pytest <path_to_test_file> -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `load_or_build_covariates`**

Create `analysis/cloud/_covariates_load.py`:

```python
"""STM patient-covariate load-or-build (mirrors _corpus_load.py).

On cache HIT: returns (cov_df, model_spec, covariate_names) from the
write-through parquet without re-fitting the ModelSpec.

On MISS (or cache_uri=None): runs build_patient_covariate_df against
the source person_df, optionally writes through the cache.

The cache key derives from (formula, person_mod, cdr, source_table, cohort)
so a cache hit is safe: same inputs ⇒ same outputs.
"""
from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, SparkSession

from _driver_common import _phase


def load_or_build_covariates(
    spark: SparkSession,
    *,
    person_df: DataFrame,
    covariate_formula: str,
    categorical_cols: list[str],
    continuous_cols: list[str],
    cdr: str,
    source_table: str,
    cohort: str | None,
    person_mod: int,
    cache_uri: str | None = None,
    max_levels: int = 10_000,
) -> tuple[DataFrame, Any, list[str]]:
    """Return (cov_df, model_spec, covariate_names) for the given formula."""
    from charmpheno.omop.covariates import build_patient_covariate_df
    from _covariates_cache import compute_cache_key, try_load, save

    if cache_uri:
        key = compute_cache_key(
            covariate_formula=covariate_formula,
            person_mod=person_mod, cdr=cdr,
            source_table=source_table, cohort=cohort,
        )
        with _phase(f"covariates-cache lookup ({cache_uri}/{key})"):
            cached = try_load(spark, cache_uri, key)
        if cached is not None:
            print("[driver]   covariates-cache HIT", flush=True)
            return cached
        print("[driver]   covariates-cache MISS, building...", flush=True)

    with _phase("build patient covariates"):
        cov_df, spec, names = build_patient_covariate_df(
            person_df,
            covariate_formula=covariate_formula,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
            max_levels=max_levels,
        )

    if cache_uri:
        with _phase(f"covariates-cache write-through ({cache_uri}/{key})"):
            save(spark, cache_uri, key,
                 cov_df=cov_df, model_spec=spec, covariate_names=names)

    return cov_df, spec, names
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest <path_to_test_file> -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/_covariates_load.py <path_to_test_file>
git commit -m "feat(cloud): load_or_build_covariates with cache hit / miss fast-paths"
```

---

## Phase 5 — STM fit driver + experiment-tracking integration

### Task 4: `stm_bigquery_cloud.py` — new fit driver

**Files:**
- Create: `analysis/cloud/stm_bigquery_cloud.py`

This is the largest single file in the plan — it mirrors `lda_bigquery_cloud.py` structurally. Implementer should `cat analysis/cloud/lda_bigquery_cloud.py` first to internalize the pattern, then write the STM variant.

- [ ] **Step 1: Inspect the LDA driver for shape**

```bash
wc -l analysis/cloud/lda_bigquery_cloud.py
head -80 analysis/cloud/lda_bigquery_cloud.py
```

Note the structural pieces:
1. argparse with `--cdr`, `--billing`, `--cohort`, `--cache-uri`, `--K`, `--max-iter`, etc.
2. `_driver_common.configure_logging` + `make_spark_session`.
3. `load_or_build_corpus` call.
4. `OnlineLDA` construction + `VIRunner.fit`.
5. Augmented `VIResult` save with `metadata['corpus_manifest']` enriched (lines ~349, ~381).

STM driver mirrors all of this and adds covariate loading + STM-specific args.

- [ ] **Step 2: Create the STM driver**

Create `analysis/cloud/stm_bigquery_cloud.py`:

```python
"""STM (prevalence-only) fit driver — analogous to lda_bigquery_cloud.py.

Loads corpus + covariates from caches (or rebuilds from BQ), broadcast-joins
by person_id, constructs StreamingSTM (Path A), runs VIRunner.fit, and
saves the augmented VIResult with corpus_manifest + covariate_manifest in
metadata.

Decision context: docs/superpowers/specs/2026-05-29-stm-prevalence-design.md
                  docs/decisions/0025-charmpheno-covariate-sidecar-parquet.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyspark.sql import functions as F

from _driver_common import _phase, configure_logging, make_spark_session
from _corpus_load import load_or_build_corpus
from _covariates_load import load_or_build_covariates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STM fit driver (prevalence-only)")
    # Mirror LDA driver flags for shared params.
    p.add_argument("--cdr", required=True)
    p.add_argument("--billing", required=True)
    p.add_argument("--source-table", default="condition_era")
    p.add_argument("--cohort", default=None)
    p.add_argument("--person-mod", type=int, default=10)
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--min-df", type=int, default=20)
    p.add_argument("--min-patient-count", type=int, default=20)
    p.add_argument("--doc-spec", default="patient_year")
    p.add_argument("--doc-min-length", type=int, default=20)
    p.add_argument("--cache-uri", default=None,
                   help="GCS/HDFS URI for the corpus + covariates caches.")
    # STM-specific.
    p.add_argument("--K", type=int, default=40)
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=5)
    p.add_argument("--covariate-formula", required=True,
                   help="R-style formula, e.g. '~ C(sex) + age'. Required for STM.")
    p.add_argument("--categorical-cols", required=True,
                   help="Comma-separated list of categorical column names referenced in the formula.")
    p.add_argument("--continuous-cols", required=True,
                   help="Comma-separated list of continuous column names referenced in the formula.")
    p.add_argument("--sigma-init", type=float, default=1.0)
    p.add_argument("--sigma-ridge", type=float, default=1e-6)
    p.add_argument("--lbfgs-max-iter", type=int, default=50)
    p.add_argument("--lbfgs-tol", type=float, default=1e-4)
    # Mini-batch SVI flags.
    p.add_argument("--subsampling-rate", type=float, default=0.2)
    p.add_argument("--tau0", type=float, default=64.0)
    p.add_argument("--kappa", type=float, default=0.7)
    p.add_argument("--random-seed", type=int, default=None)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()

    cat_cols = [c.strip() for c in args.categorical_cols.split(",") if c.strip()]
    cont_cols = [c.strip() for c in args.continuous_cols.split(",") if c.strip()]

    with make_spark_session(app_name="stm-fit") as spark:
        # --- Corpus load ---
        from charmpheno.omop.doc_spec import resolve as resolve_doc_spec
        doc_spec = resolve_doc_spec(args.doc_spec, min_length=args.doc_min_length)

        with _phase("corpus load"):
            bow_df, vocab_map, name_by_id = load_or_build_corpus(
                spark,
                doc_spec=doc_spec, cdr=args.cdr, billing=args.billing,
                source_table=args.source_table, person_mod=args.person_mod,
                vocab_size=args.vocab_size, min_df=args.min_df,
                min_patient_count=args.min_patient_count,
                cache_uri=args.cache_uri, cohort=args.cohort,
            )

        # --- Covariates load ---
        # Source person_df from BigQuery — same connection params as the corpus load.
        from charmpheno.omop import load_person_table
        with _phase("person table load"):
            person_df = load_person_table(
                spark=spark, cdr_dataset=args.cdr,
                billing_project=args.billing,
                person_sample_mod=args.person_mod,
                cohort=args.cohort,
            )

        with _phase("covariates load"):
            cov_df, model_spec, covariate_names = load_or_build_covariates(
                spark, person_df=person_df,
                covariate_formula=args.covariate_formula,
                categorical_cols=cat_cols, continuous_cols=cont_cols,
                cdr=args.cdr, source_table=args.source_table,
                cohort=args.cohort, person_mod=args.person_mod,
                cache_uri=args.cache_uri,
            )

        # --- Broadcast join: corpus + covariates ---
        with _phase("corpus + covariates join"):
            joined = bow_df.join(F.broadcast(cov_df), on="person_id", how="inner")
            n_joined = joined.count()
            print(f"[driver]   joined docs = {n_joined}", flush=True)

        # --- Construct StreamingSTM, fit ---
        from spark_vi.mllib.topic.stm import StreamingSTM

        with _phase("STM fit"):
            est = StreamingSTM(
                K=args.K,
                features_col="features",
                covariates_col="covariates",
                covariate_names=covariate_names,
                sigma_init=args.sigma_init,
                sigma_ridge=args.sigma_ridge,
                lbfgs_max_iter=args.lbfgs_max_iter,
                lbfgs_tol=args.lbfgs_tol,
                random_seed=args.random_seed,
            )
            # StreamingSTM.fit returns an STMModel (per plan 1's Task 13).
            # The fit method internally creates a VIRunner with the supplied
            # max_iter, subsampling_rate, tau0, kappa.
            stm_model = est.fit(
                joined,
                max_iter=args.max_iter,
                subsampling_rate=args.subsampling_rate,
                tau0=args.tau0, kappa=args.kappa,
                save_interval=args.save_interval,
            )

        # --- Augment VIResult metadata with corpus_manifest + covariate_manifest ---
        with _phase("augment metadata"):
            stm_model.metadata.setdefault("corpus_manifest", {
                "cdr": args.cdr,
                "source_table": args.source_table,
                "cohort": args.cohort,
                "person_mod": args.person_mod,
                "doc_spec": doc_spec.manifest(),
                "vocab_size": len(vocab_map),
                "vocab": list(vocab_map.keys()),
                "name_by_id": name_by_id,
            })
            stm_model.metadata["covariate_manifest"] = {
                "covariate_formula": args.covariate_formula,
                "categorical_cols": cat_cols,
                "continuous_cols": cont_cols,
                "covariate_names": covariate_names,
                # The fitted ModelSpec rides alongside as a sidecar file.
                # Bundle consumers prefer covariate_names for display labels.
            }
            stm_model.metadata["model_class"] = "stm"

        # --- Save ---
        with _phase("save"):
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            stm_model.save(out)
            print(f"[driver]   saved STMModel to {out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: the `load_person_table` helper is assumed to exist or be a thin wrapper around the existing OMOP loading patterns; the implementer adds it to `charmpheno.omop` if missing (check `charmpheno/charmpheno/omop/bigquery.py` and `charmpheno/charmpheno/omop/local.py` for existing person-table accessors; one likely exists as `load_omop_bigquery` already returns person-level data and only needs a `select(person_id, age, sex, ...)` projection).

- [ ] **Step 3: Quick CLI smoke**

```bash
poetry run python analysis/cloud/stm_bigquery_cloud.py --help
```

Expected: prints the argparse usage including the STM-specific flags.

- [ ] **Step 4: Commit**

```bash
git add analysis/cloud/stm_bigquery_cloud.py
git commit -m "feat(cloud): stm_bigquery_cloud.py STM fit driver wiring corpus + covariates"
```

---

### Task 5: `scripts/run_experiment.py` — STM dispatch

**Files:**
- Modify: `scripts/run_experiment.py`
- Test: `scripts/tests/test_run_experiment.py` (extend)

- [ ] **Step 1: Inspect the current model_class gate**

```bash
grep -n "model_class\|lda_bigquery_cloud\|hdp_bigquery_cloud" scripts/run_experiment.py | head -30
```

The gate at line ~562 in `validate_frontmatter` raises on `model_class != "lda"`. The fit dispatch later in `main()` runs `analysis/cloud/lda_bigquery_cloud.py` via spark-submit unconditionally.

- [ ] **Step 2: Write tests for the model_class branching**

Append to `scripts/tests/test_run_experiment.py` (file already exists per session memory):

```python
class TestModelClassDispatch:
    def test_stm_passes_validation(self):
        fm = {
            "id": "0099-test", "slug": "test", "cohort": "dementia",
            "model_class": "stm", "covariate_formula": "~ C(sex) + age",
            "categorical_cols": ["sex"], "continuous_cols": ["age"],
        }
        # Should not raise (LDA gate previously rejected anything != "lda").
        from run_experiment import validate_frontmatter
        validate_frontmatter(fm)  # idempotent — passes for stm with required keys

    def test_stm_requires_covariate_formula(self):
        fm = {
            "id": "0099-test", "slug": "test", "cohort": "dementia",
            "model_class": "stm",
            # covariate_formula missing
        }
        from run_experiment import validate_frontmatter
        with pytest.raises(SystemExit):
            validate_frontmatter(fm)

    def test_build_fit_args_dispatches_to_stm_driver(self):
        fm = {
            "id": "0099", "slug": "test", "cohort": "dementia",
            "model_class": "stm", "covariate_formula": "~ C(sex)",
            "categorical_cols": ["sex"], "continuous_cols": [],
        }
        effective = {**fm, "K": 40, "max_iter": 20}
        from run_experiment import build_fit_driver_path
        path = build_fit_driver_path(effective)
        assert path.endswith("stm_bigquery_cloud.py")

    def test_build_fit_driver_path_lda(self):
        from run_experiment import build_fit_driver_path
        path = build_fit_driver_path({"model_class": "lda"})
        assert path.endswith("lda_bigquery_cloud.py")
```

- [ ] **Step 3: Run to verify failure**

```bash
poetry run pytest scripts/tests/test_run_experiment.py::TestModelClassDispatch -v
```

Expected: FAIL — STM gate still rejects; `build_fit_driver_path` doesn't exist.

- [ ] **Step 4: Update `validate_frontmatter` to accept STM with required fields**

In `scripts/run_experiment.py`, replace the gate at lines ~557-565:

```python
def validate_frontmatter(fm: dict) -> None:
    required = ["id", "slug", "cohort", "model_class"]
    for key in required:
        if key not in fm:
            print(f"[run-exp] ERROR: frontmatter missing required field {key!r}",
                  flush=True)
            sys.exit(2)

    model_class = fm["model_class"]
    if model_class not in ("lda", "stm"):
        print(f"[run-exp] ERROR: model_class {model_class!r} not supported "
              f"(currently: lda, stm; hdp planned)", flush=True)
        sys.exit(2)

    if model_class == "stm":
        # STM requires the covariate spec at frontmatter time so the wrapper
        # can dispatch to stm_bigquery_cloud.py without further inference.
        stm_required = ["covariate_formula", "categorical_cols", "continuous_cols"]
        for key in stm_required:
            if key not in fm:
                print(f"[run-exp] ERROR: model_class=stm requires "
                      f"frontmatter field {key!r}", flush=True)
                sys.exit(2)
```

- [ ] **Step 5: Add `build_fit_driver_path` helper**

In `scripts/run_experiment.py`, add (alongside `build_fit_args` / `build_eval_args`):

```python
def build_fit_driver_path(effective: dict) -> str:
    """Return absolute path to the fit driver script for this model_class."""
    model_class = effective.get("model_class", "lda")
    base = "analysis/cloud"
    if model_class == "lda":
        return f"{base}/lda_bigquery_cloud.py"
    if model_class == "stm":
        return f"{base}/stm_bigquery_cloud.py"
    raise ValueError(f"no fit driver for model_class={model_class!r}")
```

- [ ] **Step 6: Update `build_fit_args` to thread STM-specific args**

Find the existing `build_fit_args` (which currently builds LDA args), and either rename / branch by `model_class`:

```python
def build_fit_args(effective: dict, out_dir: str) -> list[str]:
    """Build the argv for the fit driver, dispatching on model_class."""
    model_class = effective.get("model_class", "lda")
    common = [
        "--cdr", str(effective["cdr"]),
        "--billing", str(effective["billing"]),
        "--source-table", str(effective["source_table"]),
        "--cohort", str(effective.get("cohort_def", "")),
        "--person-mod", str(effective["person_mod"]),
        "--vocab-size", str(effective["vocab_size"]),
        "--min-df", str(effective["min_df"]),
        "--min-patient-count", str(effective["min_patient_count"]),
        "--doc-spec", str(effective["doc_unit"]),
        "--doc-min-length", str(effective["doc_min_length"]),
        "--K", str(effective["K"]),
        "--max-iter", str(effective["max_iter"]),
        "--save-interval", str(effective["save_interval"]),
        "--subsampling-rate", str(effective["subsampling_rate"]),
        "--tau0", str(effective["tau0"]),
        "--kappa", str(effective["kappa"]),
        "--out-dir", str(out_dir),
    ]
    if effective.get("cache_uri"):
        common.extend(["--cache-uri", str(effective["cache_uri"])])
    if effective.get("random_seed") is not None:
        common.extend(["--random-seed", str(effective["random_seed"])])

    if model_class == "lda":
        # Add any LDA-specific args here (kept as-is; existing build_fit_args
        # contents move here unchanged).
        return common  # plus LDA extras
    if model_class == "stm":
        return common + [
            "--covariate-formula", str(effective["covariate_formula"]),
            "--categorical-cols", ",".join(effective.get("categorical_cols", [])),
            "--continuous-cols", ",".join(effective.get("continuous_cols", [])),
            "--sigma-init", str(effective.get("sigma_init", 1.0)),
            "--sigma-ridge", str(effective.get("sigma_ridge", 1e-6)),
            "--lbfgs-max-iter", str(effective.get("lbfgs_max_iter", 50)),
            "--lbfgs-tol", str(effective.get("lbfgs_tol", 1e-4)),
        ]
    raise ValueError(f"unknown model_class: {model_class!r}")
```

The implementer reconciles this with the existing `build_fit_args` body — the goal is "common args identical, then dispatch on model_class for the extras." Tests in step 2 pin the shape.

- [ ] **Step 7: Update the fit dispatch in `main`**

In `main()`, find where the LDA driver path is hard-coded and replace with `build_fit_driver_path(effective)`. The spark-submit invocation otherwise stays the same.

- [ ] **Step 8: Run tests to verify pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```

Expected: ALL PASS (including existing LDA tests and the new STM dispatch tests).

- [ ] **Step 9: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): STM dispatch in run_experiment.py — validation + fit driver routing"
```

---

### Task 6: `experiments/defaults/_base.yaml` — STM defaults

**Files:**
- Modify: `experiments/defaults/_base.yaml`
- Test: ad hoc (frontmatter loading by the wrapper is exercised by Task 5's tests)

- [ ] **Step 1: Add STM keys to `_base.yaml`**

Open `experiments/defaults/_base.yaml` and append (after the existing LDA defaults):

```yaml
# --- STM (prevalence-only) defaults ---
# model_class: stm in the experiment frontmatter triggers the STM path.
# covariate_formula, categorical_cols, continuous_cols are REQUIRED in
# frontmatter when model_class: stm; no sensible default exists.
#
# Optional STM hyperparams with defaults that match OnlineSTM's constructor:
sigma_init: 1.0
sigma_ridge: 1.0e-6
lbfgs_max_iter: 50
lbfgs_tol: 1.0e-4
```

These defaults are read by `run_experiment.py`'s effective-config builder and passed through `build_fit_args` to the STM driver. Existing LDA experiments are unaffected because the LDA driver ignores these flags.

- [ ] **Step 2: Verify the defaults file still loads**

```bash
poetry run python -c "import yaml; print(yaml.safe_load(open('experiments/defaults/_base.yaml')))"
```

Expected: dict containing the new keys.

- [ ] **Step 3: Commit**

```bash
git add experiments/defaults/_base.yaml
git commit -m "feat(experiments): STM hyperparameter defaults in _base.yaml"
```

---

### Task 7: `make build-covariates EXP=N` target

**Files:**
- Modify: `analysis/cloud/Makefile`
- Test: ad hoc

- [ ] **Step 1: Inspect the existing experiment-tracking targets**

```bash
grep -n "next-exp\|exp ID\|build-dashboard-exp\|eval-exp" analysis/cloud/Makefile
```

These dispatch through `scripts/run_experiment.py`. STM doesn't need new top-level targets for fit / eval / build-dashboard — the wrapper handles model_class dispatch internally — but it does benefit from an explicit covariate-rebuild path for users iterating on formulas.

- [ ] **Step 2: Add the `build-covariates` target**

In `analysis/cloud/Makefile`, append:

```makefile
# Rebuild the covariate cache for an experiment without re-running the fit.
# Useful when iterating on covariate_formula: drops the cached covariates,
# then a subsequent `make exp ID=N` rebuilds from the new formula.
build-covariates:
ifndef EXP
	$(error EXP=NNNN required, e.g. make build-covariates EXP=0002)
endif
	@cd $(REPO_ROOT) && poetry run python scripts/run_experiment.py \
		--mode build-covariates --id $(EXP)
```

Add `build-covariates` to the `.PHONY` target list.

- [ ] **Step 3: Add `--mode build-covariates` handling in `run_experiment.py`**

In `scripts/run_experiment.py`, add a `build-covariates` mode that:
1. Loads the experiment's frontmatter + effective config.
2. Asserts model_class == "stm".
3. Constructs the covariates cache key.
4. Deletes the cache entry (if present).
5. Triggers a `load_or_build_covariates` rebuild via a small entry point in the driver — for v1 this is a small driver invocation that just rebuilds the cache without running the fit (e.g., a `--rebuild-covariates-only` flag on `stm_bigquery_cloud.py`).

For implementation simplicity in v1, the build-covariates Make target can shell to a tiny standalone script: `analysis/cloud/build_stm_covariates.py` that runs the load → build → cache write-through path. This avoids threading a special mode into the full STM driver.

Pick whichever path is cleaner — both are acceptable. The test below pins the user-facing behavior.

- [ ] **Step 4: Add a smoke test for the target**

In `scripts/tests/test_run_experiment.py`, add:

```python
class TestBuildCovariatesMode:
    def test_mode_dispatch_recognized(self):
        from run_experiment import dispatch_mode
        # Mode should resolve without crashing — the actual cluster work
        # is exercised by integration smoke (Task 10).
        result = dispatch_mode("build-covariates", id="0099", dry_run=True)
        # On dry_run, should print intended action and exit 0.
        assert result == 0
```

The implementer wires `dispatch_mode` to whichever entry point Step 3 chose.

- [ ] **Step 5: Run tests + smoke**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
make build-covariates EXP=NONEXISTENT 2>&1 | head -5
```

Expected: pytest PASS; the make invocation fails cleanly because EXP=NONEXISTENT doesn't exist (not because the target is malformed).

- [ ] **Step 6: Commit**

```bash
git add analysis/cloud/Makefile scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(cloud/make): build-covariates target for STM formula iteration"
```

---

## Phase 6 — Dashboard adapter

### Task 8: `adapt_stm` in `charmpheno.export.dashboard`

**Files:**
- Modify: `charmpheno/charmpheno/export/dashboard.py`
- Create: `charmpheno/tests/test_export_dashboard_stm.py`

- [ ] **Step 1: Inspect the existing dashboard adapter pattern**

```bash
grep -n "adapt_\|write_model_and_vocab_bundles\|model_class" charmpheno/charmpheno/export/dashboard.py
```

The LDA path is the existing `write_model_and_vocab_bundles`. STM needs a sibling that takes the same write-time inputs plus Γ̂ and covariate_names.

- [ ] **Step 2: Write the adapter test**

Create `charmpheno/tests/test_export_dashboard_stm.py`:

```python
"""Tests for the STM dashboard bundle adapter (adapt_stm).

Surfaces Γ̂ + covariate names in the bundle so the dashboard can
display per-topic per-covariate effects. β / α-equivalent surfacing
re-uses the existing LDA adapter; only the Γ̂ piece is new.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class TestAdaptSTM:
    def test_writes_gamma_json_with_covariate_labels(self, tmp_path: Path):
        from charmpheno.charmpheno.export.dashboard import adapt_stm

        K, P = 4, 3
        Gamma = np.array([
            [+1.2, -0.3, 0.0, +0.5],
            [-0.4, +0.8, -1.1, +0.2],
            [+0.0, +0.1, +0.7, -0.3],
        ])
        covariate_names = ["Intercept", "sex[T.M]", "age"]
        out_dir = tmp_path / "bundle"
        out_dir.mkdir()
        adapt_stm(
            out_dir=out_dir,
            Gamma=Gamma,
            covariate_names=covariate_names,
            K=K, P=P,
        )

        gamma_json = json.loads((out_dir / "covariate_effects.json").read_text())
        # Schema: list of {"covariate": str, "per_topic": [float, ...]} entries.
        assert len(gamma_json) == P
        assert all("covariate" in row and "per_topic" in row for row in gamma_json)
        assert [row["covariate"] for row in gamma_json] == covariate_names
        assert all(len(row["per_topic"]) == K for row in gamma_json)
        # Check first row values match Gamma row 0.
        np.testing.assert_allclose(gamma_json[0]["per_topic"], Gamma[0])

    def test_rejects_size_mismatch(self, tmp_path: Path):
        from charmpheno.charmpheno.export.dashboard import adapt_stm
        with pytest.raises(ValueError, match="covariate_names|shape"):
            adapt_stm(
                out_dir=tmp_path,
                Gamma=np.zeros((3, 4)),
                covariate_names=["a", "b"],  # wrong length: 2 vs P=3
                K=4, P=3,
            )
```

- [ ] **Step 3: Run to verify failure**

```bash
poetry run pytest charmpheno/tests/test_export_dashboard_stm.py -v
```

Expected: FAIL.

- [ ] **Step 4: Implement `adapt_stm`**

Append to `charmpheno/charmpheno/export/dashboard.py`:

```python
def adapt_stm(
    *,
    out_dir: Path,
    Gamma: np.ndarray,
    covariate_names: list[str],
    K: int,
    P: int,
) -> None:
    """Write STM-specific bundle artifact: per-covariate effect matrix Γ̂.

    Schema for covariate_effects.json:
        [{"covariate": "<name>", "per_topic": [γ_0, γ_1, ..., γ_{K-1}]}, ...]

    One row per covariate (length P); each row carries K topic-effect values.
    Companion bundle artifacts (vocab, β, α-equivalent) come from the existing
    write_model_and_vocab_bundles path; this function only adds the Γ piece.
    """
    if Gamma.shape != (P, K):
        raise ValueError(
            f"adapt_stm: Gamma shape mismatch — got {Gamma.shape}, expected ({P}, {K})"
        )
    if len(covariate_names) != P:
        raise ValueError(
            f"adapt_stm: covariate_names length {len(covariate_names)} != P={P}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {"covariate": name, "per_topic": Gamma[p].tolist()}
        for p, name in enumerate(covariate_names)
    ]
    (out_dir / "covariate_effects.json").write_text(json.dumps(payload, indent=2))
```

Add `import json` to the file's imports if not already present.

- [ ] **Step 5: Run tests to verify pass**

```bash
poetry run pytest charmpheno/tests/test_export_dashboard_stm.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add charmpheno/charmpheno/export/dashboard.py charmpheno/tests/test_export_dashboard_stm.py
git commit -m "feat(export): adapt_stm writes Gamma_hat + covariate labels into dashboard bundle"
```

---

### Task 9: `build_dashboard_cloud.py` — STM dispatch

**Files:**
- Modify: `analysis/cloud/build_dashboard_cloud.py`

- [ ] **Step 1: Inspect the existing build_dashboard_cloud structure**

```bash
grep -n "model_class\|write_model_and_vocab_bundles\|adapt_" analysis/cloud/build_dashboard_cloud.py | head -20
```

The build dashboard driver currently assumes LDA — calls `write_model_and_vocab_bundles` with LDA-shaped inputs. STM needs a parallel branch that:
1. Reads `model_class` from `VIResult.metadata`.
2. If LDA: existing path unchanged.
3. If STM: calls existing `write_model_and_vocab_bundles` for the β / vocab / α-equivalent pieces, plus `adapt_stm` for Γ̂ + covariate_names.

- [ ] **Step 2: Add STM branch**

Locate the section where `write_model_and_vocab_bundles` is called (~line 260 per session memory) and wrap with a dispatch:

```python
model_class = result.metadata.get("model_class", "lda")

if model_class == "lda":
    v_disp = write_model_and_vocab_bundles(
        out_dir=out_dir,
        beta=export.beta, alpha=export.alpha,
        vocab_ids=vocab_list,
        descriptions=descriptions, domains=domains,
        code_marginals=stats.code_marginals,
        code_doc_counts=stats.code_doc_counts,
        top_n=args.vocab_top_n,
        min_doc_count=args.min_doc_count,
    )

elif model_class == "stm":
    # β / vocab / α-equivalent reuse the LDA writer with an α-equivalent
    # computed as the corpus-empirical-mean of softmax(Γ x_d):
    #     α_eq = (1/D) Σ_d softmax(Γ x_d)
    # Approximated here as softmax(Γ x̄) where x̄ is the mean covariate vector.
    Gamma = result.global_params["Gamma"]
    covariate_manifest = result.metadata["covariate_manifest"]
    covariate_names = covariate_manifest["covariate_names"]

    # Compute α-equivalent from mean covariate (cheap; D-accurate version is a v1.x improvement).
    # Note: x̄ is the mean of the per-doc covariates used at fit time. The
    # build driver reconstructs this from the corpus + covariate cache; for
    # v1 we use the prior mean assuming x = mean-x is a reasonable LDA-ish
    # baseline. A faithful α_eq would average softmax(Γ x_d) over the actual
    # docs — tracked as a v1.x precision improvement.
    P = Gamma.shape[0]
    # For the v1 dashboard, use Γᵀ at x = 0 (intercept-only):
    #   α_eq_k = softmax(Γ[0])_k   if "Intercept" is the first covariate
    # Adjust if the formula dropped the intercept.
    intercept_idx = next(
        (i for i, n in enumerate(covariate_names) if "Intercept" in str(n)),
        None,
    )
    if intercept_idx is not None:
        eta_bar = Gamma[intercept_idx]
    else:
        eta_bar = np.zeros(Gamma.shape[1])
    alpha_eq = np.exp(eta_bar - eta_bar.max())
    alpha_eq = alpha_eq / alpha_eq.sum()

    v_disp = write_model_and_vocab_bundles(
        out_dir=out_dir,
        beta=export.beta, alpha=alpha_eq,
        vocab_ids=vocab_list,
        descriptions=descriptions, domains=domains,
        code_marginals=stats.code_marginals,
        code_doc_counts=stats.code_doc_counts,
        top_n=args.vocab_top_n,
        min_doc_count=args.min_doc_count,
    )
    # Plus the STM-specific Γ̂ artifact.
    from charmpheno.charmpheno.export.dashboard import adapt_stm
    adapt_stm(
        out_dir=out_dir, Gamma=Gamma,
        covariate_names=covariate_names,
        K=Gamma.shape[1], P=Gamma.shape[0],
    )

else:
    raise ValueError(f"build_dashboard_cloud: unsupported model_class {model_class!r}")
```

- [ ] **Step 3: Quick CLI smoke**

```bash
poetry run python analysis/cloud/build_dashboard_cloud.py --help
```

Expected: no regressions vs current help; STM-aware dispatch is silent (decided by checkpoint metadata, not CLI flag).

- [ ] **Step 4: Commit**

```bash
git add analysis/cloud/build_dashboard_cloud.py
git commit -m "feat(cloud): build_dashboard_cloud.py dispatches LDA vs STM adapter on model_class"
```

---

### Task 10: End-to-end synthetic smoke test

**Files:**
- Create: smoke test (location per existing convention — likely `charmpheno/tests/test_stm_smoke.py` or `analysis/cloud/tests/test_stm_smoke.py`)

- [ ] **Step 1: Write the smoke test**

Create the smoke test file:

```python
"""STM end-to-end smoke: synthetic data through covariates build, fit,
and dashboard bundle.

Marked slow; runs locally without BigQuery (uses the synthetic corpus
helper from spark-vi tests as a starting point).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
formulaic = pytest.importorskip("formulaic")

from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def spark():
    return (
        SparkSession.builder.master("local[2]").appName("test-stm-smoke")
        .config("spark.ui.enabled", "false").getOrCreate()
    )


@pytest.mark.slow
def test_end_to_end_synthetic_stm(spark, tmp_path: Path):
    """Build covariates → fit STM → assert bundle structure.

    Uses synthetic corpus + person tables generated locally. Does not
    exercise BigQuery; the covariates build path is exercised against
    in-memory Spark DataFrames.
    """
    from charmpheno.omop.covariates import build_patient_covariate_df
    from spark_vi.mllib.topic.stm import StreamingSTM

    # Synthetic person table.
    person_pdf = pd.DataFrame({
        "person_id": list(range(1, 51)),
        "sex":       ["M", "F"] * 25,
        "age":       np.linspace(20, 70, 50).tolist(),
    })
    person_df = spark.createDataFrame(person_pdf)

    # Synthetic BOW corpus (replicate from spark-vi's synthetic_corpus helper).
    K, V, doc_len = 4, 30, 60
    rng = np.random.default_rng(42)
    beta = rng.dirichlet(np.full(V, 0.1), size=K)
    bow_rows = []
    for pid in range(1, 51):
        x = np.array([1.0, 1.0 if person_pdf.sex.iloc[pid-1] == "M" else 0.0,
                      person_pdf.age.iloc[pid-1] / 50.0])
        eta = rng.normal(scale=0.3, size=K)
        theta = np.exp(eta - eta.max()); theta /= theta.sum()
        z = rng.choice(K, size=doc_len, p=theta)
        w = np.array([rng.choice(V, p=beta[zi]) for zi in z])
        unique, counts = np.unique(w, return_counts=True)
        from pyspark.ml.linalg import SparseVector
        bow_rows.append({
            "person_id": pid, "doc_id": pid,
            "features": SparseVector(V, unique.tolist(), counts.astype(float).tolist()),
        })
    bow_df = spark.createDataFrame(pd.DataFrame(bow_rows))

    # Build covariates from formula.
    cov_df, model_spec, names = build_patient_covariate_df(
        person_df, covariate_formula="~ C(sex) + age",
        categorical_cols=["sex"], continuous_cols=["age"],
    )

    # Join + fit.
    joined = bow_df.join(cov_df, on="person_id")
    est = StreamingSTM(
        K=K, features_col="features",
        covariates_col="covariates", covariate_names=names,
        random_seed=42,
    )
    model = est.fit(joined, max_iter=10, subsampling_rate=1.0,
                    tau0=64.0, kappa=0.7, save_interval=5)

    # Inspect bundle directly via adapt_stm.
    from charmpheno.charmpheno.export.dashboard import adapt_stm
    out = tmp_path / "bundle"
    out.mkdir()
    Gamma = model.global_params["Gamma"]
    adapt_stm(out_dir=out, Gamma=Gamma, covariate_names=names,
              K=Gamma.shape[1], P=Gamma.shape[0])
    cov_json = json.loads((out / "covariate_effects.json").read_text())
    assert len(cov_json) == len(names)
    assert all("per_topic" in row for row in cov_json)
```

- [ ] **Step 2: Run the smoke test**

```bash
poetry run pytest <path>/test_stm_smoke.py -v -m slow
```

Expected: PASS. If a piece fails it surfaces a real integration gap (formula validation, join shape, adapter output structure).

- [ ] **Step 3: Commit**

```bash
git add <path>/test_stm_smoke.py
git commit -m "test(stm): end-to-end synthetic smoke — covariates build, fit, bundle"
```

---

## Verification

After all 10 tasks land:

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
# Charmpheno unit tests (existing + new).
poetry run pytest charmpheno/tests/ -v -m "not slow and not cluster"
# Scripts tests.
poetry run pytest scripts/tests/ -v
# End-to-end synthetic smoke.
poetry run pytest -v -m slow -k stm_smoke
```

All should pass. A real cluster run is the final validation gate:

```bash
# On a Dataproc cluster with experiment 0002 set up:
make build-covariates EXP=0002
make exp ID=0002    # routes through stm_bigquery_cloud.py
make eval-exp ID=0002
make build-dashboard-exp ID=0002
```

The cluster run is the load-bearing validation that the BigQuery → covariates cache → fit → eval → dashboard pipeline composes end-to-end. Document the result in [REVIEW_LOG.md](../../REVIEW_LOG.md) with a walkthrough.

## Out of Scope / Follow-Ups

This plan closes phases 4–6 of the [STM design spec](../specs/2026-05-29-stm-prevalence-design.md). The following are deferred:

- **Faithful α-equivalent for the dashboard bundle.** The v1 build driver uses `softmax(Γ[intercept_row])` as a stand-in. A more faithful version computes `(1/D) Σ_d softmax(Γ x_d)` over the corpus's empirical covariate distribution. Tracked as a small follow-up; affects only dashboard cosmetics (α-equivalent is used for the "default topic proportion" display in the existing LDA dashboard).
- **Γ visualization in the dashboard frontend.** The bundle now contains `covariate_effects.json`; the React/D3 surface that renders it is its own design and brainstorm.
- **Per-experiment formula iteration ergonomics.** `make build-covariates EXP=N` is the v1 path; a richer "swap formula in frontmatter, auto-detect cache miss, rebuild silently on next `make exp`" is a v1.x convenience.
- **HDP-STM.** Out of scope for this entire arc; tracked in the design spec.
- **Cross-experiment covariate cache sharing.** Two experiments with the same formula + same cohort produce identical caches today; deduplication (e.g., content-addressable cache layout) is a storage optimization for v1.x.
- **Patient-table covariate column expansion.** v1 driver expects covariate columns to come from a `load_person_table` call; richer covariates that require joins (condition history, observation values) are pulled into person_df upstream by the implementer. A future helper could abstract the multi-table covariate query.
