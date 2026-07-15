"""Tests for load_or_build_covariates: cache-hit fast path + cache-miss build-through."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "cloud"))

pyspark = pytest.importorskip("pyspark")
formulaic = pytest.importorskip("formulaic")


class TestLoadOrBuildCovariates:
    def test_miss_then_hit_roundtrips(self, spark, tmp_path):
        from _covariates_load import load_or_build_covariates

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
        from _covariates_load import load_or_build_covariates
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
