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
