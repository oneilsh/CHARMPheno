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


class TestCorpusMeanProportionsFromCovariateDF:
    """corpus_mean_proportions_from_covariate_df: drives the dashboard's
    faithful α-equivalent from the (person_id, covariates) sidecar, dropping
    person_id and delegating the distributed math to spark-vi's RDD helper.
    """

    def test_matches_numpy_oracle_and_ignores_person_id(self, spark):
        from pyspark.ml.linalg import Vectors
        from charmpheno.omop.covariates import (
            corpus_mean_proportions_from_covariate_df,
        )
        from spark_vi.models.topic.stm import corpus_mean_topic_proportions

        rng = np.random.default_rng(21)
        P, K = 3, 4
        Gamma = rng.normal(size=(P, K))
        X = rng.normal(size=(6, P))
        rows = [(1000 + d, Vectors.dense(X[d].tolist())) for d in range(len(X))]
        cov_df = spark.createDataFrame(rows, ["person_id", "covariates"])

        result = corpus_mean_proportions_from_covariate_df(cov_df, Gamma)

        np.testing.assert_allclose(
            result, corpus_mean_topic_proportions(Gamma, X), rtol=1e-9
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


class TestCompositeKeyCovariates:
    def test_composite_key_one_row_per_person_cohort(self, spark):
        import pandas as pd
        from charmpheno.omop.covariates import build_patient_covariate_df
        # person 1 comorbid (two cohorts), person 2 cancer only.
        pdf = pd.DataFrame({
            "person_id":     [1, 1, 2],
            "source_cohort": ["cancer", "dementia", "cancer"],
            "sex":           ["M", "M", "F"],
            "age":           [60.0, 60.0, 70.0],
        })
        person_df = spark.createDataFrame(pdf)
        cov_df, spec, names = build_patient_covariate_df(
            person_df,
            covariate_formula="~ C(source_cohort) + C(sex) + age",
            categorical_cols=["source_cohort", "sex"],
            continuous_cols=["age"],
            key_cols=["person_id", "source_cohort"],
        )
        assert set(cov_df.columns) == {"person_id", "source_cohort", "covariates"}
        keys = {(r["person_id"], r["source_cohort"]) for r in cov_df.collect()}
        assert keys == {(1, "cancer"), (1, "dementia"), (2, "cancer")}

    def test_label_key_absent_from_formula_is_carried(self, spark):
        """The gated-STM case (exp 0027): source_cohort is a pure label key,
        NOT a formula covariate (formula '~ C(sex) + age'). The sidecar must
        still carry it as a key column so the gated dashboard prevalence can
        group by it, a comorbid person yields one row per group, and — since
        the covariate vector does not depend on source_cohort — those rows'
        vectors are identical."""
        import numpy as np
        import pandas as pd
        from charmpheno.omop.covariates import build_patient_covariate_df
        # person 1 comorbid (two groups, same sex/age), person 2 cancer only.
        pdf = pd.DataFrame({
            "person_id":     [1, 1, 2],
            "source_cohort": ["cancer", "dementia", "cancer"],
            "sex":           ["M", "M", "F"],
            "age":           [60.0, 60.0, 70.0],
        })
        cov_df, spec, names = build_patient_covariate_df(
            spark.createDataFrame(pdf),
            covariate_formula="~ C(sex) + age",   # source_cohort NOT in formula
            categorical_cols=["sex"],
            continuous_cols=["age"],
            key_cols=["person_id", "source_cohort"],
        )
        assert set(cov_df.columns) == {"person_id", "source_cohort", "covariates"}
        rows = {(r["person_id"], r["source_cohort"]): np.asarray(r["covariates"])
                for r in cov_df.collect()}
        assert set(rows) == {(1, "cancer"), (1, "dementia"), (2, "cancer")}
        # source_cohort is not a covariate, so person 1's two rows are identical.
        np.testing.assert_allclose(rows[(1, "cancer")], rows[(1, "dementia")])

    def test_default_key_cols_unchanged(self, spark):
        import pandas as pd
        from charmpheno.omop.covariates import build_patient_covariate_df
        pdf = pd.DataFrame({"person_id": [1, 2], "age": [60.0, 70.0]})
        cov_df, _, _ = build_patient_covariate_df(
            spark.createDataFrame(pdf),
            covariate_formula="~ age",
            categorical_cols=[], continuous_cols=["age"],
        )
        assert set(cov_df.columns) == {"person_id", "covariates"}
