"""Tests for the formula path of StreamingSTM (Path B).

Covers: formula parsing, validation rejecting stateful transforms,
schema-frame categorical discovery, ModelSpec construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

formulaic = pytest.importorskip("formulaic")


class TestFormulaValidation:
    def test_rejects_splines(self):
        from spark_vi.mllib.topic._formula import validate_formula
        with pytest.raises(ValueError, match="bs|spline|stateful"):
            validate_formula("~ age + bs(age, df=4)")

    def test_rejects_natural_splines(self):
        from spark_vi.mllib.topic._formula import validate_formula
        with pytest.raises(ValueError, match="ns|spline|stateful"):
            validate_formula("~ ns(age, df=4)")

    def test_rejects_standardization(self):
        from spark_vi.mllib.topic._formula import validate_formula
        with pytest.raises(ValueError, match="scale|center|stateful"):
            validate_formula("~ scale(age) + sex")

    def test_accepts_categoricals(self):
        from spark_vi.mllib.topic._formula import validate_formula
        validate_formula("~ C(cohort) + sex")  # should not raise

    def test_accepts_interactions(self):
        from spark_vi.mllib.topic._formula import validate_formula
        validate_formula("~ age * sex")

    def test_accepts_I_transforms(self):
        from spark_vi.mllib.topic._formula import validate_formula
        validate_formula("~ age + I(age**2)")

    def test_accepts_intercept_dropping(self):
        from spark_vi.mllib.topic._formula import validate_formula
        validate_formula("~ 0 + age + sex")


class TestFitModelSpec:
    def test_categorical_levels_discovered_and_applied(self):
        from spark_vi.mllib.topic._formula import fit_model_spec, apply_model_spec

        covariate_pdf = pd.DataFrame({
            "cohort": ["control", "case", "case", "control", "case"],
            "sex":    ["M", "F", "M", "F", "F"],
            "age":    [25.0, 40.0, 55.0, 30.0, 45.0],
        })
        spec, names = fit_model_spec(
            formula="~ C(cohort) + C(sex) + age",
            covariate_pdf=covariate_pdf,
        )
        applied = apply_model_spec(spec, covariate_pdf)
        # Expected columns: intercept + cohort[T.control] + sex[T.M] + age = 4
        assert applied.shape == (5, 4)
        assert "Intercept" in names or "intercept" in [n.lower() for n in names]

    def test_unseen_level_at_apply_raises(self):
        from spark_vi.mllib.topic._formula import fit_model_spec, apply_model_spec
        train = pd.DataFrame({"cohort": ["a", "b"]})
        spec, _ = fit_model_spec(formula="~ C(cohort)", covariate_pdf=train)
        test = pd.DataFrame({"cohort": ["a", "c"]})   # 'c' unseen
        with pytest.raises(Exception):
            apply_model_spec(spec, test)


class TestStreamingSTMPathBConstruction:
    def test_construct_with_formula_resolves_P_and_names(self):
        from spark_vi.mllib.topic.stm import StreamingSTM
        from spark_vi.mllib.topic._formula import fit_model_spec
        pdf = pd.DataFrame({
            "cohort": ["a", "b", "a", "b"],
            "age":    [25.0, 40.0, 55.0, 30.0],
        })
        spec, names = fit_model_spec("~ C(cohort) + age", pdf)
        est = StreamingSTM(
            K=4,
            covariate_formula="~ C(cohort) + age",
            covariate_df=pdf,
        )
        # Inject ModelSpec resolution (real .fit() will do this with Spark).
        est._resolve_model_spec_from_pandas(pdf)
        assert est.P == len(names)
        assert est.covariate_names == names
