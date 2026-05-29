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
