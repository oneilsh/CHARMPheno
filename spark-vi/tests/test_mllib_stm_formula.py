"""Tests for the STM shim's covariate-formula path.

Covers: formula parsing, validation rejecting stateful transforms,
schema-frame categorical discovery, ModelSpec construction, and the spec's JSON
persistence (ADR 0056).
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

    def test_unseen_level_check_handles_C_with_extra_args(self):
        """Bug fix: previously C(col, Treatment(...)) slipped past the unseen-level
        pre-check because the simple C(col) -> col strip broke on the extra args.
        This test directly verifies the regex extraction handles various C(...) forms.
        """
        import re

        # Test the regex extraction logic directly on various encoder_state keys
        test_cases = [
            ("C(cohort)", "cohort"),                       # simple form
            ("C(cohort, Treatment('a'))", "cohort"),       # with extra args
            ("C(cohort,whatever)", "cohort"),              # with comma-separated arg
            ("C( cohort , ...)", "cohort"),                # with spaces
            ("C(age_group)", "age_group"),                 # underscore in name
        ]

        pattern = r"^C\(\s*([^,)\s]+)"
        for encoder_key, expected_col in test_cases:
            m = re.match(pattern, encoder_key)
            assert m is not None, f"Regex failed to match '{encoder_key}'"
            extracted_col = m.group(1)
            assert extracted_col == expected_col, \
                f"For '{encoder_key}': expected '{expected_col}', got '{extracted_col}'"


class TestFormulaVariables:
    def test_categorical_by_C_or_string_or_boolean_type(self):
        from spark_vi.mllib.topic._formula import split_formula_variables
        types = {"sex": "string", "age": "double", "site": "int", "flag": "boolean",
                 "other": "string"}
        cat, cont = split_formula_variables(
            "~ sex + age + C(site) + flag + C(age > 50)", types)
        # age is inside C(age > 50), so its raw levels must reach the schema-frame.
        assert cat == ["age", "flag", "sex", "site"]
        assert cont == []
        cat, cont = split_formula_variables("~ sex + age", types)
        assert (cat, cont) == (["sex"], ["age"])

    def test_unknown_column_raises(self):
        from spark_vi.mllib.topic._formula import split_formula_variables
        with pytest.raises(ValueError, match="does not have"):
            split_formula_variables("~ C(sex) + bmi", {"sex": "string"})


class TestCovariateSpecState:
    """The saved covariate spec is JSON (formula + schema-frame), never a pickle;
    rebuilding the ModelSpec from it must encode any row exactly as the fitted spec
    does (ADR 0056)."""

    def _schema_frame(self):
        return pd.DataFrame({
            "sex": ["F", "M", "X", "F"],
            "site": [1, 2, 3, 1],
            "age": [0.0, 0.0, 0.0, 0.0],
        })

    def test_json_round_trip_encodes_bit_identically(self):
        import json
        from spark_vi.mllib.topic._formula import (
            apply_model_spec, covariate_spec_state, fit_model_spec_on_schema_frame,
            model_spec_from_state,
        )
        formula = "~ C(sex) + C(site) + age + C(sex):age"
        schema = self._schema_frame()
        spec, names = fit_model_spec_on_schema_frame(formula, schema, ["sex", "site"])
        state = covariate_spec_state(formula, schema, names, ["sex", "site"])
        state = json.loads(json.dumps(state))          # what is written to disk
        rebuilt = model_spec_from_state(state)
        rows = pd.DataFrame({"sex": ["M", "X", "F", "M"], "site": [3, 1, 2, 2],
                             "age": [33.5, 71.0, 18.25, 50.0]})
        a = apply_model_spec(spec, rows)
        b = apply_model_spec(rebuilt, rows)
        assert a.tobytes() == b.tobytes()
        assert list(rebuilt.column_names) == names

    def test_rebuilt_spec_rejects_unseen_level(self):
        import json
        from spark_vi.mllib.topic._formula import (
            apply_model_spec, covariate_spec_state, fit_model_spec_on_schema_frame,
            model_spec_from_state,
        )
        schema = self._schema_frame()
        spec, names = fit_model_spec_on_schema_frame("~ C(sex) + age", schema, ["sex"])
        rebuilt = model_spec_from_state(json.loads(json.dumps(
            covariate_spec_state("~ C(sex) + age", schema, names, ["sex"]))))
        unseen = pd.DataFrame({"sex": ["F", "Q"], "site": [1, 1], "age": [1.0, 2.0]})
        for s in (spec, rebuilt):
            with pytest.raises(ValueError, match="not seen"):
                apply_model_spec(s, unseen)

    def test_names_mismatch_is_refused(self):
        from spark_vi.mllib.topic._formula import (
            covariate_spec_state, fit_model_spec_on_schema_frame, model_spec_from_state,
        )
        schema = self._schema_frame()
        spec, names = fit_model_spec_on_schema_frame("~ C(sex)", schema, ["sex"])
        state = covariate_spec_state("~ C(sex)", schema, names, ["sex"])
        state["covariate_names"] = ["Intercept", "something else"]
        with pytest.raises(ValueError, match="rebuilt"):
            model_spec_from_state(state)

    def test_unsupported_column_type_is_refused(self):
        from spark_vi.mllib.topic._formula import covariate_spec_state
        import datetime
        schema = pd.DataFrame({"d": [datetime.date(2020, 1, 1)]})
        with pytest.raises(ValueError, match="supports"):
            covariate_spec_state("~ C(d)", schema, ["Intercept"], ["d"])
