from charmpheno.export.covariate_schema import build_covariate_schema


def _base_inputs():
    return dict(
        covariate_names=[
            "Intercept",
            "C(source_cohort)[T.dementia]",
            "C(sex)[T.M]",
            "age",
        ],
        continuous_cols=["age"],
        categorical_levels={
            "source_cohort": {"levels": ["cancer", "dementia"], "reference": "cancer"},
            "sex": {"levels": ["F", "M"], "reference": "F"},
        },
        level_counts={
            "C(source_cohort)[T.dementia]": 5000,
            "C(sex)[T.M]": 4000,
        },
        continuous_stats={"age": (40.0, 65.0, 90.0)},
        k=20,
    )


def test_builds_controls_and_recipes_aligned_to_gamma():
    s = build_covariate_schema(**_base_inputs())
    # design_columns aligned with covariate_names order
    assert [d["name"] for d in s["design_columns"]] == _base_inputs()["covariate_names"]
    kinds = [d["recipe"]["kind"] for d in s["design_columns"]]
    assert kinds == ["intercept", "dummy", "dummy", "main"]
    # controls: one per variable; continuous range/default from percentiles
    age = next(c for c in s["controls"] if c["name"] == "age")
    assert age["type"] == "continuous" and age["range"] == [40.0, 90.0] and age["default"] == 65.0
    sc = next(c for c in s["controls"] if c["name"] == "source_cohort")
    assert sc["type"] == "categorical" and sc["reference"] == "cancer"
    assert sc["levels"] == ["cancer", "dementia"]
    assert s["unsupported"] == []
    assert s["k"] == 20


def test_suppresses_categorical_level_below_k():
    inp = _base_inputs()
    inp["level_counts"]["C(sex)[T.M]"] = 3   # below k=20
    s = build_covariate_schema(**inp)
    sex = next(c for c in s["controls"] if c["name"] == "sex")
    # the under-k level M is omitted; reference F always stays
    assert "M" not in sex["levels"] and "F" in sex["levels"]


def test_unparseable_design_column_goes_to_unsupported():
    inp = _base_inputs()
    inp["covariate_names"] = inp["covariate_names"] + ["weird_basis_col_3"]
    inp["continuous_cols"] = ["age"]   # weird col is neither continuous nor C(...)
    s = build_covariate_schema(**inp)
    assert "weird_basis_col_3" in s["unsupported"]


def test_interaction_parses_to_kind_interaction():
    """A standard interaction column parses correctly and is NOT in unsupported."""
    inp = _base_inputs()
    inp["covariate_names"] = inp["covariate_names"] + ["age:C(sex)[T.M]"]
    s = build_covariate_schema(**inp)
    # find the design_column entry for the interaction
    dc = next(d for d in s["design_columns"] if d["name"] == "age:C(sex)[T.M]")
    assert dc["recipe"] == {
        "kind": "interaction",
        "factors": [
            {"kind": "main", "var": "age"},
            {"kind": "dummy", "var": "sex", "level": "M"},
        ],
    }
    assert "age:C(sex)[T.M]" not in s["unsupported"]


def test_interaction_with_colon_in_level():
    """A dummy whose level contains a colon inside an interaction must parse correctly.

    The naive name.split(':') incorrectly splits 'C(dx)[T.A:B]:age' into three
    pieces; the bracket-aware split must treat the colon inside [...] as opaque.
    """
    inp = _base_inputs()
    inp["covariate_names"] = inp["covariate_names"] + ["C(dx)[T.A:B]:age"]
    # categorical_levels entry for dx so build_covariate_schema does not crash on
    # the controls loop (dx is not referenced there unless listed, so it is safe to
    # omit — but adding it makes the fixture self-consistent)
    inp["categorical_levels"]["dx"] = {"levels": ["A:B", "ref"], "reference": "ref"}
    inp["level_counts"]["C(dx)[T.A:B]"] = 500
    s = build_covariate_schema(**inp)
    dc = next(d for d in s["design_columns"] if d["name"] == "C(dx)[T.A:B]:age")
    assert dc["recipe"] == {
        "kind": "interaction",
        "factors": [
            {"kind": "dummy", "var": "dx", "level": "A:B"},
            {"kind": "main", "var": "age"},
        ],
    }
    assert "C(dx)[T.A:B]:age" not in s["unsupported"]
