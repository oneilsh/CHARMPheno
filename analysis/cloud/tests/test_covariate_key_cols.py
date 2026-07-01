"""Guard + predicate for the covariate sidecar's key columns.

source_cohort plays ONE role now: a per-document gating/label key. It was
moved out of the covariate formula "a long while ago"; a covariate vector that
varied by source_cohort would make a person's per-(person, cohort) sidecar rows
disagree, corrupting both the corpus join and the gated dashboard prevalence.

Two units guard that invariant, shared by both producers (the fit driver and
the standalone covariate builder) so they cannot drift:

- validate_label_not_covariate: raises if source_cohort appears in the formula.
- covariate_key_cols: (person_id, source_cohort) when the fit is gated on
  source_cohort — so the sidecar carries each document's group for the gated
  dashboard prevalence — else (person_id,).

This is the fix for the bug where a gated-but-not-covariate fit (e.g. exp 0027,
covariate_formula '~ C(sex) + age', group_var source_cohort) persisted a
(person_id, covariates) sidecar the gated consumer could not group by, so
build_dashboard fell back to the intercept stand-in and dropped gating.json.
"""
import sys
from pathlib import Path

import pytest

_CLOUD = str(Path(__file__).resolve().parent.parent)
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)

from _covariates_load import (  # noqa: E402
    covariate_key_cols,
    validate_label_not_covariate,
)


def test_gated_keys_on_person_and_source_cohort():
    assert covariate_key_cols(gated=True) == ["person_id", "source_cohort"]


def test_ungated_keys_on_person_only():
    assert covariate_key_cols(gated=False) == ["person_id"]


def test_clean_formula_passes():
    # exp 0027's actual formula: sex categorical, age continuous, no label.
    validate_label_not_covariate(["sex"], ["age"])  # must not raise


def test_label_in_categorical_is_rejected():
    with pytest.raises(ValueError, match="source_cohort"):
        validate_label_not_covariate(["source_cohort", "sex"], ["age"])


def test_label_in_continuous_is_rejected():
    with pytest.raises(ValueError, match="source_cohort"):
        validate_label_not_covariate(["sex"], ["source_cohort"])
