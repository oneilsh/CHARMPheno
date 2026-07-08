"""Unit tests for charmpheno.omop.cohorts.

Data-path tests against a real CDR are deferred to the cluster smoke;
this file covers the validation surface only.
"""
import pytest

from charmpheno.omop.cohorts import (
    SUPPORTED_COHORTS,
    apply_cohort,
)


def test_supported_cohorts_includes_first_cancer_year():
    assert "first_cancer_year" in SUPPORTED_COHORTS


def test_supported_cohorts_includes_first_dementia_year():
    assert "first_dementia_year" in SUPPORTED_COHORTS


def test_apply_cohort_rejects_unknown_name():
    with pytest.raises(ValueError, match="not supported"):
        apply_cohort(
            cond_df=None,           # never reached: validation fires first
            cohort="not_a_cohort",
            spark=None,
            cdr_dataset="proj.ds",
            billing_project="bp",
            date_col="condition_start_date",
        )


def test_supported_cohorts_includes_cancer_or_dementia():
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS
    assert "cancer_or_dementia" in SUPPORTED_COHORTS


def test_cohort_metadata_has_cancer_or_dementia():
    from charmpheno.omop.cohorts import COHORT_METADATA
    assert "cancer_or_dementia" in COHORT_METADATA


def test_combine_cohorts_tags_and_unions_keeping_comorbid(spark):
    from charmpheno.omop.cohorts import _combine_cohorts
    cancer = spark.createDataFrame([(1, 10), (2, 20)], ["person_id", "concept_id"])
    dementia = spark.createDataFrame([(2, 30), (3, 40)], ["person_id", "concept_id"])
    out = _combine_cohorts(cancer, dementia)
    rows = {(r["person_id"], r["source_cohort"]) for r in out.collect()}
    # person 2 is comorbid -> appears under BOTH labels (no dedup).
    assert rows == {(1, "cancer"), (2, "cancer"), (2, "dementia"), (3, "dementia")}
    assert out.count() == 4


def test_supported_cohorts_includes_population_cancer():
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS
    assert "population_cancer" in SUPPORTED_COHORTS


def test_cohort_metadata_has_population_cancer():
    from charmpheno.omop.cohorts import COHORT_METADATA
    md = COHORT_METADATA["population_cancer"]
    assert md["id"] == "population_cancer"
    assert md["label"] and md["description"]


def test_random_event_windows_anchor_on_eligible_events_deterministically(spark):
    """The general-arm window anchors on one of the person's OWN condition-era
    dates whose forward window is fully observed; events without 365d of
    follow-up coverage are ineligible; persons with no eligible event are
    dropped; and the pick is deterministic (hash-based, not F.rand)."""
    import datetime as dt
    from charmpheno.omop.cohorts import _random_event_windows

    win = 365
    # person 1: two events, both with >365d follow-up -> one is chosen, and the
    #           chosen index_date must be one of the two ACTUAL event dates.
    # person 2: single event but only ~100d of follow-up -> ineligible -> dropped.
    # person 3: no observation period row -> dropped.
    cond = spark.createDataFrame(
        [
            (1, dt.date(2011, 3, 1)),
            (1, dt.date(2012, 6, 1)),
            (2, dt.date(2010, 11, 1)),   # op ends 2011-02-01 -> <365d ahead
            (3, dt.date(2010, 1, 1)),
        ],
        ["person_id", "condition_era_start_date"],
    )
    op = spark.createDataFrame(
        [
            (1, dt.date(2010, 1, 1), dt.date(2014, 1, 1)),
            (2, dt.date(2010, 1, 1), dt.date(2011, 2, 1)),   # too short past the event
        ],
        ["person_id", "observation_period_start_date",
         "observation_period_end_date"],
    )
    rows = {
        r["person_id"]: r["index_date"]
        for r in _random_event_windows(
            cond, op, date_col="condition_era_start_date", window_days=win,
        ).collect()
    }

    assert set(rows) == {1}                                   # 2 and 3 dropped
    assert rows[1] in {dt.date(2011, 3, 1), dt.date(2012, 6, 1)}   # a real event date

    # Deterministic: a second call yields the identical anchor.
    rows2 = {
        r["person_id"]: r["index_date"]
        for r in _random_event_windows(
            cond, op, date_col="condition_era_start_date", window_days=win,
        ).collect()
    }
    assert rows == rows2


def test_random_event_windows_enforces_prior_coverage(spark):
    """With prior_obs_days>0 an eligible anchor needs a fully-observed year
    BEFORE it, not just after — symmetric observability for the general arm."""
    import datetime as dt
    from charmpheno.omop.cohorts import _random_event_windows

    # One person, one event on 2015-06-01. Observation period 2015-01-01..2017-01-01:
    # forward year is observed (event+365 <= end), but only ~150d of prior coverage.
    events = spark.createDataFrame(
        [(1, dt.date(2015, 6, 1))], ["person_id", "condition_start_date"],
    )
    op = spark.createDataFrame(
        [(1, dt.date(2015, 1, 1), dt.date(2017, 1, 1))],
        ["person_id", "observation_period_start_date", "observation_period_end_date"],
    )
    # prior_obs_days=0 (current behavior): the anchor is eligible.
    kept = _random_event_windows(
        events, op, date_col="condition_start_date", prior_obs_days=0,
    )
    assert kept.count() == 1
    # prior_obs_days=365: insufficient prior coverage -> dropped.
    dropped = _random_event_windows(
        events, op, date_col="condition_start_date", prior_obs_days=365,
    )
    assert dropped.count() == 0


def test_window_observed_cohort_prior_lookback_is_configurable(spark):
    """prior_obs_days sets the pre-index lookback; the follow-up requirement
    (window fully observed) holds regardless. Three persons, same index, in:
      1: 90d prior, follow-up ok    -> dropped at 365d, admitted at 0d
      2: >365d prior, follow-up ok  -> admitted at both
      3: prior ok, follow-up fails  -> dropped at both
    """
    import datetime as dt
    from charmpheno.omop.cohorts import _window_observed_cohort

    first_dx = spark.createDataFrame(
        [(1, dt.date(2010, 6, 1)), (2, dt.date(2010, 6, 1)),
         (3, dt.date(2011, 12, 1))],
        ["person_id", "index_date"],
    )
    op = spark.createDataFrame(
        [(1, dt.date(2010, 3, 1), dt.date(2012, 1, 1)),   # 90d prior
         (2, dt.date(2008, 1, 1), dt.date(2012, 1, 1)),   # >365d prior
         (3, dt.date(2008, 1, 1), dt.date(2012, 1, 1))],  # follow-up fails
        ["person_id", "observation_period_start_date",
         "observation_period_end_date"],
    )

    strict = {r["person_id"] for r in
              _window_observed_cohort(first_dx, op, prior_obs_days=365).collect()}
    assert strict == {2}

    relaxed = {r["person_id"] for r in
               _window_observed_cohort(first_dx, op, prior_obs_days=0).collect()}
    assert relaxed == {1, 2}


# --- population_cancer_sparse (exp 0029: sparse general-year foreground) -----

def test_supported_cohorts_includes_population_cancer_sparse():
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS
    assert "population_cancer_sparse" in SUPPORTED_COHORTS


def test_cohort_metadata_has_population_cancer_sparse():
    from charmpheno.omop.cohorts import COHORT_METADATA
    m = COHORT_METADATA["population_cancer_sparse"]
    assert m["id"] == "population_cancer_sparse"
    assert m["label"] and m["description"]


def test_bucket_general_by_density_splits_sparse_dense_and_drops_below_min(spark):
    """The general (non-cancer) arm is split by per-person in-window event
    count: >= dense_min -> 'general' (background), sparse_min..dense_min-1 ->
    'sparse' (its own foreground), below sparse_min -> dropped. Row-preserving
    for kept persons (each event row survives, tagged)."""
    from charmpheno.omop.cohorts import _bucket_general_by_density

    rows = []
    for pid, n in [(1, 3), (2, 7), (3, 25), (4, 20), (5, 5), (6, 19)]:
        rows += [(pid, 100 + j) for j in range(n)]  # (person_id, concept_id)
    ev = spark.createDataFrame(rows, ["person_id", "concept_id"])

    out = _bucket_general_by_density(ev, sparse_min=5, dense_min=20)
    tags = {
        r["person_id"]: r["source_cohort"]
        for r in out.select("person_id", "source_cohort").distinct().collect()
    }
    assert 1 not in tags               # 3 events < sparse_min -> dropped
    assert tags[5] == "sparse"         # 5 (lower boundary)
    assert tags[2] == "sparse"         # 7
    assert tags[6] == "sparse"         # 19 (upper sparse boundary)
    assert tags[4] == "general"        # 20 (dense boundary, inclusive)
    assert tags[3] == "general"        # 25
    assert out.where(out.person_id == 3).count() == 25   # all rows retained
    assert out.where(out.person_id == 1).count() == 0    # dropped person gone
    # no leftover count column leaks into the schema
    assert "source_cohort" in out.columns and "_n_events" not in out.columns


def test_concept_set_from_ancestors_unions_inclusions_and_subtracts_exclusions(spark):
    from charmpheno.omop.cohorts import _concept_set_from_ancestors
    ca = spark.createDataFrame(
        [
            (100, 1), (100, 2), (100, 3),   # inclusion ancestor A -> {1,2,3}
            (200, 3), (200, 4),             # inclusion ancestor B -> {3,4}
            (900, 2),                       # exclusion ancestor   -> {2}
        ],
        ["ancestor_concept_id", "descendant_concept_id"],
    )
    out = _concept_set_from_ancestors(
        ca, inclusion_ancestors=[100, 200], exclusion_ancestors=[900],
    )
    # union {1,2,3,4} minus {2} = {1,3,4}; 3 is in both inclusions, not excluded
    assert {r["concept_id"] for r in out.collect()} == {1, 3, 4}


def test_concept_set_from_ancestors_no_exclusions_dedups(spark):
    from charmpheno.omop.cohorts import _concept_set_from_ancestors
    ca = spark.createDataFrame(
        [(79145, 10), (79145, 11), (79145, 11)],
        ["ancestor_concept_id", "descendant_concept_id"],
    )
    out = _concept_set_from_ancestors(ca, inclusion_ancestors=[79145])
    assert {r["concept_id"] for r in out.collect()} == {10, 11}


def test_apply_first_diagnosis_year_cohort_is_importable_with_ancestor_params():
    import inspect
    from charmpheno.omop.cohorts import (
        apply_first_diagnosis_year_cohort, _EDS_ANCESTOR,
    )
    assert _EDS_ANCESTOR == 79145
    params = inspect.signature(apply_first_diagnosis_year_cohort).parameters
    assert "inclusion_ancestors" in params
    assert "exclusion_ancestors" in params
    assert "window_days" in params


# --- disease registry + population_eds (Task 3) -------------------------------

def test_disease_registry_has_cancer_and_eds_with_expected_ancestors():
    from charmpheno.omop.cohorts import (
        _DISEASE_REGISTRY, _CANCER_ANCESTOR, _CANCER_EXCLUSION_ANCESTORS,
        _EDS_ANCESTOR,
    )
    assert _DISEASE_REGISTRY["cancer"]["inclusion_ancestors"] == (_CANCER_ANCESTOR,)
    assert _DISEASE_REGISTRY["cancer"]["exclusion_ancestors"] == _CANCER_EXCLUSION_ANCESTORS
    assert _DISEASE_REGISTRY["eds"]["inclusion_ancestors"] == (_EDS_ANCESTOR,)
    assert _DISEASE_REGISTRY["eds"]["exclusion_ancestors"] == ()


def test_supported_cohorts_includes_population_eds():
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS
    assert "population_eds" in SUPPORTED_COHORTS


def test_cohort_metadata_has_population_eds():
    from charmpheno.omop.cohorts import COHORT_METADATA
    m = COHORT_METADATA["population_eds"]
    assert m["id"] == "population_eds"
    assert m["label"] and m["description"]


# --- population_sparse (Task 4: whole-population density split, no disease arm) --

def test_supported_cohorts_includes_population_sparse():
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS
    assert "population_sparse" in SUPPORTED_COHORTS


def test_cohort_metadata_has_population_sparse():
    from charmpheno.omop.cohorts import COHORT_METADATA
    m = COHORT_METADATA["population_sparse"]
    assert m["id"] == "population_sparse"
    assert m["label"] and m["description"]
