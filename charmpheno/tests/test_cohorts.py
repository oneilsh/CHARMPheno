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
