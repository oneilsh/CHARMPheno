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


def test_disease_anchors_single_disease_returns_one_anchor():
    from charmpheno.omop.cohorts import disease_anchors
    assert disease_anchors("diabetes") == (201820,)
    assert disease_anchors("eds") == (79145,)


def test_disease_anchors_rare6_returns_six_distinct_anchors():
    from charmpheno.omop.cohorts import disease_anchors
    anchors = disease_anchors("rare6")
    assert len(anchors) == 6 and len(set(anchors)) == 6
    # EDS leads the forest; the five distinct-phenotype additions are present.
    assert anchors[0] == 79145
    assert set(anchors) == {79145, 438688, 257628, 40352976, 76685, 432595}


def test_disease_anchors_rejects_unknown_disease():
    from charmpheno.omop.cohorts import disease_anchors
    with pytest.raises(ValueError, match="not in registry"):
        disease_anchors("not_a_disease")


def test_supported_cohorts_and_metadata_include_rare6():
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS, COHORT_METADATA
    assert "population_rare6" in SUPPORTED_COHORTS
    assert "population_rare6" in COHORT_METADATA


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


# --- ingredient resolver + first-drug-era helper + drug registry (Task 2) -----

def test_ingredient_concept_ids_matches_rxnorm_ingredients_case_insensitively(spark):
    from charmpheno.omop.cohorts import _ingredient_concept_ids
    concept = spark.createDataFrame(
        [
            (11, "semaglutide", "RxNorm", "Ingredient"),
            (12, "Empagliflozin", "RxNorm", "Ingredient"),   # case differs
            (13, "semaglutide", "RxNorm", "Brand Name"),      # wrong class
            (14, "metformin", "RxNorm", "Ingredient"),        # not requested
            (15, "semaglutide", "ATC", "Ingredient"),         # wrong vocab
        ],
        ["concept_id", "concept_name", "vocabulary_id", "concept_class_id"],
    )
    out = _ingredient_concept_ids(concept, ["semaglutide", "empagliflozin"])
    assert {r["concept_id"] for r in out.collect()} == {11, 12}


def test_first_drug_era_dates_picks_earliest_era_per_person(spark):
    import datetime as dt
    from charmpheno.omop.cohorts import _first_drug_era_dates
    drug_era = spark.createDataFrame(
        [
            (1, 11, dt.date(2020, 3, 1)),
            (1, 11, dt.date(2019, 5, 1)),   # earlier -> wins for person 1
            (2, 12, dt.date(2021, 1, 1)),
            (3, 99, dt.date(2020, 1, 1)),   # ingredient not in set -> person 3 absent
        ],
        ["person_id", "drug_concept_id", "drug_era_start_date"],
    )
    ids = spark.createDataFrame([(11,), (12,)], ["concept_id"])
    out = {r["person_id"]: r["index_date"] for r in _first_drug_era_dates(drug_era, ids).collect()}
    assert out == {1: dt.date(2019, 5, 1), 2: dt.date(2021, 1, 1)}


def test_drug_registry_shape():
    from charmpheno.omop.cohorts import _DRUG_REGISTRY
    # GLP-1 comparator arms still present (unchanged shape: no class tag).
    assert {"glp1_ra", "sglt2i", "tirzepatide"} <= set(_DRUG_REGISTRY)
    assert "semaglutide" in _DRUG_REGISTRY["glp1_ra"]["ingredient_names"]
    assert _DRUG_REGISTRY["tirzepatide"]["ingredient_names"] == ("tirzepatide",)
    assert "class" not in _DRUG_REGISTRY["glp1_ra"]


# --- partition core: precedence + in-window-both-user exclusion ----------

def test_assign_drug_groups_precedence_and_in_window_exclusion(spark):
    import datetime as dt
    from charmpheno.omop.cohorts import _assign_drug_groups
    d = dt.date

    def frame(rows):  # rows: list[(person_id, date)]
        return spark.createDataFrame(rows, ["person_id", "index_date"])

    # g = GLP-1 first era, s = SGLT2i first era, t = tirzepatide first era.
    # Two arms only (glp1_ra/sglt2i); tirzepatide users are EXCLUDED (dropped),
    # and any both-user whose second drug is within window_days (365d) is excluded.
    g = frame([(1, d(2021, 1, 1)),                     # glp1 only
               (4, d(2021, 1, 1)), (5, d(2021, 1, 1)), # both, in-window -> excluded
               (6, d(2021, 1, 1)), (7, d(2021, 1, 1))])# have tirzepatide -> excluded
    s = frame([(2, d(2021, 1, 1)),                     # sglt2i only
               (4, d(2021, 2, 1)),                     # +31d -> excluded (<=365)
               (5, d(2021, 9, 1)),                     # +243d -> excluded (<=365)
               (6, d(2021, 3, 1))])                    # p6 also has t -> excluded
    t = frame([(3, d(2021, 1, 1)),                     # tirzepatide only -> excluded
               (6, d(2021, 5, 1)), (7, d(2021, 6, 1))])# t forces exclusion

    out = {r["person_id"]: (r["source_cohort"], r["index_date"])
           for r in _assign_drug_groups(g, s, t).collect()}

    assert out[1] == ("glp1_ra", d(2021, 1, 1))
    assert out[2] == ("sglt2i", d(2021, 1, 1))
    assert 3 not in out                                    # tirzepatide-only -> excluded
    assert 4 not in out                                    # both, +31d -> excluded
    assert 5 not in out                                    # both, +243d -> excluded
    assert 6 not in out                                    # has tirzepatide -> excluded
    assert 7 not in out                                    # has tirzepatide -> excluded
    # only the two clean single arms survive
    assert set(out) == {1, 2}


# --- co-initiation gap histogram (Task 4) --------------------------------

def test_coinitiation_gap_histogram_buckets(spark):
    import datetime as dt
    from charmpheno.omop.cohorts import _coinitiation_gap_histogram
    d = dt.date
    g = spark.createDataFrame(
        [(1, d(2021, 1, 1)), (2, d(2021, 1, 1)), (3, d(2021, 1, 1)),
         (4, d(2021, 1, 1)), (9, d(2021, 1, 1))],   # p9 has no s -> excluded from hist
        ["person_id", "index_date"])
    s = spark.createDataFrame(
        [(1, d(2021, 1, 4)),    # gap 3   -> 0-7
         (2, d(2021, 1, 21)),   # gap 20  -> 8-30
         (3, d(2021, 7, 20)),   # gap 200 -> 181-365
         (4, d(2022, 6, 1))],   # gap 516 -> 366+
        ["person_id", "index_date"])
    hist = {r["bucket"]: r["n"] for r in _coinitiation_gap_histogram(g, s).collect()}
    assert hist == {"0-7": 1, "8-30": 1, "181-365": 1, "366+": 1}


# --- population_glp1 orchestration + registration (Task 5) --------------

def test_supported_cohorts_includes_population_glp1():
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS
    assert "population_glp1" in SUPPORTED_COHORTS


def test_cohort_metadata_has_population_glp1():
    from charmpheno.omop.cohorts import COHORT_METADATA
    m = COHORT_METADATA["population_glp1"]
    assert m["id"] == "population_glp1"
    assert m["label"] and m["description"]


def test_apply_population_drug_cohort_importable_signature():
    import inspect
    from charmpheno.omop.cohorts import apply_population_drug_cohort
    p = inspect.signature(apply_population_drug_cohort).parameters
    assert {"window_days", "prior_obs_days", "date_col"} <= set(p)


def test_window_observed_cohort_dedups_multiple_observation_periods(spark):
    """A person with 2+ observation_period rows that each satisfy the prior +
    follow-up gates must NOT fan out into duplicate (person_id, index_date)
    rows — that would duplicate the person's documents downstream and
    over-weight multi-period patients in every cohort."""
    import datetime as dt
    from charmpheno.omop.cohorts import _window_observed_cohort
    first_dx = spark.createDataFrame(
        [(1, dt.date(2016, 1, 1))], ["person_id", "index_date"],
    )
    # Two observation periods, BOTH give >=365d prior and >=365d observed
    # follow-up around the 2016-01-01 index.
    op = spark.createDataFrame(
        [
            (1, dt.date(2014, 1, 1), dt.date(2018, 1, 1)),
            (1, dt.date(2013, 1, 1), dt.date(2019, 1, 1)),
        ],
        ["person_id", "observation_period_start_date", "observation_period_end_date"],
    )
    out = _window_observed_cohort(first_dx, op, prior_obs_days=365, window_days=365)
    assert out.count() == 1  # one surviving (person_id, index_date), not two


def test_ingredient_concept_ids_includes_explicit_extra_ids(spark):
    """Explicit concept_ids are included even when they DON'T match the
    name/class filter (pin a known ingredient id robust to vocab naming)."""
    from charmpheno.omop.cohorts import _ingredient_concept_ids
    concept = spark.createDataFrame(
        [
            (11, "semaglutide", "RxNorm", "Ingredient"),
            (779705, "tirzepatide", "RxNorm", "Precise Ingredient"),  # NOT class Ingredient
        ],
        ["concept_id", "concept_name", "vocabulary_id", "concept_class_id"],
    )
    # Name-only would miss 779705 (wrong class); the explicit id rescues it.
    out = _ingredient_concept_ids(
        concept, ["semaglutide"], extra_concept_ids=(779705,),
    )
    assert {r["concept_id"] for r in out.collect()} == {11, 779705}


def test_assign_drug_groups_long_gap_goes_to_earlier_single_arm(spark):
    """3-regime: both-users with gap > window_days start the second drug OUTSIDE
    the index year, so their index year is genuine monotherapy -> the earlier
    drug's single arm (not excluded). Middle-band gaps stay excluded."""
    import datetime as dt
    from charmpheno.omop.cohorts import _assign_drug_groups
    d = dt.date

    def frame(rows):
        return spark.createDataFrame(rows, ["person_id", "index_date"])

    g = frame([(10, d(2020, 1, 1)),   # g first, s ~547d later -> glp1_ra
               (11, d(2021, 7, 1)),   # s first, g ~547d later -> sglt2i
               (12, d(2020, 1, 1))])  # gap exactly 365 -> middle band, excluded
    s = frame([(10, d(2021, 7, 1)),
               (11, d(2020, 1, 1)),
               (12, d(2020, 12, 31))])  # 2020-01-01..2020-12-31 = 365 days
    t = spark.createDataFrame([], "person_id bigint, index_date date")

    out = {r["person_id"]: (r["source_cohort"], r["index_date"])
           for r in _assign_drug_groups(g, s, t, window_days=365).collect()}

    assert out[10] == ("glp1_ra", d(2020, 1, 1))   # earlier drug = GLP-1
    assert out[11] == ("sglt2i", d(2020, 1, 1))    # earlier drug = SGLT2i
    assert 12 not in out                            # gap 365 not > 365 -> excluded


def test_expand_descendants_includes_self_and_children(spark):
    """A drug class = descendants (incl. self) of its seed concept ids, so a
    pinned ingredient id matches drug_era whatever level it's recorded at."""
    from charmpheno.omop.cohorts import _expand_descendants
    # concept_ancestor: 779705 -> {779705(self), 900, 901}; 555 -> {555, 700}
    ca = spark.createDataFrame(
        [
            (779705, 779705), (779705, 900), (779705, 901),
            (555, 555), (555, 700),
            (999, 999),  # unrelated
        ],
        ["ancestor_concept_id", "descendant_concept_id"],
    )
    seeds = spark.createDataFrame([(779705,)], ["concept_id"])
    out = {r["concept_id"] for r in _expand_descendants(ca, seeds).collect()}
    assert out == {779705, 900, 901}


def test_lookback_feature_label_events_splits_pre_and_post_index(spark):
    import datetime as dt
    from charmpheno.omop.cohorts import lookback_feature_label_events
    events = spark.createDataFrame(
        [   # person, concept, date
            (1, 900, dt.date(2013, 6, 1)),   # 1.5y pre-index -> feature (within 5y, before)
            (1, 901, dt.date(2014, 6, 1)),   # 0.5y pre-index -> feature
            (1, 200, dt.date(2015, 1, 1)),   # index day -> label
            (1, 201, dt.date(2015, 6, 1)),   # 0.5y post -> label
            (1, 999, dt.date(2011, 1, 1)),   # 4y pre -> feature only if lookback>=~4y
        ],
        ["person_id", "concept_id", "condition_era_start_date"])
    index_df = spark.createDataFrame(
        [(1, dt.date(2015, 1, 1), "dis")], ["person_id", "index_date", "source_cohort"])
    feat, lab = lookback_feature_label_events(
        events, index_df, date_col="condition_era_start_date",
        lookback_days=365, label_window_days=365)
    fc = {r["concept_id"] for r in feat.collect()}
    lc = {r["concept_id"] for r in lab.collect()}
    assert fc == {901}                    # only within [index-1y, index)
    assert lc == {200, 201}               # only within [index, index+1y)
    assert "index_date" not in feat.columns and "source_cohort" in feat.columns
    # 5-year lookback pulls the older feature events too
    feat5, _ = lookback_feature_label_events(
        events, index_df, date_col="condition_era_start_date",
        lookback_days=1825, label_window_days=365)
    assert {r["concept_id"] for r in feat5.collect()} == {900, 901, 999}


# --- Antidepressant / MDD registry + concept map (Phase C, task 1) ------------

def test_disease_registry_has_mdd_and_anxiety_with_expected_ancestors():
    from charmpheno.omop.cohorts import (
        _DISEASE_REGISTRY, _MDD_ANCESTOR, _MDD_EXCLUSION_ANCESTORS,
        _ANXIETY_ANCESTOR,
    )
    assert _MDD_ANCESTOR == 440383
    assert _MDD_EXCLUSION_ANCESTORS == (439254, 4224940)
    assert _ANXIETY_ANCESTOR == 441542
    assert _DISEASE_REGISTRY["mdd"] == {
        "inclusion_ancestors": (440383,),
        "exclusion_ancestors": (439254, 4224940),
    }
    assert _DISEASE_REGISTRY["anxiety"] == {
        "inclusion_ancestors": (441542,),
        "exclusion_ancestors": (),
    }


def test_drug_registry_has_15_antidepressants_with_class_tags_and_pins():
    from charmpheno.omop.cohorts import (
        _DRUG_REGISTRY, _ANTIDEPRESSANT_INGREDIENTS,
    )
    assert len(_ANTIDEPRESSANT_INGREDIENTS) == 15
    assert set(_ANTIDEPRESSANT_INGREDIENTS) <= set(_DRUG_REGISTRY)
    classes = {}
    for name in _ANTIDEPRESSANT_INGREDIENTS:
        entry = _DRUG_REGISTRY[name]
        # portable-by-name + pinned-id fallback shape, plus a class tag
        assert entry["ingredient_names"] == (name,)
        assert len(entry["seed_concept_ids"]) == 1
        assert entry["class"] in {"SSRI", "SNRI", "TCA", "Atyp"}
        classes.setdefault(entry["class"], []).append(name)
    assert classes["SSRI"][:2] == ["fluoxetine", "sertraline"]
    # a couple of the AoU-validated pins
    assert _DRUG_REGISTRY["fluoxetine"]["seed_concept_ids"] == (755695,)
    assert _DRUG_REGISTRY["vortioxetine"]["seed_concept_ids"] == (44507700,)
    assert _DRUG_REGISTRY["venlafaxine"]["class"] == "SNRI"


def test_supported_cohorts_and_metadata_include_mdd_antidepressant():
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS, COHORT_METADATA
    assert "mdd_antidepressant" in SUPPORTED_COHORTS
    m = COHORT_METADATA["mdd_antidepressant"]
    assert m["id"] == "mdd_antidepressant"
    assert m["label"] and m["description"]


def test_antidepressant_concept_map_tags_expanded_concepts_with_name_and_class(spark):
    from charmpheno.omop.cohorts import _antidepressant_concept_map
    concept = spark.createDataFrame(
        [
            (755695, "fluoxetine", "RxNorm", "Ingredient"),
            (739138, "sertraline", "RxNorm", "Ingredient"),
            (739138, "sertraline", "ATC", "Ingredient"),   # wrong vocab -> ignored
        ],
        ["concept_id", "concept_name", "vocabulary_id", "concept_class_id"],
    )
    # descendants (incl. self): fluoxetine 755695 -> {755695, 111}; sertraline -> {739138}
    ca = spark.createDataFrame(
        [(755695, 755695), (755695, 111), (739138, 739138)],
        ["ancestor_concept_id", "descendant_concept_id"],
    )
    out = {
        r["concept_id"]: (r["drug_name"], r["drug_class"])
        for r in _antidepressant_concept_map(
            concept, ca, ingredients=("fluoxetine", "sertraline"),
        ).collect()
    }
    assert out == {
        755695: ("fluoxetine", "SSRI"),
        111: ("fluoxetine", "SSRI"),
        739138: ("sertraline", "SSRI"),
    }


# --- Antidepressant first-era index + MDD indication (Phase C, task 2) ---------

def _ad_concept_map(spark):
    return spark.createDataFrame(
        [(755695, "fluoxetine", "SSRI"), (739138, "sertraline", "SSRI")],
        ["concept_id", "drug_name", "drug_class"],
    )


def test_first_antidepressant_index_picks_earliest_era_and_its_ingredient(spark):
    import datetime as dt
    from charmpheno.omop.cohorts import _first_antidepressant_index
    d = dt.date
    drug_era = spark.createDataFrame(
        [
            # person 1: fluoxetine later, sertraline earlier -> index = sertraline
            (1, 755695, d(2020, 3, 1), d(2020, 9, 1)),
            (1, 739138, d(2020, 1, 1), d(2020, 6, 1)),
            # person 2: same-day co-initiation -> tie-break to lowest concept_id
            (2, 755695, d(2021, 5, 1), d(2021, 8, 1)),
            (2, 739138, d(2021, 5, 1), d(2021, 8, 1)),
        ],
        ["person_id", "drug_concept_id", "drug_era_start_date", "drug_era_end_date"],
    )
    out = {
        r["person_id"]: (r["index_date"], r["index_drug_concept_id"],
                         r["index_drug_name"], r["index_drug_class"])
        for r in _first_antidepressant_index(drug_era, _ad_concept_map(spark)).collect()
    }
    assert out[1] == (d(2020, 1, 1), 739138, "sertraline", "SSRI")
    # 739138 < 755695 -> sertraline wins the same-day tie deterministically
    assert out[2] == (d(2021, 5, 1), 739138, "sertraline", "SSRI")


def test_mdd_antidepressant_index_indication_and_new_user_bracket(spark):
    """Correct index drug + the new-user bracket + the MDD-indication rule:
    person 1 qualifies; person 2 is dropped for insufficient prior observation;
    person 3 is dropped because their MDD dx is AFTER the index date."""
    import datetime as dt
    from charmpheno.omop.cohorts import _mdd_antidepressant_index
    d = dt.date

    cond = spark.createDataFrame(
        [
            (1, 9001, d(2019, 6, 1)),   # MDD dx before index -> qualifies
            (2, 9001, d(2019, 6, 1)),   # MDD before index (but obs too short)
            (3, 9001, d(2020, 6, 1)),   # MDD AFTER index -> indication fails
        ],
        ["person_id", "concept_id", "condition_start_date"],
    )
    drug_era = spark.createDataFrame(
        [
            (1, 739138, d(2020, 1, 1), d(2020, 12, 1)),   # sertraline
            (2, 755695, d(2020, 1, 1), d(2020, 12, 1)),   # fluoxetine
            (3, 755695, d(2020, 1, 1), d(2020, 12, 1)),   # fluoxetine
        ],
        ["person_id", "drug_concept_id", "drug_era_start_date", "drug_era_end_date"],
    )
    op = spark.createDataFrame(
        [
            (1, d(2018, 1, 1), d(2022, 1, 1)),    # ample prior + follow-up
            (2, d(2019, 11, 1), d(2022, 1, 1)),   # ~61d prior -> excluded at 365
            (3, d(2018, 1, 1), d(2022, 1, 1)),    # ample, but indication fails
        ],
        ["person_id", "observation_period_start_date",
         "observation_period_end_date"],
    )
    mdd_concepts = spark.createDataFrame([(9001,)], ["concept_id"])

    out = {
        r["person_id"]: (r["index_drug_name"], r["index_drug_class"],
                         r["source_cohort"])
        for r in _mdd_antidepressant_index(
            cond, drug_era, op, _ad_concept_map(spark), mdd_concepts,
            date_col="condition_start_date", window_days=365, prior_obs_days=365,
        ).collect()
    }
    assert set(out) == {1}                                     # 2 and 3 dropped
    assert out[1] == ("sertraline", "SSRI", "mdd_antidepressant")


# --- >=90-day stability outcome labeler (Phase C, task 3) ---------------------

def test_antidepressant_stability_label_all_scenarios(spark):
    """Hand-built drug_era + index frames covering the five definitional cases:
    clean >=90d continuation (positive); early discontinuation <90d (negative);
    switch to another antidepressant within 90d (negative); a gap <= grace
    stitched to positive; a gap > grace -> negative. Plus a cohort member with
    NO era at all (uncensored -> negative)."""
    import datetime as dt
    from charmpheno.omop.cohorts import antidepressant_stability_label
    d = dt.date
    fx, sx = 755695, 739138  # fluoxetine (index drug), sertraline (a switch target)

    drug_era = spark.createDataFrame(
        [
            # p1 clean >=90d continuation -> positive
            (1, fx, d(2020, 1, 1), d(2020, 4, 30)),
            # p2 early discontinuation (45d) -> negative
            (2, fx, d(2020, 1, 1), d(2020, 2, 15)),
            # p3 index covered but SWITCH to sertraline at +31d -> negative
            (3, fx, d(2020, 1, 1), d(2020, 6, 1)),
            (3, sx, d(2020, 2, 1), d(2020, 8, 1)),
            # p4 gap 24d (<= grace 30) stitched -> coverage to 2020-05-01 -> positive
            (4, fx, d(2020, 1, 1), d(2020, 2, 15)),
            (4, fx, d(2020, 3, 10), d(2020, 5, 1)),
            # p5 gap 46d (> grace 30) broken -> first island 45d -> negative
            (5, fx, d(2020, 1, 1), d(2020, 2, 15)),
            (5, fx, d(2020, 4, 1), d(2020, 7, 1)),
        ],
        ["person_id", "drug_concept_id", "drug_era_start_date", "drug_era_end_date"],
    )
    # p6 is a cohort member with no drug_era row at all -> uncensored negative.
    index_df = spark.createDataFrame(
        [(p, d(2020, 1, 1), "fluoxetine") for p in (1, 2, 3, 4, 5, 6)],
        ["person_id", "index_date", "index_drug_name"],
    )
    concept_map = spark.createDataFrame(
        [(fx, "fluoxetine", "SSRI"), (sx, "sertraline", "SSRI")],
        ["concept_id", "drug_name", "drug_class"],
    )
    out = {
        r["person_id"]: r["worked"]
        for r in antidepressant_stability_label(
            drug_era, index_df, drug_concept_sets=concept_map,
            stability_days=90, grace_gap_days=30,
        ).collect()
    }
    assert out == {1: True, 2: False, 3: False, 4: True, 5: False, 6: False}


def test_antidepressant_stability_label_grace_gap_is_tunable(spark):
    """The stitch tolerance is the grace_gap_days knob: p5's 46d gap that breaks
    at grace=30 gets stitched at grace=60, flipping the label to positive."""
    import datetime as dt
    from charmpheno.omop.cohorts import antidepressant_stability_label
    d = dt.date
    fx = 755695
    drug_era = spark.createDataFrame(
        [
            (5, fx, d(2020, 1, 1), d(2020, 2, 15)),
            (5, fx, d(2020, 4, 1), d(2020, 7, 1)),   # 46d gap
        ],
        ["person_id", "drug_concept_id", "drug_era_start_date", "drug_era_end_date"],
    )
    index_df = spark.createDataFrame(
        [(5, d(2020, 1, 1), "fluoxetine")],
        ["person_id", "index_date", "index_drug_name"],
    )
    concept_map = spark.createDataFrame(
        [(fx, "fluoxetine", "SSRI")], ["concept_id", "drug_name", "drug_class"],
    )

    def worked(grace):
        return antidepressant_stability_label(
            drug_era, index_df, drug_concept_sets=concept_map,
            stability_days=90, grace_gap_days=grace,
        ).collect()[0]["worked"]

    assert worked(30) is False   # gap 46 > 30 -> broken, 45d island
    assert worked(60) is True    # gap 46 <= 60 -> stitched to 2020-07-01
