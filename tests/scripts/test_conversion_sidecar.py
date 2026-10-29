"""The E4 first-attestation sidecar (spec E4 / plan WP6, unit i) — build + load.

The sidecar is the only artifact in the repo that carries a POST-label-window date,
so the things worth pinning are the ones whose failure would be silent: the grain,
the `min` aggregation, the key's independence from the bundle key, and the refusal
to join a parquet that is not a sidecar.

Four groups:

  1. **The aggregation.** `min` over a person's codes, per node; a person coded at
     two descendants of the same node keeps the earlier date.
  2. **The grains (R4.3).** The two files carry two different schemas and say so;
     the loaders return exactly their own columns.
  3. **The key.** Independent of everything the bundle key folds and dependent on
     everything that changes which (person, node) pairs exist.
  4. **The refusals.** A miss is None; a parquet with the wrong columns RAISES
     rather than being joined.

Groups 1-2 use local Spark (`@slow`); 3-4 are pure or filesystem-only.
"""
import os
import sys
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUD = REPO_ROOT / "analysis" / "cloud"
for _p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
                str(REPO_ROOT / "spark-vi"), os.environ.get("PYTHONPATH", ""))
    if p)

import conversion_sidecar as cs  # noqa: E402

KEY_ARGS = dict(cdr="proj.cdr", person_mod=1, dag_source="mondo_native",
                mondo_version="2026-06-02", mondo_branch="", min_positives=100,
                code_map_identity="native:v1:2714")


# --------------------------------------------------------------------------- #
# 1. The aggregation.                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_first_attestation_is_the_min_over_a_persons_codes(spark):
    """Two codes resolving to the SAME node: the earlier date wins. That is the
    whole aggregation, and getting it backwards would silently make every
    conversion look later than it is."""
    cond = spark.createDataFrame(
        [(1, 100, date(2015, 3, 1)),
         (1, 101, date(2012, 6, 1)),          # same node, EARLIER
         (1, 200, date(2018, 1, 1)),
         (2, 100, date(2020, 5, 5))],
        "person_id long, concept_id long, condition_era_start_date date")
    code_map = cs.normalize_code_map(
        spark.createDataFrame(
            [(100, 7), (101, 7), (200, 8)],
            "std_cid long, node_cid long"),
        concept_col="std_cid", node_col="node_cid")
    got = {(r["person_id"], r["node_cid"]): r["first_attested_date"]
           for r in cs.build_conversion_sidecar(cond, code_map).collect()}
    assert got == {(1, 7): date(2012, 6, 1),
                   (1, 8): date(2018, 1, 1),
                   (2, 7): date(2020, 5, 5)}


@pytest.mark.slow
def test_a_person_with_no_mapped_code_has_no_row(spark):
    """An INNER join on purpose: 'no row' is the correct reading of 'never
    attested', and the analysis treats a missing pair as no-conversion rather than
    as missing data."""
    cond = spark.createDataFrame(
        [(1, 100, date(2015, 3, 1)), (9, 999, date(2015, 3, 1))],
        "person_id long, concept_id long, condition_era_start_date date")
    code_map = cs.normalize_code_map(
        spark.createDataFrame([(100, 7)], "std_cid long, node_cid long"),
        concept_col="std_cid", node_col="node_cid")
    got = cs.build_conversion_sidecar(cond, code_map).collect()
    assert [r["person_id"] for r in got] == [1]


@pytest.mark.slow
def test_both_code_map_flavours_normalize_to_the_same_two_columns(spark):
    """The legacy climb and the native build name the columns differently; that
    difference lives in ONE place or the aggregation forks."""
    native = cs.normalize_code_map(
        spark.createDataFrame([(100, 7)], "std_cid long, node_cid long"),
        concept_col="std_cid", node_col="node_cid")
    legacy = cs.normalize_code_map(
        spark.createDataFrame(
            [(7, 100)],
            "ancestor_concept_id long, descendant_concept_id long"),
        concept_col="descendant_concept_id", node_col="ancestor_concept_id")
    assert native.columns == legacy.columns == ["concept_id", cs.NODE_COL]
    assert native.collect() == legacy.collect()


@pytest.mark.slow
def test_the_observation_gate_keeps_the_LATEST_end_per_person(spark):
    """`_window_observed_cohort` lets a person pass if ANY period covers the
    horizon, so the binding value is the max end date — not the first row."""
    op = spark.createDataFrame(
        [(1, date(2010, 1, 1), date(2014, 1, 1)),
         (1, date(2015, 1, 1), date(2022, 1, 1)),
         (2, date(2011, 1, 1), date(2013, 1, 1))],
        "person_id long, observation_period_start_date date, "
        "observation_period_end_date date")
    got = {r["person_id"]: r[cs.OBS_END_COL]
           for r in cs.observation_gate_frame(op).collect()}
    assert got == {1: date(2022, 1, 1), 2: date(2013, 1, 1)}


# --------------------------------------------------------------------------- #
# 2. The grains (R4.3).                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_the_two_files_round_trip_with_their_own_named_grains(spark, tmp_path):
    """Two parquets under one key because they are two grains, and the loaders
    return exactly their own columns — a per-person frame can never be read back
    as a per-(person, node) one."""
    uri = str(tmp_path / "sidecars")
    key = "deadbeefdeadbeef"
    first = spark.createDataFrame(
        [(1, 7, date(2012, 6, 1)), (2, 7, date(2020, 5, 5))],
        "person_id long, node_cid long, first_attested_date date")
    horizon = spark.createDataFrame(
        [(1, date(2016, 1, 1), date(2022, 1, 1)),
         (2, date(2016, 1, 1), date(2017, 1, 1))],
        "person_id long, index_date date, observation_period_end_date date")
    cs.save_sidecar(first, uri, key)
    cs.save_sidecar(horizon, uri, key, cs.INDEX_HORIZON_FILE)

    back = cs.try_load_sidecar(spark, uri, key)
    assert tuple(back.columns) == cs.SIDECAR_COLUMNS
    assert back.count() == 2
    hz = cs.try_load_index_horizon(spark, uri, key)
    assert tuple(hz.columns) == cs.HORIZON_COLUMNS
    assert hz.count() == 2

    w = cs.sidecar_witness(key, uri, n_rows=2, n_persons=2)
    assert w["grains"] == {"first_attestation": "per (person_id, node_cid)",
                           "index_horizon": "per person_id"}
    report = cs.format_sidecar_report(w)
    assert "PER (PERSON, NODE)" in report and "PER PERSON" in report
    assert "POST-label-window" in report


@pytest.mark.slow
def test_the_index_horizon_frame_keeps_a_person_with_no_observation_row(spark):
    """A LEFT join: a NULL end date reads as 'not observed at any horizon' —
    excluded from the denominator, never counted as a non-converter."""
    index_df = spark.createDataFrame(
        [(1, date(2016, 1, 1)), (2, date(2016, 1, 1))],
        "person_id long, index_date date")
    op = spark.createDataFrame(
        [(1, date(2010, 1, 1), date(2022, 1, 1))],
        "person_id long, observation_period_start_date date, "
        "observation_period_end_date date")
    got = {r["person_id"]: r[cs.OBS_END_COL]
           for r in cs.build_index_horizon_frame(index_df, op).collect()}
    assert got == {1: date(2022, 1, 1), 2: None}


# --------------------------------------------------------------------------- #
# 3. The key (R4.2): its own, not the bundle's.                                #
# --------------------------------------------------------------------------- #
def test_the_key_is_stable_and_16_hex():
    a = cs.conversion_sidecar_key(**KEY_ARGS)
    b = cs.conversion_sidecar_key(**KEY_ARGS)
    assert a == b and len(a) == 16 and int(a, 16) >= 0


@pytest.mark.parametrize("field,value", [
    ("cdr", "proj.other_cdr"),
    ("person_mod", 10),
    ("dag_source", "mondo"),
    ("mondo_version", "2026-07-01"),
    ("mondo_branch", "MONDO:0005070"),
    ("min_positives", 50),
    ("code_map_identity", "native:v1:9999"),
])
def test_everything_that_changes_the_content_changes_the_key(field, value):
    """Five things determine which (person, node) pairs exist and when they were
    first coded; all five are folded."""
    assert cs.conversion_sidecar_key(**{**KEY_ARGS, field: value}) != \
        cs.conversion_sidecar_key(**KEY_ARGS)


def test_the_key_folds_nothing_the_bundle_key_folds_that_it_does_not_need():
    """R4.2's point: the sidecar is keyed independently, so a bundle-key move (a
    new flag, a vocabulary change, a different split) does NOT orphan it. The
    positive check is structural — the payload names five inputs and this asserts
    the signature admits no others."""
    import inspect
    params = set(inspect.signature(cs.conversion_sidecar_key).parameters)
    assert params == {"cdr", "person_mod", "dag_source", "mondo_version",
                      "mondo_branch", "min_positives", "code_map_identity"}
    for absent in ("vocab_size", "min_df", "doc_min_length", "holdout_frac",
                   "label_mask_mode", "lookback_days", "label_window_days",
                   "doc_spec", "dag_collapse", "preindex_closure"):
        assert absent not in params


def test_the_horizon_set_is_not_in_the_key():
    """The frame stores DATES; a horizon is a comparison the analysis makes.
    Keying on it would rebuild a full-history scan to answer a subtraction."""
    import inspect
    assert "horizon" not in str(inspect.signature(cs.conversion_sidecar_key))


# --------------------------------------------------------------------------- #
# 4. The refusals.                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_a_miss_is_None_and_a_wrong_artifact_RAISES(spark, tmp_path):
    """A miss is a normal outcome (a fresh bucket, a new key). A parquet that IS
    there but is not a sidecar is a wrong-artifact join, which fails silently and
    expensively if allowed through — so it raises, naming the key."""
    uri = str(tmp_path / "sidecars")
    assert cs.try_load_sidecar(spark, uri, "0000000000000000") is None
    assert cs.try_load_index_horizon(spark, uri, "0000000000000000") is None

    bogus = spark.createDataFrame([(1, 2)], "a long, b long")
    bogus.write.mode("overwrite").parquet(
        cs.sidecar_path(uri, "1111111111111111"))
    with pytest.raises(ValueError, match="not a first-attestation sidecar"):
        cs.try_load_sidecar(spark, uri, "1111111111111111")


def test_a_non_mondo_corpus_is_refused_by_name():
    """E4 is defined over the Mondo label space; a SNOMED-anchor corpus has no
    driver-side code map and must say so rather than half-building one."""
    with pytest.raises(ValueError, match="no driver-side code map"):
        cs.code_map_from_manifest(None, {"dag_source": "snomed"})


# --------------------------------------------------------------------------- #
# The index re-derivation (moved from the removed preindex_closure.py).       #
# --------------------------------------------------------------------------- #
def test_the_rederivation_calls_the_assemblers_index_builder_with_its_arguments():
    """The only way a driver-side re-derivation can diverge from the assembler
    once the pick itself is IMPORTED: a different argument. The prior-observation
    gate in particular is the intrinsic `_LOOKBACK_PRIOR_OBS_DAYS` floor, NOT the
    forward-mode `prior_obs_days` knob — a lookback config inherits
    `prior_obs_days: 0` and passing that here would silently build a different
    index than the corpus was built on."""
    from charmpheno.omop import cohorts
    from charmpheno.omop.case_finding_assembly import _LOOKBACK_PRIOR_OBS_DAYS

    seen = {}

    def _spy(cond_df, **kw):
        seen.update(kw)
        seen["cond"] = cond_df
        return "index-df"

    real = cohorts.case_finding_population_index_table
    cohorts.case_finding_population_index_table = _spy
    try:
        out = cs.feature_window_index_table(
            "cond-frame", spark="S", cdr="p.d", billing="bp",
            date_col="condition_era_start_date", label_window_days=365)
    finally:
        cohorts.case_finding_population_index_table = real

    assert out == "index-df" and seen["cond"] == "cond-frame"
    assert seen["prior_obs_days"] == _LOOKBACK_PRIOR_OBS_DAYS == 365
    assert seen["label_window_days"] == 365
    assert seen["date_col"] == "condition_era_start_date"
    assert seen["cdr_dataset"] == "p.d" and seen["billing_project"] == "bp"


def test_an_unknown_index_mode_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="index_mode"):
        cs.feature_window_index_table(
            None, spark=None, cdr="p.d", billing="bp", date_col="d",
            label_window_days=365, index_mode="episode")


class _FakeReader:
    """`case_finding_population_index_table`'s ONLY use of `spark` is one
    BigQuery read of `observation_period`; hand it a local frame instead so the
    deterministic pick can be exercised off-cluster."""

    def __init__(self, op_df):
        self._op = op_df

    def format(self, _fmt):
        return self

    def option(self, *_a, **_kw):
        return self

    def load(self):
        return self._op


class _FakeSpark:
    def __init__(self, session, op_df):
        self._session = session
        self.read = _FakeReader(op_df)

    def __getattr__(self, name):
        return getattr(self._session, name)


def _events(spark, rows):
    """(person_id, concept_id, condition_era_start_date, source_cohort=..) — the
    condition frame shape `load_omop_bigquery(source_table='condition_era')`
    returns and the assembler windows."""
    from pyspark.sql.types import (DateType, LongType, StructField, StructType)
    schema = StructType([
        StructField("person_id", LongType(), False),
        StructField("concept_id", LongType(), False),
        StructField("condition_era_start_date", DateType(), False),
    ])
    return spark.createDataFrame(
        [(int(p), int(c), date(*d)) for p, c, d in rows], schema)


def _obs(spark, rows):
    from pyspark.sql.types import (DateType, LongType, StructField, StructType)
    schema = StructType([
        StructField("person_id", LongType(), False),
        StructField("observation_period_start_date", DateType(), False),
        StructField("observation_period_end_date", DateType(), False),
    ])
    return spark.createDataFrame(
        [(int(p), date(*s), date(*e)) for p, s, e in rows], schema)


_ROWS = [
    # person 1: a long record with several eligible anchors
    (1, 100, (2012, 3, 1)), (1, 300, (2014, 6, 1)), (1, 400, (2016, 9, 1)),
    (1, 200, (2018, 2, 1)),
    # person 2: likewise, different dates so the hash pick differs
    (2, 200, (2012, 5, 1)), (2, 300, (2015, 1, 1)), (2, 100, (2017, 7, 1)),
    (2, 400, (2019, 4, 1)),
    # person 3
    (3, 400, (2013, 8, 1)), (3, 100, (2015, 11, 1)), (3, 300, (2017, 3, 1)),
]
_OBS = [(1, (2010, 1, 1), (2021, 1, 1)), (2, (2010, 1, 1), (2021, 1, 1)),
        (3, (2010, 1, 1), (2021, 1, 1))]


@pytest.mark.slow
def test_the_rederived_index_is_the_assemblers_own_pick(spark):
    """THE key risk, checked against the source of truth. `_random_event_windows`
    IS the assembler's pick (`min hash(person_id, event_date, _RANDOM_WINDOW_SALT)`,
    ties by earliest date, gated on a fully-observed forward window). The driver's
    re-derivation must land on the same index date for every person — not merely a
    plausible one."""
    from charmpheno.omop.cohorts import _random_event_windows
    from charmpheno.omop.case_finding_assembly import _LOOKBACK_PRIOR_OBS_DAYS

    cond, op = _events(spark, _ROWS), _obs(spark, _OBS)
    theirs = {r["person_id"]: r["index_date"] for r in _random_event_windows(
        cond, op, date_col="condition_era_start_date", window_days=365,
        prior_obs_days=_LOOKBACK_PRIOR_OBS_DAYS).collect()}

    mine = {r["person_id"]: r["index_date"] for r in cs.feature_window_index_table(
        cond, spark=_FakeSpark(spark, op), cdr="p.d", billing="bp",
        date_col="condition_era_start_date", label_window_days=365).collect()}

    assert mine == theirs and len(theirs) == 3
