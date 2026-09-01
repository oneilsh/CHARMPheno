"""E1's pre-index closure primitive: the closure rule, the RE-DERIVED window, the witness.

Three claims, and the second is the one the plan named as the key risk:

  1. **The closure rule is the label rule, on the other window.** `R_d` is the
     is-a closure of the pre-index frontier, emitted SPARSE (an engine-id index
     list) rather than as a third dense `array<double>` — so a patient whose
     PRE-index record carries `closure(c)` has `c ∈ R_d` and the same patient with
     the same code moved AFTER the index does not.

  2. **The driver's re-derivation reproduces the assembler's own index, exactly.**
     The driver holds no feature frame and may not edit the assembler to get one.
     It does not have to: `case_finding_population_index_table` →
     `_random_event_windows` picks each person's anchor by
     `min hash(person_id, event_date, _RANDOM_WINDOW_SALT)`, deterministic and
     resume-stable by construction. `test_the_rederived_index_is_the_assemblers_own_pick`
     checks the reproduction against `cohorts._random_event_windows` itself on a
     synthetic frame rather than arguing it, and
     `test_the_rederivation_calls_the_assemblers_index_builder_with_its_arguments`
     pins the argument list — the only way the two can diverge once the pick is
     imported rather than copied.

  3. **A bundle says whether it carries the column.** The witness round-trips
     through the cache meta, and a consumer that asks a bundle without one gets a
     named error rather than a Spark missing-column stack trace.

Local-Spark for the windowing/attestation body; pure for the closure kernel and
the witness.
"""
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUD = REPO_ROOT / "analysis" / "cloud"
for _p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# PySpark workers inherit PYTHONPATH, not the driver's sys.path (same note as
# tests/scripts/test_case_finding_cache_mondo.py). Set before the session fixture
# builds the context.
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
                str(REPO_ROOT / "spark-vi"), os.environ.get("PYTHONPATH", ""))
    if p)

import preindex_closure as pic  # noqa: E402
from charmpheno.omop.condition_dag import ConditionDag  # noqa: E402
from spark_vi.models.topic.dag_placement import DagLayout  # noqa: E402


# --------------------------------------------------------------------------- #
# A toy label DAG. Concept ids 100/200/300/400; -1 is the forest root, the same #
# `_FOREST_ROOT_CID` convention every real build uses.                          #
#                                                                              #
#        -1 root                                                                #
#       /        \                                                              #
#     100        200          (100 and 200 are two branches)                    #
#       \        /                                                              #
#         300              <- DIAMOND: 300 has BOTH as parents                  #
#          |                                                                    #
#         400                                                                   #
# --------------------------------------------------------------------------- #
_PARENTS = {100: [-1], 200: [-1], 300: [100, 200], 400: [300]}


def _toy():
    dag = ConditionDag(_PARENTS, -1, {c: f"n{c}" for c in (-1, 100, 200, 300, 400)})
    parent_int, int2cid, cid2int = dag.to_engine()
    lay = DagLayout(parent_int, n_bg=2, tpn=1)
    return dag, parent_int, int2cid, cid2int, lay


class _StubBundle:
    """The three post-prune fields R1.3 says to recover from the BUNDLE, never
    from assembler internals: parent_int -> DagLayout, cid2int, int2cid -> keep."""

    def __init__(self, parent_int, int2cid, cid2int, train_df=None, test_df=None):
        self.parent_int = parent_int
        self.int2cid = int2cid
        self.cid2int = cid2int
        self.train_df = train_df
        self.test_df = test_df


# --------------------------------------------------------------------------- #
# 1. The closure kernel (pure).                                                #
# --------------------------------------------------------------------------- #
def test_closure_of_a_frontier_node_carries_its_whole_ancestry():
    """`R_d` means "already carried closure(c)", so a doc whose pre-index record
    resolves to 400 carries 400, 300, both its parents and the root — exactly the
    positive support of `frontier_to_label`'s dense label vector."""
    _, _, _, cid2int, lay = _toy()
    got = pic.preindex_closure_ids([cid2int[400]], lay, C=len(cid2int))
    assert set(got) == {cid2int[c] for c in (-1, 100, 200, 300, 400)}
    assert got == sorted(got)                    # sparse index LIST, sorted


def test_the_diamond_closure_keeps_both_parents():
    """Multi-parent is the whole reason `R_d` is a closure and not a path: 300 is
    known via BOTH branches, so neither may be dropped."""
    _, _, _, cid2int, lay = _toy()
    got = set(pic.preindex_closure_ids([cid2int[300]], lay, C=len(cid2int)))
    assert cid2int[100] in got and cid2int[200] in got
    assert cid2int[400] not in got               # a DESCENDANT is not "known"


def test_an_empty_pre_index_frontier_carries_nothing():
    """A doc with no resolvable pre-index code carried nothing, so every node is
    incident-eligible for it. `[]`, not "the root"."""
    _, _, _, cid2int, lay = _toy()
    assert pic.preindex_closure_ids([], lay, C=len(cid2int)) == []
    assert pic.preindex_closure_ids(None, lay, C=len(cid2int)) == []


def test_closure_ids_outside_the_label_space_are_dropped():
    _, _, _, cid2int, lay = _toy()
    C = 2                                        # narrower than the DAG
    got = pic.preindex_closure_ids([cid2int[400]], lay, C=C)
    assert got and all(0 <= c < C for c in got)


def test_eligibility_removes_a_prior_carrier_from_BOTH_classes():
    """D2 is symmetric on purpose: dropping prior carriers from the positives only
    would be a different — and wrong — estimator."""
    _, _, _, cid2int, lay = _toy()
    C = len(cid2int)
    carried = pic.preindex_closure_ids([cid2int[300]], lay, C)
    elig = pic.is_incident_eligible(carried, C)
    assert not elig[cid2int[300]] and not elig[cid2int[100]]
    assert elig[cid2int[400]]                    # not carried -> still eligible
    assert pic.is_incident_eligible([], C).all()
    assert elig.dtype == np.bool_


# --------------------------------------------------------------------------- #
# 2. The re-derivation.                                                        #
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
        out = pic.feature_window_index_table(
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
        pic.feature_window_index_table(
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
        [(int(p), int(c), dt.date(*d)) for p, c, d in rows], schema)


def _obs(spark, rows):
    from pyspark.sql.types import (DateType, LongType, StructField, StructType)
    schema = StructType([
        StructField("person_id", LongType(), False),
        StructField("observation_period_start_date", DateType(), False),
        StructField("observation_period_end_date", DateType(), False),
    ])
    return spark.createDataFrame(
        [(int(p), dt.date(*s), dt.date(*e)) for p, s, e in rows], schema)


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

    mine = {r["person_id"]: r["index_date"] for r in pic.feature_window_index_table(
        cond, spark=_FakeSpark(spark, op), cdr="p.d", billing="bp",
        date_col="condition_era_start_date", label_window_days=365).collect()}

    assert mine == theirs and len(theirs) == 3


@pytest.mark.slow
def test_the_feature_window_is_strictly_pre_index(spark):
    """The other half of the re-derivation: `lookback_feature_frames` splits on
    the SAME index, so the frame this module attests over is exactly the events in
    `[index - lookback, index)` — nothing at or after the index can reach `R_d`."""
    from charmpheno.omop.cohorts import _random_event_windows
    from charmpheno.omop.case_finding_assembly import _LOOKBACK_PRIOR_OBS_DAYS

    cond, op = _events(spark, _ROWS), _obs(spark, _OBS)
    idx = {r["person_id"]: r["index_date"] for r in _random_event_windows(
        cond, op, date_col="condition_era_start_date", window_days=365,
        prior_obs_days=_LOOKBACK_PRIOR_OBS_DAYS).collect()}

    feat = pic.feature_window_condition_events(
        _FakeSpark(spark, op), cdr="p.d", billing="bp", person_mod=1,
        lookback_days=1825, label_window_days=365, _cond=cond)
    rows = feat.collect()
    assert rows
    for r in rows:
        d = r["condition_era_start_date"]
        assert d < idx[r["person_id"]]
        assert (idx[r["person_id"]] - d).days <= 1825


def _provider(spark, doc_spec=None):
    """A trivial `events_df -> attested_df` in the seam's shape: the doc attests
    exactly the concept ids it carries. Stands in for the Mondo code-map provider,
    whose only difference is which ids come out."""
    from pyspark.sql import functions as F

    def provider(events_df):
        ev = events_df.withColumn(
            "doc_id", F.concat_ws(":", F.lit("population"),
                                  F.col("person_id").cast("string")))
        roster = ev.groupBy("doc_id").agg(
            F.first("person_id").alias("person_id"),
            F.lit("population").alias("source_cohort"))
        att = ev.groupBy("doc_id").agg(
            F.collect_set(F.col("concept_id").cast("long")).alias("attested_cids"))
        return (roster.join(att, on="doc_id", how="left")
                .withColumn("attested_cids",
                            F.coalesce(F.col("attested_cids"),
                                       F.array().cast("array<bigint>"))))
    return provider


@pytest.mark.slow
def test_a_pre_index_code_lands_in_R_d_and_the_same_code_after_it_does_not(spark):
    """The spec's acceptance fixture (§11/E1), on ONE patient and ONE code moved
    across the index. Same person, same code, same everything else: before the
    index it is a TRACKING row for that node (c ∈ R_d, ineligible); after it, a
    PREDICTION row (c ∉ R_d, eligible)."""
    dag, parent_int, int2cid, cid2int, lay = _toy()
    bundle = _StubBundle(parent_int, int2cid, cid2int)

    # One person, one anchoring event stream, and node 400 coded either well
    # before the chosen index or well after it. The index is the same in both
    # frames because the anchor set is: the 400 row is not itself window-eligible
    # (its forward year runs past the observation end in the LATE case, and in the
    # EARLY case it is the same date the LATE frame lacks) — so the pick is driven
    # by the shared rows, and the assertion is about the WINDOW, not the pick.
    op = _obs(spark, [(1, (2010, 1, 1), (2019, 1, 1))])
    shared = [(1, 100, (2013, 1, 1)), (1, 200, (2014, 1, 1))]
    early = _events(spark, shared + [(1, 400, (2011, 6, 1))])
    late = _events(spark, shared + [(1, 400, (2017, 6, 1))])

    def _r_d(cond):
        feat = pic.feature_window_condition_events(
            _FakeSpark(spark, op), cdr="p.d", billing="bp", person_mod=1,
            lookback_days=1825, label_window_days=365, _cond=cond)
        frame = pic.preindex_closure_frame(
            feat, attested_provider=_provider(spark), before_dag=dag,
            bundle=bundle, n_bg=2, tpn=1)
        return {r["doc_id"]: set(r[pic.PREINDEX_CLOSURE_COL])
                for r in frame.collect()}

    from charmpheno.omop.cohorts import _random_event_windows
    from charmpheno.omop.case_finding_assembly import _LOOKBACK_PRIOR_OBS_DAYS
    idx = {r["person_id"]: r["index_date"] for r in _random_event_windows(
        early, op, date_col="condition_era_start_date", window_days=365,
        prior_obs_days=_LOOKBACK_PRIOR_OBS_DAYS).collect()}
    assert idx[1] > dt.date(2011, 6, 1), "fixture: the early 400 must be pre-index"

    late_idx = {r["person_id"]: r["index_date"] for r in _random_event_windows(
        late, op, date_col="condition_era_start_date", window_days=365,
        prior_obs_days=_LOOKBACK_PRIOR_OBS_DAYS).collect()}
    assert late_idx[1] <= dt.date(2017, 6, 1), \
        "fixture: the late 400 must be at or after the index"

    got_early, got_late = _r_d(early), _r_d(late)
    doc = "population:1"
    assert cid2int[400] in got_early[doc]
    assert cid2int[300] in got_early[doc]         # closure, not just the code
    # The same code AT OR AFTER the index is not "already known": it is the thing
    # to be predicted, and the feature window is strictly `< index`.
    assert cid2int[400] not in got_late[doc]
    assert cid2int[100] in got_late[doc]          # the genuinely-prior codes stay


# --------------------------------------------------------------------------- #
# 3. The witness (R1.4).                                                       #
# --------------------------------------------------------------------------- #
def test_a_bundle_without_the_witness_is_refused_by_name():
    """No silent mixed-vintage cache dirs: the error names the key and the uri so
    the fix is readable off the message, instead of a Spark missing-column stack
    trace from deep inside a readout."""
    b = _StubBundle({}, {}, {})
    with pytest.raises(ValueError) as e:
        pic.require_preindex_closure(b, key="abc123", cache_uri="hdfs:///c")
    msg = str(e.value)
    assert "abc123" in msg and "hdfs:///c" in msg
    assert "--preindex-closure" in msg


def test_a_witness_without_the_column_is_also_refused():
    """The other half of the mixed-vintage failure: meta says yes, parquet says
    no."""
    class _Frame:
        columns = ["doc_id", "label"]

    b = _StubBundle({}, {}, {}, train_df=_Frame())
    b.preindex_closure = pic.preindex_witness()
    with pytest.raises(ValueError, match="mixed-vintage"):
        pic.require_preindex_closure(b)


def test_a_witnessed_bundle_returns_the_column_name_to_read():
    class _Frame:
        columns = ["doc_id", pic.PREINDEX_CLOSURE_COL]

    b = _StubBundle({}, {}, {}, train_df=_Frame())
    b.preindex_closure = pic.preindex_witness()
    assert pic.require_preindex_closure(b) == pic.PREINDEX_CLOSURE_COL
    assert pic.bundle_preindex_witness(b)["version"] == pic.PREINDEX_CLOSURE_VERSION


def test_the_witness_round_trips_through_the_cache_meta():
    """`_meta_dict` -> `_restore_meta`, the path `save`/`try_load` take. Written
    ONLY when the bundle has one, so every existing entry's meta is byte-identical
    and an entry predating this restores as "no column" — which is correct."""
    import _case_finding_cache as ccache

    b = _StubBundle({0: []}, {0: -1}, {-1: 0})
    b.name_by_id = {-1: "root"}
    b.ledger = {"K_nodes": 1}
    b.vocab_maps = [{101: 0}]
    assert "preindex_closure" not in ccache._meta_dict(b)

    b.preindex_closure = pic.preindex_witness()
    meta = ccache._meta_dict(b)
    assert meta["preindex_closure"] == pic.preindex_witness()
    restored = ccache._restore_meta(
        {k: (v if k != "parent_int" else {"0": []}) for k, v in
         {"parent_int": {"0": []}, "int2cid": {"0": "-1"},
          "cid2int": {"-1": "0"}, "name_by_id": {"-1": "root"},
          "ledger": {"K_nodes": 1}, "vocab_maps": [{"101": "0"}],
          "preindex_closure": meta["preindex_closure"]}.items()})
    assert restored["preindex_closure"] == pic.preindex_witness()


@pytest.mark.slow
def test_attach_writes_a_sparse_column_and_the_witness(spark):
    """R1.5: `array<int>` index lists, not a third dense `array<double>` (label +
    labelMask are already 2x3,820 float64 at whole-Mondo). Plus the join
    convention: a doc with no pre-index attestation gets `[]`, not NULL."""
    from pyspark.sql import functions as F

    dag, parent_int, int2cid, cid2int, lay = _toy()
    op = _obs(spark, [(1, (2010, 1, 1), (2021, 1, 1)),
                      (2, (2010, 1, 1), (2021, 1, 1))])
    cond = _events(spark, [(1, 100, (2012, 1, 1)), (1, 300, (2014, 1, 1)),
                           (1, 400, (2016, 1, 1)), (1, 200, (2018, 1, 1)),
                           (2, 200, (2012, 5, 1)), (2, 300, (2015, 1, 1)),
                           (2, 100, (2017, 7, 1)), (2, 400, (2019, 4, 1))])
    docs = spark.createDataFrame(
        [("population:1",), ("population:2",), ("population:9",)], "doc_id STRING")
    bundle = _StubBundle(parent_int, int2cid, cid2int, train_df=docs,
                         test_df=docs)

    pic.attach_preindex_closure_to_bundle(
        spark, bundle, before_dag=dag, attested_provider=_provider(spark),
        cdr="p.d", billing="bp", person_mod=1, lookback_days=1825,
        label_window_days=365, n_bg=2, tpn=1,
        _feature_events=pic.feature_window_condition_events(
            _FakeSpark(spark, op), cdr="p.d", billing="bp", person_mod=1,
            lookback_days=1825, label_window_days=365, _cond=cond))

    assert pic.require_preindex_closure(bundle) == pic.PREINDEX_CLOSURE_COL
    dtype = dict(bundle.train_df.dtypes)[pic.PREINDEX_CLOSURE_COL]
    assert dtype == "array<int>"
    got = {r["doc_id"]: list(r[pic.PREINDEX_CLOSURE_COL])
           for r in bundle.train_df.collect()}
    assert got["population:9"] == []             # unknown doc -> carried nothing
    assert all(c in cid2int.values() for c in got["population:1"])
    assert bundle.train_df.where(
        F.col(pic.PREINDEX_CLOSURE_COL).isNull()).count() == 0
