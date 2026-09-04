"""Exp 0111 WP-D3 — separate the estimator's GATE from the outcome LABEL (D13/D14).

WP-D2 wired the episode / matched-random index into the Mondo fit driver; both
arms assemble with `frontier` and `label` BOTH derived from the 365-day forward
frame. WP-D3 is the driver-owned, MISS-only post-pass that swaps ONLY the
`frontier` gate to each document's 90-day presentation window [index, index+90d)
while leaving `label`/`labelMask` frozen at the 365-day frame the assembler baked.

Why a post-pass, not a HIT-time transform: the gate roll-up walks the PRE-PRUNE
DAG (`before_dag`), which lives only inside the assemble closure — never in a
loaded HIT bundle — so the swap must run before the bundle is cached. The swap
CALLS the assembler's `attach_frontiers` (never edits it) over a narrower (90-day
rather than 365-day) attested set.

What is pinned here, and why each failure would otherwise be silent:

  1. **Cache-key folding** (pure): a gated episode spec keys DIFFERENTLY from an
     un-gated one; a no-arm / un-gated spec keys BYTE-IDENTICALLY to the pinned
     Mondo key (a key move silently orphans a ~20-min bundle).
  2. **Spec + manifest** (pure): `--gate-frontier-days` sets `gate_frontier_mode`
     ONLY on an arm, and a re-readout recovers it so the key round-trips.
  3. **The gate node-set + roll-up** (`@slow`, local Spark): multimorbidity in one
     window, a later-forward-year outcome that is in the label but NOT the gate, a
     pre-index recurrence that does not open the gate, and DAG roll-up parity with
     `attach_frontiers` on the restricted attested set.
  4. **The D13 guarantee** (`@slow`): `label`/`labelMask` are byte-identical after
     the swap; only `frontier` moves.
  5. **Join integrity** (`@slow`): a missing / duplicate doc key raises loudly;
     cap integrity — only retained bundle documents receive a gate.

Groups 1-2 are pure; 3-5 use local Spark (`@slow` + the `spark` fixture).
"""
import os
import sys
import types
from datetime import date

import pytest

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
CLOUD = REPO_ROOT / "analysis" / "cloud"
for _p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
                str(REPO_ROOT / "spark-vi"), os.environ.get("PYTHONPATH", ""))
    if p)

import gated_pc_cloud as gpc  # noqa: E402
from charmpheno.omop.case_finding_assembly import (  # noqa: E402
    attach_frontiers, doc_frontier_engine_ids, frontier_to_label)
from charmpheno.omop.condition_dag import ConditionDag  # noqa: E402
from spark_vi.models.topic.dag_placement import DagLayout  # noqa: E402


# --------------------------------------------------------------------------- #
# A toy label DAG. Concept ids 100/200/300/400; -1 is the forest root.        #
#                                                                             #
#          -1 root                                                            #
#         /       \                                                           #
#       100        200                                                        #
#        |          |                                                         #
#       300        400     <- two INCOMPARABLE frontier nodes (multimorbidity) #
# --------------------------------------------------------------------------- #
_PARENTS = {100: [-1], 200: [-1], 300: [100], 400: [200]}


def _toy():
    dag = ConditionDag(_PARENTS, -1, {c: f"n{c}" for c in (-1, 100, 200, 300, 400)})
    parent_int, int2cid, cid2int = dag.to_engine()
    lay = DagLayout(parent_int, n_bg=2, tpn=1)
    return dag, parent_int, int2cid, cid2int, lay


class _StubBundle:
    """The post-prune fields `_attach_gate_frontier` reads: cid2int (== keep) and
    parent_int (-> DagLayout), plus the two frames it rewrites."""

    def __init__(self, parent_int, cid2int, train_df, test_df):
        self.parent_int = parent_int
        self.cid2int = cid2int
        self.train_df = train_df
        self.test_df = test_df


_BUNDLE_SCHEMA = ("person_id long, doc_id string, frontier array<bigint>, "
                  "label array<double>, labelMask array<double>, "
                  "features string, episode_no long")


def _doc_row(spark_dag, person, index_str, attested365):
    """One bundle row whose `frontier`/`label`/`labelMask` are baked from the FULL
    365-day attested concept-id set `attested365` — exactly what the assembler bakes
    with `emit_labels=True`, so the swap has a real 365-day baseline to preserve."""
    dag, _pi, int2cid, cid2int, lay = spark_dag
    keep = set(cid2int)
    C = len(int2cid)
    fr = doc_frontier_engine_ids(sorted(int(c) for c in attested365),
                                 dag, keep, cid2int, lay)
    label, mask = frontier_to_label(fr, lay, C)
    return (int(person), f"episode:{person}:{index_str}",
            [int(x) for x in fr], [float(v) for v in label],
            [float(v) for v in mask], "feat", 0)


def _first_attestation(spark, rows):
    """`(person_id, node_cid, first_attested_date)` — the E4 sidecar frame."""
    return spark.createDataFrame(
        [(int(p), int(n), d) for p, n, d in rows],
        "person_id long, node_cid long, first_attested_date date")


# --------------------------------------------------------------------------- #
# 1. Cache-key folding + spec/manifest (pure — no SparkSession).              #
# --------------------------------------------------------------------------- #
def _args(**over):
    base = dict(
        dag_source="mondo_native", disease="rare6", cdr="p.d", billing="proj",
        source_table="condition_era", person_mod=10, vocab_size=5000, min_df=20,
        min_patient_count=20, doc_min_length=0, min_n=50, holdout_frac=0.2,
        n_bg=2, tpn=1, strip_mode="test_only", lookback_days=365,
        label_window_days=365, label_mask_mode="full", window_mode="lookback",
        prior_obs_days=365, window_days=365, mondo_version="2026-06-02",
        mondo_branch="", min_positives=100, mondo_cache_dir="data/mondo",
        dag_collapse=False, preindex_closure=False,
        index_arm="", episode_gap_days=90, episode_cap=3, episode_salt="0111",
        episode_prior_obs_days=365, episode_window_days=365,
        episode_sidecar_uri="", gate_frontier_days=0)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_gate_mode_moves_the_episode_key_but_not_the_pinned_no_arm_key():
    from _case_finding_cache import compute_bundle_cache_key
    # The no-arm compatibility guarantee is untouched: passing no gate (or an
    # explicit 'none') keeps the tripwire's pinned Mondo key byte-identical.
    md_base = dict(
        source_table="condition_era", person_mod=10, vocab_size=5000, min_df=20,
        min_patient_count=20, doc_min_length=0, prior_obs_days=365,
        window_days=365, disease="diabetes", min_n=50, holdout_frac=0.2,
        split_salt=20260716, n_bg=2, tpn=1, cdr="p.d", multidomain=True,
        extra_domains=("drug",), index_mode="population", mondo=True,
        mondo_version="2026-06-02", mondo_branch="", min_positives=100)
    assert compute_bundle_cache_key(**md_base) == "0ba6393f0d92af07"
    assert compute_bundle_cache_key(**md_base, gate_frontier_mode=None) \
        == "0ba6393f0d92af07"
    assert compute_bundle_cache_key(**md_base, gate_frontier_mode="none") \
        == "0ba6393f0d92af07"
    # A gated episode bundle is a DISTINCT artifact from the un-gated one.
    ungated = gpc.multidomain_corpus_spec(_args(index_arm="episode"),
                                          extra_domains=())
    gated = gpc.multidomain_corpus_spec(
        _args(index_arm="episode", gate_frontier_days=90), extra_domains=())
    assert gpc.multidomain_cache_key(ungated) != gpc.multidomain_cache_key(gated)


def test_gate_mode_appears_on_the_spec_only_with_an_arm_and_a_positive_width():
    # Off by default (arm, no gate): no field, un-gated key.
    ungated = gpc.multidomain_corpus_spec(_args(index_arm="episode"),
                                          extra_domains=())
    assert "gate_frontier_mode" not in ungated
    # On (arm + width): the token names the window.
    gated = gpc.multidomain_corpus_spec(
        _args(index_arm="episode", gate_frontier_days=90), extra_domains=())
    assert gated["gate_frontier_mode"] == "gate90d"
    # A gate width WITHOUT an arm never touches the population/disease spec — the
    # gate lives only on the external path, so every existing key stays identical.
    no_arm = gpc.multidomain_corpus_spec(_args(gate_frontier_days=90),
                                         extra_domains=())
    assert "gate_frontier_mode" not in no_arm
    assert no_arm["index_mode"] == "population"


def test_gate_width_moves_the_key_and_the_episode_sampling_dict_is_unchanged():
    k = lambda a: gpc.multidomain_cache_key(
        gpc.multidomain_corpus_spec(a, extra_domains=()))
    k90 = k(_args(index_arm="episode", gate_frontier_days=90))
    k60 = k(_args(index_arm="episode", gate_frontier_days=60))
    assert k90 != k60                                   # width is corpus identity
    # WP-D2's pinned sampling-dict shape must not gain a gate field (the gate is a
    # SIBLING key, not folded into episode_sampling).
    spec = gpc.multidomain_corpus_spec(
        _args(index_arm="episode", gate_frontier_days=90), extra_domains=())
    assert spec["episode_sampling"] == {
        "arm": "episode", "gap_days": 90, "cap": 3, "salt": "0111",
        "prior_obs_days": 365, "window_days": 365}


def test_re_readout_recovers_the_gate_mode_from_the_manifest():
    import gated_pc_readout as gpr
    spec = gpc.multidomain_corpus_spec(
        _args(dag_source="mondo_native", index_arm="episode",
              gate_frontier_days=90, episode_sidecar_uri="gs://b/sc"),
        extra_domains=())
    recovered = gpr.corpus_spec_from_manifest({"corpus_manifest": dict(spec)})
    assert recovered["gate_frontier_mode"] == "gate90d"
    assert (gpc.multidomain_cache_key(recovered)
            == gpc.multidomain_cache_key(spec))


def test_gate_mode_token_round_trips_to_its_width():
    assert gpc._gate_frontier_mode(0) == "none"
    assert gpc._gate_frontier_mode(90) == "gate90d"
    assert gpc._gate_frontier_days("none") == 0
    assert gpc._gate_frontier_days(None) == 0
    assert gpc._gate_frontier_days("gate90d") == 90
    with pytest.raises(ValueError, match="unrecognized gate_frontier_mode"):
        gpc._gate_frontier_days("ninety")


# --------------------------------------------------------------------------- #
# 2-4. The gate node-set, roll-up, and the D13 label guarantee (@slow).       #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
class TestGateFrontier:
    def _gate(self, spark, first_rows, docs, *, gate_days=90):
        """Build a one-split bundle from `docs` (each (person, index_str,
        attested365)), run `_attach_gate_frontier`, return {doc_id: row-dict}."""
        toy = _toy()
        dag, parent_int, _int2cid, cid2int, _lay = toy
        train = spark.createDataFrame(
            [_doc_row(toy, p, i, a) for p, i, a in docs], _BUNDLE_SCHEMA)
        empty = spark.createDataFrame([], _BUNDLE_SCHEMA)
        bundle = _StubBundle(parent_int, cid2int, train, empty)
        first = _first_attestation(spark, first_rows)
        out = gpc._attach_gate_frontier(
            spark, bundle, first_attestation=first, before_dag=dag,
            gate_days=gate_days, n_bg=2, tpn=1)
        return toy, {r["doc_id"]: r.asDict() for r in out.train_df.collect()}

    def test_same_cluster_multimorbidity_both_land_in_the_gate(self, spark):
        # 300 and 400 first-attested in the same 90-day window -> BOTH incomparable
        # frontier nodes are in the gate frontier.
        toy, got = self._gate(
            spark,
            [(100, 300, date(2015, 1, 10)), (100, 400, date(2015, 1, 20))],
            [(100, "2015-01-01", {300, 400})])
        _dag, _pi, _i2c, cid2int, _lay = toy
        fr = set(got["episode:100:2015-01-01"]["frontier"])
        assert cid2int[300] in fr and cid2int[400] in fr

    def test_a_later_forward_year_attestation_is_in_the_label_but_not_the_gate(
            self, spark):
        # 300 attested at index+9d (in the 90d gate); 400 first-attested at
        # index+212d (in the 365d label, OUTSIDE the gate). The gate carries 300
        # only; the label — baked from the 365-day {300,400} — still carries 400.
        toy, got = self._gate(
            spark,
            [(100, 300, date(2015, 1, 10)), (100, 400, date(2015, 8, 1))],
            [(100, "2015-01-01", {300, 400})])
        _dag, _pi, _i2c, cid2int, lay = toy
        row = got["episode:100:2015-01-01"]
        gate_fr = set(row["frontier"])
        assert cid2int[300] in gate_fr
        assert cid2int[400] not in gate_fr           # OUT of the 90-day gate
        # ...but the 365-day outcome label still knows 400 (closure incl. it).
        assert row["label"][cid2int[400]] == 1.0

    def test_a_pre_index_recurrence_does_not_open_the_gate(self, spark):
        # 300's FIRST attestation is pre-index (2014-12-01); the gate uses first
        # attestation, so an in-window recurrence never reaches the sidecar and the
        # gate stays EMPTY -> [] (a valid background-only frontier, doc not dropped).
        _toyv, got = self._gate(
            spark, [(100, 300, date(2014, 12, 1))],
            [(100, "2015-01-01", {300})])
        assert got["episode:100:2015-01-01"]["frontier"] == []

    def test_dag_rollup_parity_with_attach_frontiers_on_the_restricted_set(
            self, spark):
        # The gate frontier must EQUAL `attach_frontiers` computed independently on
        # the same 90-day-restricted attested set — the roll-up is the ordinary
        # frontier path, only the input window is narrower.
        first_rows = [(100, 300, date(2015, 1, 10)), (100, 400, date(2015, 8, 1))]
        docs = [(100, "2015-01-01", {300, 400})]
        toy, got = self._gate(spark, first_rows, docs)
        dag, _pi, _i2c, cid2int, lay = toy
        keep = set(cid2int)
        # Independent path: the restricted (in-gate) attested set is {300} only.
        restricted = spark.createDataFrame(
            [(100, "episode:100:2015-01-01", [300])],
            "person_id long, doc_id string, attested_cids array<bigint>")
        ref = {r["doc_id"]: sorted(r["frontier"]) for r in
               attach_frontiers(restricted, dag, keep, cid2int, lay).collect()}
        assert sorted(got["episode:100:2015-01-01"]["frontier"]) \
            == ref["episode:100:2015-01-01"]

    def test_label_and_mask_are_byte_identical_after_the_swap(self, spark):
        # The D13 guarantee, pinned hard: the swap moves ONLY `frontier`.
        # label/labelMask/features/episode_no ride through untouched.
        toy = _toy()
        dag, parent_int, _int2cid, cid2int, _lay = toy
        docs = [(100, "2015-01-01", {300, 400}), (250, "2016-02-02", {300})]
        train = spark.createDataFrame(
            [_doc_row(toy, p, i, a) for p, i, a in docs], _BUNDLE_SCHEMA)
        before = {r["doc_id"]: r.asDict() for r in train.collect()}
        empty = spark.createDataFrame([], _BUNDLE_SCHEMA)
        bundle = _StubBundle(parent_int, cid2int, train, empty)
        # 400 out of the gate for person 100 (index+212d) -> the gate DOES differ
        # from the 365-day frontier, so a preserved label is a real invariant.
        first = _first_attestation(spark, [
            (100, 300, date(2015, 1, 10)), (100, 400, date(2015, 8, 1)),
            (250, 300, date(2016, 2, 10))])
        out = gpc._attach_gate_frontier(
            spark, bundle, first_attestation=first, before_dag=dag,
            gate_days=90, n_bg=2, tpn=1)
        after = {r["doc_id"]: r.asDict() for r in out.train_df.collect()}
        assert set(after) == set(before)
        moved = False
        for k in before:
            assert after[k]["label"] == before[k]["label"]
            assert after[k]["labelMask"] == before[k]["labelMask"]
            assert after[k]["features"] == before[k]["features"]
            assert after[k]["episode_no"] == before[k]["episode_no"]
            moved = moved or (after[k]["frontier"] != before[k]["frontier"])
        assert moved                                 # the frontier really did swap

    def test_cap_integrity_only_retained_documents_get_a_gate(self, spark):
        # A person/index present in the sidecar but NOT in the bundle (dropped by
        # the per-person cap) contributes no gate row: the output has EXACTLY the
        # bundle's documents, no phantom.
        toy, got = self._gate(
            spark,
            [(100, 300, date(2015, 1, 10)),
             (999, 400, date(2015, 1, 10))],        # 999 not in the bundle
            [(100, "2015-01-01", {300})])
        assert set(got) == {"episode:100:2015-01-01"}


# --------------------------------------------------------------------------- #
# 5. Join integrity (@slow) — the loud-failure guards on `_overwrite_frontier`. #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
class TestJoinIntegrity:
    def _df(self, spark):
        return spark.createDataFrame(
            [(100, "episode:100:2015-01-01", [1]),
             (250, "episode:250:2016-02-02", [2])],
            "person_id long, doc_id string, frontier array<bigint>")

    def test_a_missing_doc_key_raises(self, spark):
        df = self._df(spark)
        # A gate frame missing person 250's document -> a null join -> hard error,
        # never an empty-frontier fallback.
        gate = spark.createDataFrame(
            [(100, "episode:100:2015-01-01", [7])],
            "person_id long, doc_id string, frontier array<bigint>")
        with pytest.raises(ValueError, match="received no gate frontier"):
            gpc._overwrite_frontier(df, gate).collect()

    def test_a_duplicate_doc_key_raises(self, spark):
        df = self._df(spark)
        # Two gate rows for the SAME document -> an ambiguous frontier -> refused.
        gate = spark.createDataFrame(
            [(100, "episode:100:2015-01-01", [7]),
             (100, "episode:100:2015-01-01", [8]),
             (250, "episode:250:2016-02-02", [9])],
            "person_id long, doc_id string, frontier array<bigint>")
        with pytest.raises(ValueError, match="duplicate .* rows in the gate frame"):
            gpc._overwrite_frontier(df, gate).collect()

    def test_a_clean_overwrite_replaces_only_the_frontier(self, spark):
        df = self._df(spark)
        gate = spark.createDataFrame(
            [(100, "episode:100:2015-01-01", [7]),
             (250, "episode:250:2016-02-02", [])],   # an EMPTY gate is valid
            "person_id long, doc_id string, frontier array<bigint>")
        out = {r["doc_id"]: r["frontier"]
               for r in gpc._overwrite_frontier(df, gate).collect()}
        assert out == {"episode:100:2015-01-01": [7],
                       "episode:250:2016-02-02": []}
