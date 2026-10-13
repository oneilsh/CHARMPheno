"""Label redesign R2 (spec 2026-10-13): the all-ever topic gate
(`gated_pc_cloud._attach_gate_ever`, `--gate-attestation ever`)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_gate_frontier_d3 import (_BUNDLE_SCHEMA, _StubBundle, _args, _toy,  # noqa: F401
                                   gpc)
from charmpheno.omop.case_finding_assembly import (doc_frontier_engine_ids,
                                                   frontier_to_label)


def test_gate_attestation_moves_the_key_only_when_ever():
    k = lambda a: gpc.multidomain_cache_key(gpc.multidomain_corpus_spec(a, extra_domains=()))
    base = _args()
    assert "gate_attestation" not in gpc.multidomain_corpus_spec(base, extra_domains=())
    assert k(_args(gate_attestation="label")) == k(base)
    ever = gpc.multidomain_corpus_spec(_args(gate_attestation="ever"), extra_domains=())
    assert ever["gate_attestation"] == "ever"
    assert k(_args(gate_attestation="ever")) != k(base)


def _row(toy, person, attested_label_year):
    dag, _pi, int2cid, cid2int, lay = toy
    fr = doc_frontier_engine_ids(sorted(attested_label_year), dag, set(cid2int), cid2int, lay)
    label, mask = frontier_to_label(fr, lay, len(int2cid))
    return (int(person), f"population:{person}", [int(x) for x in fr],
            [float(v) for v in label], [float(v) for v in mask], "feat", 0)


@pytest.mark.slow
def test_ever_gate_rolls_up_all_history_and_leaves_label_and_test(spark):
    from pyspark.sql import functions as F
    toy = _toy()
    dag, parent_int, _i2c, cid2int, _lay = toy
    # person 1: 300 in the label year, 400 coded years earlier (prior, not re-coded);
    # person 2: nothing in the label year, 300 coded later (future);
    # person 3: never attests anything.
    train = spark.createDataFrame([_row(toy, 1, {300}), _row(toy, 2, set()),
                                   _row(toy, 3, set())], _BUNDLE_SCHEMA)
    test = spark.createDataFrame([_row(toy, 9, {300})], _BUNDLE_SCHEMA)
    events = spark.createDataFrame(
        [(1, 300), (1, 400), (2, 300), (9, 400), (3, 999)],
        "person_id long, concept_id long")

    def provider(ev):        # code map = identity on the toy's concept ids
        return (ev.where(F.col("concept_id").isin(100, 200, 300, 400))
                .withColumn("doc_id", F.concat_ws(":", "source_cohort",
                                                  F.col("person_id").cast("string")))
                .groupBy("doc_id").agg(F.first("person_id").alias("person_id"),
                                       F.first("source_cohort").alias("source_cohort"),
                                       F.collect_set("concept_id").alias("attested_cids")))

    b = _StubBundle(parent_int, cid2int, train, test)
    out = gpc._attach_gate_ever(spark, b, provider=provider, before_dag=dag, cdr="x",
                                billing="y", person_mod=1, n_bg=2, tpn=1, _events=events)
    got = {r["person_id"]: r.asDict() for r in out.train_df.collect()}
    assert set(got[1]["frontier"]) == {cid2int[300], cid2int[400]}
    assert set(got[2]["frontier"]) == {cid2int[300]}
    assert got[3]["frontier"] == []
    assert got[1]["label"][cid2int[400]] == 0.0          # label untouched
    assert got[2]["label"][cid2int[300]] == 0.0
    assert out.test_df is test                           # test split untouched
    rec = out.gate_ever_receipt
    assert rec["cells_label"] == 1 and rec["cells_ever"] == 3
    assert rec["docs_gated_label"] == 1 and rec["docs_gated_ever"] == 2


def test_re_readout_recovers_the_ever_gate_from_the_manifest():
    import gated_pc_readout as gpr
    spec = gpc.multidomain_corpus_spec(_args(gate_attestation="ever"), extra_domains=())
    rec = gpr.corpus_spec_from_manifest({"corpus_manifest": dict(spec)})
    assert gpc.multidomain_cache_key(rec) == gpc.multidomain_cache_key(spec)
    old = gpr.corpus_spec_from_manifest({"corpus_manifest": dict(
        gpc.multidomain_corpus_spec(_args(), extra_domains=()))})
    assert gpc.multidomain_cache_key(old) != gpc.multidomain_cache_key(spec)
