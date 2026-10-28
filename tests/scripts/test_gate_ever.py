"""Label redesign R2 (spec 2026-10-13): the all-ever topic gate
(`dag_gated_fit._attach_gate_ever`, `--gate-attestation ever`)."""
import os
import sys
import types
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

import dag_gated_fit as gpc  # noqa: E402
from charmpheno.omop.case_finding_assembly import (  # noqa: E402
    doc_frontier_engine_ids, frontier_to_label)
from charmpheno.omop.condition_dag import ConditionDag  # noqa: E402
from spark_vi.dag import DagLayout  # noqa: E402

# A toy label DAG: concept ids 100/200/300/400 under the forest root -1, with
# 300 under 100 and 400 under 200 (two incomparable frontier nodes).
_PARENTS = {100: [-1], 200: [-1], 300: [100], 400: [200]}


def _toy():
    dag = ConditionDag(_PARENTS, -1, {c: f"n{c}" for c in (-1, 100, 200, 300, 400)})
    parent_int, int2cid, cid2int = dag.to_engine()
    lay = DagLayout(parent_int, n_bg=2, tpn=1)
    return dag, parent_int, int2cid, cid2int, lay


class _StubBundle:
    """The post-prune fields the gate post-pass reads, plus the two frames."""

    def __init__(self, parent_int, cid2int, train_df, test_df):
        self.parent_int = parent_int
        self.cid2int = cid2int
        self.train_df = train_df
        self.test_df = test_df


_BUNDLE_SCHEMA = ("person_id long, doc_id string, frontier array<bigint>, "
                  "label array<double>, labelMask array<double>, "
                  "features string, episode_no long")


def _args(**over):
    base = dict(
        dag_source="mondo_native", disease="rare6", cdr="p.d", billing="proj",
        source_table="condition_era", person_mod=10, vocab_size=5000, min_df=20,
        min_patient_count=20, doc_min_length=0, holdout_frac=0.2,
        n_bg=2, tpn=1, strip_mode="test_only", lookback_days=365,
        label_window_days=365, label_mask_mode="full", window_mode="lookback",
        prior_obs_days=365, window_days=365, mondo_version="2026-06-02",
        mondo_branch="", min_positives=100, mondo_cache_dir="data/mondo")
    base.update(over)
    return types.SimpleNamespace(**base)


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
    import dag_gated_readout as gpr
    spec = gpc.multidomain_corpus_spec(_args(gate_attestation="ever"), extra_domains=())
    rec = gpr.corpus_spec_from_manifest({"corpus_manifest": dict(spec)})
    assert gpc.multidomain_cache_key(rec) == gpc.multidomain_cache_key(spec)
    old = gpr.corpus_spec_from_manifest({"corpus_manifest": dict(
        gpc.multidomain_corpus_spec(_args(), extra_domains=()))})
    assert gpc.multidomain_cache_key(old) != gpc.multidomain_cache_key(spec)
