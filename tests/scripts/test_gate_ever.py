"""Label redesign R2 (spec 2026-10-13; R8 of spec 2026-10-30): the all-ever gate
labels (`dag_gated_fit._attach_gate_ever`, `--gate-attestation ever`): a
`gate_labels` column on the TRAIN split beside the label-year `labels`."""
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


class _StubBundle:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df


_BUNDLE_SCHEMA = ("person_id long, doc_id string, labels array<string>, "
                  "features string, episode_no long")


def _args(**over):
    base = dict(
        dag_source="mondo_native", disease="rare6", cdr="p.d", billing="proj",
        source_table="condition_era", person_mod=10, vocab_size=5000, min_df=20,
        min_patient_count=20, doc_min_length=0, holdout_frac=0.2,
        strip_mode="test_only", lookback_days=365,
        label_window_days=365, label_mask_mode="full", window_mode="lookback",
        prior_obs_days=365, window_days=365, mondo_version="2026-06-02",
        mondo_branch="", mondo_cache_dir="data/mondo")
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


def _row(person, labels):
    return (int(person), f"population:{person}", sorted(labels), "feat", 0)


@pytest.mark.slow
def test_ever_gate_adds_all_history_labels_and_leaves_labels_and_test(spark):
    from pyspark.sql import functions as F
    # person 1: T3 in the label year, T4 coded years earlier (prior, not re-coded);
    # person 2: nothing in the label year, T3 coded later (future);
    # person 3: never attests anything.
    train = spark.createDataFrame([_row(1, {"T3"}), _row(2, set()), _row(3, set())],
                                  _BUNDLE_SCHEMA)
    test = spark.createDataFrame([_row(9, {"T3"})], _BUNDLE_SCHEMA)
    events = spark.createDataFrame(
        [(1, 300), (1, 400), (2, 300), (9, 400), (3, 999)],
        "person_id long, concept_id long")
    term = {300: "T3", 400: "T4"}

    def provider(ev):        # the code ladder: concept 300 -> T3, 400 -> T4
        m = spark.createDataFrame(list(term.items()), "concept_id long, node string")
        return (ev.join(m, "concept_id")
                .withColumn("doc_id", F.concat_ws(":", "source_cohort",
                                                  F.col("person_id").cast("string")))
                .groupBy("doc_id").agg(F.first("person_id").alias("person_id"),
                                       F.first("source_cohort").alias("source_cohort"),
                                       F.array_sort(F.collect_set("node"))
                                       .alias("labels")))

    b = _StubBundle(train, test)
    out = gpc._attach_gate_ever(spark, b, provider=provider, cdr="x", billing="y",
                                person_mod=1, _events=events)
    assert dict(out.train_df.dtypes)["gate_labels"] == "array<string>"
    got = {r["person_id"]: r.asDict() for r in out.train_df.collect()}
    assert got[1]["gate_labels"] == ["T3", "T4"]
    assert got[2]["gate_labels"] == ["T3"]
    assert got[3]["gate_labels"] == []                   # present, empty
    assert got[1]["labels"] == ["T3"] and got[2]["labels"] == []   # labels untouched
    assert out.test_df is test                           # test split untouched
    rec = out.gate_ever_receipt
    assert rec["cells_label"] == 1 and rec["cells_ever"] == 3
    assert rec["docs_gated_label"] == 1 and rec["docs_gated_ever"] == 2


@pytest.mark.slow
def test_gate_column_join_refuses_a_missing_or_duplicate_gate(spark):
    df = spark.createDataFrame([_row(1, {"T3"}), _row(2, set())], _BUNDLE_SCHEMA)
    dup = spark.createDataFrame(
        [(1, "population:1", ["T3"]), (1, "population:1", ["T4"]),
         (2, "population:2", [])], "person_id long, doc_id string, g array<string>")
    with pytest.raises(ValueError, match="duplicate"):
        gpc._attach_gate_column(df, dup, "g")
    missing = spark.createDataFrame(
        [(1, "population:1", ["T3"])], "person_id long, doc_id string, g array<string>")
    with pytest.raises(ValueError, match="received no"):
        gpc._attach_gate_column(df, missing, "g")


def test_re_readout_recovers_the_ever_gate_from_the_manifest():
    import dag_gated_readout as gpr
    spec = gpc.multidomain_corpus_spec(_args(gate_attestation="ever"), extra_domains=())
    rec = gpr.corpus_spec_from_manifest({"corpus_manifest": dict(spec)})
    assert gpc.multidomain_cache_key(rec) == gpc.multidomain_cache_key(spec)
    old = gpr.corpus_spec_from_manifest({"corpus_manifest": dict(
        gpc.multidomain_corpus_spec(_args(), extra_domains=()))})
    assert gpc.multidomain_cache_key(old) != gpc.multidomain_cache_key(spec)
