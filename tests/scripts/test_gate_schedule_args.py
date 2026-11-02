"""Leaves-first gate schedule plumbing (exp 0153/0154): front matter -> fit argv,
and the driver flags -> estimator Params."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT / "analysis" / "cloud"),
           str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_experiment as rex  # noqa: E402

BASE = {"source_table": "t", "person_mod": 1, "vocab_size": 5000, "min_df": 20,
        "min_patient_count": 20, "doc_min_length": 10, "max_iter": 100,
        "min_n": 0, "n_bg": 8, "tpn": 1, "seed": 0, "readout_mode": "distributed"}


def test_gate_warm_front_matter_is_emitted_only_when_set(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CDR", "cdr")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    plain = rex.build_dag_gated_args(dict(BASE), "/tmp/out")
    assert "--gate-warm" not in plain and "--gate-no-open-seed" not in plain
    on = rex.build_dag_gated_args(dict(BASE, gate_warm=5), "/tmp/out")
    assert on[on.index("--gate-warm") + 1] == "5" and "--gate-no-open-seed" not in on


def test_driver_flags_reach_the_estimator():
    """The driver's flags override the recommended preset; the corpus-derived inputs
    (DAG, labels column, feature columns, vocabularies, node -> terms) reach the
    estimator's Params in user ids. Unset flags keep the preset."""
    import argparse
    import dag_gated_fit as gpc
    from spark_vi.mllib.topic.dag_gated_lda import RECOMMENDED_PARAMS
    dag = {"MONDO:0000001": [], "MONDO:0000002": ["MONDO:0000001"]}
    args = argparse.Namespace(
        gate_warm=4, gate_seed_mode="balanced", gate_seed_leaves=True,
        n_bg=2, tpn=2, max_iter=7, tau0=None, kappa=None, subsampling_rate=None,
        gamma_shape=None, cavi_max_iter=None, cavi_tol=None,
        optimize_doc_concentration=None, doc_concentration=None, seed=3,
        min_docs_per_node=None,
        _dag=dag, _labels_col="gate_labels", _domain_cols=["features_0"],
        _vocabularies=[[101, 102, 103]],
        _node_terms={"MONDO:0000002": [("features_0", 103)]})
    est = gpc._build_estimator(args)
    g = est.getOrDefault
    assert type(est).__name__ == "DagGatedLDAEstimator"
    assert g("gateWarm") == 4 and g("gateOpenSeed") is True
    assert g("gateSeedMode") == "balanced" and g("gateSeedLeaves") is True
    assert g("gateNBg") == 2 and g("gateTpn") == 2 and g("maxIter") == 7
    assert g("seed") == 3
    # unset flags: the preset's values
    assert g("learningOffset") == RECOMMENDED_PARAMS["learningOffset"]
    assert g("subsamplingRate") == RECOMMENDED_PARAMS["subsamplingRate"]
    assert g("docConcentration") == RECOMMENDED_PARAMS["docConcentration"]
    assert g("optimizeDocConcentration") is False
    # the corpus' inputs, and the clinical node-set threshold (>= the egress floor)
    assert g("minDocsPerNode") == gpc.DEFAULT_MIN_DOCS_PER_NODE == 100
    assert g("labelsCol") == "gate_labels"
    assert g("featuresCols") == ["features_0"]
    assert est.getDag() == dag
    assert est.getVocabularies() == [[101, 102, 103]]
    assert est.getNodeTerms() == {"MONDO:0000002": [("features_0", 103)]}


def test_the_corpus_helpers_build_vocabularies_and_node_terms():
    import dag_gated_fit as gpc
    assert gpc.vocabularies_of([{101: 1, 100: 0}, {7: 0}]) == [[100, 101], [7]]
    nt = gpc.node_terms_of([(101, "M:1"), (102, "M:1"), (999, "M:2")],
                           {101: 0, 102: 1})
    assert nt == {"M:1": [("features_0", 101), ("features_0", 102)]}
