"""Tests for the gated_pc cloud driver: the pure pc_topics_lr scorer, the metrics
and their formatting, and the arg surface. The readout's solver checkpoint and heads
persistence moved to spark_vi.readout with ADR 0051 (spark-vi/tests/
test_readout_heads.py). The end-to-end BQ+fit run is the cluster smoke (main()
reads the CDR via the spark-bigquery connector)."""

import numpy as np
import pytest


def test_dag_children_and_min_depth():
    from spark_vi.dag import children_and_min_depth as _dag_children_and_min_depth
    # 0 root; 1,2 -> 0; 3,4 -> 1. C=5.
    parent_int = {1: [0], 2: [0], 3: [1], 4: [1]}
    children, depth = _dag_children_and_min_depth(parent_int, C=5)
    assert children[0] == [1, 2] and sorted(children[1]) == [3, 4]
    assert children[3] == [] and children[4] == []
    assert depth == {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}


def test_precision_at_recall_and_recall_at_fdr():
    """The per-node PR block the readout reports (`spark_vi.eval.metrics`, whose own
    tests hold scikit-learn as the oracle)."""
    import math
    from spark_vi.eval.metrics import core

    def _m(y, p, rt=(), fq=()):
        return core.node_metrics(core.tally(p, y), recall_targets=rt, fdr_targets=fq)
    # perfectly separable: at any recall, precision 1.0; at any FDR, recall 1.0.
    y = [0, 0, 1, 1]
    m = _m(y, [0.1, 0.2, 0.8, 0.9], (0.5, 1.0), (0.0, 0.5))
    assert m["par"] == {0.5: 1.0, 1.0: 1.0} and m["raf"] == {0.0: 1.0, 0.5: 1.0}
    # one hard positive ranked below a negative: recall 1.0 now costs some precision.
    assert _m(y, [0.1, 0.85, 0.8, 0.9], (1.0,))["par"][1.0] < 1.0
    # degenerate single-class column -> NaN, not an exception.
    assert math.isnan(_m([1, 1, 1], [0.2, 0.5, 0.9], (0.5,))["par"][0.5])


def test_score_arm_separable_and_detection_shape():
    import numpy as np
    from dag_gated_fit import score_arm
    Pi_tr = np.array([[1., 0], [.9, .1], [.1, .9], [0, 1.], [.8, .2], [.2, .8]])
    y_tr = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]], float)
    Pi_te = np.array([[.95, .05], [.05, .95], [.85, .15], [.15, .85]])
    y_te = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
    arm = score_arm(Pi_tr, y_tr, np.ones_like(y_tr), Pi_te, y_te, np.ones_like(y_te),
                    2, recall_targets=[0.5, 0.9], fdr_targets=[0.1, 0.5])
    assert arm["ranking"]["auc"] == 1.0
    assert set(arm["pr"]["par"]) == {0.5, 0.9}
    assert set(arm["pr"]["raf"]) == {0.1, 0.5}
    assert arm["pr"]["n_scored"] == 2
    # detection present with an AP + a precision@recall map (root=node0 label).
    assert "ap" in arm["detection"] and set(arm["detection"]["par"]) == {0.5, 0.9}


# --------------------------------------------------------------------------- #
# detection: the constant-column artifact (exp 0109 part B)                    #
# --------------------------------------------------------------------------- #
def test_detection_excludes_constant_node_columns():
    """A degenerate node's CONSTANT fallback column pinned detection AUC at exactly
    0.5000 in every whole-Mondo readout (exp 0104): the per-doc score is a max over
    node columns, so one constant column sitting above the informative ones makes
    the max constant too. It carries no per-document information by construction,
    so excluding it cannot lose signal — it can only stop one being masked."""
    import numpy as np
    from dag_gated_fit import detection_readout
    # node 0 = root (the foreground label); node 1 = informative; node 2 = constant
    # HIGH (0.9), which is exactly what dominates the max.
    y = np.array([[0., 0, 0], [0., 0, 0], [1., 1, 0], [1., 1, 0]])
    proba = np.array([[0.1, 0.10, 0.9], [0.1, 0.20, 0.9],
                      [0.9, 0.80, 0.9], [0.9, 0.85, 0.9]])
    det = detection_readout(proba, y, [0.5])
    assert det["skipped"] is None
    assert det["n_constant_nodes"] == 1
    assert det["auc"] == 1.0            # the informative column decides it
    # the artifact this replaces: with the constant column in the pool the max is
    # a flat 0.9 for every doc, and roc_auc_score on a constant score is 0.5000.
    from sklearn.metrics import roc_auc_score
    flat = proba[:, 1:].max(axis=1)
    assert np.ptp(flat) == 0.0
    assert roc_auc_score(y[:, 0], flat) == 0.5


def test_detection_unchanged_when_no_column_is_constant():
    """The fix is a no-op on a healthy readout — same score, same numbers."""
    import numpy as np
    from dag_gated_fit import detection_readout
    y = np.array([[0., 0, 0], [0., 0, 0], [1., 1, 0], [1., 0, 1]])
    proba = np.array([[0.1, 0.2, 0.15], [0.3, 0.1, 0.25],
                      [0.7, 0.9, 0.35], [0.8, 0.4, 0.95]])
    det = detection_readout(proba, y, [0.5])
    assert det["n_constant_nodes"] == 0
    from sklearn.metrics import roc_auc_score
    assert det["auc"] == roc_auc_score(y[:, 0], proba[:, 1:].max(axis=1))


def test_detection_skips_when_every_disease_column_is_constant():
    """Nothing to rank: say so rather than reporting a meaningless 0.5."""
    import numpy as np
    from dag_gated_fit import detection_readout
    y = np.array([[0., 0, 0], [1., 1, 0]])
    proba = np.array([[0.5, 0.4, 0.6], [0.5, 0.4, 0.6]])
    det = detection_readout(proba, y, [0.5])
    assert det["skipped"] == "all non-root node scores are constant"
    assert det["n_constant_nodes"] == 2


def test_format_arm_readout_names_the_excluded_columns():
    """Printed rather than silently applied: at whole-Mondo the count IS the
    degenerate-node count, so the line doubles as a structural read."""
    from dag_gated_fit import format_arm_readout
    arm = {"ranking": {"auc": 0.7, "ap": 0.5, "n_labels_scored": 3},
           "pr": {"par": {0.5: 0.4}, "raf": {0.1: 0.2}},
           "detection": {"skipped": None, "auc": 0.81, "ap": 0.6,
                         "prevalence": 0.3, "n_constant_nodes": 763,
                         "par": {0.5: 0.4}}}
    assert "[763 constant node col(s) excluded]" in format_arm_readout("x", arm)
    arm["detection"]["n_constant_nodes"] = 0
    assert "constant node col" not in format_arm_readout("x", arm)


def test_parse_args_surface_defaults_and_arms():
    from dag_gated_fit import parse_args
    a = parse_args([
        "--cdr", "p.d", "--billing", "bp", "--disease", "rare6",
        "--n-bg", "2", "--tpn", "1", "--out-dir", "/tmp/g",
    ])
    assert a.disease == "rare6" and a.n_bg == 2
    assert a.label_mask_mode == "full"         # full-observation default
    assert not hasattr(a, "with_dag_head")     # the fit ends at save (2026-10)
    import pytest
    for gone in ("--weight-y", "--head-l2", "--init", "--localize-head"):
        with pytest.raises(SystemExit):        # removed by ADR 0050
            parse_args(["--cdr", "p.d", "--billing", "bp", "--out-dir", "/t",
                        gone, "1"])
    with pytest.raises(SystemExit):            # no prefix matching: --k != --kappa
        parse_args(["--cdr", "p.d", "--billing", "bp", "--out-dir", "/t", "--k", "4"])


class _FakeParam:
    def __init__(self, name):
        self.name = name


class _FakeEst:
    """Just the Params surface `fit_identity` reads."""

    def __init__(self, **vals):
        self.vals = vals
        self.params = [_FakeParam(k) for k in vals]

    def isSet(self, p):
        return True

    def hasDefault(self, p):
        return True

    def getOrDefault(self, p):
        return self.vals[p.name]


def _ckpt(tmp_path, identity, n_done):
    import json
    d = tmp_path / "fit_ckpt"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"n_iterations": n_done}))
    if identity is not None:
        (d / "fit_identity.json").write_text(json.dumps(identity))
    return d


def test_fit_identity_skips_the_budget_and_hashes_long_values():
    from dag_gated_fit import fit_identity
    est = _FakeEst(maxIter=100, saveDir="/x", subsamplingRate=0.05,
                   gateParent="{" + "1: [0], " * 100 + "}")
    ident = fit_identity(est, "abc")
    assert ident["bundle_key"] == "abc"
    assert set(ident["params"]) == {"subsamplingRate", "gateParent"}
    assert ident["params"]["gateParent"].startswith("sha256:")


def test_plan_fit_resume_fresh_resume_done_and_refusals(tmp_path):
    import pytest
    from dag_gated_fit import fit_identity, plan_fit_resume
    ident = fit_identity(_FakeEst(subsamplingRate=0.05), "abc")
    assert plan_fit_resume(tmp_path / "fit_ckpt", ident, 100) == ("fresh", 0)
    d = _ckpt(tmp_path, ident, 40)
    assert plan_fit_resume(d, ident, 100) == ("resume", 40)
    assert plan_fit_resume(d, ident, 40) == ("done", 40)
    other = fit_identity(_FakeEst(subsamplingRate=0.1), "abc")
    with pytest.raises(SystemExit, match="subsamplingRate"):
        plan_fit_resume(d, other, 100)
    with pytest.raises(SystemExit, match="bundle_key"):
        plan_fit_resume(d, fit_identity(_FakeEst(subsamplingRate=0.05), "zzz"), 100)
    assert plan_fit_resume(d, other, 100, fresh=True) == ("fresh", 0)
    assert not d.exists()
    d = _ckpt(tmp_path, None, 10)                 # a checkpoint with no identity
    with pytest.raises(SystemExit, match="--fresh-fit"):
        plan_fit_resume(d, ident, 100)

