"""WP-B (spec 2026-10-07 §D1/§D4, insight 0096): the readout-side Mondo hierarchy
over a flat ribbon fit.

Toy Mondo (curies abbreviated; label nodes starred):

    disease (R)
    └── human disease (H)                      R, H cover every label -> root aliases
        ├── cardiovascular (CV)                same label set as CM -> rung, dropped
        │   └── cardiomyopathy (CM)            head: {DCM*, PPCM*, HCM*}
        │       ├── DCM*  ── PPCM*             a nested label pair (R1a)
        │       └── HCM*
        ├── connective tissue (CT)             head: {EDS*, hEDS*, Marfan*}
        │   ├── EDS* ── hEDS*
        │   └── Marfan*
        └── metabolic (X)                      one label descendant -> not a head
            └── T2D*
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "analysis" / "cloud"), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os  # noqa: E402

# PySpark workers inherit PYTHONPATH, not the driver's sys.path (same note as
# test_conversion_analysis): the lean collect ships distributed_readout's kernels
# by module reference.
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(REPO_ROOT / "analysis" / "cloud"), str(REPO_ROOT / "charmpheno"),
                str(REPO_ROOT / "spark-vi"), os.environ.get("PYTHONPATH", "")) if p)

import readout_dag as rd  # noqa: E402

R, H = "MONDO:0000001", "MONDO:0700096"
CV, CM, CT, X = "MONDO:0005267", "MONDO:0004994", "MONDO:0003900", "MONDO:0001111"
DCM, PPCM, HCM = "MONDO:0005021", "MONDO:0005243", "MONDO:0005045"
EDS, HEDS, MARFAN, T2D = "MONDO:0020066", "MONDO:0007522", "MONDO:0007947", "MONDO:0005148"

PARENT_ADJ = {H: [R], CV: [H], CM: [CV], DCM: [CM], PPCM: [DCM], HCM: [CM],
              CT: [H], EDS: [CT], HEDS: [EDS], MARFAN: [CT], X: [H], T2D: [X]}
NAMES = {R: "disease", H: "human disease", CV: "cardiovascular", CM: "cardiomyopathy",
         CT: "connective tissue", X: "metabolic"}
# the fit's bridge: root 0, then labels in engine order
INT2CID = {0: -1, 1: 5021, 2: 5243, 3: 5045, 4: 20066, 5: 7522, 6: 7947, 7: 5148}


@pytest.fixture
def rdag():
    return rd.build_readout_dag(INT2CID, PARENT_ADJ, names=NAMES)


def test_heads_are_grouping_ancestors_minus_root_aliases_and_rungs(rdag):
    assert rdag.C_fit == 8 and rdag.C == 10
    assert rdag.root_aliases == [R, H]
    # appended in curie order: CT (3900) then CM (4994)
    assert rdag.int2cid[8] == 3900 and rdag.int2cid[9] == 4994
    assert rdag.name_by_id == {3900: "connective tissue", 4994: "cardiomyopathy"}
    s = rdag.stats
    assert s["n_ancestor_heads"] == 2 and s["n_rungs_dropped"] == 1
    assert s["n_root_aliases"] == 2 and s["n_nested_labels"] == 2
    assert s["sibling_group"]["root_children"] == 3        # T2D, CT, CM


def test_readout_dag_is_the_induced_hasse_with_root_attachment(rdag):
    assert rdag.parent_int == {0: [], 1: [9], 2: [1], 3: [9], 4: [8], 5: [4],
                               6: [8], 7: [0], 8: [0], 9: [0]}


def test_positive_sources_are_label_descendants_incl_self(rdag):
    assert rdag.pos_sources == [[0], [1, 2], [2], [3], [4, 5], [5], [6], [7],
                                [4, 5, 6], [1, 2, 3]]


def test_json_round_trip_and_report(rdag, tmp_path):
    p = rd.write_readout_dag(tmp_path, rdag)
    back = rd.ReadoutDag.from_json(json.loads(p.read_text()))
    assert back == rdag
    rep = rd.format_readout_dag_report(rdag)
    assert "ancestor heads=2" in rep and "rungs dropped=1" in rep


def test_hierarchy_resolves_on_for_label_set_fits_only():
    assert rd.resolve_readout_hierarchy("auto", {"corpus_manifest": {"label_set": "dismech:x"}})
    assert not rd.resolve_readout_hierarchy("auto", {"corpus_manifest": {"label_set": ""}})
    assert not rd.resolve_readout_hierarchy("auto", {})
    assert rd.resolve_readout_hierarchy("mondo", {})
    assert not rd.resolve_readout_hierarchy("flat", {"corpus_manifest": {"label_set": "d"}})


def test_widened_bundle_view_rebinds_the_bridge_only(rdag):
    class B:
        pass
    b = B()
    b.parent_int = {0: [], 1: [0]}; b.int2cid = {0: -1, 1: 5021}; b.cid2int = {-1: 0, 5021: 1}
    b.name_by_id = {5021: "dilated cardiomyopathy"}; b.train_df = "frame"
    v = rd.widened_bundle_view(b, rdag)
    assert v.train_df == "frame" and v.parent_int == rdag.parent_int
    assert v.int2cid[9] == 4994 and v.cid2int[4994] == 9
    assert v.name_by_id[5021] == "dilated cardiomyopathy"
    assert v.name_by_id[4994] == "cardiomyopathy"
    assert b.int2cid == {0: -1, 1: 5021}          # the original is untouched


@pytest.mark.slow
def test_widening_is_closure_max_labels_and_parent_activation_masks(spark, rdag):
    from pyspark.sql import Row
    C = rdag.C_fit

    def lab(*active):
        v = [0.0] * C
        for a in active:
            v[a] = 1.0
        return v
    rows = [Row(doc="ppcm", label=lab(0, 2), labelMask=[1.0] * C),
            Row(doc="hcm", label=lab(0, 3), labelMask=[1.0] * C),
            Row(doc="heds", label=lab(0, 5), labelMask=[1.0] * C),
            Row(doc="t2d", label=lab(0, 7), labelMask=[1.0] * C),
            Row(doc="bg", label=lab(), labelMask=[0.0] * C)]
    df = spark.createDataFrame(rows)
    out = {r["doc"]: (r["label"], r["labelMask"])
           for r in rd.widen_labels_to_readout(df, rdag).collect()}
    assert set(out["ppcm"][0]) <= {0.0, 1.0} and len(out["ppcm"][0]) == 10
    # y_r: root, DCM (via PPCM), PPCM, CM (via PPCM); nothing on the CT side
    assert out["ppcm"][0] == [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    # mask_r: a node is observed iff a readout parent is active
    #   root(self) DCM(CM) PPCM(DCM) HCM(CM) EDS(CT) hEDS(EDS) Marfan(CT) T2D(root) CT(root) CM(root)
    assert out["ppcm"][1] == [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    assert out["hcm"][0] == [1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    assert out["hcm"][1] == [1, 1, 0, 1, 0, 0, 0, 1, 1, 1]       # PPCM unobserved (DCM off)
    assert out["heds"][0] == [1, 0, 0, 0, 1, 1, 0, 0, 1, 0]
    assert out["heds"][1] == [1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert out["t2d"][0] == [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    assert out["t2d"][1] == [1, 0, 0, 0, 0, 0, 0, 1, 1, 1]       # root-level siblings only
    assert out["bg"][0] == [0] * 10 and out["bg"][1] == [0] * 10
    # a full-mask fit stays full
    full = rd.widen_labels_to_readout(df, rdag, mask_mode="full").collect()
    assert all(r["labelMask"] == [1.0] * 10 for r in full)
    assert set(rd.widen_labels_to_readout(df, rdag).columns) == {"doc", "label", "labelMask"}


# --------------------------------------------------------------------------- #
# plumbing: flags, front matter, the code-map (WP-C') round trips              #
# --------------------------------------------------------------------------- #
def test_readout_tool_parses_the_hierarchy_flag():
    import dag_gated_readout as gpr
    a = gpr.build_parser().parse_args(["--run-dir", "/tmp/run"])
    assert a.readout_hierarchy == "auto"
    a = gpr.build_parser().parse_args(["--run-dir", "/tmp/run", "--readout-hierarchy", "flat"])
    assert a.readout_hierarchy == "flat"


def _two_node_model():
    """A DagGatedLDAModel whose node set is {MONDO:0005021 (DCM), its parent
    MONDO:0004995}, learned from two documents; no fit."""
    import numpy as np
    from spark_vi.core.result import VIResult
    from spark_vi.dag.nodeset import fit_node_set
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    dag = {"MONDO:0004995": [], "MONDO:0005021": ["MONDO:0004995"],
           "MONDO:0007915": ["MONDO:0004995"]}
    ns = fit_node_set(dag, [["MONDO:0005021"], ["MONDO:0004995"]])
    K = 1 + len(ns.nodes)
    res = VIResult(global_params={"lambda": np.ones((K, 2)), "alpha": np.ones(K)},
                   elbo_trace=[], n_iterations=1, converged=True,
                   metadata={"model_class": "DagGatedOnlineLDA",
                             "dag_node_set": {**ns.to_dict(), "n_bg": 1, "tpn": 1}})
    return dag, DagGatedLDAModel(res)


def test_code_map_rides_the_cache_meta_and_lands_on_the_fitted_nodes(tmp_path):
    """The bundle carries the code ladder `(std_cid, most-specific term)`; the run
    dir's code_map.tsv is that ladder landed on the FITTED node set (a term the
    fit pruned lands on its kept ancestor), in the bridge's node cids."""
    import _case_finding_cache as cfc
    import dag_gated_fit as gpc

    dag, model = _two_node_model()          # MONDO:0007915 had no document: pruned

    class B:
        name_by_id = {"MONDO:0005021": "DCM"}; ledger = {}; vocab_maps = [{101: 0}]
    b = B()
    b.dag = dag
    assert "native_code_map" not in cfc._meta_dict(b)          # absent -> no field
    b.native_code_map = [(101, "MONDO:0005021"), (102, "MONDO:0007915")]
    meta = json.loads(json.dumps(cfc._meta_dict(b)))
    assert cfc._restore_meta(meta)["native_code_map"] == b.native_code_map
    br = gpc.node_bridge(model, b.name_by_id)
    rows = gpc.landed_code_map(model, b.native_code_map, br["int2cid"])
    assert rows == [(101, 5021), (102, 4995)]  # 7915 pruned -> its parent 4995
    p = gpc.write_code_map(tmp_path, rows)
    assert p.read_text() == "std_cid\tnode_cid\n101\t5021\n102\t4995\n"


def test_bundle_meta_lands_in_the_run_dir_and_the_readout_has_bundle_only(tmp_path):
    """<run>/bundle_meta.json: the FITTED node set's integer bridge (not rebuilt from
    the bundle), the bundle's vocab maps, ledger and the DAG the fit was given."""
    import dag_gated_fit as gpc
    import dag_gated_readout as gpr
    import inspect_topics as it

    dag, model = _two_node_model()

    class B:
        name_by_id = {"MONDO:0005021": "DCM"}; ledger = {"k": 1}
        vocab_maps = [{101: 0}, {202: 0}]
    b = B()
    b.dag = dag
    p = gpc.write_bundle_meta(tmp_path, b, model)
    meta = json.loads(p.read_text())
    assert meta["vocab_maps"][0] == {"101": 0} and meta["ledger"] == {"k": 1}
    assert meta["int2cid"]["0"] == -1
    assert sorted(meta["int2cid"].values()) == [-1, 4995, 5021]   # 7915 pruned
    assert meta["name_by_id"]["5021"] == "DCM"
    assert len(meta["parent_int"]) == 2                   # the two kept nodes
    assert ["MONDO:0007915", "MONDO:0004995"] in meta["label_dag"]
    assert it.load_bundle_meta(str(p))["int2cid"] == meta["int2cid"]
    a = gpr.build_parser().parse_args(["--run-dir", "/tmp/run", "--bundle-only"])
    assert a.bundle_only


