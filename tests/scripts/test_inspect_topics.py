"""Unit tests for analysis/cloud/inspect_topics.py -- the off-cluster topics
inspector. Pure numpy; no Spark, so these run in the fast unit lane.

The load-bearing correctness claim is the topic<->node map: DagLayout lays
topics out as [0,n_bg) background then one block per non-root node in
sorted(engine-id) order, and mislabelling that silently attaches every topic to
the wrong node (the same hazard case_finding_assembly.py warns about for
render_profile). We build a synthetic fit whose per-node topics are each sharp
on a DIFFERENT vocab index, so a wrong map would surface as a wrong word.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "cloud"))
import inspect_topics as it  # noqa: E402


def _make_run(tmp_path, *, n_bg=2, tpn=1, n_nodes=3, V=10, n_dom=2):
    """A synthetic 1-domain-dominant fit. Node i's topic is sharp on vocab index
    i (in domain 0); background topics and the non-dominant domain are flat.
    Engine ids: root=0, nodes=1..n_nodes. Names: 'node{eng}'."""
    K = n_bg + n_nodes * tpn
    C = n_nodes + 1                       # incl root engine id 0
    rng = np.random.default_rng(0)
    lams = {}
    for d in range(n_dom):
        lam = np.full((K, V), 0.01)       # flat prior floor everywhere
        if d == 0:
            for i in range(n_nodes):
                t = n_bg + i * tpn        # this node's (first) topic
                lam[t, i] += 50.0         # sharp spike on vocab index i
        lams[f"lambda_{d}"] = lam
    alpha = np.full(K, 0.5)
    w_CK = np.zeros((C, K))
    for eng in range(1, C):
        t = n_bg + (eng - 1) * tpn
        w_CK[eng, t] = 2.0                # node decodes mostly from its own topic
        w_CK[eng, 0] = 0.5                # ... and a bit from BG0
    b_CK = np.full(C, -0.1)
    np.savez(tmp_path / "gated_pc_result.npz", **lams, alpha=alpha,
             w_CK=w_CK, b_CK=b_CK)

    int2cid = {str(e): 1000 + e for e in range(C)}       # engine -> concept id
    name_by_id = {str(1000 + e): f"node{e}" for e in range(C)}
    manifest = {
        "K": K, "C": C, "n_bg": n_bg, "tpn": tpn,
        "domain_names": [f"dom{d}" for d in range(n_dom)],
        "domain_vocab_sizes": [V] * n_dom,
        "corpus_manifest": {"int2cid": int2cid, "name_by_id": name_by_id},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_topic_labels_map_to_correct_node(tmp_path):
    _make_run(tmp_path)
    _, manifest = it.load_run(tmp_path)
    labels, topic2engine = it.topic_labels(manifest)
    # first two are background
    assert labels[0] == "BG0" and labels[1] == "BG1"
    assert topic2engine[0] is None
    # topic 2 -> node engine 1 -> name node1; topic 3 -> node2; topic 4 -> node3
    assert labels[2] == "node1" and topic2engine[2] == 1
    assert labels[3] == "node2" and topic2engine[3] == 2
    assert labels[4] == "node3" and topic2engine[4] == 3


def test_sharpness_separates_sharp_from_flat(tmp_path):
    _make_run(tmp_path, V=10)
    npz, manifest = it.load_run(tmp_path)
    lams = it.domain_lambdas(npz)
    sh = it.topic_sharpness(lams)
    n_bg = manifest["n_bg"]
    # background topics are flat -> support near V (10); node topics are sharp
    assert sh["support"][0] > 8.0                 # BG0 ~ uniform over 10
    assert sh["support"][n_bg] < 3.0              # node1 topic spikes
    assert sh["support_frac"][n_bg] < sh["support_frac"][0]
    # node topic accrued evidence; background did not
    assert sh["evidence"][n_bg] > sh["evidence"][0]


def test_report_renders_and_flags_starvation(tmp_path):
    _make_run(tmp_path)
    rep = it.build_report(tmp_path, top_topics=10, top_loadings=3, t_words=5)
    assert "Topics view" in rep
    assert "node1" in rep and "Node topics" in rep
    # self-weight column shows the node's own-topic readout weight (+2.000)
    assert "+2.000" in rep
    # without a vocab map the words section is explicitly skipped
    assert "--vocab-map" in rep
    # no heads/ckpt sidecar -> falls back to co-fit w_CK, and SAYS so honestly
    assert "NOT the readout decoder" in rep


def test_prefers_real_readout_heads_over_cofit(tmp_path):
    _make_run(tmp_path)
    _, manifest = it.load_run(tmp_path)
    K, C = manifest["K"], manifest["C"]
    # a heads sidecar whose decoder differs from the co-fit w_CK: node1 (engine 1,
    # topic 2) is DEGENERATE, node2 (engine 2, topic 3) loads on topic 3 at +9.
    V = np.zeros((C, K)); b = np.zeros(C)
    degen = np.zeros(C, dtype=bool); degen[1] = True
    V[2, 3] = 9.0
    np.savez(tmp_path / "readout_heads_gated_pc.npz",
             V=V, b_raw=b, degenerate=degen)
    rep = it.build_report(tmp_path, top_topics=10, top_loadings=3, t_words=5)
    assert "readout_heads_gated_pc.npz" in rep and "raw-θ decoder" in rep
    # no checkpoint present -> loadings fall back to raw V, caveated
    assert "INFLATED" in rep
    assert "degenerate heads: 1 / " in rep
    # node1's row is marked degenerate, not given a bogus weight
    for line in rep.splitlines():
        if line.startswith("| node1 "):
            assert "degenerate head" in line
    assert "+9.000" in rep                         # node2's real self-weight


def test_loadings_prefer_standardized_ckpt_over_raw_heads(tmp_path):
    _make_run(tmp_path)
    _, manifest = it.load_run(tmp_path)
    K, C = manifest["K"], manifest["C"]
    # raw heads V with an EXPLODED coefficient (the low-variance artifact) ...
    V = np.zeros((C, K)); V[2, 3] = 99999.0
    np.savez(tmp_path / "readout_heads_gated_pc.npz",
             V=V, b_raw=np.zeros(C), degenerate=np.zeros(C, dtype=bool))
    # ... and a checkpoint whose STANDARDIZED weight for the same cell is modest
    W = np.zeros((C, K)); W[2, 3] = 1.5
    np.savez(tmp_path / "readout_ckpt_gated_pc.npz",
             W_std=W, b_std=np.zeros(C), iter=np.int64(200),
             fingerprint=np.str_("fp"))
    rep = it.build_report(tmp_path, top_topics=10, top_loadings=3, t_words=5)
    # decoder is the raw heads, but loadings come from the standardized ckpt
    assert "raw-θ decoder" in rep and "standardized W_std from ckpt iter 200" in rep
    assert "+1.500" in rep and "99999" not in rep   # modest std weight, not exploded raw


def test_falls_back_to_checkpoint_when_no_heads(tmp_path):
    _make_run(tmp_path)
    _, manifest = it.load_run(tmp_path)
    K, C = manifest["K"], manifest["C"]
    W = np.zeros((C, K)); W[2, 3] = 4.0
    np.savez(tmp_path / "readout_ckpt_gated_pc.npz",
             W_std=W, b_std=np.zeros(C), iter=np.int64(120),
             fingerprint=np.str_("fp"))
    rep = it.build_report(tmp_path, top_topics=10, top_loadings=3, t_words=5)
    assert "readout_ckpt_gated_pc.npz" in rep
    assert "checkpoint iter 120" in rep
    assert "+4.000" in rep


def test_vocab_words_named_when_map_given(tmp_path):
    _make_run(tmp_path, V=10)
    # vocab map for domain 0: concept id 200+idx at position idx; domain 1 same
    vmap = [{str(200 + i): i for i in range(10)} for _ in range(2)]
    (tmp_path / "vocab.json").write_text(json.dumps(vmap))
    names = tmp_path / "names.csv"
    names.write_text("concept_id,concept_name\n"
                     + "\n".join(f"{200+i},concept_{i}" for i in range(10)))
    rep = it.build_report(tmp_path, top_topics=10, top_loadings=3, t_words=5,
                          vocab_path=tmp_path / "vocab.json", names_path=names)
    assert "Top 5 concepts" in rep
    # words render as per-domain sub-bullets under each node bullet (fixture
    # domains are named dom0/dom1)
    assert "**dom0:**" in rep and "**dom1:**" in rep
    # node1 = engine id 1 = the i=0 node, whose topic spikes on vocab index 0 of
    # domain 0, so concept_0 must be its top dom0 word (a wrong topic<->node map
    # would name a different concept). Find node1's bullet, then its dom0 line.
    lines = rep.splitlines()
    for i, line in enumerate(lines):
        if "**node1**" in line and line.lstrip().startswith("-"):
            cond = next(l for l in lines[i + 1:i + 5] if "**dom0:**" in l)
            assert "concept_0" in cond
            break
    else:
        pytest.fail("no node1 topic-words block rendered")


def test_tree_tour_indents_by_depth(tmp_path):
    _make_run(tmp_path, V=10)
    meta = {"parent_int": {"1": [0], "2": [1], "3": [2], "4": [1], "5": [1]},
            "int2cid": {str(e): 1000 + e for e in range(6)},
            "name_by_id": {str(1000 + e): f"node{e}" for e in range(6)},
            "vocab_maps": [{str(200 + i): i for i in range(10)},
                           {str(300 + i): i for i in range(10)}]}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    rep = it.build_report(tmp_path, top_topics=10, top_loadings=3, t_words=5,
                          bundle_meta_path=tmp_path / "meta.json", tour_per_depth=2)
    assert "Tree tour" in rep
    # depth marker + depth-proportional indentation (node2 is at depth 2 -> engine
    # chain 1->2->3, so node3 at depth 3 is indented deeper than node1 at depth 1)
    d1 = next(l for l in rep.splitlines() if "**node1**" in l and "`d1`" in l)
    d3 = next(l for l in rep.splitlines() if "**node3**" in l and "`d3`" in l)
    assert (len(d1) - len(d1.lstrip())) < (len(d3) - len(d3.lstrip()))


def test_bundle_meta_gives_depth_and_words(tmp_path):
    _make_run(tmp_path, V=10)
    # meta as _case_finding_cache writes it: parent_int is engine-id child->parents.
    # chain: root0 -> 1 -> 2 -> 3; node4,5 are children of 1 (shallow siblings).
    parent_int = {"1": [0], "2": [1], "3": [2], "4": [1], "5": [1]}
    vmaps = [{str(200 + i): i for i in range(10)},
             {str(300 + i): i for i in range(10)}]
    meta = {"parent_int": parent_int,
            "int2cid": {str(e): 1000 + e for e in range(6)},
            "name_by_id": {str(1000 + e): f"node{e}" for e in range(6)},
            "vocab_maps": vmaps}
    (tmp_path / "meta.json").write_text(json.dumps(meta))

    # depth is computed correctly from parent_int (longest path from root 0)
    depths = it.node_depths(parent_int)
    assert depths[1] == 1 and depths[2] == 2 and depths[3] == 3
    assert depths[4] == 2 and depths[5] == 2

    rep = it.build_report(tmp_path, top_topics=10, top_loadings=3, t_words=5,
                          bundle_meta_path=tmp_path / "meta.json", sort_by="depth")
    assert "depth" in rep and "Sharpness by depth" in rep
    # words render straight from the meta's vocab_maps (no separate --vocab-map)
    assert "Top 5 concepts" in rep
    # deepest node (engine 3 -> node3, topic 4) sorts first under sort=depth
    body = rep.split("Node topics")[1]
    assert body.index("node3") < body.index("node1")


def test_sibling_redundancy_flags_uniform_collapse(tmp_path):
    # 5 node topics over 1 domain (V=6); make two FED siblings identical and one
    # FED sibling distinct, under a common parent.
    n_bg, tpn, n_nodes, V = 1, 1, 5, 6
    K, C = n_bg + n_nodes, n_nodes + 1
    lam = np.full((K, V), 0.01)
    # nodes 1,2 (topics 1,2): identical sharp spike on word 0 (redundant)
    lam[1, 0] += 40.0
    lam[2, 0] += 40.0
    # node 3 (topic 3): distinct sharp spike on word 3
    lam[3, 3] += 40.0
    # nodes 4,5 (topics 4,5): starved (near-uniform) -> excluded from fed set
    np.savez(tmp_path / "gated_pc_result.npz", **{"lambda": lam},
             alpha=np.full(K, 0.5), w_CK=np.zeros((C, K)), b_CK=np.zeros(C))
    # engine chain: all of 1,2,3,4,5 are children of root 0 (so 0 is the parent)
    parent_int = {"1": [0], "2": [0], "3": [0], "4": [0], "5": [0]}
    meta = {"parent_int": parent_int,
            "int2cid": {str(e): 1000 + e for e in range(C)},
            "name_by_id": {str(1000 + e): f"node{e}" for e in range(C)},
            "vocab_maps": [{str(400 + i): i for i in range(V)}]}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    manifest = {"K": K, "C": C, "n_bg": n_bg, "tpn": tpn,
                "domain_names": ["dom0"], "domain_vocab_sizes": [V],
                "corpus_manifest": {
                    "int2cid": {str(e): 1000 + e for e in range(C)},
                    "name_by_id": {str(1000 + e): f"node{e}" for e in range(C)}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    # direct: parent 0 has fed children {1,2,3}; 1&2 identical (cos~1), 3 distinct
    _, mani = it.load_run(tmp_path)
    lams = it.domain_lambdas(np.load(tmp_path / "gated_pc_result.npz"))
    sh = it.topic_sharpness(lams)
    labels, t2e = it.topic_labels(mani)
    pint = {int(k): [int(p) for p in v] for k, v in parent_int.items()}
    rows = it.sibling_redundancy(pint, t2e, lams, sh, n_bg, K)
    root = next(r for r in rows if r["parent"] == 0)
    assert root["n_fed"] == 3               # topics 1,2,3 fed; 4,5 starved out
    # the identical pair pushes max fed-cos to ~1
    assert root["max_cos_fed"] > 0.99

    rep = it.build_report(tmp_path, top_topics=10, top_loadings=3, t_words=3,
                          bundle_meta_path=tmp_path / "meta.json", redundancy=10)
    assert "Sibling redundancy" in rep and "uniform collapse" in rep


def test_wrong_bundle_meta_is_flagged(tmp_path):
    _make_run(tmp_path, V=10)          # run has 2 domains of V=10
    # a meta from a DIFFERENT bundle: vocab sizes 7/7, not 10/10
    bad = {"parent_int": {"1": [0], "2": [1], "3": [1], "4": [1], "5": [1]},
           "int2cid": {str(e): 1000 + e for e in range(6)},
           "name_by_id": {str(1000 + e): f"node{e}" for e in range(6)},
           "vocab_maps": [{str(700 + i): i for i in range(7)},
                          {str(800 + i): i for i in range(7)}]}
    (tmp_path / "bad_meta.json").write_text(json.dumps(bad))
    rep = it.build_report(tmp_path, top_topics=10, top_loadings=3, t_words=5,
                          bundle_meta_path=tmp_path / "bad_meta.json")
    assert "WARNING" in rep and "DIFFERENT bundle" in rep
    assert "vocab sizes" in rep


def test_single_domain_lambda_key(tmp_path):
    # a run that stored a single `lambda` (not lambda_0) still loads
    K, V, C = 4, 6, 3
    np.savez(tmp_path / "gated_pc_result.npz",
             **{"lambda": np.full((K, V), 0.1)},
             alpha=np.full(K, 0.5), w_CK=np.zeros((C, K)), b_CK=np.zeros(C))
    manifest = {"K": K, "C": C, "n_bg": 1, "tpn": 1,
                "domain_names": ["dom0"], "domain_vocab_sizes": [V],
                "corpus_manifest": {
                    "int2cid": {"0": 1000, "1": 1001, "2": 1002},
                    "name_by_id": {"1000": "root", "1001": "a", "1002": "b"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    npz, _ = it.load_run(tmp_path)
    assert len(it.domain_lambdas(npz)) == 1
    rep = it.build_report(tmp_path, top_topics=5, top_loadings=2, t_words=3)
    assert "node topics" in rep.lower()
