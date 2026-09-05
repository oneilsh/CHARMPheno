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
    assert "no --vocab-map" in rep


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
    # node1 = engine id 1 = the i=0 node, whose topic spikes on vocab index 0,
    # so concept_0 must be its top word (a wrong topic<->node map would name a
    # different concept here).
    for line in rep.splitlines():
        if line.startswith("**node1**"):
            assert "concept_0" in line
            break
    else:
        pytest.fail("no node1 topic-words line rendered")


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
