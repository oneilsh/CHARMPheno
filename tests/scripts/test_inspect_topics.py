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


def test_grep_looks_up_named_nodes(tmp_path):
    _make_run(tmp_path)  # nodes node1/node2/node3
    rep = it.build_report(tmp_path, top_topics=10, top_loadings=3, t_words=3,
                          grep_pattern="node2")
    assert "Matched nodes" in rep and "`--grep node2`" in rep
    # node2 present in the matched section, node1/node3 not forced in by grep
    assert "node2" in rep.split("Matched nodes")[1]


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


def test_digest_is_compact_and_marks_cliff(tmp_path):
    # a fit with a fed shallow node (depth 1) and a STARVED deep node (depth 2),
    # so the digest's depth rollup shows a cliff and the verdict line names it.
    n_bg, tpn, n_nodes, V = 1, 1, 2, 10
    K, C = n_bg + n_nodes, n_nodes + 1
    lam = np.full((K, V), 0.01)          # flat prior floor everywhere
    lam[n_bg, 0] += 50.0                 # node1's topic is sharp (fed)
    # node2's topic (index n_bg+1) is left flat -> starved at depth 2
    np.savez(tmp_path / "gated_pc_result.npz", **{"lambda": lam},
             alpha=np.full(K, 0.5), w_CK=np.zeros((C, K)), b_CK=np.zeros(C))
    manifest = {"K": K, "C": C, "n_bg": n_bg, "tpn": tpn,
                "domain_names": ["condition"], "domain_vocab_sizes": [V],
                "corpus_manifest": {
                    "int2cid": {str(e): 1000 + e for e in range(C)},
                    "name_by_id": {str(1000 + e): f"node{e}" for e in range(C)}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    # node1 at depth 1 (child of root), node2 at depth 2 (child of node1)
    meta = {"parent_int": {"1": [0], "2": [1]},
            "int2cid": {str(e): 1000 + e for e in range(C)},
            "name_by_id": {str(1000 + e): f"node{e}" for e in range(C)},
            "vocab_maps": [{str(200 + i): i for i in range(V)}]}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    dig = it.build_digest(tmp_path, exemplars=3, t_words=5,
                          bundle_meta_path=tmp_path / "meta.json")
    assert "digest" in dig and "fed exemplars" in dig
    # the compact block is far smaller than the verbose report on the same fit
    rep = it.build_report(tmp_path, top_topics=10, top_loadings=3, t_words=5,
                          bundle_meta_path=tmp_path / "meta.json")
    assert len(dig) < len(rep) / 2
    # depth rollup present; depth 2 is all-flat -> cliff marker + verdict there
    assert "med-frac" in dig and "<- cliff" in dig
    assert "fed through depth 1; depth>=2" in dig
    # exemplar lines carry compact words (dominant-domain names, ·-joined)
    assert any("node1" in ln and "fed" in ln for ln in dig.splitlines())
    assert any("node2" in ln and "STARVED" in ln for ln in dig.splitlines())


def test_digest_grep_and_redundancy_one_liners(tmp_path):
    _make_run(tmp_path, V=10)
    meta = {"parent_int": {"1": [0], "2": [0], "3": [0], "4": [0], "5": [0]},
            "int2cid": {str(e): 1000 + e for e in range(6)},
            "name_by_id": {str(1000 + e): f"node{e}" for e in range(6)},
            "vocab_maps": [{str(200 + i): i for i in range(10)},
                           {str(300 + i): i for i in range(10)}]}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    dig = it.build_digest(tmp_path, exemplars=3, t_words=4,
                          bundle_meta_path=tmp_path / "meta.json",
                          grep_pattern="node2", redundancy=True)
    assert "grep 'node2'" in dig and "node2" in dig
    assert "redundancy:" in dig


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


# --- the --readout-auc credited split + paired compare (exp 0116's primary read)


def _make_readout_run(tmp_path, name, aucs_by_eng, *, mondo_cids):
    """Run dir with manifest + results_readout.json only (what build_auc_slice
    reads). Engine id e maps to concept id mondo_cids[e] — the numeric part of
    a Mondo curie, as in a mondo_native run."""
    d = tmp_path / name
    d.mkdir()
    manifest = {
        "K": 4, "C": len(mondo_cids), "n_bg": 1, "tpn": 1,
        "corpus_manifest": {
            "int2cid": {str(e): c for e, c in mondo_cids.items()},
            "name_by_id": {str(c): f"node{e}" for e, c in mondo_cids.items()},
        },
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    res = {"gated_pc": {
        "ranking": {"auc": 0.75, "ap": 0.5, "n_nodes": len(aucs_by_eng)},
        "per_node": {str(e): {"auc": a, "ap": a - 0.1}
                     for e, a in aucs_by_eng.items()},
    }}
    (d / "results_readout.json").write_text(json.dumps(res))
    return d


_MONDO_CIDS = {0: 1, 1: 4995, 2: 5010, 3: 7777, 4: 8888}


def _credited_tsv(tmp_path):
    p = tmp_path / "profile_eta.tsv"
    p.write_text(
        "mondo_id\tconcept_id\tweight\tneg\tcoverage\n"
        "MONDO:0004995\t101\t0.5\t0\t0.9\n"
        "MONDO:0004995\t102\t0.4\t1\t0.9\n"     # dup node: still one credit
        "MONDO:0005010\t103\t0.3\t0\t0.8\n"
        "MONDO:9999999\t104\t0.3\t0\t0.8\n"     # not in this run's DAG: skipped
        "OMIM:123456\t105\t0.2\t0\t0.5\n")      # non-Mondo row: skipped
    return p


def test_auc_slice_credited_split(tmp_path):
    run = _make_readout_run(tmp_path, "r1",
                            {1: 0.9, 2: 0.8, 3: 0.6, 4: 0.5},
                            mondo_cids=_MONDO_CIDS)
    rep = it.build_auc_slice(run, credited_file=str(_credited_tsv(tmp_path)))
    assert "2 credited / 2 uncredited scored node(s)" in rep
    assert "credited: median AUC=0.900" in rep       # sorted [0.8, 0.9] -> _q .5
    assert "uncredited: median AUC=0.600" in rep


def test_auc_slice_paired_compare_splits_deltas(tmp_path):
    base = _make_readout_run(tmp_path, "base",
                             {1: 0.85, 2: 0.75, 3: 0.6, 4: 0.5},
                             mondo_cids=_MONDO_CIDS)
    run = _make_readout_run(tmp_path, "r2",
                            {1: 0.9, 2: 0.8, 3: 0.6, 4: 0.5},
                            mondo_cids=_MONDO_CIDS)
    rep = it.build_auc_slice(run, credited_file=str(_credited_tsv(tmp_path)),
                             compare_dir=str(base))
    assert "4 shared scored node(s)" in rep
    assert "credited: n=2 median dAUC=+0.0500" in rep
    assert "up/down=2/0" in rep
    # the internal control is exactly flat
    assert "uncredited (internal control): n=2 median dAUC=+0.0000" in rep
    assert "up/down=0/0" in rep


def test_auc_slice_compare_without_credited_file(tmp_path):
    base = _make_readout_run(tmp_path, "base", {1: 0.8, 2: 0.7},
                             mondo_cids=_MONDO_CIDS)
    run = _make_readout_run(tmp_path, "r3", {1: 0.7, 2: 0.7},
                            mondo_cids=_MONDO_CIDS)
    rep = it.build_auc_slice(run, compare_dir=str(base))
    assert "all: n=2" in rep
    assert "up/down=0/1" in rep
    assert "credited:" not in rep


def test_auc_slice_credited_file_needs_mondo_id_column(tmp_path):
    run = _make_readout_run(tmp_path, "r4", {1: 0.8}, mondo_cids=_MONDO_CIDS)
    bad = tmp_path / "bad.tsv"
    bad.write_text("node\tweight\nMONDO:0004995\t1.0\n")
    with pytest.raises(SystemExit, match="mondo_id"):
        it.build_auc_slice(run, credited_file=str(bad))


# --- the --profile-align scorecard (insight 0084's legibility read, quantified)


def _make_align_run(tmp_path, name, *, tilt_profile):
    """2 nodes (cids 4995, 5010), n_bg=1, tpn=2, V0=10. Node 4995's boosted
    topic (t=1) is STARVED: flat floor, plus (if tilt_profile) a boost-shaped
    tilt on its profile indices {2, 3}. Node 5010's boosted topic (t=3) is FED
    on off-profile index 7."""
    d = tmp_path / name
    d.mkdir()
    V = 10
    lam0 = np.full((5, V), 0.01)
    if tilt_profile:
        lam0[1, [2, 3]] += 0.02   # mild tilt: sharp enough to see, flat enough to stay STARVED (frac>0.5)
    lam0[3, 7] += 500.0
    np.savez(d / "gated_pc_result.npz", lambda_0=lam0,
             lambda_1=np.full((5, 4), 0.01), alpha=np.full(5, 0.5),
             w_CK=np.zeros((3, 5)), b_CK=np.zeros(3))
    manifest = {
        "K": 5, "C": 3, "n_bg": 1, "tpn": 2,
        "domain_names": ["condition", "drug"], "domain_vocab_sizes": [V, 4],
        "corpus_manifest": {
            "int2cid": {"0": 1, "1": 4995, "2": 5010},
            "name_by_id": {"1": "root", "4995": "nodeA", "5010": "nodeB"}},
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def _align_meta(tmp_path):
    p = tmp_path / "meta.json"
    p.write_text(json.dumps({
        "int2cid": {"0": 1, "1": 4995, "2": 5010},
        "vocab_maps": [{str(100 + i): i for i in range(10)},
                       {str(900 + i): i for i in range(4)}],
        "parent_int": {"1": [0], "2": [0]},
    }))
    return p


def _align_tsv(tmp_path):
    p = tmp_path / "eta.tsv"
    p.write_text(
        "mondo_id\tconcept_id\tweight\tneg\tcoverage\n"
        "MONDO:0004995\t102\t0.5\t0\t0.9\n"      # vocab idx 2
        "MONDO:0004995\t103\t0.4\t0\t0.9\n"      # vocab idx 3
        "MONDO:0004995\t105\t0.4\t1\t0.9\n"      # NOT row: excluded
        "MONDO:0005010\t102\t0.3\t0\t0.8\n"      # nodeB claims idx 2 too
        "MONDO:0005010\t104\t0.3\t0\t0.8\n")
    return p


def test_profile_align_scores_boosted_starved_topic(tmp_path):
    run = _make_align_run(tmp_path, "r_tilt", tilt_profile=True)
    rep = it.build_profile_align(run, str(_align_tsv(tmp_path)),
                                 bundle_meta_path=str(_align_meta(tmp_path)))
    # nodeA's starved boosted topic holds ~all mass on its 2 profile tokens.
    assert "starved: n=1" in rep
    assert "median mass=0.429" in rep      # (0.03+0.03)/0.14
    # nodeB's fed topic concentrates on off-profile idx 7 -> ~zero mass.
    assert "fed: n=1 median mass=0.000" in rep
    assert "most aligned:  nodeA" in rep


def test_profile_align_paired_baseline_shows_flat(tmp_path):
    run = _make_align_run(tmp_path, "r2", tilt_profile=True)
    base = _make_align_run(tmp_path, "b2", tilt_profile=False)
    rep = it.build_profile_align(run, str(_align_tsv(tmp_path)),
                                 bundle_meta_path=str(_align_meta(tmp_path)),
                                 compare_dir=str(base))
    # baseline starved topic is exactly flat: mass = |profile|/V = 0.2.
    assert "baseline mass=0.200" in rep     # flat: 2 profile tokens / V=10
    assert "median mass=0.429" in rep


def test_profile_align_requires_emit_eta_columns(tmp_path):
    run = _make_align_run(tmp_path, "r3", tilt_profile=True)
    bad = tmp_path / "bad.tsv"
    bad.write_text("mondo_id\tweight\nMONDO:0004995\t1.0\n")
    with pytest.raises(SystemExit, match="concept_id"):
        it.build_profile_align(run, str(bad),
                               bundle_meta_path=str(_align_meta(tmp_path)))


def test_digest_marks_profile_tokens_and_legend(tmp_path):
    run = _make_align_run(tmp_path, "r_dig", tilt_profile=True)
    rep = it.build_digest(run, bundle_meta_path=str(_align_meta(tmp_path)),
                          profile_file=str(_align_tsv(tmp_path)),
                          grep_pattern="nodeA")
    assert "EMERGENT co-riders" in rep                  # legend line
    assert "cid:102*" in rep                            # profile token marked
    # nodeA's NOT-term concept (105 -> idx 5) must never be marked, and
    # off-profile tokens stay unmarked.
    assert "cid:105*" not in rep


def test_digest_without_profile_file_is_unchanged(tmp_path):
    run = _make_align_run(tmp_path, "r_dig2", tilt_profile=True)
    rep = it.build_digest(run, bundle_meta_path=str(_align_meta(tmp_path)))
    assert "EMERGENT" not in rep
    assert "*" not in rep.replace("**", "")             # no markers anywhere


# --------------------------------------------------------------------------- #
# --strip-audit / --collinearity                                              #
# --------------------------------------------------------------------------- #
def _make_collin_run(tmp_path, name, *, dag_source="mondo_native",
                     colliding_cid=None):
    """4 nodes under root 0, n_bg=1, tpn=2, V0=10, V1=4. Engine 3 (cid 5020,
    'parentC') is the parent of engines 1 and 2 (cids 4995/5010: the CREDITED
    pair, both boosted topics tilted onto the SAME profile tokens {2,3});
    engine 4 (cid 5030) is an uncredited FED node sharp on idx 7; parentC is
    fed on idx 8. Readout heads: the credited pair load on parentC's block and
    BG only (own share 0); the fed nodes load on their own block."""
    d = tmp_path / name
    d.mkdir()
    n_bg, tpn, n_nodes, V = 1, 2, 4, 10
    K = n_bg + tpn * n_nodes                      # 9
    C = n_nodes + 1
    lam0 = np.full((K, V), 0.01)
    lam0[1, [2, 3]] += 0.002                      # eng1 boosted (t=1): faint profile tilt, starved, at the prior floor
    lam0[3, [2, 3]] += 0.002                      # eng2 boosted (t=3): same tilt
    lam0[5, 8] += 500.0                           # eng3 parentC boosted (t=5): fed
    lam0[7, 7] += 500.0                           # eng4 boosted (t=7): fed
    lam1 = np.full((K, 4), 0.01)
    cids = {0: 1, 1: 4995, 2: 5010, 3: 5020, 4: 5030}
    if colliding_cid is not None:
        cids[4] = colliding_cid
    np.savez(d / "gated_pc_result.npz", lambda_0=lam0, lambda_1=lam1,
             alpha=np.full(K, 0.5), w_CK=np.zeros((C, K)), b_CK=np.zeros(C))
    V_heads = np.zeros((C, K))
    V_heads[1, [5, 6]] = 3.0; V_heads[1, 0] = 1.0     # eng1: ancestor + bg
    V_heads[2, [5, 6]] = 3.0; V_heads[2, 0] = 1.0     # eng2: ancestor + bg
    V_heads[3, 5] = 4.0                               # parentC: own
    V_heads[4, 7] = 4.0                               # eng4: own
    np.savez(d / "readout_heads_gated_pc.npz", V=V_heads, b_raw=np.zeros(C),
             degenerate=np.zeros(C, dtype=bool))
    int2cid = {str(e): c for e, c in cids.items()}
    manifest = {
        "K": K, "C": C, "n_bg": n_bg, "tpn": tpn,
        "domain_names": ["condition", "drug"], "domain_vocab_sizes": [V, 4],
        "strip_mode": "both", "window_mode": "lookback",
        "corpus_manifest": {
            "int2cid": int2cid, "dag_source": dag_source, "strip_mode": "both",
            "window_mode": "lookback", "index_mode": "population",
            "name_by_id": {"1": "root", "4995": "nodeA", "5010": "nodeB",
                           "5020": "parentC", "5030": "nodeD",
                           **({str(colliding_cid): "nodeD"}
                              if colliding_cid is not None else {})}},
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    meta = tmp_path / f"{name}_meta.json"
    meta.write_text(json.dumps({
        "int2cid": int2cid,
        "vocab_maps": [{str(100 + i): i for i in range(V)},
                       {str(900 + i): i for i in range(4)}],
        "parent_int": {"1": [3], "2": [3], "3": [0], "4": [0]},
    }))
    return d, meta


def test_strip_audit_native_ids_are_a_noop(tmp_path):
    run, meta = _make_collin_run(tmp_path, "sa")
    rep = it.build_strip_audit(run, bundle_meta_path=str(meta),
                               profile_file=str(_align_tsv(tmp_path)))
    assert "dag_source=mondo_native" in rep
    assert "Mondo numerics" in rep
    # no Mondo numeric (4995, 5010, ...) is an OMOP vocab key (100..109)
    assert "domain 0 (condition): 0 of 10 vocab dims" in rep
    assert "domain 1 (drug): 0 of 4 vocab dims" in rep
    assert "NO-OP" in rep
    # profile concepts 102,103,104 live in vocab 0; none equals a node id
    assert "3 in the condition vocab (live), 0 equal to a node id" in rep


def test_strip_audit_counts_a_colliding_dim(tmp_path):
    # node cid 105 == vocab concept 105 (idx 5): one dim the strip WOULD drop
    run, meta = _make_collin_run(tmp_path, "sb", colliding_cid=105)
    rep = it.build_strip_audit(run, bundle_meta_path=str(meta))
    assert "domain 0 (condition): 1 of 10 vocab dims" in rep
    assert "NO-OP" in rep                       # 1 hit is still within the coincidence budget


def test_strip_audit_anchor_path_reports_real_strip(tmp_path):
    run, meta = _make_collin_run(tmp_path, "sc", dag_source="mondo",
                                 colliding_cid=105)
    rep = it.build_strip_audit(run, bundle_meta_path=str(meta))
    assert "node ids are OMOP concept ids" in rep
    assert "VERDICT: 1 DAG-node dims stripped" in rep
    assert "NO-OP" not in rep


def test_collinearity_credited_pair_is_collinear_and_decoded_elsewhere(tmp_path):
    run, meta = _make_collin_run(tmp_path, "cl")
    rep = it.build_collinearity(run, str(_align_tsv(tmp_path)),
                                bundle_meta_path=str(meta))
    # groups: 2 credited (A,B), 2 uncredited fed (parentC, D), 0 starved
    assert "credited=2 uncredited fed=2 starved=0" in rep
    # A and B share profile token idx 2 (A={2,3}, B={2,4}) -> Jaccard 1/3
    assert "median-of-max=0.33" in rep
    # identically tilted boosted topics: peer cosine 1.00 for the credited pair;
    # the fed pair are sharp on DIFFERENT words: peer cosine ~0
    top = [l for l in rep.splitlines() if " peer=" in l]
    cred_top = [l for l in top if l.startswith("  credited:")][0]
    fed_top = [l for l in top if l.startswith("  uncredited fed:")][0]
    assert "peer=1.00" in cred_top
    # the fed pair share only their FLAT drug-domain halves (cos 0.20), well
    # below the credited pair's identical-content 1.00
    assert "peer=0.20" in fed_top
    # credited topics sit AT the flat floor (flat=1.00): their peer/bg/anc
    # agreement is the starvation-floor artefact the flat column exists to
    # expose (parentC's block has a flat 2nd topic -> anc=1.00), not content
    assert "flat=1.00" in cred_top and "anc=1.00" in cred_top and "(n_anc=2)" in cred_top
    assert "flat=0.53" in fed_top
    # decoder: credited heads put 0 on their own block, 0.75 on the ancestor
    # (parentC) block, 0.25 on BG; fed heads put 1.00 on their own block
    dec = [l for l in rep.splitlines() if l.startswith("  ")]
    cred_dec = [l for l in dec if l.startswith("  credited: n=2 own=")][0]
    # |w| = 3+3 on parentC's block + 1 on BG0: anc 6/7, bg 1/7, own 0
    assert "own=0.00" in cred_dec and "anc=0.86" in cred_dec and "bg=0.14" in cred_dec
    assert "desc=0.00" in cred_dec and "other=0.00" in cred_dec
    assert "own<0.05: 2/2" in cred_dec
    fed_dec = [l for l in dec if l.startswith("  uncredited fed: n=2 own=")][0]
    assert "own=1.00" in fed_dec
    # top-5 relation census: the credited pair's 5 largest |w| are the 2
    # ancestor topics, BG0, then zeros (classed by position: own block first)
    # fed-only view: fixture fed topics = BG0 + t5 (parentC) + t7 (D); the
    # credited pair's |w| among those is all on parentC's t5 and BG0 -> own 0
    fedl = [l for l in rep.splitlines() if "among FED topics only" in l]
    assert len(fedl) == 2 and "(3 of 9)" in fedl[0] and "own=0.00" in fedl[0]
    assert "own=1.00" in fedl[1]
    rel = [l for l in rep.splitlines() if "top-5 loaded topics by relation" in l]
    assert len(rel) == 2 and "anc=40%" in rel[0] and "bg=20%" in rel[0]
    # raw V on disk here -> the header names the inflated scale
    assert "decoder (raw-θ V (INFLATED))" in rep
    # evidence: both credited boosted topics are at the prior floor
    ev = [l for l in rep.splitlines() if "at floor" in l]
    assert any(l.startswith("  credited:") and "2/2" in l for l in ev)
    assert any(l.startswith("  uncredited fed:") and "0/2" in l for l in ev)


def test_collinearity_without_heads_says_so(tmp_path):
    run, meta = _make_collin_run(tmp_path, "cn")
    (run / "readout_heads_gated_pc.npz").unlink()
    rep = it.build_collinearity(run, str(_align_tsv(tmp_path)),
                                bundle_meta_path=str(meta))
    assert "decoder: no readout heads/checkpoint on disk" in rep


def test_heads_sidecar_w_std_is_preferred_over_raw_v(tmp_path):
    _make_run(tmp_path)
    _, manifest = it.load_run(tmp_path)
    K, C = manifest["K"], manifest["C"]
    V = np.zeros((C, K)); V[2, 3] = 900.0            # inflated raw coefficient
    W = np.zeros((C, K)); W[2, 3] = 0.9              # honest standardized one
    np.savez(tmp_path / "readout_heads_gated_pc.npz", V=V, b_raw=np.zeros(C),
             degenerate=np.zeros(C, dtype=bool), W_std=W)
    h = it.load_readout_heads(tmp_path)
    assert h["standardized"] and "heads sidecar" in h["src"]
    assert h["W_load"][2, 3] == 0.9


def test_collinearity_flags_raw_v_decoder_as_inconclusive(tmp_path):
    run, meta = _make_collin_run(tmp_path, "rv")
    rep = it.build_collinearity(run, str(_align_tsv(tmp_path)),
                                bundle_meta_path=str(meta))
    assert "INCONCLUSIVE: only the raw-θ decoder V is on disk" in rep
    z = np.load(run / "readout_heads_gated_pc.npz")
    np.savez(run / "readout_heads_gated_pc.npz", V=z["V"], b_raw=z["b_raw"],
             degenerate=z["degenerate"], W_std=z["V"])
    rep = it.build_collinearity(run, str(_align_tsv(tmp_path)),
                                bundle_meta_path=str(meta))
    assert "INCONCLUSIVE" not in rep
    assert "decoder (standardized W_std)" in rep
