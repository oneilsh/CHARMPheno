"""Tests for the lr_readout standalone driver (fork-settler): the pure arg-surface
helpers only. The Spark/bundle/npz body is cluster-covered, not unit-tested."""


def test_lr_readout_arg_surface():
    import importlib
    mod = importlib.import_module("lr_readout")
    p = mod.build_parser()
    ns = p.parse_args(["--run-dir", "/runs/0061", "--alpha-grid", "0,1,10"])
    assert ns.run_dir == "/runs/0061"
    assert mod.parse_alpha_grid(ns.alpha_grid) == [0.0, 1.0, 10.0]
    # 0 is always included even if omitted
    assert 0.0 in mod.parse_alpha_grid("1,10")


def test_render_decompose_rows_uses_concept_names():
    import importlib
    mod = importlib.import_module("lr_readout")
    # (w, count, contribution) engine tuples + a vocab-index -> concept-id -> name chain
    rows = [(0, 2.0, 5.9), (1, 3.0, -3.9)]
    idx_to_cid = {0: 111, 1: 222}
    name_by_id = {111: "Lupus erythematosus", 222: "Essential hypertension"}
    lines = mod.render_decompose_rows(rows, idx_to_cid, name_by_id)
    assert "Lupus erythematosus" in lines[0] and "+5.9" in lines[0]
    assert "Essential hypertension" in lines[1] and "-3.9" in lines[1]


def test_ranking_summary_lines_hit_and_miss():
    import importlib
    mod = importlib.import_module("lr_readout")
    nodes = [1, 2, 3, 4]
    names = {1: "Alpha", 2: "Beta", 3: "Gamma", 4: "Delta"}
    scores = [0.5, 0.1, 2.0, -0.3]                 # aligned with nodes -> node 3 highest

    # true = node 3 (the top) -> HIT, rank 1
    lines, top = mod._ranking_summary_lines(scores, nodes, [3], names, top_nodes=3)
    assert top == 3
    j = "\n".join(lines)
    assert "CALL: HIT" in j and "true best rank = 1/4" in j
    assert "1. Gamma" in j and "<- TRUE" in j
    assert j.count("\n") >= 3                       # summary + ranking header + 3 rows

    # true = node 2 (rank 3) -> MISS, top is node 3
    lines2, top2 = mod._ranking_summary_lines(scores, nodes, [2], names, top_nodes=4)
    j2 = "\n".join(lines2)
    assert top2 == 3
    assert "CALL: MISS" in j2 and "true best rank = 3/4" in j2
    assert "top = Gamma" in j2

    # background (no true node) -> ranking only, no CALL line
    lines3, top3 = mod._ranking_summary_lines(scores, nodes, [], names, top_nodes=2)
    j3 = "\n".join(lines3)
    assert "CALL:" not in j3 and "TRUE frontier" not in j3
    assert "1. Gamma" in j3 and "<- TOP" in j3


def test_render_decompose_rows_handles_routing_tuple():
    import importlib
    mod = importlib.import_module("lr_readout")
    idx_to_cid = {0: 100, 1: 200}
    name_by_id = {100: "Distinctive code", 200: "Generic code"}
    # 4-tuples: (w, count, r_u_w, contribution)
    rows = [(0, 1.0, 0.95, 3.2), (1, 4.0, 0.02, -0.01)]
    lines = mod.render_decompose_rows(rows, idx_to_cid, name_by_id)
    assert any("Distinctive code" in ln and "r=0.95" in ln for ln in lines)
    assert any("Generic code" in ln and "r=0.02" in ln for ln in lines)
    # backward compatible with the 3-tuple (no routing column)
    lines3 = mod.render_decompose_rows([(0, 1.0, 3.2)], idx_to_cid, name_by_id)
    assert any("Distinctive code" in ln for ln in lines3)


def test_classify_error_class_covers_the_2x2_plus_node_confusion():
    import importlib
    mod = importlib.import_module("lr_readout")
    c = mod._classify_error_class
    # is_fg, called_rare, hit
    assert c(False, True, False) == "background_called_rare"            # FALSE POSITIVE
    assert c(True, False, False) == "rare_called_background"            # FALSE NEGATIVE
    assert c(True, True, False) == "rare_called_rare_wrong_disease"     # node confusion
    assert c(True, True, True) == "rare_called_rare_correct"           # correct
    assert c(False, False, False) == "background_called_background"     # true negative
    # every class label appears in the display order exactly once
    labels = [k for k, _ in mod._CLASS_ORDER]
    assert set(labels) == {"background_called_rare", "rare_called_background",
                           "rare_called_rare_wrong_disease", "rare_called_rare_correct",
                           "background_called_background"}
    assert len(labels) == len(set(labels))


def test_fdr_truth_and_mm_rows_from_frontiers():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    import importlib
    mod = importlib.import_module("lr_readout")
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=1, tpn=1)      # 2,3 children of 1
    # frontiers per doc (engine ids): doc0 = {2} (deep single), doc1 = {2,3} (multimorbid),
    # doc2 = {} (background)
    frontiers = [[2], [2, 3], []]
    truth, mm_rows = mod.fdr_truth_mm_rows(frontiers, lay)
    node_idx = {u: i for i, u in enumerate(lay.nodes)}
    # doc0 frontier {2}: true for node 2 AND its ancestor node 1 (subtree membership)
    assert truth[0, node_idx[2]] and truth[0, node_idx[1]] and not truth[0, node_idx[3]]
    # doc0 is NOT multimorbid (single frontier node), despite truth having 2 trues
    assert not mm_rows[0]
    # doc1 frontier {2,3}: multimorbid (>=2 scoreable frontier nodes)
    assert mm_rows[1]
    # doc2 background: no truth, not multimorbid
    assert not truth[2].any() and not mm_rows[2]
