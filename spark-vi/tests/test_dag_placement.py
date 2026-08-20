import numpy as np
from spark_vi.models.topic.dag_placement import DagLayout, frontier_from_coded, strip_dag_node_codes, fit_gated, profile, evaluate, _auc

PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}   # root 0 -> families 1,2 -> subtypes
DIAMOND = {1: 0, 2: 0, 3: 0, 4: [1, 2], 5: [1, 3]}  # multi-parent DAG: node 4,5 have two parents

def test_daglayout_structure_and_masks():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    assert lay.nodes == [1, 2, 3, 4, 5, 6]
    assert lay.K == 2 + 6                      # bg + one topic per node
    assert lay.closure(3) == [0, 1, 3]         # root..v
    assert lay.subtree(1) == {1, 3, 4}
    assert lay.depth(3) == 2 and lay.depth(1) == 1
    # allowed(v) = bg ∪ blocks along closure(v), excluding root
    assert list(lay.allowed(3)) == [0, 1] + lay.block[1] + lay.block[3]
    assert list(lay.allowed(1)) == [0, 1] + lay.block[1]

def test_daglayout_tpn_two():
    lay = DagLayout(PARENT, n_bg=1, tpn=2)
    assert lay.K == 1 + 6 * 2
    assert len(lay.block[3]) == 2

def test_daglayout_depth_cycle_guarded():
    # DagLayout is the domain-agnostic public entry: a malformed cyclic parent map must
    # not recurse forever. The guard yields a finite depth rather than hanging.
    lay = DagLayout({1: 0, 2: [1, 3], 3: [2]})   # 2<->3 cycle
    assert lay.depth(1) == 1
    assert lay.depth(2) < float("inf") and lay.depth(3) < float("inf")

def test_strip_dag_node_codes():
    doc = np.array([10, 3, 11, 1, 12])          # 3 and 1 are DAG-node codes
    out = strip_dag_node_codes(doc, {1, 3})
    assert list(out) == [10, 11, 12]

def test_dag_placement_corpus_shapes():
    from tests._stm_synth import dag_placement_corpus

    docs, labels, node_codes = dag_placement_corpus(
        parent=PARENT, node_prev={1: .18, 2: .18, 3: .16, 4: .16, 5: .16, 6: .16},
        V=120, doc_len=40, seed=0)
    assert len(docs) == len(labels)
    assert set(labels.tolist()) <= set(PARENT.keys())
    assert set(node_codes.keys()) == set(PARENT.keys())
    # a node's exact code appears in items labeled at/below that node
    below3 = [d for d, y in zip(docs, labels) if y in {3}]
    assert any(node_codes[3] in d for d in below3)

def test_fit_gated_learns_node_signatures():
    from tests._stm_synth import dag_placement_corpus
    docs, labels, _ = dag_placement_corpus(
        parent=PARENT, node_prev={1:.18,2:.18,3:.16,4:.16,5:.16,6:.16},
        V=120, doc_len=40, seed=1)
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    rng = np.random.default_rng(3)
    beta = fit_gated(docs[:1400], labels[:1400], lay, 120, n_iter=60, burn=30, rng=rng)
    assert beta.shape == (lay.K, 120)
    assert np.allclose(beta.sum(1), 1.0, atol=1e-6)

def test_profile_returns_node_affinity():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    beta = np.full((lay.K, 30), 1e-3); beta /= beta.sum(1, keepdims=True)
    rng = np.random.default_rng(0)
    pr = profile(np.array([1, 2, 3, 4, 5]), beta, lay, n_iter=20, burn=10, rng=rng)
    assert set(pr.keys()) == set(lay.nodes)
    assert all(0.0 <= v <= 1.0 for v in pr.values())

def test_evaluate_perfect_profiles_score_high():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    labels = np.array([3, 4, 5, 6, 1, 2] * 5)
    profiles = []
    for y in labels:                              # planted "perfect" affinity: closure-loaded
        cl = [u for u in lay.closure(y) if u != 0]   # true node + all its ancestors (not root)
        profiles.append({u: (1.0 if u in cl else 0.0) for u in lay.nodes})
    m = evaluate(profiles, labels, lay)
    assert m["mrr"] == 1.0 and m["top2"] == 1.0
    assert all(v >= 0.99 for v in m["node_auc"].values())

def test_detection_metrics_separates_and_reports_operating_points():
    from spark_vi.models.topic.dag_placement import _detection_metrics
    # perfectly separable: foreground scores strictly above background scores.
    fg = [0.9, 0.8, 0.85, 0.95]
    bg = [0.1, 0.2, 0.15, 0.05, 0.3, 0.0]
    scores = fg + bg
    is_fg = [True] * len(fg) + [False] * len(bg)
    d = _detection_metrics(scores, is_fg)
    assert d["auc"] == 1.0                       # perfect ranking
    assert d["n_foreground"] == 4 and d["n_background"] == 6
    assert abs(d["prevalence"] - 0.4) < 1e-9
    op = d["operating_points"]["0.90"]
    # catches all 4 foreground (>=90%), zero background above the threshold.
    assert op["sensitivity"] >= 0.90
    assert op["bg_fpr"] == 0.0 and op["specificity"] == 1.0
    assert op["precision"] == 1.0                # perfectly separable -> all flags real


def test_detection_metrics_reports_background_false_positives():
    from spark_vi.models.topic.dag_placement import _detection_metrics
    # overlap: one background sits at 0.7, above the weakest foreground.
    fg = [0.6, 0.8, 0.9, 0.75]
    bg = [0.1, 0.7, 0.2, 0.05]
    d = _detection_metrics(fg + bg, [True] * 4 + [False] * 4)
    op = d["operating_points"]["0.90"]
    # threshold drops to catch >=90% of fg (all 4, thr=0.6); the 0.7 background
    # is then a false positive -> bg_fpr = 1/4.
    assert op["sensitivity"] >= 0.90
    assert abs(op["bg_fpr"] - 0.25) < 1e-9
    assert op["precision"] < 1.0                 # a background leaks in


def test_detection_metrics_one_class_is_nan():
    from spark_vi.models.topic.dag_placement import _detection_metrics
    d = _detection_metrics([0.1, 0.2, 0.3], [False, False, False])
    assert np.isnan(d["auc"]) and d["operating_points"] == {}


def test_evaluate_detection_block_backgrounds_park_on_background_block():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    # 4 foreground docs (each loads its own node), 6 background docs (empty
    # frontier, ~zero disease-node affinity -> mass parks on the background block).
    labels = [3, 4, 5, 6, frozenset(), frozenset(),
              frozenset(), frozenset(), frozenset(), frozenset()]
    profiles = []
    for y in labels:
        if not y or (hasattr(y, "__iter__") and not len(y)):
            profiles.append({u: 0.01 for u in lay.nodes})       # diffuse, tiny -> background
        else:
            cl = [u for u in lay.closure(y) if u != 0]
            profiles.append({u: (0.8 if u in cl else 0.0) for u in lay.nodes})
    m = evaluate(profiles, labels, lay)
    det = m["detection"]
    assert det["n_foreground"] == 4 and det["n_background"] == 6
    assert det["auc"] >= 0.99                                    # cases separate from background
    # background docs put nearly all mass on the background block; foreground far less.
    assert det["bg_mass_background_mean"] > det["bg_mass_foreground_mean"]
    assert det["bg_mass_background_mean"] > 0.9


def test_evaluate_tolerates_root_label():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    labels = np.array([3, 0, 4, 5])
    profiles = []
    for y in labels:
        if y == 0:
            profiles.append({u: 0.0 for u in lay.nodes})   # root: no informative affinity
        else:
            cl = [u for u in lay.closure(y) if u != 0]
            profiles.append({u: (1.0 if u in cl else 0.0) for u in lay.nodes})
    m = evaluate(profiles, labels, lay)                    # must not raise on the root label
    assert np.isfinite(m["mrr"])                            # 3 of 4 items are rankable

def test_render_profile_marks_true_and_shows_all_nodes():
    from spark_vi.models.topic.dag_placement import render_profile
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    aff = {1: 0.6, 2: 0.0, 3: 0.05, 4: 0.0, 5: 0.2, 6: 0.15}
    s = render_profile(aff, lay, true_node=1)
    assert "true" in s
    for u in lay.nodes:                                   # every node rendered
        assert str(u) in s
    assert s.count("\n") >= len(lay.nodes)

def test_end_to_end_recovers_family_and_subtype():
    from tests._stm_synth import dag_placement_corpus
    docs, labels, node_codes = dag_placement_corpus(
        parent=PARENT, node_prev={1:.18,2:.18,3:.16,4:.16,5:.16,6:.16},
        V=120, doc_len=40, seed=2)
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    ntr = int(0.7 * len(docs))
    rng = np.random.default_rng(5)
    beta = fit_gated(docs[:ntr], labels[:ntr], lay, 120, n_iter=80, burn=40, rng=rng)
    codes = set(node_codes.values())
    profs = [profile(strip_dag_node_codes(d, codes), beta, lay, n_iter=40, burn=20, rng=rng)
             for d in docs[ntr:]]
    m = evaluate(profs, labels[ntr:], lay)
    # gated-train places well; sim validated family ~0.99 / subtype ~0.97 (spec). Loose floors:
    assert m["auc_by_depth"][1] >= 0.85           # family level
    assert m["auc_by_depth"][2] >= 0.75           # subtype level
    assert m["mrr"] >= 0.6

def test_daglayout_multiparent_diamond():
    lay = DagLayout(DIAMOND, n_bg=2, tpn=1)
    assert lay.parents[4] == [1, 2] and lay.parents[5] == [1, 3]
    assert lay.closure(4) == [0, 1, 2, 4]          # all ancestors, depth-sorted, root first
    assert lay.closure(5) == [0, 1, 3, 5]
    assert lay.depth(4) == 2 and lay.depth(1) == 1 and lay.depth(0) == 0   # longest path
    assert lay.subtree(1) == {1, 4, 5} and lay.subtree(2) == {2, 4}
    want = {0, 1} | set(lay.block[1]) | set(lay.block[2]) | set(lay.block[3]) \
        | set(lay.block[4]) | set(lay.block[5])
    assert set(lay.allowed_set({4, 5}).tolist()) == want   # union of closures over the frontier

def test_daglayout_path_cousins_widens_contrast_set():
    # path_cousins = closure+siblings PLUS the siblings of every ancestor on the root-path.
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    sib = set(int(k) for k in lay.allowed_with_siblings(3))
    pc = set(int(k) for k in lay.allowed_with_path_cousins(3))
    assert sib <= pc                                    # superset of closure+siblings
    # node 3's ancestor 1 has sibling 2 (the "aunt"); path_cousins adds block[2], siblings does not
    assert set(lay.block[2]) <= pc
    assert not set(lay.block[2]).issubset(sib)
    assert 0 in pc and 1 in pc and len(pc) <= lay.K     # bg included, still bounded < dense K
    # multi-parent DAG: does not crash, stays a superset of the siblings support
    d = DagLayout(DIAMOND, n_bg=2, tpn=1)
    ds = set(int(k) for k in d.allowed_with_siblings(4))
    dp = set(int(k) for k in d.allowed_with_path_cousins(4))
    assert ds <= dp and len(dp) <= d.K
    # +kids adds v's OWN children's blocks (node 1 has children 3,4); superset of path_cousins
    pck = set(int(k) for k in lay.allowed_with_path_cousins_kids(1))
    assert set(int(k) for k in lay.allowed_with_path_cousins(1)) <= pck
    assert set(lay.block[3]) <= pck and set(lay.block[4]) <= pck   # node 1's children
    assert len(pck) <= lay.K


def test_daglayout_singleparent_backward_compat():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)               # scalar-parent map still works
    assert lay.closure(3) == [0, 1, 3]                   # exact old list ordering
    assert list(lay.allowed(3)) == [0, 1] + lay.block[1] + lay.block[3]
    assert list(lay.allowed(1)) == [0, 1] + lay.block[1]
    assert lay.depth(3) == 2 and lay.depth(1) == 1
    assert lay.subtree(1) == {1, 3, 4}

def test_frontier_from_coded_cases():
    lay = DagLayout(DIAMOND)
    assert frontier_from_coded([1, 4], lay) == frozenset({4})       # same-path -> most-specific
    assert frontier_from_coded([4, 5], lay) == frozenset({4, 5})    # comorbid incomparable -> set
    assert frontier_from_coded([2, 3], lay) == frozenset({2, 3})    # contradictory siblings -> set
    assert frontier_from_coded([1, 2, 4], lay) == frozenset({4})    # both parents + child -> child
    # single-parent tree: ancestor+descendant collapses to the descendant
    assert frontier_from_coded([1, 3], DagLayout(PARENT)) == frozenset({3})

def test_fit_gated_accepts_frontier_sets():
    from tests._stm_synth import dag_placement_corpus
    # comorbid training labels (sets) must be accepted and produce a valid beta_hat
    docs, labels, _ = dag_placement_corpus(
        parent=PARENT, node_prev={1:.18,2:.18,3:.16,4:.16,5:.16,6:.16},
        V=120, doc_len=40, seed=1)
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    set_labels = [frozenset({int(y)}) for y in labels[:800]]   # scalars as singleton sets
    rng = np.random.default_rng(3)
    beta = fit_gated(docs[:800], set_labels, lay, 120, n_iter=40, burn=20, rng=rng)
    assert beta.shape == (lay.K, 120)
    assert np.allclose(beta.sum(1), 1.0, atol=1e-6)

def test_evaluate_set_valued_and_instrumented():
    lay = DagLayout(DIAMOND)
    labels = [frozenset({4}), frozenset({4, 5}), frozenset({2, 3}), frozenset({1})]
    profiles = []
    for f in labels:                                   # closure-loaded perfect profiles
        load = set()
        for t in f:
            load |= (set(lay.closure(t)) - {0})
        profiles.append({u: (1.0 if u in load else 0.0) for u in lay.nodes})
    m = evaluate(profiles, labels, lay)
    assert all(v >= 0.99 for v in m["node_auc"].values())   # every node perfectly separated
    assert m["mrr"] == 1.0                                   # best true node ranks first each doc
    assert abs(m["frontier_size_mean"] - 1.5) < 1e-9
    assert abs(m["multi_frontier_rate"] - 0.5) < 1e-9        # 2 of 4 docs are comorbid
    assert np.isfinite(m["mean_hops"])


def test_evaluate_all_unrankable_labels_are_nan():
    # every doc's frontier collapses to root (0) -> no rankable true node -> mrr AND top2 are nan.
    lay = DagLayout(DIAMOND)
    labels = [frozenset({0}), frozenset({0})]
    profiles = [{u: 0.0 for u in lay.nodes} for _ in labels]
    m = evaluate(profiles, labels, lay)
    assert np.isnan(m["mrr"]) and np.isnan(m["top2"])       # not applicable, not 0.0

def test_render_profile_dag_renders_each_node_once():
    from spark_vi.models.topic.dag_placement import render_profile
    lay = DagLayout(DIAMOND)
    aff = {u: 0.1 * u for u in lay.nodes}
    s = render_profile(aff, lay, true_node=4)
    assert "true" in s
    for u in lay.nodes:                                  # every node appears
        assert str(u) in s
    # node 4 is reachable via parents 1 and 2, but its full affinity bar is rendered once
    # its numeric affinity 0.40 should appear exactly once (rendered once, referenced elsewhere)
    assert s.count("0.40") == 1

def test_render_profile_marks_every_true_frontier_node():
    from spark_vi.models.topic.dag_placement import render_profile
    lay = DagLayout(DIAMOND)
    aff = {u: 0.1 * u for u in lay.nodes}
    # set-valued frontier: BOTH true nodes must be marked
    s = render_profile(aff, lay, true_node=frozenset({2, 4}))
    assert s.count("<- true") == 2
    # a single-id true_node still marks exactly one (backward compatible)
    assert render_profile(aff, lay, true_node=2).count("<- true") == 1

def test_dag_placement_corpus_multi_shapes():
    from tests._stm_synth import dag_placement_corpus_multi

    docs, labels, node_codes = dag_placement_corpus_multi(
        parent=DIAMOND, leaf_prev={4: .5, 5: .5}, comorbid_rate=0.3,
        V=120, doc_len=48, seed=0)
    assert len(docs) == len(labels)
    assert all(isinstance(f, frozenset) and len(f) >= 1 for f in labels)
    assert set(node_codes.keys()) == set(DIAMOND.keys())
    assert any(len(f) > 1 for f in labels)                  # some comorbid patients exist


def test_end_to_end_multiparent_comorbid():
    from tests._stm_synth import dag_placement_corpus_multi
    docs, labels, node_codes = dag_placement_corpus_multi(
        parent=DIAMOND, leaf_prev={4: .5, 5: .5}, comorbid_rate=0.3,
        V=120, doc_len=48, seed=2)
    lay = DagLayout(DIAMOND, n_bg=2, tpn=1)
    ntr = int(0.7 * len(docs))
    rng = np.random.default_rng(5)
    beta = fit_gated(docs[:ntr], labels[:ntr], lay, 120, n_iter=80, burn=40, rng=rng)
    codes = set(node_codes.values())
    profs = [profile(strip_dag_node_codes(d, codes), beta, lay, n_iter=40, burn=20, rng=rng)
             for d in docs[ntr:]]
    m = evaluate(profs, labels[ntr:], lay)
    # multi-parent recovery: loose floors (investigate, do NOT loosen, if these fail).
    assert m["auc_by_depth"][1] >= 0.80          # shallow (axis parents 1,2,3)
    assert m["node_auc"][4] >= 0.70              # a multi-parent leaf is found above chance
    assert m["node_auc"][5] >= 0.70
    assert m["multi_frontier_rate"] > 0.0        # comorbid patients are present + measured
    assert np.isfinite(m["mrr"])


def test_auc_all_ties_is_half():
    # identical scores -> AUC must be 0.5 regardless of label order.
    assert abs(_auc(np.zeros(6), [1, 0, 1, 0, 1, 0]) - 0.5) < 1e-9
    assert abs(_auc(np.zeros(6), [0, 0, 0, 1, 1, 1]) - 0.5) < 1e-9


def test_auc_partial_ties_midrank():
    # scores [1,1,0,0], labels [1,0,1,0]: the two positives tie a positive with a
    # negative at each score level -> AUC 0.5 (midranks), NOT order-dependent 0/1.
    assert abs(_auc(np.array([1.0, 1.0, 0.0, 0.0]), [1, 0, 1, 0]) - 0.5) < 1e-9


def test_auc_perfect_and_degenerate():
    assert _auc(np.array([3.0, 2.0, 1.0, 0.0]), [1, 1, 0, 0]) == 1.0
    assert np.isnan(_auc(np.array([1.0, 2.0]), [1, 1]))     # one class -> nan


from spark_vi.models.topic.dag_placement import (
    DagLayout, evaluate, _average_precision,
)


def test_average_precision_perfect_and_constant():
    # perfect ranking -> AP 1.0
    assert abs(_average_precision([3.0, 2.0, 1.0, 0.0], [1, 1, 0, 0]) - 1.0) < 1e-9
    # constant scores -> AP == prevalence (2/4), NOT 0/1 (the AUC-tie failure mode)
    assert abs(_average_precision([1.0, 1.0, 1.0, 1.0], [1, 0, 1, 0]) - 0.5) < 1e-9
    import numpy as np
    assert np.isnan(_average_precision([1.0, 2.0], [0, 0]))   # no positives -> nan


def test_evaluate_adds_pr_recall_ci_keys():
    parent = {1: 0, 2: 0, 3: 1}
    lay = DagLayout(parent, n_bg=2, tpn=1)     # nodes [1,2,3], depth(3)=2
    # 4 docs: two truly under node 1 (via leaf 3), two under node 2.
    profiles = [
        {1: 0.6, 2: 0.1, 3: 0.5},   # true {3}
        {1: 0.4, 2: 0.2, 3: 0.3},   # true {3}
        {1: 0.1, 2: 0.7, 3: 0.0},   # true {2}
        {1: 0.2, 2: 0.6, 3: 0.1},   # true {2}
    ]
    labels = [{3}, {3}, {2}, {2}]
    ev = evaluate(profiles, labels, lay)
    assert set(ev["node_ap"]) == {1, 2, 3}
    assert 0.0 <= ev["ap_macro"] <= 1.0
    assert 0.0 <= ev["ap_micro"] <= 1.0
    assert 0.0 <= ev["ap_prevalence_weighted"] <= 1.0
    assert set(ev["recall_at_k"]) == {1, 2, 3}
    assert 0.0 <= ev["recall_at_k"][1] <= ev["recall_at_k"][3] <= 1.0
    # CI present for the headline metrics and brackets the point estimate.
    for key in ("ap_macro", "mrr", "top2", "recall_at_1"):
        lo, hi = ev["ci"][key]
        assert lo <= hi
    assert ev["ci"]["ap_macro"][0] <= ev["ap_macro"] <= ev["ci"]["ap_macro"][1] + 1e-9


def test_bootstrap_recall_at_1_matches_frontier_normalized_recall():
    """Regression: the CI's recall_at_1 must be the SAME frontier-normalized
    recall as the reported recall_at_k[1] (|top-1 ∩ frontier| / |frontier|), not
    top-1 accuracy. A single doc with a 2-node comorbid frontier whose argmax hits
    ONE true node has recall@1 = 0.5; with one doc every bootstrap resample is that
    doc, so the CI collapses to exactly (0.5, 0.5) — the old top-1-accuracy code
    gave 1.0."""
    from spark_vi.models.topic.dag_placement import DagLayout, evaluate
    lay = DagLayout({1: 0, 2: 0, 3: 0}, n_bg=2, tpn=1)     # nodes [1,2,3]
    # one doc, true frontier {1,2}, argmax at node 1 -> recall@1 = 1/2.
    profiles = [{1: 0.9, 2: 0.1, 3: 0.0}]
    labels = [{1, 2}]
    ev = evaluate(profiles, labels, lay)
    assert abs(ev["recall_at_k"][1] - 0.5) < 1e-9
    assert ev["ci"]["recall_at_1"] == (0.5, 0.5)


import numpy as np
from spark_vi.models.topic.dag_placement import _empirical_right_tail_p, _fdr_reject

def test_empirical_right_tail_p_counts_and_floor():
    ref = np.array([0.0, 0.1, 0.2, 0.3])          # n=4
    # value above all -> ge=0 -> p=(0+1)/(4+1)=0.2 (floored, never 0)
    # value 0.2 -> ge counts {0.2,0.3}=2 -> p=(2+1)/5=0.6
    p = _empirical_right_tail_p(np.array([0.5, 0.2, -1.0]), ref)
    assert np.allclose(p, [0.2, 0.6, 1.0])
    assert (p > 0).all()

def test_fdr_reject_bh_uniform_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=5000)                    # pure null
    rej = _fdr_reject(p, 0.1, "bh")
    assert rej.sum() <= 0.02 * len(p)             # few false rejections under the null

def test_fdr_reject_bh_planted_and_by_subset():
    p = np.concatenate([np.full(20, 1e-6), np.random.default_rng(1).uniform(size=980)])
    bh = _fdr_reject(p, 0.1, "bh")
    by = _fdr_reject(p, 0.1, "by")
    assert bh[:20].all()                          # the strong signals are found
    assert set(np.nonzero(by)[0]).issubset(set(np.nonzero(bh)[0]))   # BY ⊆ BH
    assert by.sum() <= bh.sum()


from spark_vi.models.topic.dag_placement import _assign_length_bins, per_node_discoveries

def test_assign_length_bins_quantile_and_single():
    ref = np.arange(100.0)
    b = _assign_length_bins(np.array([1.0, 50.0, 99.0]), ref, 4)
    assert b[0] == 0 and b[2] == 3 and 0 <= b[1] <= 3
    assert (_assign_length_bins(np.array([1.0, 9.0]), ref, 1) == 0).all()

def test_per_node_discoveries_recovers_planted_signal():
    rng = np.random.default_rng(0)
    n_bg, n_case, n_nodes = 500, 40, 3
    P = rng.uniform(0, 0.05, size=(n_bg + n_case, n_nodes))
    is_fg = np.zeros(n_bg + n_case, bool); is_fg[n_bg:] = True
    P[n_bg:, 1] += 0.6                       # cases are elevated on node 1
    out = per_node_discoveries(P, is_fg, np.full(n_bg + n_case, 10.0),
                               q_grid=[0.1], n_length_bins=1)
    disc = out["discoveries"][0.1]
    assert disc[n_bg:, 1].mean() > 0.7       # most true cases discovered on node 1
    assert disc[:n_bg, 1].mean() < 0.1       # few background discovered

def test_per_node_discoveries_length_conditioning_calibrates_pvalues():
    # Long records carry more node-0 mass for EVERYONE (a length confound, NOT
    # signal). Assert on the p-values, not BH discoveries: with an all-background,
    # self-referential null the BH floor (1/(n+1) > q/m when m ~ n) suppresses
    # discoveries in BOTH arms, so a discovery-count comparison is degenerate
    # (0 < 0). The p-values, however, show the effect directly: pooling the
    # short+long null makes a length-confounded long doc look falsely significant
    # (small p); conditioning on length compares it to its own (long) bin, where
    # its mass is typical, restoring a calibrated (larger) p. That correction is
    # exactly what length-conditioning is for.
    rng = np.random.default_rng(2)
    n, n_nodes = 1200, 2
    length = rng.choice([5.0, 50.0], size=n)  # ~50/50 -> the median splits the bins
    is_fg = np.zeros(n, bool)
    P = rng.uniform(0, 0.02, size=(n, n_nodes))
    P[length == 50.0, 0] += 0.3               # confound: long docs, node 0
    pooled = per_node_discoveries(P, is_fg, length, q_grid=[0.1], n_length_bins=1)
    cond = per_node_discoveries(P, is_fg, length, q_grid=[0.1], n_length_bins=2)
    assert len(np.unique(cond["bins"])) == 2                 # conditioning really split
    longm = length == 50.0
    p_pool = pooled["pmat"][longm, 0].mean()
    p_cond = cond["pmat"][longm, 0].mean()
    assert p_cond > p_pool + 0.15   # ~0.50 vs ~0.25: conditioning removes the false significance


from spark_vi.models.topic.dag_placement import _zib_empirical_gap

def test_zib_gap_small_for_beta_like_sample():
    rng = np.random.default_rng(0)
    pos = rng.beta(2.0, 8.0, size=4000)
    vals = np.concatenate([np.zeros(1000), pos])      # zero-inflated Beta by construction
    assert _zib_empirical_gap(vals) < 0.05

def test_zib_gap_large_for_non_beta_sample():
    rng = np.random.default_rng(1)
    # a bimodal positive part a single Beta cannot fit
    vals = np.concatenate([rng.uniform(0.05, 0.10, 2000), rng.uniform(0.85, 0.95, 2000)])
    assert _zib_empirical_gap(vals) > 0.15

def test_zib_gap_degenerate_returns_nan():
    assert np.isnan(_zib_empirical_gap(np.zeros(50)))

def test_zib_gap_in_simplex_returns_finite_or_nan():
    rng = np.random.default_rng(2)
    pos = rng.beta(2.0, 8.0, size=200)
    vals = np.concatenate([np.zeros(50), pos])
    gap = _zib_empirical_gap(vals)
    assert np.isnan(gap) or np.isfinite(gap)

def test_zib_gap_out_of_simplex_returns_nan():
    # ZIB models node-block mass in [0,1]; unbounded scores (e.g. LR/explain-away
    # log-ratios) are out of domain and must return nan rather than a silently
    # clipped, meaningless statistic.
    vals = np.array([-0.5, 0.3, 2.1])
    assert np.isnan(_zib_empirical_gap(vals))


def _toy_lay():
    # 3-node flat layout: root 0 with children 1,2,3. DagLayout(parent_map,
    # n_bg, tpn); parent_map is child -> [parent], root 0 has no entry.
    return DagLayout({1: [0], 2: [0], 3: [0]}, n_bg=2, tpn=1)

def test_evaluate_backward_compatible_and_fdr_block_present():
    lay = _toy_lay()
    # NOTE: the plan's original premise gave every one of the 30 patients
    # (cases and background alike) the IDENTICAL profile (0.6 on node 1, 0.05
    # elsewhere) -- with cases statistically indistinguishable from the
    # background reference, per_node_discoveries correctly finds p=1.0 for
    # every doc (each value ties every reference value) and BH rejects
    # nothing, so "n_discoveries >= 1" was unreachable regardless of
    # implementation. Fixed here by giving cases a genuinely elevated node-1
    # mass vs a distinct background reference, with a background arm large
    # enough (n_bg=60) that the BH floor 1/(n_bg+1) clears the q=0.1
    # threshold once corrected across all m=n_case+n_bg tests (verified: floor
    # ~0.016, required cases ~3.7 given q=0.1 and m=75, n_case=15 well above).
    n_case, n_bg = 15, 60
    profiles = [{u: (0.6 if u == 1 else 0.05) for u in lay.nodes} for _ in range(n_case)] + \
               [{u: 0.05 for u in lay.nodes} for _ in range(n_bg)]
    labels = [{1} for _ in range(n_case)] + [set() for _ in range(n_bg)]   # cases on node 1
    out = evaluate(profiles, labels, lay)                            # no doc_lengths
    for k in ("mrr", "top2", "auc_by_depth", "detection", "recall_at_k"):
        assert k in out                                             # prior keys intact
    assert "fdr" in out and 0.1 in out["fdr"]["by_q"]
    assert out["fdr"]["by_q"][0.1]["n_discoveries"] >= 1

def test_evaluate_fdr_multimorbidity_beats_argmax():
    lay = _toy_lay()
    # patients truly on BOTH node 1 and node 2, with mass on both blocks.
    profiles = [{1: 0.4, 2: 0.4, 3: 0.02} for _ in range(20)] + \
               [{u: 0.02 for u in lay.nodes} for _ in range(200)]
    labels = [{1, 2} for _ in range(20)] + [set() for _ in range(200)]
    out = evaluate(profiles, labels, lay, doc_lengths=[10.0] * 220)
    mm = out["fdr"]["multimorbidity"]
    # Like-for-like on CORRECT captures: argmax credits at most one true node per
    # patient (<=1); FDR credits both true nodes (~2). Both count true captures.
    assert mm["argmax_true_baseline_per_multimorbid"] <= 1.0
    assert mm["mean_true_discoveries_per_multimorbid"] > mm["argmax_true_baseline_per_multimorbid"]


def test_fdr_discovery_report_planted_and_null():
    import numpy as np
    from spark_vi.models.topic.dag_placement import fdr_discovery_report
    n_bg_docs, n_fg_docs, n_nodes = 200, 40, 3
    # Background docs are exact zeros on every node (a degenerate, zero-variance
    # null) and foreground docs are a constant elevated mass on node 0 only. This
    # is deliberately noise-free rather than rng-perturbed: _empirical_right_tail_p
    # gives every tied background doc p=1.0 (it never exceeds its own reference)
    # while every foreground doc gets the floor p=1/(n_bg+1), so the test is a
    # deterministic, seed-independent check of the planted-signal contract rather
    # than one draw of a BH procedure that (correctly, by construction) admits a
    # bounded rate of false discoveries at the reference tail under real noise.
    P = np.zeros((n_bg_docs + n_fg_docs, n_nodes))
    P[n_bg_docs:, 0] = 0.9                                                 # planted signal
    is_fg = np.zeros(n_bg_docs + n_fg_docs, dtype=bool); is_fg[n_bg_docs:] = True
    truth = np.zeros((n_bg_docs + n_fg_docs, n_nodes), dtype=bool)
    truth[n_bg_docs:, 0] = True                                            # fg docs are node-0 positives
    mm_rows = np.zeros(n_bg_docs + n_fg_docs, dtype=bool)                  # none multimorbid
    lengths = np.ones(n_bg_docs + n_fg_docs)
    rep = fdr_discovery_report(P, is_fg, lengths, truth, mm_rows,
                               q_grid=(0.05, 0.10, 0.20), n_length_bins=1)
    # planted node-0 signal -> discoveries at q=0.20 with precision 1.0 (only true node-0 docs)
    assert rep["by_q"][0.20]["n_discoveries"] >= 1
    assert rep["by_q"][0.20]["precision"] == 1.0
    assert set(rep.keys()) == {"q_grid", "by_q", "multimorbidity", "saturation_rate",
                               "zib_gap_mean", "zib_gap_max", "n_length_bins_effective"}


def test_fdr_discovery_report_all_null_no_discoveries():
    import numpy as np
    from spark_vi.models.topic.dag_placement import fdr_discovery_report
    n, n_nodes = 120, 2
    rng = np.random.default_rng(1)
    P = np.abs(rng.normal(0.0, 0.01, size=(n, n_nodes)))     # no fg/bg separation
    is_fg = np.zeros(n, dtype=bool); is_fg[100:] = True
    truth = np.zeros((n, n_nodes), dtype=bool); truth[100:, 0] = True
    mm_rows = np.zeros(n, dtype=bool)
    rep = fdr_discovery_report(P, is_fg, np.ones(n), truth, mm_rows,
                               q_grid=(0.05, 0.10, 0.20), n_length_bins=1)
    assert all(rep["by_q"][q]["n_discoveries"] == 0 for q in (0.05, 0.10, 0.20))


from spark_vi.models.topic.dag_placement import lr_placement_scores, lr_decompose

def _lr_lay():
    return DagLayout({1: [0], 2: [0]}, n_bg=1, tpn=1)   # 2 nodes, blocks [1],[2]; K=3

def test_lr_scores_distinctive_code_separates_where_thetamass_would_not():
    lay = _lr_lay()
    V = 6
    lam = np.full((3, V), 1.0)          # bg topic 0 flat
    lam[1] = np.array([1, 1, 1, 40, 1, 1.0])   # node 1 signature = code 3
    lam[2] = np.array([1, 1, 1, 1, 40, 1.0])   # node 2 signature = code 4
    # background base rate: code 3 and 4 are globally rare, code 0 common
    bg = np.array([50, 10, 10, 1, 1, 1.0]); bg = bg / bg.sum()
    case = np.zeros(V); case[3] = 1; case[0] = 5     # has the node-1 signature + common noise
    ctrl = np.zeros(V); ctrl[0] = 6                  # only the common code
    S = lr_placement_scores(np.vstack([case, ctrl]), lam, lay, alpha=1.0, background=bg)
    # Raw LR scores can be negative (the shared common-code terms are penalised
    # under every node); what matters is RANKING. The case outranks the control on
    # node 1 (has its signature), and for the case node 1 (its signature) beats
    # node 2. That separation is exactly what the θ-mass readout misses.
    assert S[0, 0] > S[1, 0]                          # case > control on node 1
    assert S[0, 0] > S[0, 1]                          # case's node 1 (signature) > its node 2

def test_lr_scores_shrinkage_pulls_toward_zero():
    lay = _lr_lay()
    V = 5
    lam = np.full((3, V), 1.0); lam[1, 2] = 20.0       # node 1 likes code 2
    bg = np.array([10, 10, 1, 10, 10.0]); bg = bg / bg.sum()
    doc = np.zeros(V); doc[2] = 1
    s_small = lr_placement_scores(doc[None], lam, lay, alpha=0.0, background=bg)[0, 0]
    s_big = lr_placement_scores(doc[None], lam, lay, alpha=1e6, background=bg)[0, 0]
    assert s_small > s_big                              # strong shrinkage -> toward 0
    assert abs(s_big) < 1e-2                             # alpha huge -> ~neutral

def test_lr_scores_alpha_zero_unseen_code_is_finite():
    lay = _lr_lay()
    V = 4
    lam = np.full((3, V), 1.0); lam[1, 1] = 5.0
    lam[1, 3] = 0.0                                     # node 1 NEVER saw code 3
    bg = np.array([1, 1, 1, 1.0]) / 4
    doc = np.zeros(V); doc[3] = 1                       # patient has the unseen code
    s = lr_placement_scores(doc[None], lam, lay, alpha=0.0, background=bg)[0, 0]
    assert np.isfinite(s)                               # epsilon floor, not -inf

def test_lr_decompose_sums_to_score():
    lay = _lr_lay()
    V = 5
    lam = np.full((3, V), 1.0); lam[1] = np.array([1, 1, 20, 1, 5.0])
    bg = np.array([20, 10, 1, 5, 2.0]); bg = bg / bg.sum()
    doc = np.array([0, 1, 2, 0, 3.0])                   # counts
    parts = lr_decompose(doc, lam, lay, 1, alpha=1.0, background=bg)
    score = lr_placement_scores(doc[None], lam, lay, alpha=1.0, background=bg)[0, 0]
    assert abs(sum(c for _, _, c in parts) - score) < 1e-9
    assert all(cnt > 0 for _, cnt, _ in parts)          # only present codes listed


def test_lr_placement_scores_infinite_alpha_is_finite_and_separates():
    # alpha=inf is the parameter-free limit (score direction nc/bg - Σλ). It must
    # stay finite and still rank a distinctive-code case above a common-code control.
    lay = _lr_lay()
    V = 6
    lam = np.full((3, V), 1.0); lam[1, 3] = 40.0          # node 1 signature = code 3
    bg = np.array([50, 10, 10, 1, 1, 1.0]); bg = bg / bg.sum()
    case = np.zeros(V); case[3] = 1; case[0] = 5
    ctrl = np.zeros(V); ctrl[0] = 6
    S = lr_placement_scores(np.vstack([case, ctrl]), lam, lay,
                            alpha=float("inf"), background=bg)
    assert np.isfinite(S).all()
    assert S[0, 0] > S[1, 0]                              # separates at the limit


from spark_vi.models.topic.dag_placement import lr_auc_sweep

def test_lr_auc_sweep_separates_planted_cases():
    rng = np.random.default_rng(0)
    lay = _lr_lay()                                  # 2 nodes; BOTH informative so
    V = 8                                            # max-over-nodes has no flat node
    lam = np.full((3, V), 1.0)                       # to win at ~0 (a real-data hazard,
    lam[1, 5] = 60.0                                 # noted in the caveats)
    lam[2, 6] = 60.0                                 # node 1 -> code 5, node 2 -> code 6
    bg = np.full(V, 1.0) / V
    rows, is_fg = [], []
    for _ in range(20):                              # node-1 cases: code 5 + light noise
        d = np.zeros(V); d[5] = 1; d[0] = rng.integers(0, 2); rows.append(d); is_fg.append(True)
    for _ in range(20):                              # node-2 cases: code 6
        d = np.zeros(V); d[6] = 1; d[0] = rng.integers(0, 2); rows.append(d); is_fg.append(True)
    for _ in range(300):                             # controls: only the common code
        d = np.zeros(V); d[0] = rng.integers(1, 4); rows.append(d); is_fg.append(False)
    bow = np.array(rows)
    out = lr_auc_sweep(bow, lam, lay, np.array(is_fg),
                       alpha_grid=[0.0, 1.0, 10.0, 100.0], background=bg)
    assert set(out) == {0.0, 1.0, 10.0, 100.0}
    assert max(out.values()) > 0.9                   # SOME alpha cleanly separates the signal


def test_routing_rows_soft_responsibility_and_conservation():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout, _routing_rows
    lay = DagLayout({1: 0, 2: 0}, n_bg=1, tpn=1)               # K=3: [bg, node1, node2]
    V = 3
    lam = np.zeros((3, V))
    lam[0] = [8.0, 0.0, 2.0]     # background topic: code0 (and some code2)
    lam[1] = [0.0, 5.0, 0.0]     # node1 topic: code1 only (distinctive)
    lam[2] = [0.0, 0.0, 6.0]     # node2 topic: code2 only
    r = _routing_rows(lam, lay)                                # [2 nodes x V]
    # code1 is emitted only by node1's topic -> fully node1
    assert np.isclose(r[0, 1], 1.0) and np.isclose(r[1, 1], 0.0)
    # code0 is emitted only by background -> neither node claims it
    assert np.isclose(r[0, 0], 0.0) and np.isclose(r[1, 0], 0.0)
    # code2 is shared by background (P=2/10=0.2) and node2 (P=6/6=1.0):
    # node2 responsibility = 1.0 / (0.2 + 1.0) = 0.8333...
    assert np.isclose(r[1, 2], 1.0 / 1.2, atol=1e-6)
    # conservation: node responsibilities + background responsibility = 1 per seen code
    #   (background resp = 1 - sum of node resp); must be in [0,1].
    node_sum = r.sum(axis=0)
    assert np.all(node_sum <= 1.0 + 1e-9) and np.all(node_sum >= -1e-9)
    assert np.isclose(node_sum[2], 1.0 / 1.2, atol=1e-6)       # only node2 (+bg) see code2

def test_explain_away_suppresses_comorbid_negatives():
    # A doc = 1 distinctive rare code (d, emitted by node1) + several generic codes
    # (g*, emitted by background). Plain LR docks node1 for the generic codes (they
    # are below base rate under node1); explain-away routes them to background, so
    # they contribute ~0 -> explain-away score(node1) >= plain LR score(node1).
    import numpy as np
    from spark_vi.models.topic.dag_placement import (
        DagLayout, explain_away_placement_scores, lr_placement_scores)
    lay = DagLayout({1: 0, 2: 0}, n_bg=1, tpn=1)               # K=3
    V = 4                                                      # d=0, g1=1, g2=2, g3=3
    lam = np.zeros((3, V))
    lam[0] = [0.0, 40.0, 40.0, 40.0]   # background: the generic codes
    lam[1] = [30.0, 0.0, 0.0, 0.0]     # node1: distinctive code d only
    lam[2] = [0.0, 1.0, 1.0, 1.0]      # node2: weak/uniform
    bow = np.zeros((1, V)); bow[0] = [1, 1, 1, 1]              # d + 3 generic codes
    bg = np.array([0.10, 0.30, 0.30, 0.30])                   # base rate (d rarer)
    i = lay.nodes.index(1)
    lr = lr_placement_scores(bow, lam, lay, alpha=float("inf"), background=bg)[0, i]
    ea = explain_away_placement_scores(bow, lam, lay, alpha=float("inf"),
                                       background=bg)[0, i]
    assert ea >= lr - 1e-9        # comorbid negatives suppressed
    assert ea > 0.0               # the distinctive code still carries positive signal

def test_explain_away_background_only_doc_scores_near_zero():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout, explain_away_placement_scores
    lay = DagLayout({1: 0, 2: 0}, n_bg=1, tpn=1)
    V = 3
    lam = np.zeros((3, V)); lam[0] = [5.0, 5.0, 5.0]           # only background has mass
    bow = np.zeros((1, V)); bow[0] = [1, 1, 1]
    s = explain_away_placement_scores(bow, lam, lay, alpha=float("inf"),
                                      background=np.array([0.34, 0.33, 0.33]))
    assert np.allclose(s, 0.0, atol=1e-6)                      # nodes have no routing -> ~0


def test_explain_away_decompose_shows_routing_and_sums_to_score():
    import numpy as np
    from spark_vi.models.topic.dag_placement import (
        DagLayout, explain_away_decompose, explain_away_placement_scores)
    lay = DagLayout({1: 0, 2: 0}, n_bg=1, tpn=1)
    V = 4
    lam = np.zeros((3, V))
    lam[0] = [0.0, 40.0, 40.0, 40.0]; lam[1] = [30.0, 0.0, 0.0, 0.0]; lam[2] = [0.0, 1.0, 1.0, 1.0]
    row = np.array([1.0, 1.0, 1.0, 1.0])
    bg = np.array([0.10, 0.30, 0.30, 0.30])
    rows = explain_away_decompose(row, lam, lay, 1, alpha=float("inf"), background=bg)
    by_w = {w: (cnt, r, c) for (w, cnt, r, c) in rows}
    # distinctive code d=0 routes to node1 (r ~ 1), positive contribution
    assert by_w[0][1] > 0.9 and by_w[0][2] > 0.0
    # generic codes route to background (r ~ 0) -> contribution ~ 0 (not negative)
    for g in (1, 2, 3):
        assert abs(by_w[g][1]) < 0.05 and abs(by_w[g][2]) < 1e-3
    # Σ contribution == the node score
    total = sum(c for (_w, _cnt, _r, c) in rows)
    score = explain_away_placement_scores(row[None], lam, lay, alpha=float("inf"),
                                          background=bg)[0, lay.nodes.index(1)]
    assert np.isclose(total, score, atol=1e-6)


def test_log1p_dense_bow_does_not_crash_and_matches_sparse():
    # Regression for the broken sparse/dense discriminator: `hasattr(X, "data")`
    # is true for BOTH scipy sparse matrices AND numpy ndarrays (ndarray.data is a
    # read-only memoryview), so a dense bow with count_mode="log1p" used to take the
    # sparse in-place-mutate branch (`X.data = np.log1p(X.data)`) and crash with
    # AttributeError: attribute 'data' of 'numpy.ndarray' objects is not writable.
    # The driver's per-case viewer builds exactly this kind of dense bow row.
    import scipy.sparse
    from spark_vi.models.topic.dag_placement import explain_away_placement_scores

    lay = _lr_lay()
    V = 6
    lam = np.full((3, V), 1.0)
    lam[1] = np.array([1, 1, 1, 40, 1, 1.0])
    lam[2] = np.array([1, 1, 1, 1, 40, 1.0])
    bg = np.array([50, 10, 10, 1, 1, 1.0]); bg = bg / bg.sum()
    dense = np.zeros((2, V))
    dense[0, 3] = 1; dense[0, 0] = 5
    dense[1, 0] = 6

    lr_dense = lr_placement_scores(dense, lam, lay, alpha=1.0, background=bg,
                                   count_mode="log1p")
    ea_dense = explain_away_placement_scores(dense, lam, lay, alpha=1.0, background=bg,
                                             count_mode="log1p")
    assert lr_dense.shape == (2, 2) and np.isfinite(lr_dense).all()
    assert ea_dense.shape == (2, 2) and np.isfinite(ea_dense).all()

    # sparse/dense equivalence: same scores either way under count_mode="log1p"
    sparse = scipy.sparse.csr_matrix(dense)
    lr_sparse = lr_placement_scores(sparse, lam, lay, alpha=1.0, background=bg,
                                    count_mode="log1p")
    ea_sparse = explain_away_placement_scores(sparse, lam, lay, alpha=1.0, background=bg,
                                              count_mode="log1p")
    assert np.allclose(lr_dense, lr_sparse, atol=1e-9)
    assert np.allclose(ea_dense, ea_sparse, atol=1e-9)


def test_daglayout_descendants_is_proper_and_mirrors_closure():
    from spark_vi.models.topic.dag_placement import DagLayout
    # DAG: 1 -> 2 -> 4, 1 -> 3, and 4 also a child of 3 (multi-parent diamond)
    lay = DagLayout({2: 1, 3: 1, 4: 2}, n_bg=1, tpn=1)
    lay.parents.setdefault(4, [])
    if 3 not in lay.parents[4]:
        lay.parents[4].append(3)          # 4 has parents {2,3}
    lay.children.setdefault(3, [])
    if 4 not in lay.children[3]:
        lay.children[3].append(4)
    # node 1 (anchor) has every other node as a descendant
    assert set(lay.descendants(1)) == {2, 3, 4}
    # node 4 (leaf) has none
    assert lay.descendants(4) == []
    # descendants excludes u itself and is disjoint from proper ancestors
    assert 2 not in lay.descendants(2)
    assert set(lay.descendants(2)) == {4}
    # sorted by (depth, id)
    d = lay.descendants(1)
    assert d == sorted(d, key=lambda x: (lay.depth(x), x))


def test_allowed_with_siblings_adds_sibling_blocks():
    """Localized-head support: allowed(v) + siblings' blocks (the closure contrast
    set). Layout {1:0,2:0,3:1}, n_bg=2, tpn=1 -> block[1]=[2],block[2]=[3],block[3]=[4]."""
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=2, tpn=1)
    def s(v):
        return set(int(k) for k in lay.allowed_with_siblings(v))
    # node 1: allowed {0,1,2} + sibling 2's block {3}  (both are root's children)
    assert s(1) == {0, 1, 2, 3}
    assert s(2) == {0, 1, 2, 3}
    # node 3: parent is 1, which has only child 3 -> no siblings -> == allowed(3)
    assert s(3) == {0, 1, 2, 4} == set(int(k) for k in lay.allowed(3))
    # root: no parents, no siblings -> background only
    assert s(0) == {0, 1}
    # siblings only ADD (superset of allowed), and stay within [0, K)
    for v in (0, 1, 2, 3):
        assert set(int(k) for k in lay.allowed(v)) <= s(v)
        assert max(s(v)) < lay.K


def test_cost_report_flags_support_and_dense_vs_localized():
    """Pre-flight cost profile: support sizes, fan-out, and dense-vs-localized head
    matrix costs. A deep bounded-fan-out tree localizes well (support << K); a flat
    high-fan-out DAG does not (a node's siblings ~ K, so localized ~ dense)."""
    from spark_vi.models.topic.dag_placement import DagLayout
    # deep, bounded fan-out: root -> A,B ; A -> A1,A2 ; B -> B1,B2  (each node few sibs)
    deep = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    lay = DagLayout(deep, n_bg=2, tpn=1)                    # K = 2 + 6 = 8
    d, s = lay.cost_report(C=7, vocab_size=100, localized=True)
    assert d["C"] == 7 and d["K"] == 8
    assert d["support_max"] <= d["K"]                       # never exceeds K
    assert d["localized_head_mem_bytes"] <= d["dense_head_mem_bytes"]
    # collected = the padded (C, S, S) actually emitted: packed floor <= collected
    # <= dense wall (this is the size that hits driver.maxResultSize).
    assert d["collected_head_mem_bytes"] == 7 * d["support_max"] ** 2 * 8
    assert (d["localized_head_mem_bytes"] <= d["collected_head_mem_bytes"]
            <= d["dense_head_mem_bytes"])
    assert d["lambda_bytes"] == 8 * 100 * 8                 # K*V*8
    assert "SIZE/COST PROFILE" in s and "support/node" in s and "collected" in s

    # flat high-fan-out: 6 nodes all under root -> each sees ~5 siblings ~ K.
    flat = {c: 0 for c in range(1, 7)}
    lay2 = DagLayout(flat, n_bg=2, tpn=1)                   # K = 8
    d2, _ = lay2.cost_report(C=7, localized=True)
    assert d2["fanout_max"] == 6
    # a root child's support = bg + own + 5 siblings = 2+1+5 = 8 = K (localization
    # gives nothing on a flat DAG — the report makes that visible).
    assert d2["support_max"] == d2["K"]
    assert d2["lambda_bytes"] is None                       # vocab_size omitted
