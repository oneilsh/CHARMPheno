"""Framing 1 — differential with reasons (differential.py): the four engines on small
synthetic worlds with known answers."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "analysis" / "cloud"), str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import differential as df  # noqa: E402

V = 12


def test_filter_record_drops_excluded_codes_and_counts_distinct():
    bags, lens = df.filter_record({0: {1: 3.0, 2: 1.0}, 1: {5: 2.0}}, {0: {2}})
    assert bags == {0: {1: 3.0}, 1: {5: 2.0}} and lens == {0: 1.0, 1: 1.0}


def _presence_world():
    # topics: 0 background (flat); 1 = A's aortic subtype (code 0); 2 = A's skeletal
    # subtype (code 1); 3 = B's profile (code 2). Every profile also carries the
    # shared codes 5 and 6, as real signatures carry common care codes.
    lam = np.full((4, V), 1.0)
    lam[1:, 5] = lam[1:, 6] = 60.0
    lam[1, 0] = 200.0
    lam[2, 1] = 200.0
    lam[3, 2] = 200.0
    ev = np.array([100.0, 1000.0, 200.0, 800.0])
    subtypes = {10: [(1, 1000.0), (2, 200.0)], 20: [(3, 800.0)]}
    return [lam], subtypes, ev


def test_subtype_mixture_recognizes_the_minor_subtype_that_averaging_dilutes():
    lams, subtypes, ev = _presence_world()
    eng = df.PresenceSubtypeEngine(lams, subtypes, 1, ev, [10, 20], {})
    # a skeletal-type A patient: code 1 present (and a few shared codes), no code 0
    bags, lens = df.filter_record({0: {1: 1.0, 5: 1.0, 6: 1.0}}, {})
    s = eng.scores(eng.subtype_ll(bags, lens))
    assert s[0] > s[1]                       # A wins through its skeletal subtype
    # an averaged-profile A (one subtype with both codes diluted) does worse here
    avg = df.PresenceSubtypeEngine(
        [np.vstack([lams[0], (1000 * lams[0][1] / lams[0][1].sum()
                              + 200 * lams[0][2] / lams[0][2].sum())[None, :] * 50])],
        {10: [(4, 1.0)], 20: [(3, 1.0)]}, 1, np.append(ev, 1.0), [10, 20], {})
    s_avg = avg.scores(avg.subtype_ll(bags, lens))
    assert (s[0] - s[1]) > (s_avg[0] - s_avg[1])
    r = eng.reasons(eng.subtype_ll(bags, lens), 10)
    assert max(r, key=r.get) == (0, 1)


def test_excluded_codes_leave_the_absent_term_consistent():
    lams, subtypes, ev = _presence_world()
    eng = df.PresenceSubtypeEngine(lams, subtypes, 1, ev, [10, 20], {0: {0}})
    bags, lens = df.filter_record({0: {0: 2.0, 1: 1.0}}, {0: {0}})
    s = eng.scores(eng.subtype_ll(bags, lens))
    assert np.all(np.isfinite(s)) and lens[0] == 1.0


def _gated_world():
    # K = 4: 0, 1 background; 2 = block of member 10 (code 0); 3 = block of 20 (code 1)
    lam = np.full((4, V), 0.5)
    lam[0, 3:] = 40.0
    lam[1, 3:] = 40.0
    lam[2, 0] = 200.0
    lam[3, 1] = 200.0
    from scipy.special import digamma
    eb = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    return eb, np.full(4, 0.5), {10: [0, 1, 2], 20: [0, 1, 3]}


def test_gated_likelihood_prefers_the_member_whose_block_explains_the_record():
    eb, alpha, gates = _gated_world()
    eng = df.GatedLikelihoodEngine(eb, alpha, gates, [10, 20])
    e, lphi = eng.elbos([3, 4, 1], [2.0, 1.0, 3.0])          # marker of member 20
    assert e[1] > e[0]
    r = eng.reasons([3, 4, 1], [2.0, 1.0, 3.0], lphi, 20)
    assert max(r, key=r.get) == 1
    e0, _ = eng.elbos([3, 4], [2.0, 1.0])                    # no marker: near tie
    assert abs(e0[0] - e0[1]) < abs(e[0] - e[1])


def test_multiclass_head_learns_a_separable_problem_and_predicts_simplex_rows():
    rng = np.random.default_rng(0)
    n, K = 400, 30
    y = rng.integers(0, 3, n)
    theta = rng.dirichlet(np.ones(K), n)
    theta[np.arange(n), 10 + y] += 0.5
    theta /= theta.sum(axis=1, keepdims=True)
    h = df.MulticlassHead(np.arange(K), 3).fit(theta[:300], y[:300])
    P = h.predict(theta[300:])
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.mean(P.argmax(axis=1) == y[300:]) > 0.9


def test_fit_tau_shrinks_overconfident_scores_and_metrics_are_pooled():
    rng = np.random.default_rng(1)
    truths, scores = [], []
    for i in range(200):
        t = int(rng.integers(0, 2))
        # scores say the truth by a margin of 30 nats but are right only 70% of the time
        good = rng.random() < 0.7
        s = np.zeros(2); s[t if good else 1 - t] = 30.0
        truths.append([t]); scores.append(s)
    lp = np.log([0.5, 0.5])
    tau, _ = df.fit_tau_scores(lp, scores, truths)
    assert tau < 0.1
    posts = [df.posterior_from_scores(lp, s, tau) for s in scores]
    m = df.metrics(posts, truths)
    assert 0.6 < m["top1"] < 0.8 and m["ece"] < 0.1 and m["top3"] == 1.0


def test_pool_reasons_suppresses_small_members():
    per = [(1, {"a": 1.0, "b": 0.5})] * 25 + [(2, {"c": 1.0})] * 5
    out = df.pool_reasons(per)
    assert out == {1: ["a", "b"]}


def test_run_differential_end_to_end_on_a_tiny_run(tmp_path, monkeypatch):
    import json
    sys.path.insert(0, str(REPO_ROOT / "tests" / "scripts"))
    sys.path.insert(0, str(REPO_ROOT / "charmpheno"))
    from test_inspect_topics import _make_census_run
    import gated_pc_readout as gpr
    _make_census_run(tmp_path)                       # node1 sig idx 0; node2 sigs idx 2, 3
    K, C = 10, 4
    (tmp_path / "readout_dag.json").write_text(json.dumps({
        "version": 1, "C_fit": 4, "C": 4,
        "parent_int": {"1": [0], "2": [0], "3": [0]},
        "int2cid": {str(e): 1000 + e for e in range(4)},
        "name_by_id": {str(1000 + e): f"node{e}" for e in range(4)},
        "pos_sources": [[0], [1], [2], [3]]}))
    Vh = np.zeros((C, K)); Vh[1, 1] = 30.0; Vh[2, 4] = 30.0; Vh[2, 5] = 30.0
    np.savez(tmp_path / "readout_heads_gated_pc_stacked_mass99.npz", V=Vh,
             b_raw=np.array([5.0, -2.0, -2.0, -2.0]), degenerate=np.zeros(C, bool),
             const=np.zeros(C), theta_topm=0, C=C, K=K)
    (tmp_path / "bundle_meta.json").write_text(json.dumps({
        "int2cid": {str(e): 1000 + e for e in range(4)},
        "vocab_maps": [{str(500 + i): i for i in range(12)}],
        "name_by_id": {str(1000 + e): f"node{e}" for e in range(4)}}))
    rng = np.random.default_rng(0)

    def fake(df, members, n_domains, *, max_patients=600, seed=0, **kw):
        n = 120 if df == "test" else 100
        out = []
        for i in range(n):
            t = int(rng.random() < 0.5)
            bag = {11: 3.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0}
            bag.update({0: 2.0} if t == 0 else {2: 2.0, 3: 1.0})
            out.append(([t], {0: bag}, {0: float(len(bag))}))
        out.append(([0, 1], {0: {11: 1.0}}, {0: 1.0}))   # multi-positive: set aside
        return out, n + 1
    monkeypatch.setattr(gpr, "collect_set_patients", fake)
    # a concept NAMED like a member (code 7, in every record) is attestation (0141)
    (tmp_path / "concept_names.csv").write_text("concept_id,concept_name\n507,node1\n"
                                                "500,marker one\n")
    js = gpr.run_differential("test", "train", tmp_path, "^node1$|^node2$", adhoc=True,
                              tag="t", next_question=True, nq_pool=6)
    rep = (tmp_path / "differential_t.md").read_text()
    m = js[next(iter(js))]["metrics"]
    for k in ("E1 presence subtypes", "E2 stacked heads", "E3 multiclass head",
              "E4 gated likelihood"):
        assert k in m and k in rep
    assert m["E1 presence subtypes"]["top1"] > 0.9
    assert m["E4 gated likelihood"]["top1"] > 0.9
    assert m["E3 multiclass head"]["top1"] > 0.9
    assert js[next(iter(js))]["n_eval"] == 60
    assert "E1 reasons" in rep and "E3 reasons" in rep
    assert "### Next question" in rep and "| VOI (E3) | 3 |" in rep
    assert "| VOI (E1) | 3 |" in rep
    for k in ("E5 code head", "E6 θ + codes"):                 # exp 0148
        assert k in m and m[k]["top1"] > 0.9
    assert "| VOI (codes) | 3 |" in rep and "code differential:" in rep
    assert js[next(iter(js))]["nq_code_engine"]["chosen"] in ("E5 codes", "E6 θ + codes")
    assert js[next(iter(js))]["next_question"]["n"] == 60
    reasons = rep.split("E1 reasons")[1].split("###")[0]
    assert "marker one" in reasons and "· node1" not in reasons and ": node1" not in reasons
    # exp 0149: multi-positive patients as soft targets + cross-fit over both halves
    js2 = gpr.run_differential("test", "train", tmp_path, "^node1$|^node2$", adhoc=True,
                               tag="t2", multi=True, crossfit=True)
    g2 = js2[next(iter(js2))]
    assert g2["n_eval"] == 120 and g2["crossfit"] and len(g2["tau_e1"]) == 2
    assert g2["n_train_multi"] == 1 and "E6 θ + codes" in g2["metrics_multi"]
    assert g2["metrics"]["E3 multiclass head"]["top1"] > 0.9
    # exp 0150 survey mode: fast engines, framing 2 on E5, the survey table
    js3 = gpr.run_differential("test", "train", tmp_path, "^node1$|^node2$", adhoc=True, tag="t3",
                               next_question=True, nq_pool=6, fast=True, max_members=5,
                               crossfit=True)
    sv = (tmp_path / "differential_t3_survey.md").read_text()
    assert "differential survey" in sv and "groups scored" in sv
    g3 = js3[next(iter(js3))]
    assert "E4 gated likelihood" not in g3["metrics"] and "E6 θ + codes" in g3["metrics"]


def test_hide_split_and_question_values():
    rng = np.random.default_rng(0)
    fb = {0: {i: 1.0 for i in range(8)}, 1: {0: 2.0, 1: 1.0}}
    vis, hid = df.hide_split(fb, 0.2, rng)
    assert len(hid) == 2 and sum(len(b) for b in vis.values()) == 8
    assert df.hide_split({0: {1: 1.0}}, 0.2, rng) is None
    q0 = np.array([0.5, 0.5])
    qa = np.array([[0.5, 0.5], [0.9, 0.1], [0.9, 0.1]])
    v = df.question_values(q0, qa, [1.0, 1.0, 0.1])
    assert v[0] == 0.0 and v[1] > v[2] > 0.0


def _toy_engine_and_head():
    import model_acquisition as ma
    # K = 4 topics: 0, 1 background; 2 emits code 0 (member A); 3 emits code 1 (member B)
    lam = np.full((4, V), 0.5)
    lam[0, 3:] = lam[1, 3:] = 40.0
    lam[2, 0] = 200.0
    lam[3, 1] = 200.0
    heads = (np.zeros((3, 4)), np.zeros(3), np.zeros(3, bool), np.zeros(3), 0)
    eng = ma.ModelEngine([lam], np.full(4, 0.5), heads, {1: [0, 1], 2: [0, 2]}, [1, 2])
    rng = np.random.default_rng(0)
    pats = []
    for i in range(160):
        t = int(rng.random() < 0.5)
        bag = {int(x): 1.0 for x in rng.choice(np.arange(3, V), 5, replace=False)}
        bag[0 if t == 0 else 1] = 2.0
        pats.append(([t], {0: bag}))
    th = []
    for _, b in pats:
        g = eng.fold_in([eng.gidx(0, v) for v in b[0]], list(b[0].values()))
        th.append(g / g.sum())
    head = df.MulticlassHead(np.arange(4), 2).fit(np.stack(th[:100]),
                                                  [p[0][0] for p in pats[:100]])
    return eng, head, pats[100:], np.stack(th[100:])


def test_e3_reasons_point_at_the_marker_code():
    eng, head, pats, th = _toy_engine_and_head()
    truth, b = pats[0]
    idx = [eng.gidx(0, v) for v in b[0]]
    cnt = list(b[0].values())
    c = df.e3_reasons(head, th[0], eng.alpha.sum() + sum(cnt), eng.eb[:, idx], cnt,
                      truth[0])
    marker = 0 if truth[0] == 0 else 1
    assert list(b[0])[int(np.argmax(c))] == marker


def test_next_question_voi_finds_hidden_markers_more_than_random():
    eng, head, pats, _ = _toy_engine_and_head()
    pool = [(0, v) for v in range(V)]
    r = df.evaluate_next_question(eng, head, pats, pool, {}, np.random.default_rng(1),
                                  ks=(1, 3), frac=0.5)
    assert r["n"] == len(pats)
    assert 0.0 <= r["by"]["VOI (E3)@1"]["yield"] <= 1.0
    # revealing every hidden code restores the full-record call
    assert r["full"]["top1"] >= r["visible"]["top1"]
    assert r["by"]["random hidden@3"]["yield"] == 1.0
    assert sum(r["recommended"].values()) == 3 * len(pats)


def test_e1_candidate_scores_match_a_full_recompute():
    lams, subtypes, ev = _presence_world()
    eng = df.PresenceSubtypeEngine(lams, subtypes, 1, ev, [10, 20], {0: {9}})
    bags = {0: {5: 1.0, 6: 1.0}}
    lens = {0: 2.0}
    cand = [(0, 0), (0, 1), (0, 2)]
    S, P = eng.candidate_scores(bags, lens, cand)
    for c, (d, v) in enumerate(cand):
        b = {0: dict(bags[0])}
        b[0][v] = 1.0
        full = eng.scores(eng.subtype_ll(b, {0: 3.0}))
        assert np.allclose(S[c], full)
    assert np.all((P > 0) & (P < 1))
    # code 0 (A's aortic marker) is far more likely present under A than under B
    assert P[0, 0] > P[0, 1]


def test_next_question_with_the_e1_engine_reports_both_differentials():
    eng, head, pats, _ = _toy_engine_and_head()
    lams = [np.exp(np.log(eng.eb[:, :V]))]                     # any positive rows
    e1 = df.PresenceSubtypeEngine(lams, {1: [(2, 1.0)], 2: [(3, 1.0)]}, 2,
                                  np.ones(4), [1, 2], {})
    r = df.evaluate_next_question(eng, head, pats, [(0, v) for v in range(V)], {},
                                  np.random.default_rng(2), ks=(1,), frac=0.8, e1=e1,
                                  e1_lp=np.log([0.5, 0.5]), e1_tau=1.0)
    assert "VOI (E1)" in r["strategies"] and r["frac_hidden"] == 0.8
    x = r["by"]["VOI (E1)@1"]
    assert 0.0 <= x["p_true_e1"] <= 1.0 and 0.0 <= x["p_true"] <= 1.0


def _marker_bags(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        t = int(rng.random() < 0.5)
        bag = {int(x): 1.0 for x in rng.choice(np.arange(3, V), 5, replace=False)}
        bag[0 if t == 0 else 1] = 1.0
        out.append((t, {0: bag}))
    return out


def test_code_head_learns_markers_and_candidates_match_predict():
    tr = _marker_bags(200, 3)
    h = df.CodeHead(2, min_df=3).fit([b for _, b in tr], [t for t, _ in tr])
    te = _marker_bags(60, 4)
    P = h.predict([b for _, b in te])
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.mean(P.argmax(axis=1) == np.array([t for t, _ in te])) > 0.9
    # reasons: the marker code carries the call
    t, b = te[0]
    r = h.reasons(b, t)
    assert max(r, key=r.get) == (0, 0 if t == 0 else 1)
    # a candidate's differential == predict on the record with the code added
    base = {0: {5: 1.0, 6: 1.0}}
    cand = [(0, 0), (0, 1), (0, 7)]
    Q = h.candidates(base, cand)
    for i, (d, v) in enumerate(cand):
        bb = {0: dict(base[0])}
        bb[0][v] = 1.0
        assert np.allclose(Q[i], h.predict([bb])[0])
    py = h.p_yes(np.array([1.0, 0.0]), cand)
    assert py[0] > py[1]                                   # marker of class 0


def test_hybrid_code_head_and_next_question_with_codes():
    eng, head, pats, th = _toy_engine_and_head()
    tr = _marker_bags(200, 5)
    thr = []
    for _, b in tr:
        g = eng.fold_in([eng.gidx(0, v) for v in b[0]], list(b[0].values()))
        thr.append(g / g.sum())
    h6 = df.CodeHead(2, min_df=3, theta_topics=np.arange(4)).fit(
        [b for _, b in tr], [t for t, _ in tr], np.stack(thr))
    assert np.allclose(h6.predict([pats[0][1]], th[:1]).sum(), 1.0)
    for h in (df.CodeHead(2, min_df=3).fit([b for _, b in tr], [t for t, _ in tr]), h6):
        r = df.evaluate_next_question(eng, head, pats, [(0, v) for v in range(V)], {},
                                      np.random.default_rng(1), ks=(1, 3), frac=0.5,
                                      code_head=h)
        assert "VOI (codes)" in r["strategies"]
        x = r["by"]["VOI (codes)@1"]
        assert 0.0 <= x["p_true_c"] <= 1.0 and 0.0 <= x["yield"] <= 1.0
        assert r["full"]["top1_c"] >= r["visible"]["top1_c"] - 1e-9
        assert sum(r["recommended_codes"].values()) == 3 * r["n"]


def test_soft_targets_spread_weight_over_true_members():
    Y = df.label_matrix([[0, 2], [1]], 3)
    assert np.allclose(Y, [[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
    assert np.allclose(df.label_matrix([2, 0], 3), [[0, 0, 1], [1, 0, 0]])


def test_next_question_codes_only():
    tr = _marker_bags(200, 7)
    h = df.CodeHead(2, min_df=3).fit([b for _, b in tr], [t for t, _ in tr])
    pats = [([t], b) for t, b in _marker_bags(60, 8)]
    r = df.evaluate_next_question_codes(h, pats, [(0, v) for v in range(V)],
                                        np.random.default_rng(0), ks=(1, 3), frac=0.5)
    assert r["n"] == 60 and r["by"]["random hidden@3"]["yield"] == 1.0
    assert r["full"]["top1"] >= r["visible"]["top1"]
    assert r["by"]["VOI (codes)@3"]["p_true"] >= r["by"]["random@3"]["p_true"]


def test_most_specific_members_drops_ancestors():
    import gated_pc_readout as gpr
    pi = {"1": [0], "2": [1], "3": [1], "4": [0]}
    assert gpr.most_specific_members([1, 2, 3, 4], pi) == [2, 3, 4]
    assert gpr.most_specific_members([2, 4], pi) == [2, 4]


def test_top_members_rolls_up_subtypes_and_drops_the_group():
    import gated_pc_readout as gpr
    # group 5 -> {1 (hep C), 4 (hep A)}; 1 -> {2 (chronic), 3 (acute)}
    pi = {"1": [5], "2": [1], "3": [1], "4": [5], "5": [0]}
    assert gpr.top_members([1, 2, 3, 4, 5], pi, group=5) == [1, 4]
    assert gpr.top_members([2, 3, 4], pi, group=5) == [2, 3, 4]
