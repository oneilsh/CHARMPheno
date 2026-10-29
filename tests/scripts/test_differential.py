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


def test_metrics_are_pooled_over_patients():
    posts = [np.array([0.9, 0.1]), np.array([0.6, 0.4]), np.array([0.2, 0.8])]
    m = df.metrics(posts, [[0], [1], [1]])
    assert m["top1"] == pytest.approx(2 / 3) and m["top3"] == 1.0
    assert m["n"] == 3


def test_pool_reasons_suppresses_small_members():
    per = [(1, {"a": 1.0, "b": 0.5})] * 25 + [(2, {"c": 1.0})] * 5
    out = df.pool_reasons(per)
    assert out == {1: ["a", "b"]}


@pytest.mark.slow
def test_run_differential_end_to_end_on_a_tiny_run(tmp_path, monkeypatch):
    import json
    sys.path.insert(0, str(REPO_ROOT / "tests" / "scripts"))
    sys.path.insert(0, str(REPO_ROOT / "charmpheno"))
    from test_inspect_topics import _make_census_run
    import framings as gpr
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
                              tag="t")
    rep = (tmp_path / "differential_t.md").read_text()
    g = js[next(iter(js))]
    m = g["metrics"]
    for k in ("E2 stacked heads", "E3 multiclass head", "E5 code head", "E6 θ + codes"):
        assert k in m and k in rep
    for k in ("E3 multiclass head", "E5 code head", "E6 θ + codes"):
        assert m[k]["top1"] > 0.9
    assert g["n_eval"] == 60
    assert "E3 reasons" in rep and "E5 reasons" in rep
    reasons = rep.split("E5 reasons")[1].split("###")[0]
    assert "marker one" in reasons and "· node1" not in reasons and ": node1" not in reasons
    # cross-fit over both halves, and the survey table
    js2 = gpr.run_differential("test", "train", tmp_path, "^node1$|^node2$", adhoc=True,
                               tag="t2", crossfit=True, max_members=5)
    g2 = js2[next(iter(js2))]
    assert g2["n_eval"] == 120 and g2["crossfit"]
    assert g2["metrics"]["E3 multiclass head"]["top1"] > 0.9
    sv = (tmp_path / "differential_t2_survey.md").read_text()
    assert "differential survey" in sv and "groups scored" in sv


def _toy_engine_and_head():
    from model_engine import ModelEngine
    # K = 4 topics: 0, 1 background; 2 emits code 0 (member A); 3 emits code 1 (member B)
    lam = np.full((4, V), 0.5)
    lam[0, 3:] = lam[1, 3:] = 40.0
    lam[2, 0] = 200.0
    lam[3, 1] = 200.0
    heads = (np.zeros((3, 4)), np.zeros(3), np.zeros(3, bool), np.zeros(3), 0)
    eng = ModelEngine([lam], np.full(4, 0.5), heads, {1: [0, 1], 2: [0, 2]}, [1, 2])
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


@pytest.mark.slow
def test_hybrid_code_head_reads_codes_and_theta():
    eng, head, pats, th = _toy_engine_and_head()
    tr = _marker_bags(200, 5)
    thr = []
    for _, b in tr:
        g = eng.fold_in([eng.gidx(0, v) for v in b[0]], list(b[0].values()))
        thr.append(g / g.sum())
    h6 = df.CodeHead(2, min_df=3, theta_topics=np.arange(4)).fit(
        [b for _, b in tr], [t for t, _ in tr], np.stack(thr))
    assert np.allclose(h6.predict([pats[0][1]], th[:1]).sum(), 1.0)
    assert np.mean(h6.predict([b for _, b in pats], th).argmax(axis=1)
                   == np.array([t[0] for t, _ in pats])) > 0.9


def test_soft_targets_spread_weight_over_true_members():
    Y = df.label_matrix([[0, 2], [1]], 3)
    assert np.allclose(Y, [[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
    assert np.allclose(df.label_matrix([2, 0], 3), [[0, 0, 1], [1, 0, 0]])


def test_most_specific_members_drops_ancestors():
    import framings as gpr
    pi = {"1": [0], "2": [1], "3": [1], "4": [0]}
    assert gpr.most_specific_members([1, 2, 3, 4], pi) == [2, 3, 4]
    assert gpr.most_specific_members([2, 4], pi) == [2, 4]


def test_top_members_rolls_up_subtypes_and_drops_the_group():
    import framings as gpr
    # group 5 -> {1 (hep C), 4 (hep A)}; 1 -> {2 (chronic), 3 (acute)}
    pi = {"1": [5], "2": [1], "3": [1], "4": [5], "5": [0]}
    assert gpr.top_members([1, 2, 3, 4, 5], pi, group=5) == [1, 4]
    assert gpr.top_members([2, 3, 4], pi, group=5) == [2, 3, 4]


def _shared_and_marker_bags(n, seed):
    """Two members. Codes 3..5 are on nearly everyone (frequent, uninformative); code 1
    marks member 1 on 40% of its patients; codes 6..V are noise."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        t = int(rng.random() < 0.5)
        bag = {v: 1.0 for v in (3, 4, 5) if rng.random() < 0.95}
        noise = rng.choice(np.arange(6, V), 2, replace=False)
        if t == 1 and rng.random() < 0.4:
            noise[0] = 1                    # the marker REPLACES a noise code, so the
        bag.update({int(x): 1.0 for x in noise})    # distinct-count feature can't see it
        out.append((t, {0: bag}))
    return out


def test_relevance_selection_keeps_the_marker_over_shared_codes():
    tr = _shared_and_marker_bags(300, 11)
    bags, y = [b for _, b in tr], [t for t, _ in tr]
    dfh = df.CodeHead(2, min_df=3, max_codes=3).fit(bags, y)
    rel = df.CodeHead(2, min_df=3, max_codes=3, select="relevance",
                      rel_lambda=0.3).fit(bags, y)
    assert dfh.codes == [(0, 3), (0, 4), (0, 5)]           # frequency: the shared codes
    assert (0, 1) in rel.codes                              # relevance: the marker
    # Within a group lift is capped near 1 / P(m) = 2, so the readout's λ = 0.6 is
    # already frequency-dominated (threshold ~0.45 for this marker), as is λ = 1.
    for lam in (0.6, 1.0):
        h = df.CodeHead(2, min_df=3, max_codes=3, select="relevance",
                        rel_lambda=lam).fit(bags, y)
        assert (0, 1) not in h.codes
    te = _shared_and_marker_bags(200, 12)
    acc = lambda h: np.mean(h.predict([b for _, b in te]).argmax(axis=1)
                            == np.array([t for t, _ in te]))
    assert acc(rel) > acc(dfh) + 0.05


def test_relevance_selection_support_is_within_member_and_round_robin():
    # code 1: on 4 of member 1's patients only (group df 4 >= 3, member support 4)
    # code 2: on 2 of member 0's patients only (member support 2 < min_df 3 -> dropped)
    bags = [{0: {9: 1.0}} for _ in range(40)]
    y = [0] * 20 + [1] * 20
    for i in (20, 21, 22, 23):
        bags[i][0][1] = 1.0
    for i in (0, 1):
        bags[i][0][2] = 1.0
    for i in range(10):                                     # code 3: member 0's marker
        bags[i][0][3] = 1.0
    h = df.CodeHead(2, min_df=3, select="relevance").fit(bags, y)
    assert (0, 2) not in h.codes
    assert {(0, 1), (0, 3)} <= set(h.codes)
    # round robin: with room for two codes each member gets its best one
    h2 = df.CodeHead(2, min_df=3, max_codes=2, select="relevance", rel_lambda=0.0).fit(
        bags, y)
    assert set(h2.codes) == {(0, 1), (0, 3)}
    with pytest.raises(ValueError):
        df.CodeHead(2, select="lift")


# --- framing 2 as workup (exp 0167) -------------------------------------------

def test_workup_answers_categories_and_filters():
    toks = [(0, 7), (0, 8), (2, 5000 * 100 + 2), (2, 5000 * 100 + 3), (2, 6000 * 100),
            (2, 7000 * 100 + 11), (1, 9)]
    a = df.workup_answers(toks, meas_dom=2, excl={0: {8}})
    assert a == {("c", 0, 7): 1, ("m", 5000): df.WK_HIGH, ("m", 6000): df.WK_DONE,
                 ("m", 7000): df.WK_HIGH, ("c", 1, 9): 1}
    b = df.workup_answers(toks, meas_dom=2, drop_done=True,
                          vocab_sets={0: {7}, 1: set(), 2: {500002, 500003, 600000}})
    assert b == {("c", 0, 7): 1, ("m", 5000): df.WK_HIGH}


def _workup_toy(n, seed):
    """Two members. Test 100 is HIGH for member 1 (70%) vs 10% for member 0; code 5 is
    recorded for 50% of everyone (uninformative); code 6 for 30% of member 0 only."""
    rng = np.random.default_rng(seed)
    ans, y = [], []
    for _ in range(n):
        m = int(rng.random() < 0.5)
        a = {}
        if rng.random() < (0.7 if m else 0.1):
            a[("m", 100)] = df.WK_HIGH
        elif rng.random() < 0.5:
            a[("m", 100)] = df.WK_NORMAL
        if rng.random() < 0.5:
            a[("c", 0, 5)] = 1
        if m == 0 and rng.random() < 0.3:
            a[("c", 0, 6)] = 1
        ans.append(a)
        y.append(m)
    return ans, np.array(y)


def test_workup_eig_ranks_the_separating_test_and_is_calibrated():
    tr, ytr = _workup_toy(4000, 0)
    cands = df.workup_candidates(tr)
    assert [k for k, _ in cands] == [("c", 0, 5), ("c", 0, 6), ("m", 100)]
    L = df.answer_likelihoods(tr, np.eye(2)[ytr], cands)
    np.testing.assert_allclose(L.sum(axis=1), 1.0)
    q0 = np.array([0.5, 0.5])
    post, pa = df.workup_posteriors(q0, L)
    eig = df.workup_eig(q0, post, pa)
    assert np.argmax(eig) == 2 and eig[0] < 1e-3            # test > code 6 > code 5
    te, yte = _workup_toy(1500, 1)
    A = np.stack([df.answer_vector(a, cands) for a in te])
    q0s = np.tile(q0, (len(te), 1))
    tau = df.fit_workup_tau(q0s, yte, A, L)
    assert 0.7 < tau < 1.4                                   # independent answers
    r = df.evaluate_workup(q0s, yte, A, L, np.array([0.5, 0.5]), tau)
    assert r["R2"]["top1"] > r["R2"]["random"] > 0           # the test pays most
    assert r["R2"]["oracle"] >= r["R2"]["top1"]
    assert 0.7 < r["R3"]["slope"] < 1.3                      # calibrated EIG
    assert r["R1_spearman_quartiles"][1] > 0
    assert r["facts"]["n_facts"] == 3
    assert r["facts"]["spearman_mean_eig_vs_mean_gain"] > 0.9   # the test ranks first
    assert r["facts"]["top5_mean_gain"] >= r["facts"]["all_mean_gain"] - 1e-12
    cov = df.workup_coverage([10, 20, None, 40], A[:4])
    assert cov["window_days_quartiles"][1] == 20.0


def test_workup_block_on_synthetic_rows():
    """`dag_gated_readout._workup_block`: windows -> answers -> candidates ->
    likelihoods -> per-engine R1-R4 over cross-fit halves, coverage, top facts.
    Measurement token 900*100+3 (test 900 high) marks member 1; condition code 501 is
    attestation (excluded); 502 is noise; prevalent rows (no window) are skipped."""
    from types import SimpleNamespace
    import framings as gpr

    rng = np.random.default_rng(0)
    inv = [{0: 501, 1: 502}, {}, {0: 90003, 1: 90002, 2: 90000}]
    ctx = SimpleNamespace(inv_maps=inv,
                          code_name=lambda d, v: {(0, 1): "noise", (2, 0): "T900 [high]"}
                          .get((d, v), f"{d}:{v}"))
    wk = {"tok2idx": [{t: i for i, t in m.items()} for m in inv],
          "vocab_sets": {d: set(m.values()) for d, m in enumerate(inv)}, "meas_dom": 2}

    def rows(n):
        out = []
        for i in range(n):
            m = int(rng.random() < 0.5)
            toks = [{"d": 0, "t": 501}]
            if rng.random() < (0.8 if m else 0.1):
                toks.append({"d": 2, "t": 90003})
            elif rng.random() < 0.5:
                toks.append({"d": 2, "t": 90002})
            if rng.random() < 0.5:
                toks.append({"d": 0, "t": 502})
            ex = None if i % 10 == 9 else {"__wk": toks, "__wdays": 30 + i % 50}
            # pre-index record: half the patients already carry the noise code 502
            # (vocab idx 1 in dom 0), so 502 is new for only the other half
            out.append(([m], {0: ({1: 1.0} if i % 2 else {})}, None, None, None, ex))
        return out
    tr, te = rows(600), rows(300)
    ytr = [r[0][0] for r in tr]
    truths = [r[0] for r in te]
    halves = [(list(range(0, 300, 2)), list(range(1, 300, 2)))]
    halves.append((halves[0][1], halves[0][0]))
    q = np.tile([0.5, 0.5], (300, 1))
    w = gpr._workup_block(df, ctx, wk, tr, te, ytr, [10, 11], {0: {0}}, halves, truths,
                          {"E6": q, "E3": q}, np.array([0.5, 0.5]))
    assert w["n_test_windows"] == 270 and w["n_candidates"] == 2   # 501 excluded
    assert w["coverage"]["eligible_median"] in (1.0, 1.5, 2.0)     # 502 new for half
    wb = gpr._workup_block(df, ctx, wk, tr, te, ytr, [10, 11], {0: {0}}, halves,
                           truths, {"E6": q}, np.array([0.5, 0.5]),
                           length_edges=(40,))
    assert len(wb["train_windows_per_bin"]) == 2 and wb["E6"]["n"] == 270
    assert wb["E6"]["R2"]["top1"] > wb["E6"]["R2"]["random"]
    assert w["E6"]["n"] == 270 and len(w["E6"]["tau"]) == 2
    assert w["E6"]["R2"]["top1"] > w["E6"]["R2"]["random"]
    assert w["top_facts"][0] == "test T900"
    text = "\n".join(gpr._format_workup(w, "strict"))
    assert "| E6 | 270 |" in text and "test T900" in text
    assert w["diagnostics"]["engine"] == "E6" and "diagnostic (E6)" in text
    assert len(w["diagnostics"]["by_activity"]) == 3


def test_known_facts_and_new_only_likelihoods():
    inv = [{0: 501, 1: 502}, {}, {0: 90003, 1: 90002}]
    kn = df.known_facts({0: {1: 1.0}, 2: {1: 1.0}}, inv, meas_dom=2)
    assert kn == {("c", 0, 502), ("m", 900)}
    cands = [(("c", 0, 502), 2), (("m", 900), df.WK_MAX_A)]
    assert list(df.eligible_vector(kn, cands)) == [False, False]
    # two training patients of member 0: one already knew 502 (and has it again),
    # the other is new to it and does not have it -> P(recorded | m0, new) is low
    ans = [{("c", 0, 502): 1}, {}]
    L = df.answer_likelihoods(ans, np.eye(2)[[0, 0]], cands[:1],
                              known=[{("c", 0, 502)}, set()])
    assert L[0, 1, 0] == pytest.approx(0.5 / 2.0)        # (0 + .5) / (1 + 1)
    assert df.workup_candidates(ans * 30, known=[{("c", 0, 502)}, set()] * 30) == []
    assert df.length_bin(10, (30, 120)) == 0 and df.length_bin(30, (30, 120)) == 1
    assert df.length_bin(500, (30, 120)) == 2


def test_workup_diagnostics_find_the_activity_confound():
    """Member 0 gets more care: every one of 20 codes is recorded more often for it,
    scaled by a per-patient activity level. Absences are then correlated within a
    patient, so a low-activity member-0 patient's high-EIG facts all point to member
    1: R1 is lowest in the low-activity tercile. No train/test shift: the log-
    likelihoods agree."""
    rng = np.random.default_rng(0)

    def draw(n):
        ans, ys, act = [], [], []
        for _ in range(n):
            m = int(rng.random() < 0.5)
            lvl = rng.choice([0.2, 1.0, 1.8])
            base = (0.45 if m == 0 else 0.25) * lvl
            a = {("c", 0, v): 1 for v in range(20)
                 if rng.random() < min(base * (1 + v / 20), 0.95)}
            ans.append(a); ys.append(m); act.append(len(a))
        return ans, np.array(ys), act
    tr, ytr, _ = draw(3000)
    cands = df.workup_candidates(tr)
    L = df.answer_likelihoods(tr, np.eye(2)[ytr], cands)
    te, yte, act = draw(900)
    A = [df.answer_vector(a, cands) for a in te]
    q0 = [np.array([0.5, 0.5])] * len(te)
    d = df.workup_diagnostics(q0, yte, A, L, 1.0, activity=act,
                              train=([df.answer_vector(a, cands) for a in tr], ytr,
                                     [np.ones(len(cands), bool)] * len(tr)))
    assert abs(d["ll_test"] - d["ll_train"]) < 0.02
    assert abs(d["pred_nondefault"] - d["obs_nondefault"]) < 0.02
    terc = {b["tercile"]: b for b in d["by_activity"]}
    assert terc["low"]["R1_median"] < terc["high"]["R1_median"]
    assert d["default_pairs"]["n"] + d["nondefault_pairs"]["n"] == len(te) * len(cands)


def test_fact_level_ranking_survives_the_per_patient_sign_artifact():
    """The activity-correlated toy: per-patient R1 is near zero, but pooled over
    patients each fact's mean realized gain ranks with its mean EIG."""
    rng = np.random.default_rng(0)

    def draw(n):
        ans, ys = [], []
        for _ in range(n):
            m = int(rng.random() < 0.5)
            lvl = rng.choice([0.2, 1.0, 1.8])
            base = (0.45 if m == 0 else 0.25) * lvl
            ans.append({("c", 0, v): 1 for v in range(20)
                        if rng.random() < min(base * (1 + v / 20), 0.95)})
            ys.append(m)
        return ans, np.array(ys)
    tr, ytr = draw(3000)
    cands = df.workup_candidates(tr)
    L = df.answer_likelihoods(tr, np.eye(2)[ytr], cands)
    te, yte = draw(2000)
    A = [df.answer_vector(a, cands) for a in te]
    r = df.evaluate_workup([np.array([0.5, 0.5])] * len(te), yte, A, L,
                           np.array([0.5, 0.5]), 1.0)
    assert abs(r["R1_spearman_quartiles"][1]) < 0.15
    # pooled per fact the gains are EIG in level (calibrated); the 20 facts' EIGs
    # are close together, so their order is only moderately recovered at n = 2000
    assert 0.85 < r["facts"]["slope"] < 1.15
    assert r["facts"]["spearman_mean_eig_vs_mean_gain"] > 0.2
