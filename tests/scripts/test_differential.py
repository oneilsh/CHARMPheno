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
            bag = {11: 3.0, 7: 1.0}
            bag.update({0: 2.0} if t == 0 else {2: 2.0, 3: 1.0})
            out.append(([t], {0: bag}, {0: float(len(bag))}))
        out.append(([0, 1], {0: {11: 1.0}}, {0: 1.0}))   # multi-positive: set aside
        return out, n + 1
    monkeypatch.setattr(gpr, "collect_set_patients", fake)
    js = gpr.run_differential("test", "train", tmp_path, "^node1$|^node2$", adhoc=True,
                              tag="t")
    rep = (tmp_path / "differential_t.md").read_text()
    m = js[next(iter(js))]["metrics"]
    for k in ("E1 presence subtypes", "E2 stacked heads", "E3 multiclass head",
              "E4 gated likelihood"):
        assert k in m and k in rep
    assert m["E1 presence subtypes"]["top1"] > 0.9
    assert m["E4 gated likelihood"]["top1"] > 0.9
    assert m["E3 multiclass head"]["top1"] > 0.9
    assert js[next(iter(js))]["n_eval"] == 60
    assert "E1 reasons" in rep
