"""WP-D part 3: simulated acquisition on held-out patients (voi_acquisition.py) and
the driver's patient collect (gated_pc_readout.collect_set_patients)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "analysis" / "cloud"), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(REPO_ROOT / "analysis" / "cloud"),
                str(REPO_ROOT / "charmpheno"), str(REPO_ROOT / "spark-vi"),
                os.environ.get("PYTHONPATH", "")) if p)

import voi_acquisition as va  # noqa: E402

V = 30


def _world(rng, n=200):
    """Two diseases; code 0 marks A, code 1 marks B, codes 2.. are shared noise.
    Patients are drawn from the presence model itself, so the background keeps each
    separator present in ~64% of the OTHER disease's records too (absence is weak)."""
    bg = np.full(V, 1.0 / V)
    a = np.full(V, 0.2 / (V - 2)); a[0] = 0.8
    b = np.full(V, 0.2 / (V - 2)); b[1] = 0.8
    prof = np.stack([a, b])
    prior = np.array([0.7, 0.3])
    pool, mix = va.build_pool(prior, {0: prof}, {0: bg}, {}, pool=V)
    pats = []
    for i in range(n):
        t = 0 if rng.random() < 0.7 else 1
        L = 40.0
        p = 1 - np.exp(-L * (0.3 * prof[t] + 0.7 * bg))
        bag = set(np.flatnonzero(rng.random(V) < p).tolist())
        pats.append(([t], {0: bag}, {0: L}))
    return prior, pool, mix, pats


def test_pool_puts_the_separators_first_and_honours_exclusions():
    rng = np.random.default_rng(0)
    prior, pool, mix, _ = _world(rng)
    assert {pool[0][1], pool[1][1]} == {0, 1}
    bg = np.full(V, 1.0 / V)
    pool2, _ = va.build_pool(prior, {0: np.stack([bg, bg])}, {0: bg}, {0: {0, 1, 2}},
                             pool=V)
    assert not {0, 1, 2} & {v for _, v in pool2}


def test_greedy_beats_random_early_and_full_is_the_ceiling():
    rng = np.random.default_rng(1)
    prior, pool, mix, pats = _world(rng)
    res = va.simulate_set(prior, pool, mix, pats, steps=5, seed=3)
    g = res["strategies"]["greedy"]["by_step"]
    r = res["strategies"]["random"]["by_step"]
    assert g[0]["top1"] == pytest.approx(r[0]["top1"])          # same prior
    assert g[2]["p_true"] > r[2]["p_true"]
    assert g[2]["top1"] > g[0]["top1"] + 0.05                   # beats the majority
    assert res["full"]["top1"] >= g[5]["top1"] - 0.05
    assert 0.0 <= res["full"]["ece"] <= 1.0


def test_an_empty_domain_record_is_uninformative():
    prior = np.array([0.5, 0.5])
    P = np.zeros((2, 3))
    traj, hits = va.simulate_patient(prior, P, np.zeros(3, bool), strategy="greedy",
                                     steps=3)
    assert np.allclose(traj[-1], prior) and not hits.any()


def test_format_suppresses_under_the_egress_floor():
    rng = np.random.default_rng(2)
    prior, pool, mix, pats = _world(rng, n=10)
    res = va.simulate_set(prior, pool, mix, pats, steps=3)
    assert "suppressed" in va.format_acquisition("x", ["a", "b"], prior, res)
    prior, pool, mix, pats = _world(rng, n=40)
    txt = va.format_acquisition("x", ["a", "b"], prior,
                                va.simulate_set(prior, pool, mix, pats, steps=3))
    assert "| greedy | 3 |" in txt and "full pool" in txt


@pytest.mark.slow
def test_collect_set_patients_filters_members_and_reads_sparse_bags(spark):
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import Row
    from gated_pc_readout import collect_set_patients
    rows = [Row(label=[1.0, 1.0, 0.0, 0.0], features_0=Vectors.sparse(5, [0, 3], [2.0, 1.0]),
                features_1=Vectors.sparse(4, [1], [5.0])),
            Row(label=[1.0, 0.0, 0.0, 1.0], features_0=Vectors.sparse(5, [4], [1.0]),
                features_1=Vectors.sparse(4, [], [])),
            Row(label=[1.0, 0.0, 1.0, 0.0], features_0=Vectors.sparse(5, [2], [3.0]),
                features_1=Vectors.sparse(4, [0, 2], [1.0, 1.0]))]
    df = spark.createDataFrame(rows)
    pats, n = collect_set_patients(df, [1, 3], 2)
    assert n == 2
    got = sorted((t, sorted(b[0]), l[0], l[1]) for t, b, l in pats)
    assert got == [([0], [0, 3], 2.0, 1.0), ([1], [4], 1.0, 0.0)]       # distinct codes
    pats, _ = collect_set_patients(df, [1, 3], 2, length="tokens")
    got = sorted((t, l[0], l[1]) for t, b, l in pats)
    assert got == [([0], 3.0, 5.0), ([1], 1.0, 0.0)]


def test_auto_tau_fits_on_one_half_scores_the_other_and_tempers_overconfidence():
    """Naive Bayes double-counts correlated codes: here codes 0-9 are ONE finding
    copied ten times (all present or all absent together, and weakly informative),
    so tau = 1 counts it ten times over and the fitted temper must come down. Codes
    10-19 are the mirror profile, never observed (absent for both classes alike)."""
    rng = np.random.default_rng(4)
    bg = np.full(V, 1.0 / V)
    a = np.full(V, 1.0 / V); a[:10] *= 1.6; a[10:20] *= 0.4
    b = np.full(V, 1.0 / V); b[:10] *= 0.6; b[10:20] *= 1.4    # both sum to 1
    prof = np.stack([a, b])
    prior = np.array([0.5, 0.5])
    pool, mix = va.build_pool(prior, {0: prof}, {0: bg}, {}, pool=V)
    L = 40.0
    pats = []
    for i in range(400):
        t = int(rng.random() < 0.5)
        p0 = 1 - np.exp(-L * (0.3 * prof[t][0] + 0.7 * bg[0]))
        hit = rng.random() < p0
        bag = set(range(10)) if hit else set()
        bag |= set(range(20, V))                    # shared codes: always present
        pats.append(([t], {0: bag}, {0: L}))
    raw = va.simulate_set(prior, pool, mix, pats, steps=3, tau=1.0)
    auto = va.simulate_set(prior, pool, mix, pats, steps=3, tau="auto")
    assert auto["n"] == 200 and auto["tau_fit"]["fit_n"] == 200
    assert auto["tau"] <= 0.2
    assert auto["full"]["ece"] < raw["full"]["ece"]
    assert "tau fit on the other 200" in va.format_acquisition("x", ["a", "b"], prior, auto)
