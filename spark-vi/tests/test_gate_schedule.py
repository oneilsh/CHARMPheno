"""Leaves-first gate schedule (GatedOnlineLDA.set_gate_schedule; report
2026-10-19-leaves-first-gate-sim): heights, the capped gate, frozen/opened/seeded
blocks, multi-domain lambda, the driver-only seed, and the seed-count pass."""
import pickle

import numpy as np
import pytest

from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.gated_lda import GatedOnlineLDA
from spark_vi.models.topic.types import GatedBOWDocument

# diamond: 1 and 2 under root; 3 under both 1 and 2; 4 under 1
PARENT = {1: 0, 2: 0, 3: [1, 2], 4: 1}


def _docs(V=12, n=60, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        f = [frozenset(), frozenset({3}), frozenset({4}), frozenset({2})][i % 4]
        idx = np.unique(rng.integers(0, V, size=8)).astype(np.int32)
        out.append(GatedBOWDocument(idx, np.ones(len(idx)), len(idx), frontier=f))
    return out


def test_height_is_longest_path_down_on_a_multi_parent_dag():
    lay = DagLayout(PARENT, n_bg=1, tpn=1)
    assert {u: lay.height(u) for u in lay.nodes} == {1: 1, 2: 1, 3: 0, 4: 0}


def test_allowed_set_height_cap():
    lay = DagLayout(PARENT, n_bg=1, tpn=1)
    full = lay.allowed_set(frozenset({3}))
    assert list(full) == sorted([0] + lay.block[1] + lay.block[2] + lay.block[3])
    assert list(lay.allowed_set(frozenset({3}), max_height=0)) == sorted([0] + lay.block[3])
    assert list(lay.allowed_set(frozenset({3}), max_height=None)) == list(full)


def _fit(m, docs, iters, lr=0.5):
    gp = m.initialize_global(None)
    hist = [gp]
    for _ in range(iters):
        gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=lr)
        hist.append(gp)
    return hist


def test_warm_zero_is_the_production_gate():
    lay = DagLayout(PARENT, n_bg=1, tpn=1)
    m = GatedOnlineLDA(lay, 12, alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global(None)
    assert "gate_iter" not in gp
    m.set_gate_schedule(0)
    assert "gate_iter" not in m.initialize_global(None)


def test_schedule_freezes_then_opens_and_seeds():
    lay = DagLayout(PARENT, n_bg=1, tpn=1)
    m = GatedOnlineLDA(lay, 12, alpha=0.1, eta=0.02, random_seed=0)
    seed = np.zeros((lay.K, 12))
    seed[lay.block[1][0], :3] = 1.0                    # node 1's seed: words 0..2
    m.set_gate_schedule(2, open_seed=seed)
    hist = _fit(m, _docs(), 5)
    k1, k3 = lay.block[1][0], lay.block[3][0]
    lam0 = hist[0]["lambda"]
    # iters 0,1: node 1 (height 1) closed -> frozen; leaf 3 learns
    assert np.array_equal(hist[1]["lambda"][k1], lam0[k1])
    assert not np.array_equal(hist[1]["lambda"][k3], lam0[k3])
    # opened at iteration 2 with the seed: its mass sits on words 0..2
    assert int(hist[2]["gate_open_at"][k1]) == 2
    row = hist[2]["lambda"][k1]
    assert row[:3].sum() > 0.9 * row.sum()
    assert [int(h["gate_iter"]) for h in hist] == [0, 1, 2, 3, 4, 5]


def test_schedule_multidomain_seeds_each_domain():
    lay = DagLayout(PARENT, n_bg=1, tpn=1)
    m = GatedOnlineLDA(lay, 12, domains=[8, 4], alpha=0.1, eta=0.02, random_seed=0)
    seed = np.zeros((lay.K, 12))
    seed[lay.block[2][0], 1] = 5.0                     # domain 0, local word 1
    seed[lay.block[2][0], 9] = 5.0                     # domain 1, local word 1
    m.set_gate_schedule(1, open_seed=seed)
    hist = _fit(m, _docs(), 2)
    k2 = lay.block[2][0]
    for d in (0, 1):
        row = hist[1]["lambda"][d][k2]
        assert np.argmax(row) == 1 and row[1] > 0.8 * row.sum()


def test_seed_is_driver_only():
    lay = DagLayout(PARENT, n_bg=1, tpn=1)
    m = GatedOnlineLDA(lay, 12, alpha=0.1, eta=0.02, random_seed=0)
    m.set_gate_schedule(2, open_seed=np.ones((lay.K, 12)))
    clone = pickle.loads(pickle.dumps(m))
    assert clone._open_seed is None and m._open_seed is not None
    assert clone.gate_warm == 2


@pytest.mark.slow
def test_gate_open_seed_counts_sums_admitted_documents(spark):
    from spark_vi.models.topic.gated_init import gate_open_seed_counts
    lay = DagLayout(PARENT, n_bg=1, tpn=1)
    docs = [GatedBOWDocument(np.array([0, 5], np.int32), np.array([2.0, 1.0]), 3,
                             frontier=frozenset({3})),
            GatedBOWDocument(np.array([1], np.int32), np.array([4.0]), 4,
                             frontier=frozenset({4})),
            GatedBOWDocument(np.array([2], np.int32), np.array([7.0]), 7,
                             frontier=frozenset())]
    out = gate_open_seed_counts(spark.sparkContext.parallelize(docs, 2), lay, 8)
    k1, k2, k3 = lay.block[1][0], lay.block[2][0], lay.block[3][0]
    # node 1 admits docs at 3 and 4; node 2 admits the doc at 3 (multi-parent child)
    assert out[k1].tolist() == [2.0, 4.0, 0, 0, 0, 1.0, 0, 0]
    assert out[k2].tolist() == [2.0, 0, 0, 0, 0, 1.0, 0, 0]
    assert not out[k3].any() and not out[0].any()     # leaves, background: no seed


def test_init_seed_replaces_only_the_domain_it_covers():
    lay = DagLayout(PARENT, n_bg=1, tpn=1)
    m = GatedOnlineLDA(lay, 12, domains=[8, 4], alpha=0.1, eta=0.02, random_seed=0)
    init = np.zeros((lay.K, 12))
    k3 = lay.block[3][0]                               # leaf, open at iteration 0
    init[k3, 2] = 1.0                                  # condition domain only
    m.set_gate_schedule(2, init_seed=init)
    base = GatedOnlineLDA(lay, 12, domains=[8, 4], alpha=0.1, eta=0.02, random_seed=0)
    base.set_gate_schedule(2)
    np.random.seed(0); gp = m.initialize_global(None)
    np.random.seed(0); gb = base.initialize_global(None)
    row = gp["lambda"][0][k3]
    assert np.argmax(row) == 2 and row[2] > 0.9 * row.sum()
    assert np.array_equal(gp["lambda"][1][k3], gb["lambda"][1][k3])   # other domain kept
    k1 = lay.block[1][0]                               # closed parent: untouched
    assert np.array_equal(gp["lambda"][0][k1], gb["lambda"][0][k1])
