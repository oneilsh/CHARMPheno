"""Sparse per-topic eta boost (profile-eta word-side prior) — WP-1 tests.

Covers the four pre-registered obligations of the 2026-09-06 profile-eta-prior
plan's WP-1, at every eta site the WP-0 scout found (the micro-design comment
above `_resolve_eta_boost` in gated_lda.py):

  1. a boost TILTS a zero-count (starved-floor) topic's E[log beta] toward the
     boosted vocab indices, because the prior re-enters every lambda update;
  2. an absent boost (None / {}) is BYTE-IDENTICAL to the un-boosted model
     (in-process here; additionally verified out-of-band against a fit captured
     on the pre-change tree — same docs/seed, bit-identical lambda/alpha/ELBO);
  3. the lambda update target equals eta_vec + counts (hand-computable fixture,
     learning_rate=1.0 makes the target the update);
  4. non-boosted topics and non-condition domains are bit-identical to the
     scalar-eta values.

Plus: the eta-dependent ELBO term against a hand-computed per-topic prior, the
fail-fast validation guards, the ADR-0047 pickle exclusion (the boost must not
ride a task closure), the PC delegation path, and the estimator-shim JSON
passthrough (the WP-3 seam).
"""
import pickle

import numpy as np
import pytest

from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.gated_lda import GatedOnlineLDA, _resolve_eta_boost
from spark_vi.models.topic.lda import _dirichlet_kl
from spark_vi.models.topic.types import GatedBOWDocument

PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
V = 30

# With n_bg=2 / tpn=1, node u's block is topic [u + 1]; docs below carry
# frontiers {3}, {} and {5}, whose allowed sets never include node 6's topic
# (index 7) — the starved-floor topic every boost in this suite targets.
STARVED_TOPIC = 7
BOOST_IDX = np.array([2, 5, 9], dtype=np.int64)
BOOST_W = np.array([1.0, 0.5, 0.25])


def _lay():
    return DagLayout(PARENT, n_bg=2, tpn=1)


def _boost():
    return {STARVED_TOPIC: (BOOST_IDX.copy(), BOOST_W.copy())}


def _docs():
    return [
        GatedBOWDocument(indices=np.array([5, 6], dtype=np.int32),
                         counts=np.array([2.0, 1.0]), length=3,
                         frontier=frozenset({3})),
        GatedBOWDocument(indices=np.array([1, 2, 7], dtype=np.int32),
                         counts=np.array([1.0, 3.0, 2.0]), length=6,
                         frontier=frozenset()),
        GatedBOWDocument(indices=np.array([0, 8, 9], dtype=np.int32),
                         counts=np.array([1.0, 1.0, 4.0]), length=6,
                         frontier=frozenset({5})),
    ]


def _fit(model, n_iters=5, lr=0.5):
    gp = model.initialize_global(None)
    docs = _docs()
    elbos = []
    for _ in range(n_iters):
        stats = model.local_update(docs, gp)
        gp = model.update_global(gp, stats, learning_rate=lr)
        elbos.append(model.compute_elbo(gp, stats))
    return gp, np.array(elbos)


# -- 2: absent boost is byte-identical ---------------------------------------

def test_boost_none_and_empty_are_byte_identical_to_no_kwarg():
    """None and {} resolve to the SAME no-boost state as omitting the kwarg,
    and a full 5-step fit is bit-identical across all three (same seed). The
    None/{} paths never branch into boost code, so this pins that boost-absent
    behavior IS the original code path."""
    lay = _lay()
    m_plain = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    m_none = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0,
                            eta_boost=None)
    m_empty = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0,
                             eta_boost={})
    assert m_plain._eta_boost is None
    assert m_none._eta_boost is None
    assert m_empty._eta_boost is None
    gp0, e0 = _fit(m_plain)
    for m in (m_none, m_empty):
        gp, e = _fit(m)
        np.testing.assert_array_equal(gp["lambda"], gp0["lambda"])
        assert gp["lambda"].dtype == gp0["lambda"].dtype
        np.testing.assert_array_equal(gp["alpha"], gp0["alpha"])
        np.testing.assert_array_equal(e, e0)
    # Display/provenance paths stay byte-identical too.
    assert m_none.iteration_summary(gp0) == m_plain.iteration_summary(gp0)
    assert m_none.get_metadata() == m_plain.get_metadata()
    assert "eta_boost_topics" not in m_plain.get_metadata()


def test_boost_none_multidomain_fit_is_byte_identical():
    lay = _lay()
    kw = dict(alpha=0.1, eta=[0.02, 0.05], domains=[20, 10], random_seed=0)
    gp0, e0 = _fit(GatedOnlineLDA(lay, V, **kw))
    gp1, e1 = _fit(GatedOnlineLDA(lay, V, eta_boost=None, **kw))
    for m in range(2):
        np.testing.assert_array_equal(gp1["lambda"][m], gp0["lambda"][m])
    np.testing.assert_array_equal(e1, e0)


# -- 1: floor tilt ------------------------------------------------------------

def test_zero_count_boosted_topic_elogbeta_tilts_toward_boosted_vocab():
    """The persistent-prior mechanism: the boosted topic's node (6) is in no
    doc's frontier closure, so its lambda row sees ZERO counts every iteration
    and rho-blends toward its effective prior. With a boost that prior is
    eta + boost, so E[log beta] ends tilted onto the boosted indices; without
    it the row is flat and the same inequality fails."""
    lay = _lay()
    m_b = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0,
                         eta_boost=_boost())
    m_p = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp_b, _ = _fit(m_b, n_iters=40)
    gp_p, _ = _fit(m_p, n_iters=40)

    # The starved row's lambda has converged (geometrically) onto the prior.
    np.testing.assert_allclose(
        gp_b["lambda"][STARVED_TOPIC][BOOST_IDX], 0.02 + BOOST_W, rtol=1e-9)
    others = np.setdiff1d(np.arange(V), BOOST_IDX)
    np.testing.assert_allclose(
        gp_b["lambda"][STARVED_TOPIC][others], 0.02, rtol=1e-9)

    def elogbeta_row(gp, k):
        from scipy.special import digamma
        lam = gp["lambda"]
        return digamma(lam[k]) - digamma(lam[k].sum())

    row_b = elogbeta_row(gp_b, STARVED_TOPIC)
    row_p = elogbeta_row(gp_p, STARVED_TOPIC)
    assert row_b[BOOST_IDX].min() > row_p.max()          # tilted well above flat
    assert row_b[BOOST_IDX].min() > row_b[others].max()  # tilt is AT the profile
    assert not (row_p[BOOST_IDX].min() > row_p[others].max())  # no tilt unboosted

    # And the tilt did not leak: every other topic's row is bit-identical to
    # the un-boosted fit (obligation 4, trained rows included).
    for k in range(lay.K):
        if k == STARVED_TOPIC:
            continue
        np.testing.assert_array_equal(gp_b["lambda"][k], gp_p["lambda"][k])


# -- 3: lambda update = eta_vec + counts --------------------------------------

def test_lambda_update_equals_eta_vec_plus_counts_single_domain():
    """At learning_rate=1.0 the update IS the natural-gradient target
    eta_vec + expElogbeta * counts-stat; the boost adds its weights on the
    boosted topic's listed indices and nothing else. The starved topic's row
    (zero counts) lands EXACTLY on eta + boost — the tilted floor."""
    lay = _lay()
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0,
                       eta_boost=_boost())
    gp = m.initialize_global(None)
    stats = m.local_update(_docs(), gp)

    from scipy.special import digamma
    lam = gp["lambda"]
    eb = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    expected = gp["eta"] + eb * stats["lambda_stats"]
    expected[STARVED_TOPIC, BOOST_IDX] += BOOST_W

    new_gp = m.update_global(gp, stats, learning_rate=1.0)
    np.testing.assert_array_equal(new_gp["lambda"], expected)
    # Zero-count row: exactly the effective prior.
    floor = np.full(V, 0.02)
    floor[BOOST_IDX] += BOOST_W
    np.testing.assert_array_equal(new_gp["lambda"][STARVED_TOPIC], floor)


def test_lambda_update_equals_eta_vec_plus_counts_multidomain():
    lay = _lay()
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=[0.02, 0.05], domains=[20, 10],
                       random_seed=0, eta_boost=_boost())
    gp = m.initialize_global(None)
    stats = m.local_update(_docs(), gp)

    eb = m._assemble_expElogbeta(gp["lambda"])
    expected = m._eta_vocab_vector() + eb * stats["lambda_stats"]
    expected[STARVED_TOPIC, BOOST_IDX] += BOOST_W

    new_gp = m.update_global(gp, stats, learning_rate=1.0)
    np.testing.assert_array_equal(new_gp["lambda"][0], expected[:, :20])
    np.testing.assert_array_equal(new_gp["lambda"][1], expected[:, 20:])
    # Starved row, condition domain: exactly eta_0 + boost; drug domain: eta_1.
    floor0 = np.full(20, 0.02)
    floor0[BOOST_IDX] += BOOST_W
    np.testing.assert_array_equal(new_gp["lambda"][0][STARVED_TOPIC], floor0)
    np.testing.assert_array_equal(new_gp["lambda"][1][STARVED_TOPIC],
                                  np.full(10, 0.05))


# -- 4: non-boosted topics / non-condition domains untouched ------------------

def test_one_step_non_boosted_rows_and_other_domains_bit_identical():
    """One M-step from the same global params, with vs without the boost:
    every non-boosted topic row is bit-identical; the non-condition domain is
    bit-identical even on the boosted topic; and the boosted row moves by
    exactly rho * w on the boosted indices, 0 elsewhere."""
    lay = _lay()
    rho = 0.7
    kw = dict(alpha=0.1, eta=[0.02, 0.05], domains=[20, 10], random_seed=0)
    m_b = GatedOnlineLDA(lay, V, eta_boost=_boost(), **kw)
    m_p = GatedOnlineLDA(lay, V, **kw)
    gp = m_p.initialize_global(None)
    stats = m_p.local_update(_docs(), gp)

    out_b = m_b.update_global(gp, stats, learning_rate=rho)
    out_p = m_p.update_global(gp, stats, learning_rate=rho)

    np.testing.assert_array_equal(out_b["lambda"][1], out_p["lambda"][1])
    for k in range(lay.K):
        if k == STARVED_TOPIC:
            continue
        np.testing.assert_array_equal(out_b["lambda"][0][k],
                                      out_p["lambda"][0][k])
    diff = out_b["lambda"][0][STARVED_TOPIC] - out_p["lambda"][0][STARVED_TOPIC]
    expected = np.zeros(20)
    expected[BOOST_IDX] = rho * BOOST_W
    np.testing.assert_allclose(diff, expected, rtol=1e-12, atol=0.0)


# -- ELBO: eta-dependent global KL under the boosted prior --------------------

def test_elbo_matches_manual_boosted_prior_single_domain():
    lay = _lay()
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=3,
                       eta_boost=_boost())
    gp = m.initialize_global(None)
    agg = {"doc_loglik_sum": np.array(-12.3), "doc_theta_kl_sum": np.array(1.7)}

    manual_kl = 0.0
    for k in range(lay.K):
        prior = np.full(V, 0.02)
        if k == STARVED_TOPIC:
            prior[BOOST_IDX] += BOOST_W
        manual_kl += _dirichlet_kl(gp["lambda"][k], prior)
    expected = -12.3 - 1.7 - manual_kl
    np.testing.assert_allclose(m.compute_elbo(gp, agg), expected)

    # The boost genuinely moves the eta-dependent term (vs the flat prior).
    m_p = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=3)
    assert m.compute_elbo(gp, agg) != m_p.compute_elbo(gp, agg)


def test_elbo_matches_manual_boosted_prior_multidomain():
    lay = _lay()
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=[0.5, 0.2], domains=[20, 10],
                       random_seed=3, eta_boost=_boost())
    gp = m.initialize_global(None)
    agg = {"doc_loglik_sum": np.array(-12.3), "doc_theta_kl_sum": np.array(1.7)}

    manual_kl = 0.0
    for md, (V_m, eta_m) in enumerate([(20, 0.5), (10, 0.2)]):
        for k in range(lay.K):
            prior = np.full(V_m, eta_m)
            if md == 0 and k == STARVED_TOPIC:
                prior[BOOST_IDX] += BOOST_W
            manual_kl += _dirichlet_kl(gp["lambda"][md][k], prior)
    expected = -12.3 - 1.7 - manual_kl
    np.testing.assert_allclose(m.compute_elbo(gp, agg), expected)


# -- validation guards --------------------------------------------------------

@pytest.mark.parametrize("boost, match", [
    ({8: (np.array([1]), np.array([1.0]))}, "outside"),          # K=8 -> max 7
    ({-1: (np.array([1]), np.array([1.0]))}, "outside"),
    ({3.5: (np.array([1]), np.array([1.0]))}, "integer"),
    ({7: (np.array([1, 1]), np.array([1.0, 2.0]))}, "duplicate"),
    ({7: (np.array([1]), np.array([0.0]))}, "finite and > 0"),
    ({7: (np.array([1]), np.array([-1.0]))}, "finite and > 0"),
    ({7: (np.array([1]), np.array([np.nan]))}, "finite and > 0"),
    ({7: (np.array([1, 2]), np.array([1.0]))}, "shape"),
    ({7: (np.array([1.5]), np.array([1.0]))}, "integer array"),
    ({7: (np.array([-1]), np.array([1.0]))}, "condition"),
    ({7: (np.array([30]), np.array([1.0]))}, "condition"),       # V=30 -> max 29
    ({7: np.array([1])}, "pair"),
    ([7], "dict"),
])
def test_eta_boost_validation_rejects_malformed_input(boost, match):
    lay = _lay()
    with pytest.raises(ValueError, match=match):
        GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, eta_boost=boost)


def test_eta_boost_indices_validated_against_condition_domain_width():
    """Multi-domain: indices must lie inside DOMAIN 0 ([0, V_0)); an index in a
    later domain's range is the boost-leak wiring bug and must fail fast."""
    lay = _lay()
    with pytest.raises(ValueError, match="condition"):
        GatedOnlineLDA(lay, V, domains=[20, 10], eta_boost={
            7: (np.array([25]), np.array([1.0]))})
    # The same index is legal when the whole vocabulary is one domain.
    m = GatedOnlineLDA(lay, V, eta_boost={7: (np.array([25]), np.array([1.0]))})
    assert m._eta_boost is not None


def test_resolve_eta_boost_copies_and_coerces_dtypes():
    idx = np.array([3, 1], dtype=np.int32)
    w = np.array([2.0, 1.0])
    out = _resolve_eta_boost({np.int64(4): (idx, w)}, K=8, boost_v=30)
    (oi, ow) = out[4]
    assert oi.dtype == np.int64 and ow.dtype == np.float64
    idx[0] = 9
    w[0] = 99.0
    np.testing.assert_array_equal(oi, [3, 1])   # detached from caller arrays
    np.testing.assert_array_equal(ow, [2.0, 1.0])


# -- ADR 0047: the boost must not ride a task closure -------------------------

def test_pickle_excludes_boost_and_executor_path_is_boost_free():
    """The model rides every task closure (VIRunner's `_model=model` capture);
    ADR 0047's closure clause forbids array-shaped closure payloads, and no
    executor-side method reads eta — so a pickled copy carries NO boost, and
    its local_update (the executor's method) is bit-identical anyway."""
    lay = _lay()
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0,
                       eta_boost=_boost())
    clone = pickle.loads(pickle.dumps(m))
    assert clone._eta_boost is None
    assert m._eta_boost is not None            # the driver's instance keeps it

    gp = m.initialize_global(None)
    out_m = m.local_update(_docs(), gp)
    out_c = clone.local_update(_docs(), gp)
    for key in out_m:
        np.testing.assert_array_equal(out_m[key], out_c[key])


# -- provenance / display -----------------------------------------------------

def test_metadata_and_iteration_summary_surface_the_boost():
    lay = _lay()
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0,
                       eta_boost=_boost())
    md = m.get_metadata()
    assert md["eta_boost_topics"] == [STARVED_TOPIC]
    assert md["eta_boost_nnz"] == 3
    np.testing.assert_allclose(md["eta_boost_mass"], float(BOOST_W.sum()))
    gp = m.initialize_global(None)
    assert "η_boost[topics=1 nnz=3" in m.iteration_summary(gp)


# -- PC delegation (the Gated-PC composition inherits the boost) --------------

def test_pc_wrapped_engine_honors_boost_via_delegation():
    from spark_vi.models.topic.pc import OnlinePCLDA
    from spark_vi.models.topic.types import GatedPCDocument
    lay = _lay()
    eng = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0,
                         eta_boost=_boost())
    m = OnlinePCLDA(K=lay.K, vocab_size=V, C=2, weight_y=0.0, topic_engine=eng)
    gp = m.initialize_global(None)
    docs = [GatedPCDocument(indices=d.indices, counts=d.counts, length=d.length,
                            y=np.zeros(2), label_mask=np.zeros(2),
                            frontier=d.frontier)
            for d in _docs()]
    stats = m.local_update(docs, gp)
    new_gp = m.update_global(gp, stats, learning_rate=1.0)
    floor = np.full(V, 0.02)
    floor[BOOST_IDX] += BOOST_W
    np.testing.assert_array_equal(new_gp["lambda"][STARVED_TOPIC], floor)
    assert m.get_metadata()["eta_boost_topics"] == [STARVED_TOPIC]


# -- estimator shim passthrough (the WP-3 seam) -------------------------------

def _pc_estimator(**kw):
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    return OnlinePCLDAEstimator(**kw)


def test_estimator_eta_boost_json_roundtrip_builds_the_boost():
    from spark_vi.mllib.topic.pc import _build_model_and_config
    est = _pc_estimator(numLabels=7)
    est.setGateParent(PARENT)
    est.setEtaBoost({STARVED_TOPIC: (BOOST_IDX.tolist(), BOOST_W.tolist())})
    model, _cfg = _build_model_and_config(est, vocab_size=V)
    boost = model._lda._eta_boost
    assert set(boost) == {STARVED_TOPIC}
    np.testing.assert_array_equal(boost[STARVED_TOPIC][0], BOOST_IDX)
    np.testing.assert_array_equal(boost[STARVED_TOPIC][1], BOOST_W)


def test_estimator_eta_boost_requires_gate_parent():
    from spark_vi.mllib.topic.pc import _build_model_and_config
    est = _pc_estimator(numLabels=7)
    est.setEtaBoost({STARVED_TOPIC: (BOOST_IDX.tolist(), BOOST_W.tolist())})
    with pytest.raises(ValueError, match="gateParent"):
        _build_model_and_config(est, vocab_size=V)


def test_estimator_eta_boost_default_and_reset_are_unboosted():
    from spark_vi.mllib.topic.pc import _build_model_and_config
    est = _pc_estimator(numLabels=7)
    est.setGateParent(PARENT)
    model, _cfg = _build_model_and_config(est, vocab_size=V)
    assert model._lda._eta_boost is None
    est.setEtaBoost({STARVED_TOPIC: (BOOST_IDX.tolist(), BOOST_W.tolist())})
    est.setEtaBoost(None)                       # reset restores the flat prior
    model, _cfg = _build_model_and_config(est, vocab_size=V)
    assert model._lda._eta_boost is None


def test_estimator_eta_boost_rejects_malformed_json():
    from spark_vi.mllib.topic.pc import _build_model_and_config
    est = _pc_estimator(numLabels=7)
    est.setGateParent(PARENT)
    est.setEtaBoost("{not json")
    with pytest.raises(ValueError, match="JSON"):
        _build_model_and_config(est, vocab_size=V)
