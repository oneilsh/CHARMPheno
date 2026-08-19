"""Domain-aware supervised topic correction for multi-domain Gated-PC.

When OnlinePCLDA's injected topic engine is a multi-domain ``GatedOnlineLDA``,
``global_params["lambda"]`` is a per-domain dict ``{m: (K, V_m)}`` and the
supervised gradient stat ``grad_topics_stat`` is the CONCATENATED ``(K, V_total)``
∂loss_y/∂expElogbeta. ``update_global`` must SCATTER that gradient back to the
right per-domain λ block (``_split_to_domains``) and transform+cap each block
against its OWN domain's λ — never pooling the digamma normalizer across domains.

These are pure-numpy unit tests (no Spark): one call each of ``local_update`` +
``update_global`` on a tiny 2-domain gated batch.
"""
import copy

import numpy as np


def _two_domain_gated_pc(V0, V1, C, *, weight_y=50.0, topic_trust=0.1):
    """A 2-domain Gated-PC: OnlinePCLDA wrapping a 2-domain GatedOnlineLDA."""
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.pc import OnlinePCLDA
    parent = {1: 0, 2: 0}                                  # flat DAG: 2 disease nodes
    lay = DagLayout(parent, n_bg=2, tpn=1)
    V = V0 + V1
    engine = GatedOnlineLDA(lay, vocab_size=V, domains=[V0, V1], random_seed=0)
    m = OnlinePCLDA(K=engine.K, vocab_size=V, C=C, weight_y=weight_y,
                    topic_trust=topic_trust, head_optimizer="sgd",
                    random_seed=0, topic_engine=engine)
    return m, engine, V


def _doc(rng, lo, hi, C, frontier, nnz=4):
    """A GatedPCDocument whose tokens fall in the half-open vocab range [lo, hi)."""
    from spark_vi.models.topic.types import GatedPCDocument
    idx = np.sort(rng.choice(np.arange(lo, hi), size=nnz, replace=False)).astype(np.int32)
    cnt = rng.integers(1, 5, size=nnz).astype(np.float64)
    y = rng.integers(0, 2, size=C).astype(np.float64)
    mask = np.ones(C, dtype=np.float64)                    # all cells observed
    return GatedPCDocument(indices=idx, counts=cnt, length=int(cnt.sum()),
                           y=y, label_mask=mask, frontier=frozenset(frontier))


def test_multidomain_correction_routes_gradient_to_the_right_domain_block():
    """Docs carry tokens in DOMAIN 0 ONLY, so grad_topics_stat is zero on domain
    1's columns. The domain-aware correction must then leave domain 1's λ block
    EXACTLY at its unsupervised step (zero correction) while moving domain 0's
    block — proving the scatter routes each domain's gradient to its own block."""
    V0, V1, C = 30, 12, 3
    m, engine, V = _two_domain_gated_pc(V0, V1, C)
    rng = np.random.default_rng(0)
    # every doc's tokens are in [0, V0) — domain 0 only; frontiers vary to gate.
    fronts = [{1}, {2}, set(), {1}, {2}, {1, 2}]
    docs = [_doc(rng, 0, V0, C, f) for f in fronts]

    gp = m.initialize_global(None)
    # A non-trivial head: the supervised topic gradient ∂loss_y/∂θ ∝ w_CK, so it
    # is EXACTLY zero at the w_CK=0 seed. Seed a head so the correction has signal.
    gp["w_CK"] = np.random.default_rng(7).standard_normal((C, engine.K)) * 0.5
    assert isinstance(gp["lambda"], dict) and set(gp["lambda"]) == {0, 1}
    assert gp["lambda"][0].shape == (engine.K, V0)
    assert gp["lambda"][1].shape == (engine.K, V1)

    stats = m.local_update(docs, gp)
    # the fused (K, V_total) supervised topic gradient (local_update ran the
    # per-domain-assembled expElogbeta through the head).
    assert stats["grad_topics_stat"].shape == (engine.K, V)
    # domain 1's columns saw no tokens -> exactly zero gradient there.
    assert np.allclose(stats["grad_topics_stat"][:, V0:], 0.0)
    assert not np.allclose(stats["grad_topics_stat"][:, :V0], 0.0)

    lr = 0.5
    # the pure unsupervised M-step (what m.update_global computes internally before
    # the PC correction) — deterministic given (gp, stats, lr).
    unsup = engine.update_global(copy.deepcopy(gp), copy.deepcopy(stats), lr)
    new_gp = m.update_global(copy.deepcopy(gp), copy.deepcopy(stats), lr)

    assert isinstance(new_gp["lambda"], dict) and set(new_gp["lambda"]) == {0, 1}
    # domain 1: zero gradient -> zero correction -> identical to the unsup step.
    np.testing.assert_allclose(new_gp["lambda"][1], unsup["lambda"][1],
                               rtol=1e-12, atol=1e-12)
    # domain 0: nonzero gradient -> corrected away from the unsup step.
    assert not np.allclose(new_gp["lambda"][0], unsup["lambda"][0])


def test_multidomain_correction_preserves_dict_and_positivity_both_domains():
    """A general batch with tokens in BOTH domains: the NATURAL-gradient correction
    keeps λ a valid per-domain dict (right keys/shapes), every block finite &
    strictly positive, and BOTH blocks move off the unsupervised step (each domain's
    own gradient is applied). A moderate weight_y keeps the (now live, no longer
    trust-clipped) correction a sub-unit relative move."""
    V0, V1, C = 24, 16, 3
    m, engine, V = _two_domain_gated_pc(V0, V1, C, weight_y=2.0)
    rng = np.random.default_rng(1)
    docs = []
    fronts = [{1}, {2}, {1}, {2}, {1, 2}, set()]
    for f in fronts:                                       # tokens span both domains
        d0 = _doc(rng, 0, V0, C, f, nnz=3)
        d1 = _doc(rng, V0, V, C, f, nnz=2)
        idx = np.sort(np.concatenate([d0.indices, d1.indices])).astype(np.int32)
        cnt = rng.integers(1, 5, size=len(idx)).astype(np.float64)
        from spark_vi.models.topic.types import GatedPCDocument
        docs.append(GatedPCDocument(indices=idx, counts=cnt, length=int(cnt.sum()),
                                    y=d0.y, label_mask=d0.label_mask, frontier=d0.frontier))

    gp = m.initialize_global(None)
    gp["w_CK"] = np.random.default_rng(7).standard_normal((C, engine.K)) * 0.5
    stats = m.local_update(docs, gp)
    assert stats["grad_topics_stat"].shape == (engine.K, V)

    lr = 0.5
    unsup = engine.update_global(copy.deepcopy(gp), copy.deepcopy(stats), lr)
    new_gp = m.update_global(copy.deepcopy(gp), copy.deepcopy(stats), lr)

    assert isinstance(new_gp["lambda"], dict) and set(new_gp["lambda"]) == {0, 1}
    for md, Vm in ((0, V0), (1, V1)):
        blk = new_gp["lambda"][md]
        assert blk.shape == (engine.K, Vm)
        assert np.isfinite(blk).all() and (blk > 0).all()          # valid pseudocounts
        assert not np.allclose(blk, unsup["lambda"][md])           # this domain moved


def test_supervised_correction_is_scale_stable_natural_gradient():
    """insight 0072 regression guard: the NATURAL-gradient λ-correction holds a
    STEADY relative move as the corpus grows (λ and the corpus-scaled supervised
    gradient scale together) — unlike the OLD raw-gradient step, which vanished as
    1/λ² and silently zeroed weight_y at whole-population λ."""
    from spark_vi.models.topic.pc import _grad_topics_to_lambda
    m, engine, V = _two_domain_gated_pc(8, 8, 2, weight_y=1.0)
    rng = np.random.default_rng(3)
    K = engine.K
    base_grad = rng.normal(size=(K, 8))
    base_lam = np.abs(rng.normal(size=(K, 8))) + 0.5

    def _rel(new, lam):
        return float(np.sqrt(((new - lam) ** 2).sum()) / np.sqrt((lam * lam).sum()))

    nat, raw = [], []
    for cs in (1e2, 1e4, 1e6):                       # realistic whole-corpus scales
        lam, grad = base_lam * cs, base_grad * cs    # both scale with the corpus
        nat.append(_rel(m._corrected_lambda_block(grad, lam, lam, 0.1, 1.0), lam))
        raw.append(_rel(np.maximum(lam - 0.1 * _grad_topics_to_lambda(grad, lam), 1e-30),
                        lam))                        # the OLD raw-gradient step
    # NATURAL: flat across 4 decades; OLD raw: vanishes as 1/λ².
    assert max(nat) / min(nat) < 1.1
    assert raw[-1] / raw[0] < 1e-3                    # raw collapsed with scale
    assert nat[-1] > 1e3 * raw[-1]                    # natural survives where raw died


def test_correction_is_mass_preserving_and_bounded_at_high_weight_y():
    """exp 0098 regression guard: the EXPONENTIATED-GRADIENT λ-correction preserves each
    topic-row's total pseudocount Σλ_k (to machine precision) and keeps λ strictly
    positive at ANY weight_y — so a large weight_y cannot starve a topic (Σλ_k→~0, the
    −4.5e27 ELBO detonation at weight_y=16) or bloat one. The ADDITIVE step it replaces
    could not guarantee either. Reduces to the additive move at small weight_y."""
    m, engine, V = _two_domain_gated_pc(20, 20, 2, weight_y=1.0)
    rng = np.random.default_rng(5)
    K = engine.K
    cs = 1e5                                          # whole-Mondo-like corpus scale
    lam = (np.abs(rng.normal(size=(K, 20))) + 0.2) * cs
    grad = rng.normal(size=(K, 20)) * cs             # corpus-scaled supervised gradient
    mass0 = lam.sum(axis=1)
    for wy in (2.0, 16.0, 1000.0):
        out = m._corrected_lambda_block(grad, lam, lam, 0.11, wy)
        assert (out > 0).all() and np.isfinite(out).all()          # strictly positive
        # per-topic total mass conserved to machine precision at EVERY weight_y
        np.testing.assert_allclose(out.sum(axis=1), mass0, rtol=1e-10)
    # small weight_y ≈ the additive natural-gradient step (EG reduces to it)
    eg2 = m._corrected_lambda_block(grad, lam, lam, 0.11, 2.0)
    from scipy.special import digamma
    eb = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    add2 = lam - 0.11 * 2.0 * (grad * eb)
    assert np.abs(eg2 - add2).max() / lam.max() < 0.02             # agree at small move
