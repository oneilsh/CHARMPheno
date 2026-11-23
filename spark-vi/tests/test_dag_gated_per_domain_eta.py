"""Per-domain eta in the gated engine (no Spark): the scalar form is unchanged, a
per-domain eta equal to the scalar in every domain is the same arithmetic, and
distinct values reach each domain's lambda intercept and the ELBO prior."""
import numpy as np
import pytest

from spark_vi.dag.layout import DagLayout
from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA


def _engine(eta, domains=(4, 3)):
    lay = DagLayout({1: [0], 2: [1]}, n_bg=2, tpn=1)
    return DagGatedOnlineLDA(lay, sum(domains), domains=list(domains), eta=eta,
                             alpha=np.full(lay.K, 0.5), random_seed=3)


def _stats(K, V, seed=0):
    rng = np.random.default_rng(seed)
    return {"lambda_stats": rng.gamma(1.0, 1.0, size=(K, V))}


def test_equal_per_domain_eta_is_the_scalar_fit():
    a, b = _engine(0.1), _engine([0.1, 0.1])
    ga, gb = a.initialize_global(None), b.initialize_global(None)
    assert np.ndim(ga["eta"]) == 0 and gb["eta"].shape == (2,)
    st = _stats(a.K, a.V)
    ua = a.update_global(ga, st, 0.3)
    ub = b.update_global(gb, st, 0.3)
    for m in (0, 1):
        np.testing.assert_array_equal(ua["lambda"][m], ub["lambda"][m])
    agg = {"doc_loglik_sum": -1.0, "doc_theta_kl_sum": 0.5}
    assert a.compute_elbo(ua, agg) == b.compute_elbo(ub, agg)


def test_distinct_etas_reach_each_domain():
    e = _engine([0.01, 2.0])
    gp = e.initialize_global(None)
    zero = {"lambda_stats": np.zeros((e.K, e.V))}
    out = e.update_global(gp, zero, 1.0)            # rho = 1: lambda = eta exactly
    np.testing.assert_allclose(out["lambda"][0], 0.01)
    np.testing.assert_allclose(out["lambda"][1], 2.0)
    assert "η=[0.01, 2]" in e.iteration_summary(out)
    np.testing.assert_allclose(e.iteration_diagnostics(out)["eta"], [0.01, 2.0])
    agg = {"doc_loglik_sum": 0.0, "doc_theta_kl_sum": 0.0}
    # lambda == the prior in each domain: the global KL is zero.
    assert abs(e.compute_elbo(out, agg)) < 1e-9


def test_per_domain_eta_validation():
    with pytest.raises(ValueError, match="one value per domain"):
        _engine([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="> 0"):
        _engine([0.1, 0.0])
    lay = DagLayout({1: [0]}, n_bg=1, tpn=1)
    with pytest.raises(ValueError, match="one value per domain"):
        DagGatedOnlineLDA(lay, 5, eta=[0.1, 0.2])       # no domains
