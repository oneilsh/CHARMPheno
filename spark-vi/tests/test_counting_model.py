"""Contract-exercise tests: CountingModel implements VIModel fully.

CountingModel: each data row is either 0 (tail) or 1 (head); the 'global'
parameters are the Beta-posterior counts (alpha, beta) over the coin bias.
One global step aggregates counts; convergence is reaching max_iterations.
"""
import numpy as np
import pytest


def test_counting_model_is_a_vimodel():
    from spark_vi.core import VIModel
    from spark_vi.models.counting import CountingModel

    assert issubclass(CountingModel, VIModel)


def test_counting_model_initialize_global_returns_prior_counts():
    from spark_vi.models.counting import CountingModel

    m = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    g = m.initialize_global(data_summary=None)
    np.testing.assert_allclose(g["alpha"], 1.0)
    np.testing.assert_allclose(g["beta"], 1.0)


def test_counting_model_local_update_returns_sufficient_stats():
    from spark_vi.models.counting import CountingModel

    m = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    g = m.initialize_global(data_summary=None)
    # Partition of three heads and two tails.
    stats = m.local_update(rows=[1, 1, 1, 0, 0], global_params=g)
    # Sufficient stats: number of heads, number of tails.
    np.testing.assert_allclose(stats["heads"], 3.0)
    np.testing.assert_allclose(stats["tails"], 2.0)


def test_counting_model_combine_stats_sums_elementwise():
    from spark_vi.models.counting import CountingModel

    m = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    combined = m.combine_stats(
        {"heads": np.array(3.0), "tails": np.array(2.0)},
        {"heads": np.array(1.0), "tails": np.array(4.0)},
    )
    np.testing.assert_allclose(combined["heads"], 4.0)
    np.testing.assert_allclose(combined["tails"], 6.0)


def test_counting_model_update_global_applies_natural_gradient():
    """One step: lambda_new = (1 - rho) * lambda_old + rho * (prior + stats).

    With rho=1.0 the update jumps directly to (prior + stats).
    """
    from spark_vi.models.counting import CountingModel

    m = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    old = m.initialize_global(data_summary=None)
    stats = {"heads": np.array(10.0), "tails": np.array(5.0)}
    new = m.update_global(old, stats, learning_rate=1.0)
    # rho=1.0: new = prior + stats = (1 + 10, 1 + 5)
    np.testing.assert_allclose(new["alpha"], 11.0)
    np.testing.assert_allclose(new["beta"], 6.0)


def test_counting_model_update_global_interpolates_partial_step():
    from spark_vi.models.counting import CountingModel

    m = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    old = {"alpha": np.array(2.0), "beta": np.array(2.0)}
    stats = {"heads": np.array(10.0), "tails": np.array(0.0)}
    new = m.update_global(old, stats, learning_rate=0.5)
    # new = 0.5 * old + 0.5 * (prior + stats) = 0.5 * (2, 2) + 0.5 * (11, 1)
    np.testing.assert_allclose(new["alpha"], 6.5)
    np.testing.assert_allclose(new["beta"], 1.5)


def test_counting_model_elbo_at_exact_posterior_equals_log_marginal_likelihood():
    """ELBO is tight at the exact posterior.

    For Beta-Bernoulli, the log marginal likelihood has a closed form:
        log p(x) = betaln(a0 + h, b0 + t) - betaln(a0, b0)
    The ELBO is a lower bound on log p(x), and the bound is tight (equality)
    precisely when q == true posterior. This is the strongest analytic
    correctness check available for an ELBO implementation.
    """
    from scipy.special import betaln

    from spark_vi.models.counting import CountingModel

    a0, b0 = 2.0, 3.0
    h, t = 30.0, 10.0
    m = CountingModel(prior_alpha=a0, prior_beta=b0)

    # Exact posterior: Beta(a0 + h, b0 + t)
    posterior = {"alpha": np.array(a0 + h), "beta": np.array(b0 + t)}
    stats = {"heads": np.array(h), "tails": np.array(t)}
    elbo_at_posterior = m.compute_elbo(posterior, stats)

    log_marginal = betaln(a0 + h, b0 + t) - betaln(a0, b0)
    np.testing.assert_allclose(elbo_at_posterior, log_marginal, rtol=1e-10)


def test_counting_model_elbo_is_below_log_marginal_when_q_is_off():
    """ELBO is a strict lower bound: any q != posterior gives ELBO < log p(x)."""
    from scipy.special import betaln

    from spark_vi.models.counting import CountingModel

    a0, b0 = 1.0, 1.0
    h, t = 30.0, 10.0
    m = CountingModel(prior_alpha=a0, prior_beta=b0)

    log_marginal = betaln(a0 + h, b0 + t) - betaln(a0, b0)
    stats = {"heads": np.array(h), "tails": np.array(t)}

    # An off-posterior q that puts most mass near p = 1, contradicting 30/40 data.
    off_q = {"alpha": np.array(100.0), "beta": np.array(1.0)}
    elbo_off = m.compute_elbo(off_q, stats)
    assert np.isfinite(elbo_off)
    assert elbo_off < log_marginal


def test_counting_model_elbo_increases_along_a_run_toward_posterior():
    """Within a single run, ELBO is monotonically non-decreasing as q
    converges to the true posterior under repeated full-batch updates with
    learning_rate=1.0 (which is the CAVI fixed-point jump).
    """
    from spark_vi.models.counting import CountingModel

    m = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    g = m.initialize_global(data_summary=None)
    stats = {"heads": np.array(30.0), "tails": np.array(10.0)}

    elbo0 = m.compute_elbo(g, stats)
    g = m.update_global(g, stats, learning_rate=1.0)  # jumps to exact posterior
    elbo1 = m.compute_elbo(g, stats)

    assert np.isfinite(elbo0) and np.isfinite(elbo1)
    assert elbo1 >= elbo0  # tight at posterior; should be strictly greater here


def test_counting_model_required_methods_surface_on_base():
    """The VIModel ABC refuses instantiation unless the three required
    methods are implemented."""
    from spark_vi.core import VIModel

    with pytest.raises(TypeError):
        # Missing required abstract methods → abstract class cannot be instantiated.
        VIModel()
