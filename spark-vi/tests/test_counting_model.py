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


def test_counting_model_elbo_is_increasing_with_more_data():
    """Crude sanity: posterior concentration should monotonically raise the
    (surrogate) ELBO along a sequence of updates with consistent evidence.

    This is a smoke check of the ELBO method returning a finite number, not
    a correctness proof of the log-marginal likelihood itself.
    """
    from spark_vi.models.counting import CountingModel

    m = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    g = m.initialize_global(data_summary=None)
    elbo0 = m.compute_elbo(g, {"heads": np.array(0.0), "tails": np.array(0.0)})
    g = m.update_global(g, {"heads": np.array(30.0), "tails": np.array(10.0)},
                        learning_rate=1.0)
    elbo1 = m.compute_elbo(g, {"heads": np.array(30.0), "tails": np.array(10.0)})
    assert np.isfinite(elbo0) and np.isfinite(elbo1)
    # Under CountingModel's surrogate ELBO, concentration from (1,1) -> (31,11)
    # strictly increases the score (the (a+b) term dominates the -betaln term).
    assert elbo1 > elbo0


def test_counting_model_required_methods_surface_on_base():
    """The VIModel ABC refuses instantiation unless the three required
    methods are implemented."""
    from spark_vi.core import VIModel

    with pytest.raises(TypeError):
        # Missing required abstract methods → abstract class cannot be instantiated.
        VIModel()
