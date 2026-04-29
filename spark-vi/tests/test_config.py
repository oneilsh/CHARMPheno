"""VIConfig holds training-loop hyperparameters."""
import pytest


def test_vi_config_defaults_are_sensible():
    from spark_vi.core import VIConfig

    cfg = VIConfig()
    # Hoffman-style defaults (see docs/architecture/SPARK_VI_FRAMEWORK.md).
    assert cfg.max_iterations >= 1
    assert 0.0 < cfg.learning_rate_tau0
    assert 0.0 < cfg.learning_rate_kappa < 1.0
    assert cfg.convergence_tol > 0.0
    assert cfg.checkpoint_interval is None or cfg.checkpoint_interval > 0
    # Mini-batch defaults: full-batch (None) and MLlib-style with-replacement.
    assert cfg.mini_batch_fraction is None
    assert cfg.sample_with_replacement is True
    assert cfg.random_seed is None


def test_vi_config_rejects_invalid_values():
    from spark_vi.core import VIConfig

    with pytest.raises(ValueError):
        VIConfig(max_iterations=0)
    with pytest.raises(ValueError):
        VIConfig(learning_rate_kappa=1.5)
    with pytest.raises(ValueError):
        VIConfig(convergence_tol=-1.0)


def test_vi_config_is_frozen_dataclass():
    from spark_vi.core import VIConfig

    cfg = VIConfig()
    with pytest.raises(Exception):
        cfg.max_iterations = 999  # frozen


def test_vi_config_accepts_valid_mini_batch_fraction():
    from spark_vi.core import VIConfig

    # boundary and interior values
    for f in (0.01, 0.05, 0.5, 1.0):
        cfg = VIConfig(mini_batch_fraction=f)
        assert cfg.mini_batch_fraction == f


def test_vi_config_rejects_invalid_mini_batch_fraction():
    from spark_vi.core import VIConfig

    with pytest.raises(ValueError):
        VIConfig(mini_batch_fraction=0.0)
    with pytest.raises(ValueError):
        VIConfig(mini_batch_fraction=-0.1)
    with pytest.raises(ValueError):
        VIConfig(mini_batch_fraction=1.5)


def test_vi_config_accepts_random_seed_int_or_none():
    from spark_vi.core import VIConfig

    assert VIConfig(random_seed=None).random_seed is None
    assert VIConfig(random_seed=0).random_seed == 0
    assert VIConfig(random_seed=42).random_seed == 42


def test_vi_config_rejects_non_int_random_seed():
    from spark_vi.core import VIConfig

    with pytest.raises(ValueError):
        VIConfig(random_seed="42")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        VIConfig(random_seed=3.14)  # type: ignore[arg-type]
