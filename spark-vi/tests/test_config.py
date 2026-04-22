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
