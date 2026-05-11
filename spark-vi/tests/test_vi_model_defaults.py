"""Default behavior of VIModel optional hooks (get_metadata, iteration_diagnostics)."""
import numpy as np


def _make_stub_class():
    from spark_vi.core.model import VIModel

    class _ToyModel(VIModel):
        def initialize_global(self, data_summary=None):
            return {"lambda": np.array(1.0)}

        def local_update(self, rows, global_params):
            return {"x": np.array(0.0)}

        def update_global(self, global_params, target_stats, learning_rate):
            return global_params

    return _ToyModel


def test_vi_model_get_metadata_default_returns_empty_dict():
    cls = _make_stub_class()
    model = cls()
    assert model.get_metadata() == {}


def test_vi_model_iteration_diagnostics_default_returns_empty_dict():
    cls = _make_stub_class()
    model = cls()
    assert model.iteration_diagnostics({"foo": np.array([1.0])}) == {}
