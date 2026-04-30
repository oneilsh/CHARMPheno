"""VIModel contract tests for the optional infer_local capability."""
import pytest


def test_vimodel_default_infer_local_raises_clear_error():
    """A VIModel that doesn't override infer_local must raise NotImplementedError
    with a message naming the concrete class — no silent fallback to None/NaN.
    """
    from spark_vi.models.counting import CountingModel

    m = CountingModel()
    with pytest.raises(NotImplementedError) as exc:
        m.infer_local(row=1, global_params={"alpha": 1.0, "beta": 1.0})
    msg = str(exc.value)
    assert "CountingModel" in msg
    assert "transform" in msg.lower() or "inference" in msg.lower()
