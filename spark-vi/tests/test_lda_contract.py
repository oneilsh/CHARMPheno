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


def test_vanilla_lda_is_a_vimodel():
    from spark_vi.core import VIModel
    from spark_vi.models.lda import VanillaLDA
    assert issubclass(VanillaLDA, VIModel)


def test_vanilla_lda_default_alpha_eta_match_one_over_k():
    """Default symmetric alpha and eta both default to 1/K, matching MLlib."""
    from spark_vi.models.lda import VanillaLDA
    m = VanillaLDA(K=4, vocab_size=100)
    assert m.alpha == pytest.approx(0.25)
    assert m.eta == pytest.approx(0.25)


def test_vanilla_lda_explicit_alpha_eta_respected():
    from spark_vi.models.lda import VanillaLDA
    m = VanillaLDA(K=10, vocab_size=100, alpha=0.1, eta=0.2)
    assert m.alpha == pytest.approx(0.1)
    assert m.eta == pytest.approx(0.2)


def test_vanilla_lda_rejects_invalid_hyperparams():
    from spark_vi.models.lda import VanillaLDA
    with pytest.raises(ValueError):
        VanillaLDA(K=0, vocab_size=10)
    with pytest.raises(ValueError):
        VanillaLDA(K=2, vocab_size=0)
    with pytest.raises(ValueError):
        VanillaLDA(K=2, vocab_size=10, alpha=-1.0)
    with pytest.raises(ValueError):
        VanillaLDA(K=2, vocab_size=10, eta=0.0)
    with pytest.raises(ValueError):
        VanillaLDA(K=2, vocab_size=10, cavi_max_iter=0)
    with pytest.raises(ValueError):
        VanillaLDA(K=2, vocab_size=10, cavi_tol=0.0)
