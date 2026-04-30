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


def test_vanilla_lda_initialize_global_returns_lambda_of_correct_shape():
    import numpy as np
    from spark_vi.models.lda import VanillaLDA

    m = VanillaLDA(K=5, vocab_size=20, gamma_shape=100.0)
    g = m.initialize_global(data_summary=None)
    assert "lambda" in g
    assert g["lambda"].shape == (5, 20)
    # Gamma(100, 1/100) draws are positive with mean ~1; sanity-check positivity.
    assert (g["lambda"] > 0).all()


def test_vanilla_lda_initialize_global_is_seedable_via_numpy():
    """Seeding numpy.random produces reproducible lambda init.

    The model's lambda init draws from numpy's default Gamma RNG; tests can
    pin reproducibility by seeding np.random before construction.
    """
    import numpy as np
    from spark_vi.models.lda import VanillaLDA

    np.random.seed(42)
    g1 = VanillaLDA(K=3, vocab_size=10).initialize_global(None)
    np.random.seed(42)
    g2 = VanillaLDA(K=3, vocab_size=10).initialize_global(None)
    np.testing.assert_array_equal(g1["lambda"], g2["lambda"])


def test_vanilla_lda_local_update_returns_expected_keys():
    """local_update returns the four keys the runner + ELBO need."""
    import numpy as np
    from spark_vi.core import BOWDocument
    from spark_vi.models.lda import VanillaLDA

    np.random.seed(0)
    m = VanillaLDA(K=3, vocab_size=5)
    g = m.initialize_global(None)
    docs = [
        BOWDocument(indices=np.array([0, 2], dtype=np.int32),
                    counts=np.array([1.0, 2.0]), length=3),
        BOWDocument(indices=np.array([1, 4], dtype=np.int32),
                    counts=np.array([3.0, 1.0]), length=4),
    ]
    stats = m.local_update(rows=iter(docs), global_params=g)
    assert set(stats.keys()) == {"lambda_stats", "doc_loglik_sum", "doc_theta_kl_sum", "n_docs"}
    assert stats["lambda_stats"].shape == (3, 5)
    assert isinstance(float(stats["doc_loglik_sum"]), float)
    assert isinstance(float(stats["doc_theta_kl_sum"]), float)
    assert int(stats["n_docs"]) == 2


def test_vanilla_lda_local_update_lambda_stats_is_nonzero_only_on_seen_columns():
    """Lambda stats accumulate only on columns whose token indices appeared."""
    import numpy as np
    from spark_vi.core import BOWDocument
    from spark_vi.models.lda import VanillaLDA

    np.random.seed(0)
    m = VanillaLDA(K=2, vocab_size=6)
    g = m.initialize_global(None)
    # Only indices 1 and 3 ever appear.
    docs = [BOWDocument(indices=np.array([1, 3], dtype=np.int32),
                         counts=np.array([2.0, 1.0]), length=3)]
    stats = m.local_update(rows=iter(docs), global_params=g)
    untouched_cols = [0, 2, 4, 5]
    np.testing.assert_array_equal(stats["lambda_stats"][:, untouched_cols], 0.0)
    # The seen columns received some mass.
    assert (stats["lambda_stats"][:, [1, 3]] > 0).any()


def test_vanilla_lda_local_update_handles_empty_partition():
    """Empty rows iterator returns zero stats and n_docs=0."""
    import numpy as np
    from spark_vi.models.lda import VanillaLDA

    m = VanillaLDA(K=2, vocab_size=4)
    g = m.initialize_global(None)
    stats = m.local_update(rows=iter([]), global_params=g)
    np.testing.assert_array_equal(stats["lambda_stats"], np.zeros((2, 4)))
    assert int(stats["n_docs"]) == 0
    assert float(stats["doc_loglik_sum"]) == 0.0
    assert float(stats["doc_theta_kl_sum"]) == 0.0


def test_vanilla_lda_update_global_at_lr_zero_is_identity():
    import numpy as np
    from spark_vi.models.lda import VanillaLDA
    np.random.seed(0)
    m = VanillaLDA(K=2, vocab_size=4)
    g = m.initialize_global(None)
    target = {"lambda_stats": np.ones((2, 4)) * 5.0}
    new_g = m.update_global(g, target_stats=target, learning_rate=0.0)
    np.testing.assert_array_equal(new_g["lambda"], g["lambda"])


def test_vanilla_lda_update_global_at_lr_one_jumps_to_target():
    """At rho=1.0, lambda becomes (eta + lambda_stats)."""
    import numpy as np
    from spark_vi.models.lda import VanillaLDA
    np.random.seed(0)
    m = VanillaLDA(K=2, vocab_size=4, eta=0.05)
    g = m.initialize_global(None)
    target = {"lambda_stats": np.full((2, 4), 7.0)}
    new_g = m.update_global(g, target_stats=target, learning_rate=1.0)
    np.testing.assert_allclose(new_g["lambda"], 0.05 + 7.0)


def test_vanilla_lda_combine_stats_is_associative():
    """treeReduce relies on associativity: combine(a, combine(b, c)) == combine(combine(a, b), c)."""
    import numpy as np
    from spark_vi.models.lda import VanillaLDA
    rng = np.random.default_rng(0)
    m = VanillaLDA(K=2, vocab_size=3)
    def _stats():
        return {
            "lambda_stats": rng.normal(size=(2, 3)),
            "doc_loglik_sum": np.array(rng.normal()),
            "doc_theta_kl_sum": np.array(rng.normal()),
            "n_docs": np.array(float(rng.integers(0, 100))),
        }
    a, b, c = _stats(), _stats(), _stats()
    left = m.combine_stats(a, m.combine_stats(b, c))
    right = m.combine_stats(m.combine_stats(a, b), c)
    for k in left:
        np.testing.assert_allclose(left[k], right[k])
