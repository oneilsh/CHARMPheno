"""LocalRunner, the in-memory engine (ADR 0051 §2): fast, Spark-free checks.

Agreement with VIRunner on Spark is the one slow test in
test_local_runner_agreement.py; these pin the sampler, the shared-loop bookkeeping
(resume, checkpoint) and that model-held schedules advance.
"""
import numpy as np
import pytest

from spark_vi.core import LocalRunner, VIConfig
from spark_vi.core.local_runner import sample_rows
from spark_vi.models.topic.counting import CountingModel


@pytest.mark.parametrize("with_replacement", [True, False])
def test_sample_rows_matches_rdd_sample_rate(with_replacement):
    """Batch size has mean n * fraction under both of RDD.sample's samplers."""
    n, f = 2000, 0.1
    sizes = [sample_rows(n, f, with_replacement, s).size for s in range(200)]
    assert abs(np.mean(sizes) - n * f) < 5.0
    idx = sample_rows(n, f, with_replacement, 7)
    np.testing.assert_array_equal(idx, sample_rows(n, f, with_replacement, 7))
    assert idx.min() >= 0 and idx.max() < n
    if not with_replacement:
        assert np.unique(idx).size == idx.size


def test_with_replacement_draws_duplicates():
    idx = np.concatenate([sample_rows(500, 0.5, True, s) for s in range(5)])
    assert np.unique(idx).size < idx.size


class _DiagCounting(CountingModel):
    def iteration_diagnostics(self, global_params):
        return {"alpha": float(global_params["alpha"])}


def test_resume_matches_continuous_run(tmp_path):
    """Checkpoint + resume is the uninterrupted fit: the Robbins-Monro counter, the
    minibatch seed stream and the diagnostic traces all continue."""
    from spark_vi.io.export import save_result

    rows = [1] * 60 + [0] * 40
    cfg = dict(convergence_tol=1e-12, mini_batch_fraction=0.3, random_seed=3)
    continuous = LocalRunner(_DiagCounting(), VIConfig(max_iterations=6, **cfg)).fit(rows)
    first = LocalRunner(_DiagCounting(), VIConfig(max_iterations=3, **cfg)).fit(rows)
    save_result(first, tmp_path / "ckpt")
    resumed = LocalRunner(_DiagCounting(), VIConfig(max_iterations=3, **cfg)).fit(
        rows, resume_from=tmp_path / "ckpt")
    assert resumed.n_iterations == continuous.n_iterations == 6
    np.testing.assert_allclose(resumed.global_params["alpha"],
                               continuous.global_params["alpha"])
    np.testing.assert_allclose(resumed.elbo_trace, continuous.elbo_trace)
    assert resumed.diagnostic_traces["alpha"] == pytest.approx(
        continuous.diagnostic_traces["alpha"])
    assert len(resumed.diagnostic_traces["alpha"]) == 6
    # Full batch (no seeds) resumes exactly too.
    full = dict(convergence_tol=1e-12)
    cont = LocalRunner(CountingModel(), VIConfig(max_iterations=6, **full)).fit(rows)
    r3 = LocalRunner(CountingModel(), VIConfig(max_iterations=3, **full)).fit(rows)
    save_result(r3, tmp_path / "full")
    res = LocalRunner(CountingModel(), VIConfig(max_iterations=3, **full)).fit(
        rows, resume_from=tmp_path / "full")
    np.testing.assert_allclose(res.global_params["alpha"], cont.global_params["alpha"])


def test_an_empty_minibatch_counts_with_a_nan_elbo():
    """A tiny fraction of a tiny corpus draws empty batches: those iterations leave
    the parameters unchanged and record NaN, so every trace has n_iterations entries."""
    cfg = VIConfig(max_iterations=20, mini_batch_fraction=0.1, random_seed=0,
                   convergence_tol=1e-12)
    res = LocalRunner(_DiagCounting(), cfg).fit([1, 0] * 5)   # P(empty) = e^-1
    elbo = np.asarray(res.elbo_trace)
    assert res.n_iterations == len(elbo) == len(res.diagnostic_traces["alpha"]) == 20
    empty = np.isnan(elbo)
    assert empty.any() and not empty.all()
    a = np.asarray(res.diagnostic_traces["alpha"])
    for i in np.flatnonzero(empty):
        assert a[i] == (a[i - 1] if i > 0 else 1.0)       # unchanged (prior alpha 1)


def test_checkpoint_and_final_save(tmp_path):
    from spark_vi.io.export import load_result

    cfg = VIConfig(max_iterations=5, convergence_tol=1e-12, checkpoint_interval=2,
                   checkpoint_dir=tmp_path / "ck")
    res = LocalRunner(CountingModel(early_stop=False), cfg).fit([1, 0, 1])
    on_disk = load_result(tmp_path / "ck")
    assert on_disk.n_iterations == res.n_iterations == 5
    np.testing.assert_allclose(on_disk.global_params["alpha"], res.global_params["alpha"])


def test_chunking_does_not_change_the_fit():
    """Chunks are the local analogue of partitions: additive stats, same answer."""
    from spark_vi.models.topic import BOWDocument
    from spark_vi.models.topic.lda import OnlineLDA

    rng = np.random.default_rng(0)
    docs = []
    for _ in range(40):
        w = rng.integers(0, 20, size=15)
        idx, cnt = np.unique(w, return_counts=True)
        docs.append(BOWDocument(indices=idx.astype(np.int32),
                                counts=cnt.astype(np.float64), length=int(cnt.sum())))
    cfg = VIConfig(max_iterations=4, mini_batch_fraction=0.5, random_seed=1,
                   convergence_tol=1e-12)
    a = LocalRunner(OnlineLDA(K=3, vocab_size=20, random_seed=0), cfg).fit(docs)
    b = LocalRunner(OnlineLDA(K=3, vocab_size=20, random_seed=0), cfg,
                    chunk_size=7).fit(docs)
    np.testing.assert_allclose(a.global_params["lambda"], b.global_params["lambda"],
                               rtol=1e-10)


def test_gate_schedule_advances_through_global_params():
    """The leaves-first gate's counter rides global_params on this engine too."""
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.types import GatedBOWDocument

    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=1, tpn=1)
    m = DagGatedOnlineLDA(lay, 12, random_seed=0)
    m.set_gate_schedule(warm=1)
    rng = np.random.default_rng(0)
    docs = []
    for i in range(30):
        idx, cnt = np.unique(rng.integers(0, 12, size=10), return_counts=True)
        docs.append(GatedBOWDocument(idx.astype(np.int32), cnt.astype(np.float64),
                                     int(cnt.sum()), frontier=frozenset({[2, 3][i % 2]})))
    res = LocalRunner(m, VIConfig(max_iterations=3, convergence_tol=1e-12)).fit(docs)
    assert int(res.global_params["gate_iter"]) == 3
    assert (res.global_params["gate_open_at"] >= 0).all()


def test_rejects_empty_corpus_and_non_model():
    with pytest.raises(ValueError):
        LocalRunner(CountingModel()).fit([])
    with pytest.raises(TypeError):
        LocalRunner(object())
