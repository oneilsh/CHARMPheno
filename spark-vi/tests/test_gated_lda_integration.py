"""Spark-local end-to-end smoke tests for a MULTI-DOMAIN GatedOnlineLDA fit.

Why this file exists: every other multi-domain test on this branch drives the
E-step / M-step directly, and two runner-facing breaks were found by code reading
alone (`iteration_summary` raised on a per-domain dict lambda; `save_result`
silently wrote an unreadable one). Both were in code the fit itself never touches
-- which is the signal that the class was never CLOSED against the real driver
loop. These tests run the actual VIRunner over an actual RDD once, which is the
only thing that exercises broadcast of a dict-valued global param, the default
elementwise `combine_stats` over the flat sufficient statistics, the per-iteration
`iteration_summary` call, and the final VIResult assembly together.

Scope: SMOKE, not recovery. One iteration, six documents, ten vocabulary ids.
Recovery and oracle equivalence are asserted by the dense tests in
test_gated_lda.py; nothing here should ever be tightened into a quality gate.

Uses the session-scoped `spark` fixture from tests/conftest.py (Java
security-manager option and PYSPARK_PYTHON pinning are applied there).
Deliberately NOT marked `slow` (the default addopts is -m 'not slow'): a
one-iteration six-document fit costs the Spark session and nothing else, and a
closed-against-the-runner check that never runs in the default suite is the same
gap this file was written to close. The other Spark-local smoke file,
test_gated_lda_shim.py, is unmarked for the same reason.
"""
import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")


def _tiny_two_domain_rdd(spark):
    """Six hand-built two-domain gated docs (domain 0 = ids 0..5, domain 1 = 6..9),
    parallelized over 2 slices so the reduce combines more than one partition."""
    from spark_vi.models.topic.types import GatedBOWDocument
    specs = [
        ([0, 1, 6], [2.0, 1.0, 3.0], frozenset({2})),
        ([1, 2, 7], [1.0, 2.0, 1.0], frozenset({2})),
        ([3, 4, 8], [3.0, 1.0, 2.0], frozenset({3})),
        ([4, 5, 9], [1.0, 1.0, 4.0], frozenset({3})),
        ([0, 3, 6], [1.0, 1.0, 1.0], frozenset({1})),
        ([2, 5, 9], [2.0, 1.0, 1.0], frozenset()),      # labeled background doc
    ]
    docs = [
        GatedBOWDocument(indices=np.array(idx, dtype=np.int32),
                         counts=np.array(cnt, dtype=np.float64),
                         length=int(sum(cnt)), frontier=front)
        for idx, cnt, front in specs
    ]
    rdd = spark.sparkContext.parallelize(docs, numSlices=2).persist()
    rdd.count()             # materialize for VIRunner's strict cache precondition
    return rdd


def _tiny_model(**kw):
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    return lay, GatedOnlineLDA(lay, vocab_size=10, domains=[6, 4],
                               eta=[0.05, 0.2], omega=[1.0, 0.5],
                               alpha=0.1, random_seed=0, **kw)


def test_multidomain_gated_fit_runs_one_iteration_through_vi_runner(spark):
    """A one-iteration multi-domain fit completes through the real VIRunner and
    returns a well-formed per-domain dict lambda.

    What this actually exercises beyond the direct-call tests: dict-valued
    global_params through the driver->executor broadcast; the flat (K, V)
    lambda_stats plus the flat (n_domains,) instrument through the default
    elementwise combine and the runner's corpus-equivalent stats scaling; the
    per-iteration iteration_summary call (which raised on dict lambda before it was
    overridden); compute_elbo on aggregated stats; and VIResult assembly.
    """
    from spark_vi.core import VIConfig, VIRunner
    lay, model = _tiny_model()
    rdd = _tiny_two_domain_rdd(spark)

    cfg = VIConfig(max_iterations=1, random_seed=0, convergence_tol=1e-9)
    result = VIRunner(model, config=cfg).fit(rdd)

    lam = result.global_params["lambda"]
    assert isinstance(lam, dict) and sorted(lam) == [0, 1]
    assert lam[0].shape == (lay.K, 6) and lam[1].shape == (lay.K, 4)
    for m in (0, 1):
        assert np.all(np.isfinite(lam[m])) and np.all(lam[m] > 0)
    assert result.global_params["alpha"].shape == (lay.K,)
    assert len(result.elbo_trace) == 1 and np.isfinite(result.elbo_trace[0])
    assert result.n_iterations == 1

    # The runner calls iteration_summary every iteration; assert the multi-domain
    # line it produced from these params is well formed (it is the read the
    # override exists for, and a per-domain eta / dict lambda breaks the inherited
    # implementation outright).
    line = model.iteration_summary(result.global_params)
    assert "η_m=[0.05, 0.2]" in line, line
    assert "Σλ_k[m0:" in line and "Σλ_k[m1:" in line, line
    assert "θ_contrib_m=[" in line, line


def test_multidomain_gated_fit_checkpoint_round_trips_the_dict_lambda(spark, tmp_path):
    """A checkpointed multi-domain fit writes a checkpoint that loads back with its
    per-domain lambda intact -- the whole round trip, through the real runner.

    History, because this test used to pin the OPPOSITE contract: checkpointing a
    multi-domain fit was unsupported, and `VIConfig.checkpoint_dir` had to raise
    `UnsupportedGlobalParamError`. The reason was that io.export.save_result had no
    per-domain dict lambda writer, so np.save pickled the dict into a ~573-byte 0-d
    object array -- the fit reported success and the checkpoint was unreadable,
    because load_result reads params with allow_pickle=False. SP3 added the writer:
    save_result is now format_version 2 and records a dict param's domain keys under
    the manifest's "dict_param_keys", one params/<name>_<key>.npy per domain, with
    load_result converting those JSON string keys back to int. The guard therefore
    no longer fires, and what is worth pinning is the round trip it stood in for --
    that the on-disk checkpoint is byte-exact and not silently pickled or lossy.

    The runner's final-save guarantee means a configured checkpoint_dir also holds
    the FINAL VIResult after fit() returns, which is what this reads back.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.io.export import load_result
    _lay, model = _tiny_model()
    rdd = _tiny_two_domain_rdd(spark)

    ckpt = tmp_path / "ckpt"
    cfg = VIConfig(max_iterations=1, random_seed=0, convergence_tol=1e-9,
                   checkpoint_interval=1, checkpoint_dir=ckpt)
    result = VIRunner(model, config=cfg).fit(rdd)

    reloaded = load_result(ckpt)
    lam, lam_back = result.global_params["lambda"], reloaded.global_params["lambda"]
    # int domain keys (not the JSON strings), one sidecar per domain, byte-exact.
    assert isinstance(lam_back, dict)
    assert sorted(lam_back) == sorted(lam) == [0, 1]
    for m in (0, 1):
        assert lam_back[m].shape == lam[m].shape
        np.testing.assert_array_equal(lam_back[m], lam[m])
    # the flat params round-trip alongside the dict one
    np.testing.assert_array_equal(reloaded.global_params["alpha"],
                                  result.global_params["alpha"])
