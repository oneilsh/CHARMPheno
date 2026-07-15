"""Tests for the refit-loop dynamics building blocks (Task B2).

1. ``gated_ln_corpus``'s new ``eta_scale`` kwarg scales the PLANTED covariance
   used to draw eta (and the returned ``Sigma_true``) -- default 1.0 must be
   byte-identical to omitting the kwarg, and eta_scale=9.0 must return exactly
   9x the eta_scale=1.0 covariance on the same seed (same draws before scaling,
   since eta_scale only rescales the covariance passed to
   ``rng.multivariate_normal``, and the RNG stream up to that call is
   independent of eta_scale).
2. A tiny end-to-end smoke of the fit -> calibrate -> refit -> recalibrate
   wiring (not asserting dynamics, just that it runs and returns a finite grid
   value) -- see scripts/refit_dynamics_synthetic.py for the real experiment.
"""
from __future__ import annotations

import numpy as np

from tests._stm_synth import gated_ln_corpus, fit_stm
from spark_vi.mllib.topic.stm import corpus_heldout_scale_sweep_gated


class TestEtaScaleScalesVariance:
    def test_eta_scale_scales_variance(self):
        _, _, sig1, _ = gated_ln_corpus(
            group_weights={"A": 0.7, "B": 0.3}, fg_per_group=1, bg_k=2,
            V=40, D=20, doc_len=15, seed=0,
        )
        _, _, sig9, _ = gated_ln_corpus(
            group_weights={"A": 0.7, "B": 0.3}, fg_per_group=1, bg_k=2,
            V=40, D=20, doc_len=15, eta_scale=9.0, seed=0,
        )
        np.testing.assert_allclose(sig9, 9.0 * sig1)

    def test_default_eta_scale_matches_omitted_kwarg(self):
        docs_a, part_a, sig_a, beta_a = gated_ln_corpus(
            group_weights={"A": 0.7, "B": 0.3}, fg_per_group=1, bg_k=2,
            V=40, D=20, doc_len=15, seed=0,
        )
        docs_b, part_b, sig_b, beta_b = gated_ln_corpus(
            group_weights={"A": 0.7, "B": 0.3}, fg_per_group=1, bg_k=2,
            V=40, D=20, doc_len=15, eta_scale=1.0, seed=0,
        )
        np.testing.assert_array_equal(sig_a, sig_b)
        np.testing.assert_array_equal(beta_a, beta_b)
        for da, db in zip(docs_a, docs_b):
            np.testing.assert_array_equal(da.indices, db.indices)
            np.testing.assert_array_equal(da.counts, db.counts)


class TestOneRefitRoundRuns:
    def test_one_refit_round_runs(self):
        """TINY smoke: plant -> fit(unit) -> calibrate -> refit(pin=c0) ->
        recalibrate. Not asserting dynamics -- just that the loop wiring
        (sigma_diagonal_pin refit + fresh-seed sweep) is sound end to end."""
        docs, part, sigma_true, beta_true = gated_ln_corpus(
            group_weights={"A": 0.7, "B": 0.3}, fg_per_group=1, bg_k=2,
            V=60, D=200, doc_len=20, eta_scale=5.0, seed=0,
        )
        K = part.K
        c_grid = [1, 3, 5]

        gp0 = fit_stm(docs, K=K, V=60, sigma_init=1.0, n_iter=30,
                     partition=part, reference_topic=True)
        sweep0 = corpus_heldout_scale_sweep_gated(
            docs, gp0, part, c_grid=c_grid, holdout_frac=0.5, reference=0, seed=0,
        )
        c0 = sweep0["argmax_c"]
        assert c0 in c_grid

        gp1 = fit_stm(docs, K=K, V=60, sigma_init=1.0, n_iter=30,
                     partition=part, reference_topic=True,
                     sigma_diagonal_pin=c0)
        sweep1 = corpus_heldout_scale_sweep_gated(
            docs, gp1, part, c_grid=c_grid, holdout_frac=0.5, reference=0, seed=1,
        )
        c1 = sweep1["argmax_c"]
        assert c1 in c_grid
        assert np.isfinite(c1)
