"""Tests for spark_vi.eval.topic.concentration.

doc_concentration computes (top_mass, eff_topics) for a single document's
topic-proportion vector theta; ConcentrationAcc is the scalable histogram +
streaming-moments accumulator that combines across partitions (mirrors the
Welford accumulator in spark_vi.mllib.topic.stm, but for a fixed-bin
histogram since percentiles need combinable state, not just mean/variance).
"""
from __future__ import annotations

import numpy as np
import pytest


def test_doc_concentration_onehot():
    from spark_vi.eval.topic.concentration import doc_concentration

    theta = np.array([0.0, 0.0, 1.0, 0.0])
    top_mass, eff_topics = doc_concentration(theta)
    assert top_mass == pytest.approx(1.0)
    assert eff_topics == pytest.approx(1.0)


def test_doc_concentration_uniform():
    from spark_vi.eval.topic.concentration import doc_concentration

    # Uniform over m=4 topics, embedded in a larger K-length vector (rest 0).
    m = 4
    theta = np.zeros(10)
    theta[:m] = 1.0 / m
    top_mass, eff_topics = doc_concentration(theta)
    assert top_mass == pytest.approx(1.0 / m)
    assert eff_topics == pytest.approx(4.0)


def test_doc_concentration_known_mix():
    from spark_vi.eval.topic.concentration import doc_concentration

    theta = np.array([0.5, 0.3, 0.2])
    top_mass, eff_topics = doc_concentration(theta)
    assert top_mass == pytest.approx(0.5, abs=1e-12)
    expected_eff = 1.0 / (0.25 + 0.09 + 0.04)
    assert eff_topics == pytest.approx(expected_eff, abs=1e-6)
    assert eff_topics == pytest.approx(2.6316, abs=1e-4)


def test_doc_concentration_degenerate():
    from spark_vi.eval.topic.concentration import doc_concentration

    theta = np.zeros(5)
    top_mass, eff_topics = doc_concentration(theta)
    assert np.isnan(top_mass)
    assert np.isnan(eff_topics)


class TestConcentrationAcc:
    def test_acc_summary_moments(self):
        from spark_vi.eval.topic.concentration import ConcentrationAcc

        acc = ConcentrationAcc.zeros(n_bins=20, eff_max=5.0)
        thetas = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.3, 0.2]),
        ]
        for theta in thetas:
            acc.add(theta)

        summary = acc.summary()
        assert summary["n_docs"] == 3

        tops = [1.0, 0.5, 0.5]
        effs = [1.0, 2.0, 1.0 / (0.25 + 0.09 + 0.04)]
        assert summary["top_mass"]["mean"] == pytest.approx(np.mean(tops))
        assert summary["eff_topics"]["mean"] == pytest.approx(np.mean(effs))

        for key in ("top_mass", "eff_topics"):
            p10 = summary[key]["p10"]
            p25 = summary[key]["p25"]
            p50 = summary[key]["p50"]
            p75 = summary[key]["p75"]
            p90 = summary[key]["p90"]
            for v in (p10, p25, p50, p75, p90):
                assert np.isfinite(v)
            assert p10 <= p25 <= p50 <= p75 <= p90

    def test_acc_skips_degenerate_docs(self):
        from spark_vi.eval.topic.concentration import ConcentrationAcc

        acc = ConcentrationAcc.zeros(n_bins=10, eff_max=3.0)
        acc.add(np.array([1.0, 0.0]))
        acc.add(np.zeros(2))  # degenerate -> must be skipped, not counted
        summary = acc.summary()
        assert summary["n_docs"] == 1

    def test_acc_empty_summary_is_json_safe_no_div_by_zero(self):
        from spark_vi.eval.topic.concentration import ConcentrationAcc

        acc = ConcentrationAcc.zeros(n_bins=10, eff_max=3.0)
        summary = acc.summary()
        assert summary["n_docs"] == 0
        # Must not raise, and means/percentiles should be None or nan.
        mean = summary["top_mass"]["mean"]
        assert mean is None or (isinstance(mean, float) and np.isnan(mean))

    def test_acc_combine_matches_single(self):
        from spark_vi.eval.topic.concentration import ConcentrationAcc

        rng = np.random.default_rng(0)
        thetas = []
        for _ in range(25):
            raw = rng.random(5)
            thetas.append(raw / raw.sum())

        i = 11
        acc_a = ConcentrationAcc.zeros(n_bins=15, eff_max=5.0)
        for theta in thetas[:i]:
            acc_a.add(theta)
        acc_b = ConcentrationAcc.zeros(n_bins=15, eff_max=5.0)
        for theta in thetas[i:]:
            acc_b.add(theta)

        acc_full = ConcentrationAcc.zeros(n_bins=15, eff_max=5.0)
        for theta in thetas:
            acc_full.add(theta)

        # combine() must not mutate its inputs.
        n_a_before = acc_a.n
        n_b_before = acc_b.n
        combined = acc_a.combine(acc_b)
        assert acc_a.n == n_a_before
        assert acc_b.n == n_b_before

        combined_summary = combined.summary()
        full_summary = acc_full.summary()

        assert combined_summary["n_docs"] == full_summary["n_docs"]
        for key in ("top_mass", "eff_topics"):
            assert combined_summary[key]["mean"] == pytest.approx(full_summary[key]["mean"])
            assert combined_summary[key]["std"] == pytest.approx(full_summary[key]["std"])
            np.testing.assert_array_equal(
                combined_summary[key]["hist"], full_summary[key]["hist"]
            )
            np.testing.assert_allclose(
                combined_summary[key]["bin_edges"], full_summary[key]["bin_edges"]
            )

    def test_acc_combine_mismatch_raises(self):
        from spark_vi.eval.topic.concentration import ConcentrationAcc

        acc_a = ConcentrationAcc.zeros(n_bins=10, eff_max=5.0)
        acc_b = ConcentrationAcc.zeros(n_bins=20, eff_max=5.0)
        with pytest.raises(ValueError):
            acc_a.combine(acc_b)

        acc_c = ConcentrationAcc.zeros(n_bins=10, eff_max=8.0)
        with pytest.raises(ValueError):
            acc_a.combine(acc_c)
