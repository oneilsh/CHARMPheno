"""Tests for the general per-document concentration-heterogeneity + burstiness
diagnostic (spark_vi.eval.topic.concentration_heterogeneity).

Uses MOCK infer_theta callables (never a real STM/LDA inference) so these
tests exercise only the diagnostic's own math: aggregation, spread ratio,
rank correlation (Spearman), and burstiness correlation (Pearson). See
spark_vi.eval.topic.concentration for the underlying top_mass/eff_topics
(inverse-Simpson, Hill 1973 / Jost 2006) definitions being reused here.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

from spark_vi.eval.topic.concentration_heterogeneity import (
    concentration_raw_vs_dedup,
    dedup_counts,
    doc_burstiness,
)


def _doc(indices, counts):
    return SimpleNamespace(
        indices=np.asarray(indices, dtype=np.int32),
        counts=np.asarray(counts, dtype=np.float64),
    )


class TestDocBurstiness:
    def test_repeat_fraction_and_max_token_share(self):
        out = doc_burstiness([0, 1, 2], [3, 1, 1])
        assert out["total"] == pytest.approx(5.0)
        assert out["unique"] == 3
        assert out["repeat_fraction"] == pytest.approx(0.4)
        assert out["max_token_share"] == pytest.approx(0.6)

    def test_all_distinct_zero_repeat_fraction(self):
        out = doc_burstiness([0, 1, 2], [1, 1, 1])
        assert out["repeat_fraction"] == pytest.approx(0.0)
        assert out["max_token_share"] == pytest.approx(1.0 / 3.0)


class TestDedupCounts:
    def test_caps_at_one(self):
        assert np.allclose(dedup_counts(np.array([3.0, 1.0, 2.0])), [1.0, 1.0, 1.0])

    def test_already_all_ones_unchanged(self):
        assert np.allclose(dedup_counts(np.array([1.0, 1.0])), [1.0, 1.0])


class TestConcentrationRawVsDedupMath:
    """Hand (numpy/scipy)-computed reference values from a prescribed
    per-doc theta lookup, keyed on the exact (indices, counts) tuple so the
    mock can return different raw vs. dedup theta without depending on the
    real inference implementation at all.
    """

    def _docs(self):
        return [
            _doc([0, 1, 2], [3, 1, 1]),   # total=5 unique=3 rep=0.4
            _doc([0, 1, 3], [2, 1, 1]),   # total=4 unique=3 rep=0.25
            _doc([0, 2, 3], [1, 1, 1]),   # total=3 unique=3 rep=0.0
            _doc([1, 2, 3], [4, 2, 1]),   # total=7 unique=3 rep=1-3/7
        ]

    def _theta_lookup(self):
        # keyed by (tuple(indices), tuple(counts)) -- unambiguously
        # distinguishes the raw call from the dedup call for every doc here
        # (no doc's raw counts already equal its own dedup'd counts vector
        # except doc index 2, whose theta is set equal for both so the
        # lookup is self-consistent regardless of which key resolves).
        return {
            ((0, 1, 2), (3.0, 1.0, 1.0)): np.array([0.7, 0.2, 0.1]),
            ((0, 1, 2), (1.0, 1.0, 1.0)): np.array([0.5, 0.3, 0.2]),
            ((0, 1, 3), (2.0, 1.0, 1.0)): np.array([0.6, 0.3, 0.1]),
            ((0, 1, 3), (1.0, 1.0, 1.0)): np.array([0.4, 0.35, 0.25]),
            ((0, 2, 3), (1.0, 1.0, 1.0)): np.array([0.5, 0.3, 0.2]),
            ((1, 2, 3), (4.0, 2.0, 1.0)): np.array([0.8, 0.15, 0.05]),
            ((1, 2, 3), (1.0, 1.0, 1.0)): np.array([0.6, 0.25, 0.15]),
        }

    def _infer_theta(self, indices, counts):
        lookup = self._theta_lookup()
        key = (tuple(int(i) for i in indices), tuple(float(c) for c in counts))
        return lookup[key]

    def test_math_matches_hand_computation(self):
        docs = self._docs()
        result = concentration_raw_vs_dedup(docs, self._infer_theta)

        assert result["n_docs"] == 4
        assert result["n_skipped"] == 0

        # Reference values computed directly from the prescribed thetas.
        top_mass_raw_ref = np.array([0.7, 0.6, 0.5, 0.8])
        top_mass_dedup_ref = np.array([0.5, 0.4, 0.5, 0.6])

        assert np.allclose(result["top_mass_raw"], top_mass_raw_ref)
        assert np.allclose(result["top_mass_dedup"], top_mass_dedup_ref)

        expected_spread_ratio = np.std(top_mass_dedup_ref) / np.std(top_mass_raw_ref)
        assert result["spread_ratio_top_mass"] == pytest.approx(expected_spread_ratio)

        expected_rank_corr = spearmanr(top_mass_raw_ref, top_mass_dedup_ref).statistic
        assert result["rank_corr_top_mass"] == pytest.approx(expected_rank_corr)

        expected_p50_raw = float(np.percentile(top_mass_raw_ref, 50))
        expected_p50_dedup = float(np.percentile(top_mass_dedup_ref, 50))
        assert result["top_mass_raw_summary"]["p50"] == pytest.approx(expected_p50_raw)
        assert result["top_mass_dedup_summary"]["p50"] == pytest.approx(expected_p50_dedup)

        repeat_fraction_ref = np.array([1 - 3 / 5, 1 - 3 / 4, 1 - 3 / 3, 1 - 3 / 7])
        assert np.allclose(result["repeat_fraction"], repeat_fraction_ref)
        expected_burstiness_corr = pearsonr(top_mass_raw_ref, repeat_fraction_ref).statistic
        assert result["burstiness_corr_top_mass"] == pytest.approx(expected_burstiness_corr)

    def test_skips_degenerate_docs(self):
        docs = self._docs() + [
            _doc([5], [1.0]),        # single unique token -> skip
            _doc([5, 6], [0.5, 0.5]),  # total < 2 -> skip
        ]
        # Extend the lookup so the four "good" docs still resolve; the
        # skipped docs must never reach infer_theta.
        good_docs = self._docs()

        def infer_theta_or_fail(indices, counts):
            key = (tuple(int(i) for i in indices), tuple(float(c) for c in counts))
            lookup = self._theta_lookup()
            if key not in lookup:
                raise AssertionError(f"infer_theta called on a doc that should have been skipped: {key}")
            return lookup[key]

        result = concentration_raw_vs_dedup(docs, infer_theta_or_fail)
        assert result["n_docs"] == 4
        assert result["n_skipped"] == 2


class TestDiscrimination:
    """Same 6-document burstiness profile in both scenarios; only the mock
    infer_theta (standing in for "how the model responds to burstiness")
    differs. Only RELATIONAL inequalities are asserted -- no magic
    thresholds, per the general-library constraint.
    """

    def _docs(self):
        return [
            _doc([0, 1, 2], [1, 1, 1]),   # rep=0.0    (raw counts == dedup counts)
            _doc([0, 1, 3], [2, 1, 1]),   # rep=0.25
            _doc([0, 2, 3], [3, 1, 1]),   # rep=0.4
            _doc([1, 2, 3], [4, 1, 1]),   # rep=0.5
            _doc([0, 1, 4], [6, 1, 1]),   # rep=0.625
            _doc([0, 3, 4], [9, 1, 1]),   # rep=1 - 3/11
        ]

    def _key(self, indices, counts):
        return (tuple(int(i) for i in indices), tuple(float(c) for c in counts))

    def _genuine_infer_theta(self, indices, counts):
        # Ignores counts entirely: the model's inferred theta reflects
        # genuine cross-topic structure keyed only on which tokens are
        # present, not how many times. So raw and dedup calls -- which
        # share the same indices -- always agree, by construction.
        by_indices = {
            (0, 1, 2): np.array([0.40, 0.35, 0.25]),
            (0, 1, 3): np.array([0.60, 0.25, 0.15]),
            (0, 2, 3): np.array([0.42, 0.38, 0.20]),
            (1, 2, 3): np.array([0.70, 0.20, 0.10]),
            (0, 1, 4): np.array([0.40, 0.35, 0.25]),
            (0, 3, 4): np.array([0.55, 0.25, 0.20]),
        }
        return by_indices[tuple(int(i) for i in indices)]

    def _bursty_infer_theta(self, indices, counts):
        # Peaky on raw (repeated tokens inflate the apparent top topic
        # share) but collapses to a near-flat theta once deduped -- the
        # canonical burstiness artifact this diagnostic targets.
        lookup = {
            self._key([0, 1, 2], [1, 1, 1]): np.array([0.340, 0.330, 0.330]),
            self._key([0, 1, 3], [2, 1, 1]): np.array([0.450, 0.300, 0.250]),
            self._key([0, 1, 3], [1, 1, 1]): np.array([0.352, 0.330, 0.318]),
            self._key([0, 2, 3], [3, 1, 1]): np.array([0.550, 0.250, 0.200]),
            self._key([0, 2, 3], [1, 1, 1]): np.array([0.344, 0.336, 0.320]),
            self._key([1, 2, 3], [4, 1, 1]): np.array([0.650, 0.200, 0.150]),
            self._key([1, 2, 3], [1, 1, 1]): np.array([0.356, 0.330, 0.314]),
            self._key([0, 1, 4], [6, 1, 1]): np.array([0.750, 0.150, 0.100]),
            self._key([0, 1, 4], [1, 1, 1]): np.array([0.331, 0.335, 0.334]),
            self._key([0, 3, 4], [9, 1, 1]): np.array([0.850, 0.100, 0.050]),
            self._key([0, 3, 4], [1, 1, 1]): np.array([0.348, 0.330, 0.322]),
        }
        return lookup[self._key(indices, counts)]

    def test_diagnostic_discriminates_genuine_vs_bursty(self):
        docs = self._docs()
        genuine = concentration_raw_vs_dedup(docs, self._genuine_infer_theta)
        bursty = concentration_raw_vs_dedup(docs, self._bursty_infer_theta)

        assert genuine["n_docs"] == 6
        assert bursty["n_docs"] == 6

        assert genuine["spread_ratio_top_mass"] > bursty["spread_ratio_top_mass"]
        assert bursty["burstiness_corr_top_mass"] > genuine["burstiness_corr_top_mass"]
