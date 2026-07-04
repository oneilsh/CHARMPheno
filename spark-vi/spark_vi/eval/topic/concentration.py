"""Per-document topic-concentration diagnostics.

Given a per-document topic-proportion vector theta (theta_k >= 0, sum(theta)
= 1), two summary statistics describe how "peaked" vs "diffuse" the document
is across topics:

- top_mass = max_k theta_k -- the single largest topic's share.
- eff_topics = 1 / sum_k theta_k^2 -- the inverse-Simpson index, i.e. the
  Hill number of order 2 (Hill 1973, "Diversity and Evenness: A Unifying
  Notation and Its Consequences", Ecology 54(2); see also Jost 2006,
  "Entropy and diversity", Oikos 113(2)): the count of equally-weighted
  topics that would produce the same concentration as theta. A one-hot theta
  has eff_topics = 1; a uniform theta over m topics has eff_topics = m.

This module is pure numpy + stdlib (no Spark, no mllib imports) so it can be
imported from both the eval path and the fit drivers without creating a
circular import; see spark_vi.mllib.topic.stm for the STM producers
(corpus_concentration_stm / corpus_concentration_stm_rdd) that feed
ConcentrationAcc from a distributed corpus.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def doc_concentration(theta: np.ndarray) -> tuple[float, float]:
    """Per-document (top_mass, eff_topics). See module docstring for definitions.

    Assumes theta is a nonnegative topic-proportion vector. If it does not
    sum to a positive value (empty/degenerate doc), returns (nan, nan) so the
    caller can skip it.
    """
    theta = np.asarray(theta, dtype=np.float64)
    s = float(theta.sum())
    if not np.isfinite(s) or s <= 0.0:
        return (float("nan"), float("nan"))
    p = theta / s
    top_mass = float(p.max())
    eff_topics = float(1.0 / np.sum(p * p))
    return (top_mass, eff_topics)


def _percentile_from_hist(hist: np.ndarray, edges: np.ndarray, q: float) -> float:
    """Percentile q (in [0, 100]) via linear interpolation within the
    cumulative-histogram bin that contains it.

    ``hist`` is length n_bins (counts per bin), ``edges`` is length
    n_bins + 1 (bin boundaries). Returns NaN if the histogram is empty
    (total count 0).
    """
    total = float(hist.sum())
    if total <= 0.0:
        return float("nan")
    target = (q / 100.0) * total
    cum = np.cumsum(hist)
    # First bin whose cumulative count reaches the target.
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, len(hist) - 1)
    prev_cum = float(cum[idx - 1]) if idx > 0 else 0.0
    bin_count = float(hist[idx])
    lo, hi = float(edges[idx]), float(edges[idx + 1])
    if bin_count <= 0.0:
        # Empty bin containing the target (can happen at the boundary of an
        # all-zero histogram region) -- fall back to the bin's lower edge.
        return lo
    frac = (target - prev_cum) / bin_count
    frac = min(max(frac, 0.0), 1.0)
    return lo + frac * (hi - lo)


@dataclass
class ConcentrationAcc:
    """Scalable, combinable per-document concentration accumulator.

    Holds fixed-bin histograms (so percentiles scale to any corpus size
    without collecting per-doc scalars to the driver -- mirrors the
    histogram approach in charmpheno.charmpheno.export.theta_aggregates)
    plus streaming sums for mean/std, over both top_mass (support [0, 1])
    and eff_topics (support [1, eff_max]). All state is plain numpy/float,
    so an instance is picklable and safe to move across Spark partitions
    and treeReduce.
    """

    n_bins: int
    eff_max: float
    n: int = 0
    top_sum: float = 0.0
    top_sumsq: float = 0.0
    top_hist: np.ndarray = field(default=None)
    eff_sum: float = 0.0
    eff_sumsq: float = 0.0
    eff_hist: np.ndarray = field(default=None)

    @classmethod
    def zeros(cls, n_bins: int, eff_max: float) -> "ConcentrationAcc":
        """All-zero histograms/sums for n_bins bins, eff_topics support
        [1, eff_max] (eff_max is typically K, the topic count). Guards
        eff_max > 1 (a degenerate K=1 corpus would otherwise produce an
        empty/invalid bin range) by nudging eff_max to 1 + 1e-9."""
        if not (eff_max > 1.0):
            eff_max = 1.0 + 1e-9
        return cls(
            n_bins=n_bins,
            eff_max=eff_max,
            n=0,
            top_sum=0.0,
            top_sumsq=0.0,
            top_hist=np.zeros(n_bins, dtype=np.int64),
            eff_sum=0.0,
            eff_sumsq=0.0,
            eff_hist=np.zeros(n_bins, dtype=np.int64),
        )

    def add(self, theta: np.ndarray) -> None:
        """Accumulate one document's (top_mass, eff_topics). Degenerate docs
        (doc_concentration returns nan) are skipped entirely -- n is not
        incremented and no histogram bin is touched."""
        top, eff = doc_concentration(theta)
        if not (np.isfinite(top) and np.isfinite(eff)):
            return
        self.n += 1
        self.top_sum += top
        self.top_sumsq += top * top
        self.eff_sum += eff
        self.eff_sumsq += eff * eff

        top_bin = int(np.clip(top * self.n_bins, 0, self.n_bins - 1))
        self.top_hist[top_bin] += 1

        eff_bin = int(np.clip(
            (eff - 1.0) / (self.eff_max - 1.0) * self.n_bins, 0, self.n_bins - 1
        ))
        self.eff_hist[eff_bin] += 1

    def combine(self, other: "ConcentrationAcc") -> "ConcentrationAcc":
        """Return a NEW accumulator summing self and other (functional --
        neither input is mutated), so it is safe as a Spark treeReduce
        combiner. Raises ValueError if n_bins or eff_max differ."""
        if self.n_bins != other.n_bins:
            raise ValueError(
                f"ConcentrationAcc.combine: n_bins mismatch ({self.n_bins} != {other.n_bins})"
            )
        if self.eff_max != other.eff_max:
            raise ValueError(
                f"ConcentrationAcc.combine: eff_max mismatch ({self.eff_max} != {other.eff_max})"
            )
        return ConcentrationAcc(
            n_bins=self.n_bins,
            eff_max=self.eff_max,
            n=self.n + other.n,
            top_sum=self.top_sum + other.top_sum,
            top_sumsq=self.top_sumsq + other.top_sumsq,
            top_hist=self.top_hist + other.top_hist,
            eff_sum=self.eff_sum + other.eff_sum,
            eff_sumsq=self.eff_sumsq + other.eff_sumsq,
            eff_hist=self.eff_hist + other.eff_hist,
        )

    def summary(self) -> dict:
        """JSON-serializable summary (plain floats/ints/lists, no ndarrays).

        mean = sum/n; std = sqrt(max(sumsq/n - mean^2, 0)); percentiles via
        cumulative-histogram linear interpolation (_percentile_from_hist).
        If n == 0, means/percentiles are None (still valid JSON) rather than
        computed via a zero division.
        """
        top_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        eff_edges = np.linspace(1.0, self.eff_max, self.n_bins + 1)

        def _stat_block(total_sum, total_sumsq, hist, edges):
            if self.n == 0:
                return {
                    "mean": None,
                    "std": None,
                    "p10": None, "p25": None, "p50": None,
                    "p75": None, "p90": None,
                    "hist": hist.tolist(),
                    "bin_edges": edges.tolist(),
                }
            mean = total_sum / self.n
            std = float(np.sqrt(max(total_sumsq / self.n - mean * mean, 0.0)))
            pcts = {
                f"p{q}": _percentile_from_hist(hist, edges, q)
                for q in (10, 25, 50, 75, 90)
            }
            return {
                "mean": float(mean),
                "std": std,
                **pcts,
                "hist": hist.tolist(),
                "bin_edges": edges.tolist(),
            }

        return {
            "n_docs": int(self.n),
            "top_mass": _stat_block(self.top_sum, self.top_sumsq, self.top_hist, top_edges),
            "eff_topics": _stat_block(self.eff_sum, self.eff_sumsq, self.eff_hist, eff_edges),
        }
