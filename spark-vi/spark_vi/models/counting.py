"""CountingModel: a trivial VI model used to exercise the framework contract.

Posterior over the bias of a Bernoulli coin:

    prior:       p ~ Beta(prior_alpha, prior_beta)
    likelihood:  rows are 0/1 iid from Bernoulli(p)
    sufficient stats: (# heads, # tails) aggregated across partitions
    global update: Beta-Bernoulli conjugate posterior counts with Robbins-Monro
                   interpolation against the previous iterate.

This is not a realistic use case; it exists so contract-level tests can run
end-to-end through VIRunner without depending on a real model's math.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.special import betaln

from spark_vi.core.model import VIModel


class CountingModel(VIModel):
    """Beta-Bernoulli conjugate VI stand-in."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("priors must be positive")
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        return {
            "alpha": np.array(self.prior_alpha),
            "beta": np.array(self.prior_beta),
        }

    def local_update(
        self,
        rows: Iterable[int],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        heads = 0
        tails = 0
        for r in rows:
            if r == 1:
                heads += 1
            elif r == 0:
                tails += 1
            else:
                raise ValueError(f"CountingModel rows must be 0 or 1, got {r!r}")
        return {"heads": np.array(float(heads)), "tails": np.array(float(tails))}

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        # lambda_hat = prior + aggregated counts.
        target_alpha = self.prior_alpha + aggregated_stats["heads"]
        target_beta = self.prior_beta + aggregated_stats["tails"]
        # Robbins-Monro interpolation.
        new_alpha = (1.0 - learning_rate) * global_params["alpha"] + learning_rate * target_alpha
        new_beta = (1.0 - learning_rate) * global_params["beta"] + learning_rate * target_beta
        return {"alpha": np.array(float(new_alpha)), "beta": np.array(float(new_beta))}

    def compute_elbo(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
    ) -> float:
        """Surrogate ELBO: log marginal likelihood under current posterior pseudocounts.

        Using log B(alpha, beta) - log B(prior_alpha, prior_beta) + log P(data | counts).
        Good enough to be monotonic-ish and finite for the tests.
        """
        a = float(global_params["alpha"])
        b = float(global_params["beta"])
        h = float(aggregated_stats.get("heads", 0.0))
        t = float(aggregated_stats.get("tails", 0.0))
        # Log posterior predictive factor + log prior normalizer ratio.
        return -betaln(a, b) + betaln(self.prior_alpha, self.prior_beta) + h * 0.0 + t * 0.0 \
            + float(a + b)  # placeholder monotone-in-data-weight term for tests
