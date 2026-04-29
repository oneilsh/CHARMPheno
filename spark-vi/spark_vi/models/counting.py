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
from scipy.special import betaln, digamma

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
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        # lambda_hat = prior + (pre-scaled) aggregated counts.
        target_alpha = self.prior_alpha + target_stats["heads"]
        target_beta = self.prior_beta + target_stats["tails"]
        # Robbins-Monro interpolation.
        new_alpha = (1.0 - learning_rate) * global_params["alpha"] + learning_rate * target_alpha
        new_beta = (1.0 - learning_rate) * global_params["beta"] + learning_rate * target_beta
        return {"alpha": np.array(float(new_alpha)), "beta": np.array(float(new_beta))}

    def compute_elbo(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
    ) -> float:
        """Exact Beta-Bernoulli ELBO for q(p) = Beta(alpha, beta).

        For prior Beta(a0, b0) and observed counts (h, t):

            ELBO(q) = E_q[log p(x | p)] - KL(q || prior)

        with:
            E_q[log p(x | p)] = h * (psi(a) - psi(a+b))
                              + t * (psi(b) - psi(a+b))

            KL(Beta(a,b) || Beta(a0,b0))
                = (a - a0) * (psi(a) - psi(a+b))
                + (b - b0) * (psi(b) - psi(a+b))
                + betaln(a0, b0) - betaln(a, b)

        At the exact posterior (a = a0 + h, b = b0 + t) the ELBO equals the
        log marginal likelihood log p(x) = betaln(a0+h, b0+t) - betaln(a0, b0).
        """
        a = float(global_params["alpha"])
        b = float(global_params["beta"])
        h = float(aggregated_stats.get("heads", 0.0))
        t = float(aggregated_stats.get("tails", 0.0))
        a0 = self.prior_alpha
        b0 = self.prior_beta

        psi_ab = digamma(a + b)
        e_log_p = digamma(a) - psi_ab          # E_q[log p]
        e_log_1mp = digamma(b) - psi_ab        # E_q[log(1 - p)]

        expected_log_likelihood = h * e_log_p + t * e_log_1mp
        kl_q_prior = (
            (a - a0) * e_log_p
            + (b - b0) * e_log_1mp
            + betaln(a0, b0)
            - betaln(a, b)
        )
        return float(expected_log_likelihood - kl_q_prior)
