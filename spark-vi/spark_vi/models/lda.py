"""VanillaLDA: Hoffman 2010 Online LDA as a VIModel.

Generative model for each document d (= one row in the RDD):
    theta_d ~ Dirichlet(alpha · 1_K)
    for n in 1..N_d:
        z_dn ~ Categorical(theta_d)
        w_dn ~ Categorical(beta_{z_dn})

Globally:
    beta_k ~ Dirichlet(eta · 1_V)

Variational mean field:
    q(beta_k) = Dirichlet(lambda_k)         # global, shape (K, V)
    q(theta_d) = Dirichlet(gamma_d)         # local, shape (K,)
    q(z_dn) = Categorical(phi_dn)           # local, collapsed via Lee/Seung 2001

Symbols:
    K           number of topics
    V           vocabulary size
    D           number of documents (corpus_size)
    N_d         total tokens in document d (with repeats)
    lambda      (K, V) global variational Dirichlet for topic-word
    gamma_d     (K,) local variational Dirichlet for doc-topic
    expElogbeta (K, V) precomputed exp(E[log beta_kv]) under q(beta)
    expElogthetad (K,) precomputed exp(E[log theta_dk]) under q(theta_d)
    phi_norm    (n_unique,) implicit phi-normalizer for the Lee/Seung trick
    alpha, eta  symmetric Dirichlet concentrations

References:
    Hoffman, Blei, Bach 2010. Online learning for LDA. NIPS.
    Hoffman, Blei, Wang, Paisley 2013. Stochastic VI. JMLR.
    Lee, Seung 2001. Algorithms for non-negative matrix factorization. NIPS.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from spark_vi.core.model import VIModel
from spark_vi.core.types import BOWDocument


class VanillaLDA(VIModel):
    """Vanilla LDA fittable by VIRunner with mini-batch SVI.

    Hyperparameters match Spark MLlib's pyspark.ml.clustering.LDA defaults
    so head-to-head comparisons are apples-to-apples.
    """

    def __init__(
        self,
        K: int,
        vocab_size: int,
        alpha: float | None = None,
        eta: float | None = None,
        gamma_shape: float = 100.0,
        cavi_max_iter: int = 100,
        cavi_tol: float = 1e-3,
    ) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if alpha is None:
            alpha = 1.0 / K
        if eta is None:
            eta = 1.0 / K
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if gamma_shape <= 0:
            raise ValueError(f"gamma_shape must be > 0, got {gamma_shape}")
        if cavi_max_iter < 1:
            raise ValueError(f"cavi_max_iter must be >= 1, got {cavi_max_iter}")
        if cavi_tol <= 0:
            raise ValueError(f"cavi_tol must be > 0, got {cavi_tol}")

        self.K = int(K)
        self.V = int(vocab_size)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.gamma_shape = float(gamma_shape)
        self.cavi_max_iter = int(cavi_max_iter)
        self.cavi_tol = float(cavi_tol)

    # Contract methods (filled in over subsequent tasks).

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """Random Gamma(gamma_shape, 1/gamma_shape) init for lambda (K, V).

        gamma_shape=100 (MLlib default) gives draws tightly concentrated near 1;
        this is the variational analog of an "uninformative" topic-word prior
        with a small amount of symmetry-breaking noise.
        """
        lam = np.random.gamma(
            shape=self.gamma_shape,
            scale=1.0 / self.gamma_shape,
            size=(self.K, self.V),
        )
        return {"lambda": lam}

    def local_update(
        self,
        rows: Iterable[BOWDocument],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("Implemented in Task 8")

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("Implemented in Task 10")
