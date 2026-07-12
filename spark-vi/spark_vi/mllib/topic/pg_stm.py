"""Distributed PG-STM (sub-project 2).

Phase 1 — ``StreamingPGSTM``: distributed mini-batch PG-SVI over an RDD of
``STMDocument`` (the runaway-cure test; ``sigma_mode`` iw|mle). The per-document math
lives in ``spark_vi.models.topic.pg_stm`` (pure functions); this module only ORCHESTRATES
it across Spark using the same ``mapPartitions(_local).treeReduce(_combine)`` idiom as
``spark_vi.mllib.topic.stm``. Each worker folds ``pg_accumulate_doc(pg_estep_doc(...))``
over its partition into ONE small ``PGSuffStats``; the tree-reduce sums them; the driver
runs the shared ``pg_mstep``. So ``StreamingPGSTM(batch="all").fit`` reproduces the
single-machine ``PGSTMVI.fit`` by construction (Hoffman et al. 2013, "Stochastic
Variational Inference", JMLR 14:1303-1347).

Phase 2 — ``pg_stm_sigma_readout``: the comorbidity Sigma-correlation read-out (which
mean-field VI cannot produce — insight 0044). Sigma is a single small (K-1)x(K-1) global,
so it does NOT need full-corpus distribution: a large driver-collected SUBSAMPLE, run
through the VALIDATED exact single-machine ``pg_stm_gibbs``, estimates it (recovers the
planted stick correlation where mean-field flips its sign). A fully-distributed exact
sampler is deferred — a naive port drifts (label-switching across the shared background
block was not pinned), and the subsample is statistically sufficient for a global Sigma.
"""
from __future__ import annotations

import numpy as np

from spark_vi.models.topic.pg_stm import (
    pg_estep_doc, pg_empty_stats, pg_accumulate_doc, pg_combine_stats, pg_mstep,
    stick_layout, pg_stm_gibbs,
)


def _scale_stats(st, f):
    """Scale every sufficient statistic by ``f`` (the SVI D/|batch| inflation so a
    mini-batch estimates the full-corpus stats)."""
    return {"wts": st["wts"] * f, "XtX": st["XtX"] * f, "XtM": st["XtM"] * f,
            "S": st["S"] * f,
            "group_counts": {g: n * f for g, n in st["group_counts"].items()},
            "D": st["D"] * f}


def _blend_stats(a, b, rho):
    """Robbins-Monro convex blend (1-rho)*a + rho*b of two stat dicts (natural-gradient
    step on the sufficient statistics)."""
    keys = set(a["group_counts"]) | set(b["group_counts"])
    gc = {g: (1.0 - rho) * a["group_counts"].get(g, 0.0)
          + rho * b["group_counts"].get(g, 0.0) for g in keys}
    return {"wts": (1.0 - rho) * a["wts"] + rho * b["wts"],
            "XtX": (1.0 - rho) * a["XtX"] + rho * b["XtX"],
            "XtM": (1.0 - rho) * a["XtM"] + rho * b["XtM"],
            "S": (1.0 - rho) * a["S"] + rho * b["S"],
            "group_counts": gc, "D": (1.0 - rho) * a["D"] + rho * b["D"]}


class StreamingPGSTM:
    """Distributed mini-batch PG-SVI for the gated nested stick-breaking logistic-normal
    topic model. Mirrors ``StreamingSTM`` but wraps the PG-VI kernel; the IW block-Sigma
    M-step (``sigma_mode="iw"``) is the runaway cure, ``"mle"`` the un-regularized
    contrast arm.

    Constructor mirrors ``PGSTMVI`` defaults so ``StreamingPGSTM(batch="all").fit``
    reproduces ``PGSTMVI.fit`` exactly (same init, same M-step)."""

    def __init__(self, K, V, partition, *, P, beta_eta=0.1, gamma_ridge=1e-6,
                 sigma_mode="iw", Psi0_scale=1.0, nu0=None, inner_rounds=8,
                 inner_tol=1e-3, seed=0):
        if sigma_mode not in ("iw", "mle"):
            raise ValueError(f"sigma_mode must be 'iw' or 'mle', got {sigma_mode!r}")
        self.K = int(K); self.V = int(V); self.partition = partition
        self.P = int(P)
        self.beta_eta = float(beta_eta); self.gamma_ridge = float(gamma_ridge)
        self.sigma_mode = sigma_mode
        self.Psi0_scale = float(Psi0_scale)
        self.nu0 = float(nu0) if nu0 is not None else float((K - 1) + 2)
        self.inner_rounds = int(inner_rounds); self.inner_tol = float(inner_tol)
        self.seed = int(seed)
        self.layout = stick_layout(partition)

    def _init_globals(self):
        """PGSTMVI.fit's init verbatim: beta from smoothed uniform-random counts (seed),
        Gamma=0, Sigma=I."""
        rng = np.random.default_rng(self.seed)
        beta = rng.random((self.K, self.V)) + self.beta_eta
        beta /= beta.sum(axis=1, keepdims=True)
        Gamma = np.zeros((self.P, self.K - 1))
        Sigma = np.eye(self.K - 1)
        return beta, Gamma, Sigma

    def _reduce_stats(self, work_rdd, log_beta, Gamma, Sigma, depth):
        """One distributed E-step + sufficient-stat reduction over ``work_rdd``."""
        K, B, V, P = self.K, self.layout["B"], self.V, self.P
        layout = self.layout
        groups = tuple(self.partition.groups)
        inner_rounds, inner_tol = self.inner_rounds, self.inner_tol
        bc = work_rdd.context.broadcast((log_beta, Gamma, Sigma))

        def _local(rows, _bc=bc, _K=K, _B=B, _V=V, _P=P, _layout=layout,
                   _groups=groups, _ir=inner_rounds, _it=inner_tol):
            lb, G, Sg = _bc.value
            st = pg_empty_stats(_K, _V, _P, _groups)
            for doc in rows:
                (g,) = tuple(doc.groups)
                estep = pg_estep_doc(doc, _layout["groups"][g], lb, G, Sg,
                                     K=_K, B=_B, inner_rounds=_ir, inner_tol=_it)
                pg_accumulate_doc(st, doc, estep[:6], K=_K)   # drop n_clips
            return [st]

        return work_rdd.mapPartitions(_local).treeReduce(pg_combine_stats, depth=depth)

    def fit(self, doc_rdd, *, max_iter=100, batch="all", tau0=64.0, kappa=0.7,
            depth=2, on_iteration=None):
        """Fit over an RDD of STMDocument. ``batch="all"`` -> full-batch (rho=1 each
        iter, one reduce over all docs). A float in (0,1] -> mini-batch fraction with
        Robbins-Monro rho_t=(t+tau0)^-kappa blend of the sufficient stats scaled by
        D/|batch|. Returns {beta, Gamma, Sigma, sigma_max_trace}."""
        beta, Gamma, Sigma = self._init_globals()
        full_batch = (batch == "all")
        D = None if full_batch else int(doc_rdd.count())
        running = None
        sigma_max_trace = []
        for t in range(int(max_iter)):
            log_beta = np.log(beta)
            if full_batch:
                work = doc_rdd
            else:
                work = doc_rdd.sample(False, float(batch), seed=self.seed + t)
            stats = self._reduce_stats(work, log_beta, Gamma, Sigma, depth)
            if full_batch:
                mstep_stats = stats
            else:
                bsize = max(int(stats["D"]), 1)
                hat = _scale_stats(stats, D / bsize)
                rho = (t + tau0) ** (-kappa)
                running = hat if running is None else _blend_stats(running, hat, rho)
                mstep_stats = running
            beta, Gamma, Sigma = pg_mstep(
                mstep_stats, beta_eta=self.beta_eta, gamma_ridge=self.gamma_ridge,
                sigma_mode=self.sigma_mode, Psi0_scale=self.Psi0_scale, nu0=self.nu0,
                partition=self.partition, layout=self.layout)
            sigma_max_trace.append(float(np.max(np.abs(Sigma))))
            if on_iteration is not None:
                on_iteration(t, {"beta": beta, "Gamma": Gamma, "Sigma": Sigma})
        return {"beta": beta, "Gamma": Gamma, "Sigma": Sigma,
                "sigma_max_trace": sigma_max_trace}


def pg_stm_sigma_readout(doc_rdd, *, K, V, partition, P, subsample_n=20000,
                         n_iter=600, burn=300, Psi0_scale=1.0, nu0=None,
                         gamma_ridge=1e-6, beta_eta=0.1, seed=0):
    """Phase-2 comorbidity Sigma-correlation read-out. Collects a driver-side SUBSAMPLE
    of ``subsample_n`` documents (Sigma is a small (K-1)x(K-1) global — a representative
    subsample estimates it) and runs the VALIDATED exact single-machine
    ``pg_stm_gibbs`` on it, returning the trustworthy-correlation Sigma (+ beta, Gamma,
    Sigma_samples). This is the correlation read-out mean-field VI cannot produce
    (insight 0044). ``subsample_n`` <= 0 uses the whole corpus (only for small corpora).

    Reuses the exact sampler rather than a bespoke distributed one because a naive
    distributed exact-Gibbs drifts (unpinned label-switching across the shared background
    block) — deferred as a future optimization; the subsample is statistically sufficient
    for a single global Sigma."""
    if subsample_n and subsample_n > 0:
        total = doc_rdd.count()
        if total > subsample_n:
            frac = min(1.0, (subsample_n * 1.2) / total)     # oversample, then trim
            docs = doc_rdd.sample(False, frac, seed=seed).take(int(subsample_n))
        else:
            docs = doc_rdd.collect()
    else:
        docs = doc_rdd.collect()
    return pg_stm_gibbs(docs, K=K, V=V, partition=partition, P=P, n_iter=n_iter,
                        burn=burn, seed=seed, Psi0_scale=Psi0_scale, nu0=nu0,
                        gamma_ridge=gamma_ridge, beta_eta=beta_eta)
