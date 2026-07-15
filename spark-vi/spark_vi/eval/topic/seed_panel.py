"""Seed-panel acceptance test for the STM generative concentration scale c.

A gated logistic-normal topic model (OnlineSTM) fits Sigma pinned to a
unit-diagonal correlation R for stability (ADR 0034/0036), discarding the
generative concentration SCALE: the shipped generative covariance is
Sigma_gen = c * R for a scalar c recovered separately (held-out predictive-LL
calibration; see spark_vi.mllib.topic.stm.corpus_heldout_scale_sweep_gated).

This module answers a narrower, harder question about that scale: does it
over-commit on the tool's hard case, a tiny 1-2 token "seed" prefix (a doc
whose only observed tokens are one or two vocabulary codes)? Conditioned
generation for a seed prefix IS a per-doc gated Laplace E-step (Blei &
Lafferty 2007, "A Correlated Topic Model of Science", Ann. Appl. Stat. 1(1)
-- the logistic-normal Laplace approximation) with the seed tokens as the
visible data: infer eta_hat under prior precision (1/c) * safe_inverse(R over
the allowed set), then theta_hat = softmax over the allowed set. A seed with
almost no data should leave the E-step still influenced by the prior;
if it doesn't -- if theta_hat collapses onto the seed's own topic with ~100%
of the mass and every secondary topic reads as impossible -- the tool's
completions will look overconfident on short real records too.

All three exported functions are pure numpy/stdlib composition over the
EXISTING inference primitives (_stm_doc_inference, _gated_mode_theta,
doc_concentration, safe_inverse) -- this module does not reimplement
inference, only wires up the "seed prefix -> conditioned theta -> discovered
concentration" pipeline and the sweep over c.

eff_topics / top_mass definitions and their Hill 1973 / Jost 2006 citations
live in spark_vi.eval.topic.concentration (doc_concentration); not repeated
here.
"""
from __future__ import annotations

import numpy as np

from spark_vi.eval.topic.concentration import doc_concentration
from spark_vi.models.topic._linalg import safe_inverse
from spark_vi.models.topic.stm import _stm_doc_inference
from spark_vi.mllib.topic.stm import _gated_mode_theta


def signature_seeds(
    beta: np.ndarray,
    partition,
    *,
    group: str,
    n_codes: int = 2,
    reference: int | None = None,
    exclude_background: bool = True,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Top-n_codes vocabulary "seed" prefix for each topic of interest.

    For each FOREGROUND topic of ``group`` (``partition.block_indices(group)``)
    -- and, if ``exclude_background`` is False, ALSO each background topic
    (``partition.background_indices()``) -- takes that topic's top ``n_codes``
    vocabulary indices by beta[topic] (descending probability) as a seed
    prefix. The reference topic (if any) is always skipped: pinned at eta=0,
    it has no "own" concentration to recover and seeding it is degenerate.

    ``exclude_background=True`` (the default) restricts seeding to the
    group's foreground topics -- the rare, clinically interesting topics the
    tool exists to surface, and the ones a reviewer would worry about
    over-committing on. ``exclude_background=False`` widens the panel to the
    shared background topics too.

    beta is (K, V), row-normalized topic -> vocab probabilities.

    Returns a list of (topic_id, seed_indices, seed_counts) tuples, one per
    seeded topic: seed_indices is the (n_codes,) array of top vocabulary ids;
    seed_counts is np.ones(n_codes) (each seed code observed once).
    """
    topics = list(partition.block_indices(group))
    if not exclude_background:
        topics = list(partition.background_indices()) + topics
    seeds = []
    for topic_id in topics:
        if reference is not None and topic_id == reference:
            continue
        row = beta[topic_id]
        # argsort ascending, take the last n_codes (largest), then reverse so
        # index 0 is the single highest-probability code.
        top = np.argsort(row)[-n_codes:][::-1]
        seed_indices = np.asarray(top, dtype=np.int64)
        seed_counts = np.ones(n_codes, dtype=np.float64)
        seeds.append((int(topic_id), seed_indices, seed_counts))
    return seeds


def conditioned_theta(
    beta: np.ndarray,
    Gamma: np.ndarray,
    R: np.ndarray,
    partition,
    *,
    group: str,
    seed_indices: np.ndarray,
    seed_counts: np.ndarray,
    c: float,
    x: np.ndarray | None = None,
    reference: int | None = None,
    lbfgs_max_iter: int = 50,
    lbfgs_tol: float = 1e-4,
) -> np.ndarray:
    """Gated conditioned-mode theta for one seed prefix at generative scale c.

    allowed = partition.allowed_indices({group}) (background union the
    group's foreground block -- the reference topic, if it lives in
    background, is automatically included). The prior precision restricted to
    that allowed set is the MARGINAL precision (1/c) * safe_inverse(R[allowed,
    allowed]) (Sigma_gen = c*R, so Sigma_gen_inv restricted to a topic subset
    is the inverse of R's sub-block scaled by 1/c -- see
    _stm_doc_inference's docstring on marginal vs conditional precision).

    beta is used LINEARLY as expElogbeta in the data term (it is the
    lambda-normalized probability matrix, not exp-digamma of a Dirichlet
    parameter -- there is no lambda for a frozen/reconstructed model, and the
    data term only ever multiplies expElogbeta against counts, so a plain
    probability matrix is the correct substitute; see
    corpus_heldout_scale_sweep_gated's docstring for the same distinction).

    x defaults to a covariate vector of zeros with a leading 1.0 (the
    intercept), length P = Gamma.shape[0] -- i.e. the population mean/
    reference-covariate document. Callers with a real covariate design pass
    their own x.

    Returns theta, shape (K,): zero outside the allowed set, softmax(eta_hat)
    over the allowed set inside it (via _gated_mode_theta).
    """
    K = beta.shape[0]
    if x is None:
        P = Gamma.shape[0]
        x = np.zeros(P, dtype=np.float64)
        x[0] = 1.0

    allowed = partition.allowed_indices(frozenset({group}))
    Sigma_inv_allowed = (1.0 / c) * safe_inverse(R[np.ix_(allowed, allowed)])

    eta_hat, _, _ = _stm_doc_inference(
        indices=np.asarray(seed_indices, dtype=np.int64),
        counts=np.asarray(seed_counts, dtype=np.float64),
        expElogbeta=beta,
        Gamma=Gamma,
        Sigma_inv_allowed=Sigma_inv_allowed,
        x=x,
        max_iter=lbfgs_max_iter,
        tol=lbfgs_tol,
        allowed=allowed,
        reference=reference,
    )
    return _gated_mode_theta(eta_hat, allowed, K)


def seed_panel_sweep(
    beta: np.ndarray,
    Gamma: np.ndarray,
    R: np.ndarray,
    partition,
    *,
    group: str,
    c_grid,
    n_codes: int = 2,
    x: np.ndarray | None = None,
    reference: int | None = None,
) -> list[dict]:
    """Run signature_seeds, then conditioned_theta for every (seed, c) pair.

    For each seed topic's own top-n_codes prefix and each c in c_grid: infers
    theta via conditioned_theta, reads off (top_mass, eff_topics) via
    doc_concentration, and records whether the seed recovers its own source
    topic (recovered_topic == argmax(theta)) plus second_mass, the
    second-largest theta entry -- the "how much is left for a secondary
    interest" readout the acceptance test is built around.

    Returns a list of per-(seed, c) dict rows:
    {seed_topic, c, recovered_topic, recovers_self, top_mass, eff_topics,
    second_mass}.
    """
    seeds = signature_seeds(
        beta, partition, group=group, n_codes=n_codes, reference=reference,
    )
    rows = []
    for seed_topic, seed_indices, seed_counts in seeds:
        for c in c_grid:
            theta = conditioned_theta(
                beta, Gamma, R, partition,
                group=group, seed_indices=seed_indices, seed_counts=seed_counts,
                c=c, x=x, reference=reference,
            )
            recovered_topic = int(np.argmax(theta))
            sorted_theta = np.sort(theta)[::-1]
            top_mass, eff_topics = doc_concentration(theta)
            second_mass = float(sorted_theta[1]) if sorted_theta.shape[0] > 1 else 0.0
            rows.append({
                "seed_topic": seed_topic,
                "c": c,
                "recovered_topic": recovered_topic,
                "recovers_self": recovered_topic == seed_topic,
                "top_mass": top_mass,
                "eff_topics": eff_topics,
                "second_mass": second_mass,
            })
    return rows
