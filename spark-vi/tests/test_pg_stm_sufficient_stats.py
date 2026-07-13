"""Task 1 — the PG M-step refactored into distributable sufficient statistics.

The single-machine `PGSTMVI.fit` accumulates per-doc contributions into a few
global-shaped arrays (word-topic stats, the covariate moments XᵀX / XᵀM for the ridge
regression, the block scatter S, group counts, D) and then runs a pure M-step. Pulling
those into standalone functions lets the distributed driver reduce the SAME small arrays
via mapPartitions+treeReduce and call the SAME M-step — so full-batch StreamingPGSTM is
byte-identical to PGSTMVI by construction, not by parallel re-derivation.

These tests pin the two things the refactor must preserve exactly: (1) the moment-form
Γ equals the stacked-matrix ridge regression, and (2) accumulate→combine→M-step over a
corpus reproduces `PGSTMVI`'s own one-iteration M-step outputs.
"""
import numpy as np

from spark_vi.models.topic.pg_stm import (
    gamma_ridge, pg_gamma_ridge_moments,
    PGSTMVI, pg_empty_stats, pg_accumulate_doc, pg_combine_stats, pg_mstep,
    stick_layout,
)
from spark_vi.models.topic.types import STMDocument
from tests._stm_synth import gated_ln_corpus


def test_gamma_ridge_moment_form_matches_stacked():
    """Γ from accumulated moments (XᵀX, XᵀM) equals Γ from the stacked (M, X) form."""
    rng = np.random.default_rng(0)
    D, P, Km1 = 50, 3, 5
    X = rng.normal(size=(D, P))
    M = rng.normal(size=(D, Km1))
    ref = gamma_ridge(M, X, ridge=1e-6)
    got = pg_gamma_ridge_moments(X.T @ X, X.T @ M, ridge=1e-6)
    assert np.allclose(ref, got, atol=1e-12)


def test_suffstats_combine_is_associative_sum():
    """pg_combine_stats is a pure elementwise sum (a valid treeReduce combiner)."""
    docs, part, _St, _b = gated_ln_corpus(
        group_weights={"A": 0.6, "B": 0.4}, fg_per_group=1, bg_k=3,
        V=40, D=20, doc_len=30, seed=0)
    K, V = part.K, 40
    P = docs[0].x.shape[0]
    layout = stick_layout(part)
    model = PGSTMVI(K=K, V=V, partition=part, P=P, n_iter=1, seed=0)
    log_beta = np.log(np.full((K, V), 1.0 / V))
    Gamma = np.zeros((P, K - 1))
    Sigma = np.eye(K - 1)

    def build(subset):
        st = pg_empty_stats(K, V, P, part.groups)
        for doc in subset:
            (g,) = tuple(doc.groups)
            estep = model._e_step_doc(doc, layout["groups"][g], log_beta, Gamma, Sigma)
            pg_accumulate_doc(st, doc, estep, K=K)
        return st

    whole = build(docs)
    combined = pg_combine_stats(build(docs[:11]), build(docs[11:]))
    assert np.allclose(whole["wts"], combined["wts"])
    assert np.allclose(whole["XtX"], combined["XtX"])
    assert np.allclose(whole["XtM"], combined["XtM"])
    assert np.allclose(whole["S"], combined["S"])
    assert whole["group_counts"] == combined["group_counts"]
    assert whole["D"] == combined["D"] == len(docs)


def test_accumulate_then_mstep_matches_single_machine_one_iter():
    """One full-corpus accumulate → pg_mstep reproduces PGSTMVI's first-iteration
    β/Γ/Σ (the M-step the streaming driver will call)."""
    docs, part, _St, _b = gated_ln_corpus(
        group_weights={"A": 0.6, "B": 0.4}, fg_per_group=1, bg_k=3,
        V=40, D=60, doc_len=30, seed=1)
    K, V = part.K, 40
    P = docs[0].x.shape[0]
    layout = stick_layout(part)
    model = PGSTMVI(K=K, V=V, partition=part, P=P, n_iter=1, seed=0)

    # PGSTMVI's own init (mirror fit's first iteration exactly).
    rng = np.random.default_rng(0)
    beta0 = rng.random((K, V)) + model.beta_eta
    beta0 /= beta0.sum(axis=1, keepdims=True)
    Gamma0 = np.zeros((P, K - 1))
    Sigma0 = np.eye(K - 1)

    st = pg_empty_stats(K, V, P, part.groups)
    for doc in docs:
        (g,) = tuple(doc.groups)
        estep = model._e_step_doc(doc, layout["groups"][g], np.log(beta0), Gamma0, Sigma0)
        pg_accumulate_doc(st, doc, estep, K=K)
    beta, Gamma, Sigma = pg_mstep(
        st, beta_eta=model.beta_eta, gamma_ridge=model.gamma_ridge,
        sigma_mode="iw", Psi0_scale=model.Psi0_scale, nu0=model.nu0,
        partition=part, layout=layout)

    # PGSTMVI.fit for one iter from the identical init:
    ref = PGSTMVI(K=K, V=V, partition=part, P=P, n_iter=1, seed=0).fit(docs)
    assert np.allclose(beta, ref["beta"], atol=1e-10)
    assert np.allclose(Gamma, ref["Gamma"], atol=1e-10)
    assert np.allclose(Sigma, ref["Sigma"], atol=1e-10)


def test_pg_accumulate_doc_tolerates_group_less_document():
    """Deterministic structure check; no empirical or transfer claim. A background-only
    document (no foreground group) accumulates its word/covariate/scatter stats and
    increments D, but touches no per-group count."""
    K, V, P = 4, 5, 1
    stats = pg_empty_stats(K, V, P, groups=("A",))
    doc = STMDocument(indices=np.array([0, 1], dtype=np.int32),
                      counts=np.array([2.0, 1.0]), length=3,
                      x=np.array([1.0]), groups=frozenset())
    active = np.array([0, 1], dtype=np.int64)          # 2 background sticks
    m = np.array([0.1, -0.2]); Vd = np.eye(2)
    phi = np.array([[0.6, 0.4, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0]])   # (n_tok, K)
    allowed = np.array([0, 1, 2], dtype=np.int64); mu_active = np.zeros(2)
    pg_accumulate_doc(stats, doc, (m, Vd, phi, active, allowed, mu_active), K=K)
    assert stats["D"] == 1
    assert stats["group_counts"] == {"A": 0}           # no group incremented
    assert np.allclose(stats["XtX"], np.array([[1.0]]))  # x x^T for x=[1]
