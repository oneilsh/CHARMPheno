"""Task 6: gated full-batch PG-VI driver — end-to-end planted recovery gate.

The corpus (`gated_ln_corpus`) is softmax-planted (logistic-normal over the doc's
allowed set), while PGSTMVI fits the nested stick-breaking logistic-normal. Only
beta recovery is cross-model comparable — Sigma is checked for shape/finiteness/
boundedness only (never asserted ~ planted Sigma_true)."""
import numpy as np

from spark_vi.models.topic.pg_stm import PGSTMVI, stick_layout
from tests._stm_synth import (gated_ln_corpus, planted_recovery,
                              foreground_recovers_group)


def _corpus(seed=0):
    docs, part, Sigma_true, beta = gated_ln_corpus(
        group_weights={"A": 0.5, "B": 0.5}, fg_per_group=1, bg_k=2,
        V=60, D=600, doc_len=40, seed=seed)
    return docs, {"beta": beta, "Sigma": Sigma_true}, part


def test_pgvi_recovers_planted_structure():
    docs, planted, part = _corpus(seed=0)
    P = docs[0].x.shape[0]
    out = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=150, seed=0).fit(docs)
    assert planted_recovery(out["beta"], planted["beta"]) >= 0.75
    for g in part.groups:
        assert foreground_recovers_group(out["beta"], part, g, planted["beta"])
    assert out["Sigma"].shape == (part.K - 1, part.K - 1)
    assert np.all(np.isfinite(out["Sigma"])) and np.max(np.abs(out["Sigma"])) < 1e3
    assert len(out["sigma_max_trace"]) == 150


def test_stick_layout_indices():
    docs, planted, part = _corpus(seed=0)
    lay = stick_layout(part)
    # B=2 background topics -> 1 background stick (global index 0)
    assert list(lay["bg_sticks"]) == [0]
    # group A: gate=1, no fg sticks (m_g=1); active = [bg stick 0, gate 1]
    assert lay["groups"]["A"]["gate"] == 1
    assert list(lay["groups"]["A"]["fg_sticks"]) == []
    assert list(lay["groups"]["A"]["active"]) == [0, 1]
    assert list(lay["groups"]["A"]["allowed"]) == [0, 1, 2]
    # group B: gate=2
    assert lay["groups"]["B"]["gate"] == 2
    assert list(lay["groups"]["B"]["active"]) == [0, 2]
    assert list(lay["groups"]["B"]["allowed"]) == [0, 1, 3]


def test_pgvi_output_shapes():
    docs, planted, part = _corpus(seed=1)
    P = docs[0].x.shape[0]
    out = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=20, seed=0).fit(docs)
    K, V = part.K, 60
    D = len(docs)
    assert out["beta"].shape == (K, V)
    assert out["Gamma"].shape == (P, K - 1)
    assert out["Sigma"].shape == (K - 1, K - 1)
    assert out["psi_mean"].shape == (D, K - 1)
    assert out["psi_var"].shape == (D, K - 1)
    # inactive stick entries left at prior mean / var
    assert np.all(np.isfinite(out["psi_mean"]))
    assert np.all(np.isfinite(out["psi_var"]))


def test_sigma_mode_mle_runs():
    docs, planted, part = _corpus(seed=2)
    P = docs[0].x.shape[0]
    out = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=20, seed=0,
                  sigma_mode="mle").fit(docs)
    assert np.all(np.isfinite(out["Sigma"]))
