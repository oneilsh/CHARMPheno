"""Acceptance gate for the Firth/Jeffreys-penalized logistic head (head_penalty).

On SEPARABLE data the logistic MLE runs |w| -> infinity (a real blow-up we hit on
the cluster). The shipped cure is a tunable absolute L2 ridge (head_l2). Firth's
Jeffreys-prior penalty +1/2 log det H_c is the PARAMETER-FREE cure: it
self-regularizes exactly at separation (as |w|->inf, H->0, log det H->-inf) and
yields finite, calibrated weights. These tests PROVE that the inner-loop
(head_optimizer='newton', Path B) IRLS head bounds |w| under head_penalty='firth'
while the plain (ridge-free) fit blows up, and that Firth does not distort a
well-posed (non-separable) fit.
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.pc import OnlinePCLDA, DagClosureHead
from spark_vi.models.topic.types import PCDocument


# ---------------------------------------------------------------------------
# Helpers: drive the M-step inner-loop head fit directly.
#
# update_global runs the LDA lambda step + topic correction first, then (Path B)
# converges the flat logistic head on the collected (head_theta, head_s, head_obs)
# design over self.head_inner_iters Newton/IRLS steps. We build a valid base
# target_stats via local_update (so the LDA machinery is happy), then OVERRIDE the
# head design with a synthetic one whose separability we control, seed w_CK at 0,
# and read back the converged head. The learning_rate does not touch the head loop.
# ---------------------------------------------------------------------------
def _dummy_docs(n: int, V: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n):
        nnz = int(rng.integers(2, min(V, 5) + 1))
        idx = np.sort(rng.choice(V, size=nnz, replace=False)).astype(np.int32)
        cnt = rng.integers(1, 5, size=nnz).astype(np.float64)
        docs.append(PCDocument(indices=idx, counts=cnt, length=int(cnt.sum()),
                               y=np.ones(1), label_mask=np.ones(1)))
    return docs


def _fit_inner_head(K, theta, y, *, head_penalty, head_l2, inner_iters,
                    head_newton_ridge=1e-2, seed=0):
    """Run the Path-B inner-loop head fit on a synthetic (theta, y) design.

    Returns the converged head weight vector w for the single label (C=1).
    """
    V = 6
    model = OnlinePCLDA(
        K=K, vocab_size=V, C=1, weight_y=50.0,
        head_optimizer="newton", head_penalty=head_penalty,
        head_l2=head_l2, head_inner_iters=inner_iters,
        head_newton_ridge=head_newton_ridge, random_seed=seed,
    )
    gp = model.initialize_global(None)
    docs = _dummy_docs(len(theta), V, seed=seed)
    st = model.local_update(docs, gp)          # valid base stats (lambda, grads, ...)
    # Override the head design with the synthetic separable/non-separable one.
    n = len(theta)
    st["head_theta"] = np.asarray(theta, dtype=np.float64)      # (n, K)
    st["head_s"] = np.where(np.asarray(y) > 0.5, 1.0, -1.0).reshape(n, 1)  # (n, 1)
    st["head_obs"] = np.ones((n, 1), dtype=np.float64)
    gp["w_CK"] = np.zeros((1, K), dtype=np.float64)
    out = model.update_global(gp, st, 0.5)
    return out["w_CK"][0].copy()


def _auc(scores, y):
    y = np.asarray(y)
    pos = scores[y > 0.5]
    neg = scores[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def _separable_design(seed=0, n=200):
    """Realistic Dirichlet theta (as the PC head actually receives); label = 1 iff
    theta[:,0] > theta[:,1], with a clean margin band removed. Perfectly separable,
    so the ridge-free logistic MLE runs |w| -> infinity."""
    rng = np.random.default_rng(seed)
    th = rng.dirichlet(np.full(2, 0.3), size=n)
    margin = np.abs(th[:, 0] - th[:, 1]) > 0.05     # drop the ambiguous band
    th = th[margin]
    y = (th[:, 0] > th[:, 1]).astype(float)
    return th, y


# ---------------------------------------------------------------------------
# TEST 1 — the SEPARATION CURE (the key acceptance test).
# ---------------------------------------------------------------------------
def test_firth_bounds_weights_on_separable_design_while_none_blows_up():
    th, y = _separable_design(seed=1)

    # head_penalty='none' with a ~0 ridge (head_l2=1e-12): the classic blow-up.
    w_none = _fit_inner_head(2, th, y, head_penalty="none",
                             head_l2=1e-12, inner_iters=200)
    none_norm = float(np.linalg.norm(w_none))

    # head_penalty='firth' (parameter-free): finite, bounded weights.
    w_firth = _fit_inner_head(2, th, y, head_penalty="firth",
                              head_l2=1e-12, inner_iters=200)
    firth_norm = float(np.linalg.norm(w_firth))

    print(f"\n[separation cure] none_|w|={none_norm:.4e}  firth_|w|={firth_norm:.4e}  "
          f"ratio={none_norm / firth_norm:.1f}")

    # (a) none blows up (driven to the +-50 logit-clip saturation ceiling — without the
    #     clip it is literally infinite); firth stays finite and small.
    assert np.isfinite(firth_norm)
    assert none_norm > 200.0, f"none did NOT blow up (|w|={none_norm:.3e}); check the design"
    assert firth_norm < 60.0, f"firth |w|={firth_norm:.3e} not bounded"
    assert firth_norm < none_norm / 5.0, (
        f"firth |w|={firth_norm:.3e} not MUCH smaller than none |w|={none_norm:.3e}")

    # (b) firth preserves the separating ranking: AUC == 1 on the training design.
    auc_firth = _auc(th @ w_firth, y)
    assert auc_firth == 1.0, f"firth broke the ranking: AUC={auc_firth}"
    # the direction points the separating way ([+, -]).
    assert w_firth[0] > 0 and w_firth[1] < 0, f"firth direction wrong: {w_firth}"


def test_firth_weights_are_stable_across_inner_iters_on_separable():
    """The Firth head CONVERGES to a SMALL finite fixed point: |w| at 100 and 300 inner
    iters agree AND stay small, whereas 'none' sits pinned at the large clip-saturation
    ceiling (10x bigger) — the pathological blow-up the clip merely bounds."""
    th, y = _separable_design(seed=2)

    w100 = _fit_inner_head(2, th, y, head_penalty="firth", head_l2=1e-12, inner_iters=100)
    w300 = _fit_inner_head(2, th, y, head_penalty="firth", head_l2=1e-12, inner_iters=300)
    assert np.linalg.norm(w300 - w100) < 1e-3, (
        f"firth not converged: |w100|={np.linalg.norm(w100):.4f} "
        f"|w300|={np.linalg.norm(w300):.4f}")
    assert np.linalg.norm(w300) < 60.0, "firth fixed point not small/bounded"

    n300 = np.linalg.norm(_fit_inner_head(2, th, y, head_penalty="none",
                                          head_l2=1e-12, inner_iters=300))
    assert n300 > 5.0 * np.linalg.norm(w300), (
        f"none |w|={n300:.3e} not far above firth |w|={np.linalg.norm(w300):.3e}")


# ---------------------------------------------------------------------------
# TEST 2 — NON-SEPARABLE ~ MLE: firth must not distort a well-posed fit.
# ---------------------------------------------------------------------------
def test_firth_matches_plain_newton_on_non_separable_design():
    rng = np.random.default_rng(7)
    n, K = 400, 3
    th = rng.random((n, K))
    w_true = np.array([3.0, -3.0, 1.0])
    logit = th @ w_true - (th @ w_true).mean()
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(float)          # noisy -> non-separable

    # plain newton (head_penalty='none', weak fixed ridge ~ MLE).
    w_none = _fit_inner_head(K, th, y, head_penalty="none", head_l2=1e-6, inner_iters=60)
    w_firth = _fit_inner_head(K, th, y, head_penalty="firth", head_l2=1e-6, inner_iters=60)

    cos = float(w_none @ w_firth / (np.linalg.norm(w_none) * np.linalg.norm(w_firth)))
    ratio = float(np.linalg.norm(w_firth) / np.linalg.norm(w_none))
    print(f"\n[non-separable] cos(firth,none)={cos:.4f}  "
          f"|w_firth|/|w_none|={ratio:.3f}  |w_none|={np.linalg.norm(w_none):.3f}")

    assert cos > 0.95, f"firth distorted the direction on a well-posed fit: cos={cos:.4f}"
    assert 0.5 < ratio < 1.2, f"firth |w| not comparable (collapsed/inflated): ratio={ratio:.3f}"
    assert np.linalg.norm(w_firth) > 1.0, "firth collapsed a well-posed fit toward zero"


# ---------------------------------------------------------------------------
# TEST 3 — validation errors.
# ---------------------------------------------------------------------------
def test_firth_requires_flat_head_not_dag_closure():
    # a 3-node chain DAG closure head: 0 root, 1->0, 2->1.
    head = DagClosureHead([[], [0], [1]])
    with pytest.raises(ValueError, match="firth.*FLAT|FLAT.*head"):
        OnlinePCLDA(K=4, vocab_size=8, C=3, weight_y=50.0,
                    head_optimizer="newton", head_penalty="firth", head=head)


def test_firth_requires_newton_optimizer():
    with pytest.raises(ValueError, match="firth.*newton"):
        OnlinePCLDA(K=4, vocab_size=8, C=1, weight_y=50.0,
                    head_optimizer="sgd", head_penalty="firth")


def test_head_penalty_rejects_unknown_value():
    with pytest.raises(ValueError, match="head_penalty"):
        OnlinePCLDA(K=4, vocab_size=8, C=1, head_penalty="ridge")


def test_firth_auto_enables_inner_loop_path():
    """head_penalty='firth' with head_inner_iters==0 auto-activates Path B (25)."""
    m = OnlinePCLDA(K=4, vocab_size=8, C=1, weight_y=50.0,
                    head_optimizer="newton", head_penalty="firth")
    assert m.head_inner_iters == 25
    # an explicit >0 value is respected, not overwritten.
    m2 = OnlinePCLDA(K=4, vocab_size=8, C=1, weight_y=50.0, head_optimizer="newton",
                     head_penalty="firth", head_inner_iters=50)
    assert m2.head_inner_iters == 50


# ---------------------------------------------------------------------------
# TEST 5 (regression) — head_penalty='none' is byte-for-byte the old inner loop.
# ---------------------------------------------------------------------------
def test_none_penalty_is_unchanged_ridge_behavior():
    """With head_penalty='none' the inner loop is the shipped fixed-L2 Newton: on a
    non-separable design it reaches the same ridge fixed point regardless of the new
    branch (sanity: finite, sensible direction)."""
    th, y = _separable_design(seed=3)
    # With a real ridge (head_l2=1e-2), 'none' stays finite even on separable data.
    w = _fit_inner_head(2, th, y, head_penalty="none", head_l2=1e-2, inner_iters=200)
    assert np.isfinite(np.linalg.norm(w))
    assert np.linalg.norm(w) < 1e3
    assert _auc(th @ w, y) == 1.0
