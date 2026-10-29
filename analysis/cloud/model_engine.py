"""ModelEngine: the saved DAG-gated fit and readout heads, restricted to one candidate
set's members (framing 1's engines E2 / E3, `framings.run_differential`).

* θ from a record: the deployment fold-in, exactly — UNGATED full-K CAVI over the
  per-domain-normalized concatenated E[log β], with the fitted α
  (`DagGatedLDAModel._transform`; `_cavi_doc_inference` itself). The empty record
  folds in to θ = α: the model's own "nothing known yet" state.
* Heads: the run's saved readout heads (`readout_heads_gated_pc_<tag>.npz`):
  p_c = σ(clip(V_c·θ + b_c, ±50)), degenerate nodes at their constant. Only the heads
  in the closure of the set's members are evaluated.
* Which member: q_m ∝ P_stack(m) = Π_{a ∈ closure(m)} p_a, normalized over the set.
  For a set under a common ancestor G every member's closure contains G's, so the
  shared factor cancels and q is the stacked conditional P(m | G) renormalized.

(Born in the acquisition simulation `model_acquisition.py`, removed in the 2026-10-29
cleanup with `voi_acquisition.py`; last at commit 29556f3d.)"""
from __future__ import annotations

import numpy as np

__all__ = ["ModelEngine"]

_SCORE_CLIP = 50.0


class ModelEngine:
    """`lams` is the per-domain λ list (K, V_d) in domain order; `alpha` (K,); heads
    `(V (C,K), b (C,), degenerate (C,) bool, const (C,), topm)`; `closure` a
    `{member: [readout nodes in its closure]}` map; `members` the set's label nodes."""

    def __init__(self, lams, alpha, heads, closure, members, *, cavi_max_iter=100,
                 cavi_tol=1e-3):
        from spark_vi.models.topic.dag_gated_lda import expElogbeta_from_lambda
        sizes = [int(l.shape[1]) for l in lams]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        self.eb = expElogbeta_from_lambda(dict(enumerate(lams)))
        self.alpha = np.asarray(alpha, dtype=np.float64)
        V, b, deg, const, topm = heads
        nodes = sorted({a for m in members for a in closure[m]})
        self.Vs = np.asarray(V, dtype=np.float64)[nodes]
        self.bs = np.asarray(b, dtype=np.float64)[nodes]
        self.deg = np.asarray(deg, dtype=bool)[nodes]
        self.const = np.asarray(const, dtype=np.float64)[nodes]
        self.topm = int(topm)
        pos = {a: j for j, a in enumerate(nodes)}
        self.Msub = np.zeros((len(members), len(nodes)))
        for i, m in enumerate(members):
            for a in closure[m]:
                self.Msub[i, pos[a]] = 1.0
        self.max_iter, self.tol = int(cavi_max_iter), float(cavi_tol)

    def gidx(self, d, v):
        """Global column of vocab index `v` in domain `d`."""
        return int(self.offsets[d] + v)

    def fold_in(self, idx, cnt, gamma0=None):
        """γ for a bag (global ids, counts); the deployment CAVI."""
        from spark_vi.models.topic.lda import _cavi_doc_inference
        if len(idx) == 0:
            return self.alpha.copy()
        g0 = self.alpha + 1.0 if gamma0 is None else gamma0
        gamma, _, _, _ = _cavi_doc_inference(
            indices=np.asarray(idx, dtype=np.int64), counts=np.asarray(cnt, np.float64),
            expElogbeta=self.eb, alpha=self.alpha, gamma_init=g0,
            max_iter=self.max_iter, tol=self.tol)
        return gamma

    def posterior(self, theta):
        """q over members for θ (K,) or a batch (Q, K)."""
        th = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        if self.topm and self.topm < th.shape[1]:
            cut = np.partition(th, th.shape[1] - self.topm, axis=1)[:, [-self.topm]]
            th = np.where(th >= cut, th, 0.0)
        z = np.clip(th @ self.Vs.T + self.bs, -_SCORE_CLIP, _SCORE_CLIP)
        p = 1.0 / (1.0 + np.exp(-z))
        if self.deg.any():
            p[:, self.deg] = self.const[self.deg]
        logq = np.log(np.clip(p, 1e-12, 1.0)) @ self.Msub.T
        logq -= logq.max(axis=1, keepdims=True)
        q = np.exp(logq)
        q /= q.sum(axis=1, keepdims=True)
        return q[0] if np.ndim(theta) == 1 else q
