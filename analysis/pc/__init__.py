"""In-memory, full-batch reference implementation of a faithful, flat
Prediction-Constrained (PC) topic model (Hughes, Hope, Weiner, McCoy, Perlis,
Sudderth & Doshi-Velez 2017/2018).

This subpackage is the **correctness oracle** for PC training: a numpy/scipy-only,
id-agnostic, gradient-based reference (NOT variational inference — it deliberately
does not live in ``spark-vi``). It optimizes the PC Lagrangian by full-batch
gradient descent (``scipy.optimize.minimize`` / L-BFGS-B) over unconstrained
parameters, with hand-coded, finite-difference-checked gradients.

Model (documents d = 1..D, bag-of-words counts X (D, V); K topics; C classes;
labels y_d given only on a labeled subset L):

    beta_k = softmax(w_k)          topic-word rows on the simplex,  w in R^{K x V}
    pi_d   = softmax(u_d)          doc-topic point estimate,        u in R^{D x K}
    yhat_d = softmax(eta @ pi_d + b)   supervised head,  eta in R^{C x K}, b in R^C

Objective (minimized over w, u, eta, b):

    L = GEN + lam * PRED
    GEN  = sum_{d=1..D} [ -sum_v x_dv log( sum_k pi_dk beta_kv )
                          - (alpha - 1) sum_k log pi_dk ]        # ALL docs
    PRED = sum_{d in L} crossentropy( y_d, softmax(eta @ pi_d + b) )   # LABELED docs

The three faithful invariants that distinguish PC from naive sLDA up-weighting:
  1. PRED is summed over labeled docs and scaled by a free scalar ``lam``,
     decoupled from per-token counts (the label is never folded into the
     per-token word likelihood where N_d tokens would drown one label).
  2. Semi-supervised asymmetry: unlabeled docs contribute GEN only; labeled docs
     contribute GEN + lam * PRED. GEN is identical for both.
  3. Prediction is read off pi_d (the expected topic frequencies z-bar), not a
     separate variable.

The head and the objective are factored as standalone pure functions so the
future VI-native port re-wires them over ``OnlineLDA`` rather than rewriting.

References:
    Hughes, Hope, Weiner, McCoy, Perlis, Sudderth, Doshi-Velez 2017. Prediction-
        constrained topic models for antidepressant recommendation. NeurIPS ML4H.
    Hughes, Hope, Weiner, McCoy, Perlis, Sudderth, Doshi-Velez 2018. Semi-
        supervised prediction-constrained topic models. AISTATS.
"""
from __future__ import annotations

from analysis.pc.head import softmax_head_loss
from analysis.pc.generative import generative_neg_loglik
from analysis.pc.objective import pack_params, unpack_params, pc_objective
from analysis.pc.model import PCTopicModel

__all__ = [
    "softmax_head_loss",
    "generative_neg_loglik",
    "pack_params",
    "unpack_params",
    "pc_objective",
    "PCTopicModel",
]
