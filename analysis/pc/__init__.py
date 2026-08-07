"""In-memory, full-batch reference implementation of a faithful, flat
Prediction-Constrained (PC) topic model (Hughes, Hope, Weiner, McCoy, Perlis,
Sudderth & Doshi-Velez 2017/2018).

This subpackage is the **correctness oracle** for PC training: a numpy/scipy-only,
id-agnostic, gradient-based reference (NOT variational inference — it deliberately
does not live in ``spark-vi``; the faithful path additionally uses ``autograd``).

There are TWO models here, and it matters which is which:

**PRIMARY — the faithful reference (:class:`~analysis.pc.model.PCTopicModel`,
:mod:`analysis.pc.slda_reference`).** Mirrors the authors' ``calc_loss__slda``
term-for-term. Its defining move: each doc's topic vector ``pi_d`` is a
**generative MAP estimate from the words ONLY** (an unrolled NEF exponentiated-
gradient inference), *label-free and byte-for-byte identical at train and test*.
The supervised loss (a per-label logistic head ``sigmoid(w_CK @ pi_d)``, no bias)
reshapes the **global topics** by ``autograd``-differentiating *through* that
pi-inference. Terms: ``loss_x`` (multinomial gen NLL, all docs) + ``loss_pi``
(Dirichlet MAP prior) + ``weight_y * loss_y`` (logistic, labeled docs only) +
``loss_topics`` (Dirichlet-on-beta) + ``loss_w`` (head L2). ``weight_y`` is a
free multiplier decoupled from token counts; ``weight_y = 0`` is unsupervised
LDA-MAP. This is what removes the train/test ``pi`` mismatch (see the plan doc).

**VARIANT — the free-pi PC-family model (:class:`~analysis.pc.variants.PCTopicModelFreePi`,
built on the factored A1 pieces below).** Gives each doc a FREE ``pi_d`` that the
label shapes at train time; ``beta_k = softmax(w_k)``, ``pi_d = softmax(u_d)``,
softmax head, objective ``GEN + lam * PRED`` over free ``(w, u, eta, b)``. It
works (beats two-stage) but is NOT Hughes' algorithm and has a train/test ``pi``
mismatch. It is kept as the **seed for the future VI-native port** (a label-shaped
E-step is the natural, tractable supervised-VI move) — see the plan's parked
VI-port fork. The A1 factored functions it uses (``pc_objective`` etc.) are
standalone pure ``(params, data) -> (value, grad)`` so that port re-wires rather
than rewrites.

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
from analysis.pc.variants import PCTopicModelFreePi
from analysis.pc.slda_reference import (
    calc_loss__slda,
    nef_map_pi_DK,
    make_convex_alpha_minus_1,
)

__all__ = [
    # Faithful reference (primary).
    "PCTopicModel",
    "calc_loss__slda",
    "nef_map_pi_DK",
    "make_convex_alpha_minus_1",
    # Free-pi PC-family variant (NOT Hughes' algorithm; VI-port seed).
    "PCTopicModelFreePi",
    # Factored A1 objective pieces (shared by the variant / future VI port).
    "softmax_head_loss",
    "generative_neg_loglik",
    "pack_params",
    "unpack_params",
    "pc_objective",
]
