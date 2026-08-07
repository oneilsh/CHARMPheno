"""Finite-difference gradient checking helpers shared by the PC tests.

The PC factored functions each return analytic gradients w.r.t. several
arguments simultaneously. To gradient-check one argument we freeze the others,
build a scalar-valued closure over the flattened argument, and compare the
analytic gradient against a central/forward finite-difference approximation.

We report a *relative* error ||approx - analytic|| / (||analytic|| + eps) so
the < 1e-5 gate is scale-invariant across the very different magnitudes of the
generative vs. prediction gradients.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _central_fprime(f_val: Callable[[np.ndarray], float], x0: np.ndarray,
                    eps: float) -> np.ndarray:
    """Central-difference gradient (O(eps^2) truncation, vs approx_fprime's
    forward O(eps)) so the < 1e-5 correctness gate is not limited by the
    finite-difference approximation itself."""
    g = np.empty_like(x0)
    for i in range(x0.size):
        step = np.zeros_like(x0)
        step[i] = eps
        g[i] = (f_val(x0 + step) - f_val(x0 - step)) / (2.0 * eps)
    return g


def rel_grad_error(
    f_val: Callable[[np.ndarray], float],
    analytic_grad: np.ndarray,
    x0: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Relative L2 error between analytic_grad and a central finite-diff gradient.

    f_val: flat vector -> scalar. analytic_grad: the claimed gradient, same
    shape as x0 (flattened). Returns ||approx - analytic|| / (||analytic|| + 1e-12).
    """
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    approx = _central_fprime(f_val, x0, eps)
    analytic = np.asarray(analytic_grad, dtype=np.float64).ravel()
    denom = np.linalg.norm(analytic) + 1e-12
    return float(np.linalg.norm(approx - analytic) / denom)
