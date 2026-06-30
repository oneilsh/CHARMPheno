import numpy as np
from spark_vi.models.topic.stm import (
    _stm_neg_log_joint, _stm_neg_log_joint_grad, _stm_neg_log_joint_hessian,
)

def _toy():
    rng = np.random.default_rng(0)
    K, Vn = 3, 4
    eta = rng.normal(size=K)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    counts = np.array([2.0, 1.0, 3.0, 1.0])
    expElogbeta = np.abs(rng.normal(size=(K, Vn))) + 0.1
    Gamma = rng.normal(size=(2, K))
    x = np.array([1.0, 0.5])
    return dict(eta=eta, indices=indices, counts=counts,
               expElogbeta=expElogbeta, Gamma=Gamma, x=x), K

def test_full_precision_prior_matches_diagonal_special_case():
    common, K = _toy()
    eta = common.pop("eta")
    sigma_diag = np.array([1.5, 2.0, 0.5])
    Sigma_inv = np.diag(1.0 / sigma_diag)
    # Prior term with full precision == diagonal hand-computation.
    diff = eta - common["Gamma"].T @ common["x"]
    expected_prior = 0.5 * float(diff @ Sigma_inv @ diff)
    f = _stm_neg_log_joint(eta, Sigma_inv=Sigma_inv, **common)
    # Recompute the data term alone by zeroing the prior (Sigma_inv = 0).
    data_only = _stm_neg_log_joint(eta, Sigma_inv=np.zeros((K, K)), **common)
    assert np.isclose(f - data_only, expected_prior)

def test_full_precision_grad_and_hessian_use_Sigma_inv():
    common, K = _toy()
    eta = common.pop("eta")
    Sigma_inv = np.array([[2.0, 0.3, 0.0],
                          [0.3, 1.0, 0.1],
                          [0.0, 0.1, 4.0]])
    diff = eta - common["Gamma"].T @ common["x"]
    g = _stm_neg_log_joint_grad(eta, Sigma_inv=Sigma_inv, **common)
    g_data = _stm_neg_log_joint_grad(eta, Sigma_inv=np.zeros((K, K)), **common)
    assert np.allclose(g - g_data, Sigma_inv @ diff)
    H = _stm_neg_log_joint_hessian(eta, Sigma_inv=Sigma_inv, **common)
    H_data = _stm_neg_log_joint_hessian(eta, Sigma_inv=np.zeros((K, K)), **common)
    assert np.allclose(H - H_data, Sigma_inv)
