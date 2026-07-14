import numpy as np
from spark_vi.models.topic.pg_stm_dag import dag_offset_ridge
from spark_vi.models.topic.pg_stm_dag_gibbs import dag_offset_ridge_draw


def test_offset_draw_mean_is_the_ridge_and_covariance_is_matrix_normal():
    rng = np.random.default_rng(0)
    Pw, Ksm1 = 4, 3
    W = rng.standard_normal((200, Pw))
    M = rng.standard_normal((200, Ksm1))
    WtW, WtM = W.T @ W, W.T @ M
    penalty = np.array([1e-6, 0.5, 0.5, 0.5])
    Sigma = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    ridge_mean = dag_offset_ridge(WtW, WtM, penalty=penalty)

    draws = np.array([dag_offset_ridge_draw(WtW, WtM, Sigma, penalty=penalty, rng=rng)
                      for _ in range(4000)])           # (4000, Pw, Ksm1)
    # (a) Monte-Carlo mean == the ridge point
    assert np.abs(draws.mean(axis=0) - ridge_mean).max() < 0.05
    # (b) row-covariance of a fixed column c == A^{-1} * Sigma[c, c]
    Ainv = np.linalg.inv(WtW + np.diag(penalty))
    col = 1
    emp_cov = np.cov(draws[:, :, col].T)
    assert np.abs(emp_cov - Ainv * Sigma[col, col]).max() < 0.05
