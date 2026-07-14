"""Co-sampled Gibbs read-out engine over the DAG-offset PG-STM (step 2). Emits
offset-INCREMENT posterior draws on the compiler's identified quotient DAG.

Inference substrate: stick-breaking + Polya-Gamma augmentation (Polson, Scott &
Windle 2013; Linderman, Johnson & Adams 2015). The offset block is drawn from its
matrix-normal conditional each sweep (a proper joint chain, not a ridge point).
"""
import numpy as np

from spark_vi.models.topic.pg_stm_dag import dag_offset_ridge


def dag_offset_ridge_draw(WtW, WtM, Sigma, *, penalty, rng):
    """One matrix-normal draw of the offset coefficient block:
    C ~ MN(mean = (WtW + diag(penalty))^{-1} WtM, row_cov = (WtW + diag(penalty))^{-1},
    col_cov = Sigma). Its expectation is exactly dag_offset_ridge(WtW, WtM, penalty).
    Depth-scaled `penalty` is the diagonal Gaussian prior precision on the coefficient
    rows; Sigma is the stick-space residual covariance shared across the K-1 columns."""
    WtW = np.asarray(WtW, dtype=np.float64)
    WtM = np.asarray(WtM, dtype=np.float64)
    A = WtW + np.diag(np.asarray(penalty, dtype=np.float64))
    mean = dag_offset_ridge(WtW, WtM, penalty=penalty)
    Ainv = np.linalg.inv(A)
    L_row = np.linalg.cholesky((Ainv + Ainv.T) / 2.0)
    L_col = np.linalg.cholesky((Sigma + Sigma.T) / 2.0)
    Z = rng.standard_normal(mean.shape)
    return mean + L_row @ Z @ L_col.T
