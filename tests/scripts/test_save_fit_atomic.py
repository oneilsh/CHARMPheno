"""_save_fit writes gated_pc_result.npz + manifest.json ATOMICALLY.

_save_fit doubles as the periodic fit checkpoint (--fit-save-interval): it is
called mid-fit while the SVI loop continues, so a crash during a write must not
tear the very artifact that is the crash insurance. These tests assert the happy
path leaves both files, no stray .tmp, and a re-readable npz — the observable
contract of the tmp+os.replace atomic write.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUD = REPO_ROOT / "analysis" / "cloud"
for _p in (str(REPO_ROOT), str(CLOUD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gated_pc_cloud as gpc  # noqa: E402


def _gp(C=6, K=4, V=10):
    return {
        "lambda": np.random.default_rng(0).random((K, V)),
        "alpha": np.full(K, 0.5),
        "w_CK": np.zeros((C, K)),
        "b_CK": np.zeros(C),
    }


def test_save_fit_writes_both_files_and_no_tmp(tmp_path):
    gpc._save_fit(tmp_path, _gp(), C=6, manifest_fields={"C": 6}, partial="fit-only")
    names = {p.name for p in tmp_path.iterdir()}
    assert "gated_pc_result.npz" in names
    assert "manifest.json" in names
    # No torn temp files survive a clean write.
    assert not any(n.endswith(".tmp") for n in names), names


def test_save_fit_npz_reloads_and_manifest_marks_partial(tmp_path):
    gp = _gp()
    gpc._save_fit(tmp_path, gp, C=6, manifest_fields={"C": 6}, partial="fit-only")
    with np.load(tmp_path / "gated_pc_result.npz") as z:
        assert np.allclose(z["lambda"], gp["lambda"])
        assert np.allclose(z["alpha"], gp["alpha"])
        assert z["w_CK"].shape == (6, 4)
        assert z["b_CK"].shape == (6,)
    import json
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["partial"] == "fit-only"
    assert man["results"] is None


def test_save_fit_multidomain_lambda_dict(tmp_path):
    gp = _gp()
    gp["lambda"] = {0: np.ones((4, 10)), 1: np.full((4, 7), 2.0)}
    gpc._save_fit(tmp_path, gp, C=6, manifest_fields={"C": 6}, partial="fit-only")
    with np.load(tmp_path / "gated_pc_result.npz") as z:
        assert np.allclose(z["lambda_0"], 1.0)
        assert np.allclose(z["lambda_1"], 2.0)


def test_save_fit_overwrites_in_place_atomically(tmp_path):
    # A second call (the final full save, or the next checkpoint) replaces the
    # same paths and still leaves no .tmp behind.
    gpc._save_fit(tmp_path, _gp(), C=6, manifest_fields={"C": 6}, partial="fit-only")
    gpc._save_fit(tmp_path, _gp(), C=6, manifest_fields={"C": 6}, partial=None)
    names = {p.name for p in tmp_path.iterdir()}
    assert not any(n.endswith(".tmp") for n in names), names
    import json
    assert json.loads((tmp_path / "manifest.json").read_text())["partial"] is None


def test_readout_heads_sidecar_carries_w_std_when_given(tmp_path):
    """The heads sidecar stores the solve's STANDARDIZED weights beside V when
    the caller passes them (inspect_topics reads them as the honest loadings
    scale once the solver checkpoint is gone), and stays readable without."""
    C, K = 3, 5
    V = np.arange(C * K, dtype=float).reshape(C, K)
    W = V / 7.0
    ok = gpc._write_readout_heads(tmp_path, "gated_pc", V, np.zeros(C),
                                  np.zeros((C, K), dtype=bool),
                                  np.zeros(C, dtype=bool), C, K, 0, W_std=W)
    assert ok
    z = np.load(tmp_path / "readout_heads_gated_pc.npz")
    assert "W_std" in z.files and np.array_equal(z["W_std"], W)
    assert np.array_equal(z["V"], V)
    ok = gpc._write_readout_heads(tmp_path, "other", V, np.zeros(C),
                                  np.zeros((C, K), dtype=bool),
                                  np.zeros(C, dtype=bool), C, K, 0)
    assert ok and "W_std" not in np.load(tmp_path / "readout_heads_other.npz").files


def test_feature_mask_solve_is_an_exact_reduced_solve():
    """`_apply_feature_mask` on the batched solver: masked coordinates stay
    EXACTLY 0 through the whole L-BFGS path, and the kept coordinates land on
    the optimum of the reduced problem (a solve over the kept features only)."""
    from analysis.pc.batched_lr import solve_batched_lr
    rng = np.random.default_rng(3)
    C, K, n = 2, 5, 400
    X = rng.normal(size=(n, K))
    w_true = np.array([[1.5, -1.0, 0.0, 0.8, 0.0], [0.0, 2.0, -1.2, 0.0, 0.5]])
    Y = np.stack([(rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ w_true[c] + 0.2))))
                  for c in range(C)], axis=1).astype(float)

    def stats_fn(W, b, node_mask=None):
        z = X @ W.T + b                                   # (n, C)
        p = 1 / (1 + np.exp(-z))
        loss = -(Y * np.log(p + 1e-300) + (1 - Y) * np.log(1 - p + 1e-300)).sum(0)
        gW = (p - Y).T @ X                                # (C, K)
        gb = (p - Y).sum(0)
        return loss, gW, gb

    mask = np.ones((C, K), dtype=bool)
    mask[0, [1, 4]] = False                               # node 0 loses 2 features
    mask[1, [0, 2, 3]] = False                            # node 1 keeps 2
    Wm, bm, im = solve_batched_lr(gpc._apply_feature_mask(stats_fn, mask), C, K,
                                  l2=1.0, max_iter=300, gtol=1e-8)
    assert np.all(Wm[~mask] == 0.0)                       # exactly, not approximately
    # reduced problem per node: solve on the kept columns only, compare
    for c in range(C):
        keep = np.flatnonzero(mask[c])
        Xr = X[:, keep]

        def sf_r(W, b, node_mask=None, _Xr=Xr, _y=Y[:, c]):
            z = _Xr @ W[0] + b[0]
            p = 1 / (1 + np.exp(-z))
            loss = -(_y * np.log(p + 1e-300) + (1 - _y) * np.log(1 - p + 1e-300)).sum()
            return (np.array([loss]), ((p - _y) @ _Xr)[None, :],
                    np.array([(p - _y).sum()]))
        Wr, br, _ = solve_batched_lr(sf_r, 1, len(keep), l2=1.0, max_iter=300,
                                     gtol=1e-8)
        assert np.allclose(Wm[c, keep], Wr[0], atol=1e-4), c
        assert abs(bm[c] - br[0]) < 1e-4
