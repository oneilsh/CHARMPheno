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
