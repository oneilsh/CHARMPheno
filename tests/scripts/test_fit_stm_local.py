# tests/scripts/test_fit_stm_local.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "local"))

import json
import numpy as np


def _make_sim(tmp_path):
    from simulate_gated_omop import main as sim_main
    sim_main([
        "--n-patients", "300", "--seed", "5",
        "--background-k", "3", "--foreground", "rare_dx:2",
        "--group-props", "common:0.7,rare_dx:0.3",
        "--age-means", "common:55,rare_dx:72",
        "--n-background-concepts", "12", "--n-group-concepts", "6",
        "--codes-per-visit-mean", "6", "--output-dir", str(tmp_path)])
    return (tmp_path / "gated_omop_N300_seed5.parquet",
            tmp_path / "gated_person_N300_seed5.parquet")


def test_fit_stm_local_writes_gated_checkpoint(tmp_path):
    from fit_stm_local import main as fit_main
    omop, person = _make_sim(tmp_path)
    out = tmp_path / "ckpt"
    rc = fit_main([
        "--omop", str(omop), "--person", str(person),
        "--K", "5", "--background-k", "3", "--foreground", "rare_dx:2",
        "--covariate-formula", "~ C(sex) + age",
        "--max-iter", "8", "--out-dir", str(out)])
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text())
    cm = manifest["metadata"]["corpus_manifest"]
    assert cm["topic_block_spec"]["background_k"] == 3
    assert cm["topic_block_spec"]["foreground"] == [["rare_dx", 2]]
    assert manifest["metadata"]["model_class"] == "stm"
    assert (out / "covariates.parquet").exists()
    lam = np.load(out / "params" / "lambda.npy")
    assert lam.shape[0] == 5  # K rows
