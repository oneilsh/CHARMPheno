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
    assert manifest["metadata"]["vocab"]            # top-level vocab present
    assert "name_by_id" in manifest["metadata"]     # top-level name_by_id present
    assert (out / "covariates.parquet").exists()
    lam = np.load(out / "params" / "lambda.npy")
    assert lam.shape[0] == 5  # K rows


def test_fit_recovers_planted_rare_foreground(tmp_path):
    from simulate_gated_omop import main as sim_main
    from fit_stm_local import main as fit_main
    from spark_vi.io import load_result

    sim_main([
        "--n-patients", "600", "--seed", "7",
        "--background-k", "3", "--foreground", "rare_dx:2",
        "--group-props", "common:0.7,rare_dx:0.3",
        "--age-means", "common:55,rare_dx:72",
        "--n-background-concepts", "12", "--n-group-concepts", "6",
        "--codes-per-visit-mean", "6", "--output-dir", str(tmp_path)])
    omop = tmp_path / "gated_omop_N600_seed7.parquet"
    person = tmp_path / "gated_person_N600_seed7.parquet"
    oracle = json.loads((tmp_path / "gated_oracle_N600_seed7.json").read_text())
    out = tmp_path / "ckpt"
    fit_main([
        "--omop", str(omop), "--person", str(person),
        "--K", "5", "--background-k", "3", "--foreground", "rare_dx:2",
        "--covariate-formula", "~ C(sex) + age",
        "--max-iter", "40", "--out-dir", str(out)])

    result = load_result(out)
    lam = result.global_params["lambda"]
    beta = lam / lam.sum(axis=1, keepdims=True)
    vocab = result.metadata["corpus_manifest"]["vocab"]  # index -> concept_id
    cid_to_idx = {int(c): i for i, c in enumerate(vocab)}
    rare_cids = oracle["group_concepts"]["rare_dx"]
    rare_cols = [cid_to_idx[c] for c in rare_cids if c in cid_to_idx]
    # foreground topics are indices 3,4 (after 3 background); at least one must
    # concentrate on the rare-dx distinctive concepts.
    fg_mass = beta[3:][:, rare_cols].sum(axis=1).max()
    bg_mass = beta[:3][:, rare_cols].sum(axis=1).max()
    assert fg_mass > 0.4, fg_mass
    assert bg_mass < 0.15, bg_mass


def test_fit_stm_local_reference_topic_end_to_end(tmp_path):
    """--reference-topic threads through to the engine: the saved Gamma has its
    reference column zeroed and the metadata records the hardening config."""
    import json
    import numpy as np
    from fit_stm_local import main as fit_main
    omop, person = _make_sim(tmp_path)
    out = tmp_path / "ckpt_ref"
    rc = fit_main([
        "--omop", str(omop), "--person", str(person),
        "--K", "5", "--background-k", "3", "--foreground", "rare_dx:2",
        "--covariate-formula", "~ C(sex) + age",
        "--reference-topic",
        "--sigma-prior-scale", "2.0", "--sigma-prior-count", "500.0",
        "--max-iter", "8", "--out-dir", str(out)])
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["metadata"]["stm_hardening"] == {
        "reference_topic": True,
        "sigma_prior_scale": 2.0,
        "sigma_prior_count": 500.0,
        "spectral_init": False,
    }
    Gamma = np.load(out / "params" / "Gamma.npy")
    assert np.allclose(Gamma[:, 0], 0.0)


def test_fit_stm_local_spectral_init_end_to_end(tmp_path):
    """--spectral-init threads through to the engine: the fit completes on the
    gated sim corpus (exercising the block-aware anchor-word path) and the
    metadata records the hardening config."""
    import json
    from fit_stm_local import main as fit_main
    omop, person = _make_sim(tmp_path)
    out = tmp_path / "ckpt_spectral"
    rc = fit_main([
        "--omop", str(omop), "--person", str(person),
        "--K", "5", "--background-k", "3", "--foreground", "rare_dx:2",
        "--covariate-formula", "~ C(sex) + age",
        "--spectral-init",
        "--max-iter", "8", "--out-dir", str(out)])
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["metadata"]["stm_hardening"]["spectral_init"] is True
