"""build_dag_placement_args covariate forwarding (2x2 prediction axis).

The gated case-finding driver gained --covariate-formula / --pred-cov et al.;
run_experiment must forward them from experiment frontmatter, mirroring the STM
frontmatter keys. The critical subtlety: YAML 1.1 parses `pred_cov: on` as the
boolean True, which must serialize back to the literal "on" (not "True")."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_experiment  # noqa: E402


def _base_fm():
    return {
        "source_table": "condition_era", "person_mod": 1, "vocab_size": 5000,
        "min_df": 20, "min_patient_count": 20, "doc_min_length": 10,
        "min_n": 20, "n_bg": 40, "tpn": 2, "max_iter": 100,
    }


def _pairs(args):
    return {args[i]: args[i + 1] for i in range(0, len(args) - 1)}


def test_dag_placement_forwards_covariate_args(monkeypatch):
    monkeypatch.setattr(run_experiment, "_require_workspace_env",
                        lambda: ("proj.ds", "billing"))
    fm = {**_base_fm(),
          "covariate_formula": "age_std + C(sex)",
          "categorical_cols": ["sex"], "continuous_cols": ["age_std"],
          "pred_cov": True,                 # YAML `on` -> boolean True
          "known_sex_only": True}
    args = run_experiment.build_dag_placement_args(fm, "/tmp/out")
    p = _pairs(args)
    assert p["--covariate-formula"] == "age_std + C(sex)"
    assert p["--covariate-categorical"] == "sex"
    assert p["--covariate-continuous"] == "age_std"
    assert "--known-sex-only" in args
    # YAML `on` -> True must serialize to the literal "on", never "True".
    assert p["--pred-cov"] == "on"


def test_dag_placement_pred_cov_off_string(monkeypatch):
    monkeypatch.setattr(run_experiment, "_require_workspace_env",
                        lambda: ("p.d", "b"))
    fm = {**_base_fm(), "pred_cov": "off"}
    args = run_experiment.build_dag_placement_args(fm, "/tmp/out")
    assert _pairs(args)["--pred-cov"] == "off"


def test_dag_placement_baseline_omits_covariates(monkeypatch):
    monkeypatch.setattr(run_experiment, "_require_workspace_env",
                        lambda: ("p.d", "b"))
    args = run_experiment.build_dag_placement_args(_base_fm(), "/tmp/out")
    assert "--covariate-formula" not in args
    assert "--pred-cov" not in args
    assert "--known-sex-only" not in args
