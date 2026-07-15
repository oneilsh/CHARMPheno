# tests/scripts/test_build_dashboard_gated.py
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis" / "local"))


def test_local_dashboard_emits_gating_files(tmp_path):
    from simulate_gated_omop import main as sim_main
    from fit_stm_local import main as fit_main
    import build_dashboard

    sim_main(["--n-patients", "400", "--seed", "9",
              "--background-k", "3", "--foreground", "rare_dx:2",
              "--group-props", "common:0.7,rare_dx:0.3",
              "--age-means", "common:55,rare_dx:72",
              "--n-background-concepts", "12", "--n-group-concepts", "6",
              "--codes-per-visit-mean", "6", "--output-dir", str(tmp_path)])
    omop = tmp_path / "gated_omop_N400_seed9.parquet"
    person = tmp_path / "gated_person_N400_seed9.parquet"
    ckpt = tmp_path / "ckpt"
    fit_main(["--omop", str(omop), "--person", str(person),
              "--K", "5", "--background-k", "3", "--foreground", "rare_dx:2",
              "--covariate-formula", "~ C(sex) + age",
              "--max-iter", "10", "--out-dir", str(ckpt)])

    out = tmp_path / "bundle"
    rc = build_dashboard.main(["--checkpoint", str(ckpt), "--input", str(omop),
                               "--out-dir", str(out)])
    assert rc == 0
    gating = json.loads((out / "gating.json").read_text())
    assert gating["group_var"] == "source_cohort"
    assert "rare_dx" in gating["groups"]              # >=k at this size
    assert len(gating["topic_blocks"]) == 5
    assert (out / "covariate_effects.json").exists()
    assert (out / "covariate_schema.json").exists()

    # Categorical reference level must be non-empty — this catches the bug where
    # result.model_spec is None so _categorical_levels_from_spec returns {}.
    schema = json.loads((out / "covariate_schema.json").read_text())
    sex_control = next(
        (c for c in schema["controls"] if c["name"] == "sex" and c["type"] == "categorical"),
        None,
    )
    assert sex_control is not None, "covariate_schema.json has no categorical 'sex' control"
    assert sex_control["reference"], (
        "sex control has empty reference level — categorical_levels not persisted at fit time"
    )
    # The reference level must be one of the two fitted categories.
    assert sex_control["reference"] in ("F", "M"), (
        f"unexpected reference level: {sex_control['reference']!r}"
    )
