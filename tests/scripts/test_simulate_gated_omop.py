# tests/scripts/test_simulate_gated_omop.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import numpy as np
import pandas as pd

from simulate_gated_omop import build_gated_beta, simulate_gated


def _gb():
    rng = np.random.default_rng(0)
    return build_gated_beta(
        n_background_concepts=10, n_group_concepts=5,
        background_k=3, foreground=(("rare_dx", 2),), rng=rng, bleed=0.05)


def test_beta_is_row_stochastic_and_blocks_labeled():
    gb = _gb()
    assert gb.beta.shape == (5, 15)  # 3 bg + 2 fg topics; 10 bg + 5 group concepts
    np.testing.assert_allclose(gb.beta.sum(axis=1), 1.0, atol=1e-9)
    assert gb.topic_blocks == ["background", "background", "background",
                               "rare_dx", "rare_dx"]


def test_foreground_topics_concentrate_on_group_concepts():
    gb = _gb()
    grp_cols = [list(gb.concept_ids).index(c) for c in gb.group_concepts["rare_dx"]]
    # each rare_dx foreground topic puts the majority of its mass on rare concepts
    fg = gb.beta[3:]
    assert (fg[:, grp_cols].sum(axis=1) > 0.8).all()


def test_simulate_emits_expected_columns_and_gating():
    gb = _gb()
    events, persons, oracle = simulate_gated(
        gb, n_patients=200, group_props={"common": 0.8, "rare_dx": 0.2},
        foreground=(("rare_dx", 2),), visits_per_patient_mean=3.0,
        codes_per_visit_mean=6.0, age_means={"common": 55.0, "rare_dx": 70.0},
        theta_alpha=0.3, seed=1)
    assert set(events.columns) == {
        "person_id", "visit_occurrence_id", "concept_id", "concept_name",
        "source_cohort", "true_topic_id", "true_block"}
    assert set(persons.columns) == {"person_id", "source_cohort", "sex", "age"}
    # common patients (no foreground block) NEVER emit a foreground topic
    common_pids = set(persons.loc[persons.source_cohort == "common", "person_id"])
    common_ev = events[events.person_id.isin(common_pids)]
    assert (common_ev.true_block == "background").all()
    # rare_dx patients DO emit some foreground
    rare_ev = events[~events.person_id.isin(common_pids)]
    assert (rare_ev.true_block == "rare_dx").any()
    assert oracle["background_k"] == 3


def test_cli_writes_three_files(tmp_path):
    import json
    from simulate_gated_omop import main
    rc = main([
        "--n-patients", "150", "--seed", "2",
        "--background-k", "3", "--foreground", "rare_dx:2",
        "--group-props", "common:0.8,rare_dx:0.2",
        "--age-means", "common:55,rare_dx:70",
        "--n-background-concepts", "10", "--n-group-concepts", "5",
        "--output-dir", str(tmp_path)])
    assert rc == 0
    ev = tmp_path / "gated_omop_N150_seed2.parquet"
    pe = tmp_path / "gated_person_N150_seed2.parquet"
    orc = tmp_path / "gated_oracle_N150_seed2.json"
    assert ev.exists() and pe.exists() and orc.exists()
    persons = pd.read_parquet(pe)
    assert set(persons.source_cohort.unique()) <= {"common", "rare_dx"}
    oracle = json.loads(orc.read_text())
    assert oracle["foreground"] == [["rare_dx", 2]]
