"""Tests for scripts/simulate_lda_omop.py."""
import numpy as np
import pandas as pd
import pytest


def _tiny_beta() -> pd.DataFrame:
    """Three topics, four concepts each, sharply peaked distributions."""
    return pd.DataFrame({
        "topic_id":     [0, 0, 0, 0,  1, 1, 1, 1,  2, 2, 2, 2],
        "concept_id":   [1, 2, 3, 4,  1, 2, 3, 4,  1, 2, 3, 4],
        "concept_name": ["a", "b", "c", "d"] * 3,
        # Topic 0 loves concept 1; topic 1 loves 2; topic 2 loves 3.
        "weight":       [0.97, 0.01, 0.01, 0.01,
                         0.01, 0.97, 0.01, 0.01,
                         0.01, 0.01, 0.97, 0.01],
    })


def test_simulate_produces_expected_schema():
    from simulate_lda_omop import simulate

    df = simulate(
        beta=_tiny_beta(),
        n_patients=5,
        theta_alpha=0.1,
        visits_per_patient_mean=2,
        codes_per_visit_mean=3,
        seed=42,
    )
    assert set(df.columns) == {
        "person_id", "visit_occurrence_id", "concept_id",
        "concept_name", "true_topic_id",
    }
    assert df["person_id"].nunique() == 5
    assert len(df) > 0


def test_simulate_is_deterministic_given_seed():
    from simulate_lda_omop import simulate

    args = dict(
        beta=_tiny_beta(),
        n_patients=5,
        theta_alpha=0.1,
        visits_per_patient_mean=2,
        codes_per_visit_mean=3,
        seed=123,
    )
    a = simulate(**args)
    b = simulate(**args)
    pd.testing.assert_frame_equal(a, b)


def test_simulate_concept_ids_come_from_beta_vocab():
    from simulate_lda_omop import simulate

    df = simulate(
        beta=_tiny_beta(),
        n_patients=20,
        theta_alpha=0.1,
        visits_per_patient_mean=3,
        codes_per_visit_mean=4,
        seed=7,
    )
    assert set(df["concept_id"].unique()).issubset({1, 2, 3, 4})


def test_simulate_true_topic_id_matches_a_valid_topic():
    from simulate_lda_omop import simulate

    df = simulate(
        beta=_tiny_beta(),
        n_patients=10,
        theta_alpha=0.1,
        visits_per_patient_mean=2,
        codes_per_visit_mean=3,
        seed=0,
    )
    assert set(df["true_topic_id"].unique()).issubset({0, 1, 2})


def test_simulate_concentrated_theta_recovers_expected_concept():
    """With very low alpha and peaked beta, a patient is dominated by one topic,
    which should concentrate their emitted concepts on that topic's favored
    concept. A sanity check that the generative process works as intended."""
    from simulate_lda_omop import simulate

    df = simulate(
        beta=_tiny_beta(),
        n_patients=100,
        theta_alpha=0.001,           # push θ nearly to a corner per patient
        visits_per_patient_mean=3,
        codes_per_visit_mean=10,
        seed=1,
    )
    # For each patient, the modal concept should be the favored concept of
    # their modal true_topic_id (topic 0 → concept 1, etc.).
    favored = {0: 1, 1: 2, 2: 3}
    correct = 0
    groups = df.groupby("person_id")
    for _, pdf in groups:
        modal_topic = pdf["true_topic_id"].mode().iloc[0]
        modal_concept = pdf["concept_id"].mode().iloc[0]
        if modal_concept == favored[modal_topic]:
            correct += 1
    # >=80% match is comfortably above chance for this scenario.
    assert correct / groups.ngroups >= 0.80
