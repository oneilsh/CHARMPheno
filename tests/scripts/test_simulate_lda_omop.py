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


def _tiny_topic_metadata() -> pd.DataFrame:
    """Topic 1 carries 10x as much corpus mass as topics 0 and 2."""
    return pd.DataFrame({
        "topic_id": [0, 1, 2],
        "usage_pct": [0.1, 1.0, 0.1],
        "uniformity_h": [0.5, 0.6, 0.7],
        "coherence_c": [0.0, 0.1, 0.2],
    })


def test_asymmetric_alpha_matches_renormalized_usage_with_total_K_alpha():
    """α_k / Σα = Ũ_k and Σα = K * theta_alpha."""
    from simulate_lda_omop import _asymmetric_alpha

    alpha = _asymmetric_alpha(
        beta=_tiny_beta(),
        topic_metadata=_tiny_topic_metadata(),
        theta_alpha=0.1,
    )
    K = 3
    np.testing.assert_allclose(alpha.sum(), K * 0.1)
    expected_u = np.array([0.1, 1.0, 0.1])
    expected_u = expected_u / expected_u.sum()
    np.testing.assert_allclose(alpha / alpha.sum(), expected_u)


def test_asymmetric_alpha_renormalizes_over_present_topics_only():
    """If beta omits topic 0, Ũ should renormalize over the surviving topics."""
    from simulate_lda_omop import _asymmetric_alpha

    beta = _tiny_beta()
    beta = beta[beta["topic_id"] != 0]  # drop topic 0
    alpha = _asymmetric_alpha(
        beta=beta,
        topic_metadata=_tiny_topic_metadata(),
        theta_alpha=0.1,
    )
    # Surviving topics 1 and 2 with usage 1.0 and 0.1 → Ũ = 10/11, 1/11.
    expected_u = np.array([1.0, 0.1])
    expected_u = expected_u / expected_u.sum()
    np.testing.assert_allclose(alpha / alpha.sum(), expected_u)
    np.testing.assert_allclose(alpha.sum(), 2 * 0.1)


def test_asymmetric_alpha_rejects_missing_topic_id():
    from simulate_lda_omop import _asymmetric_alpha

    bad_metadata = pd.DataFrame({
        "topic_id": [0, 1],  # missing topic 2
        "usage_pct": [0.1, 0.5],
        "uniformity_h": [0.5, 0.6],
        "coherence_c": [0.0, 0.1],
    })
    with pytest.raises(ValueError, match="missing usage_pct"):
        _asymmetric_alpha(
            beta=_tiny_beta(), topic_metadata=bad_metadata, theta_alpha=0.1,
        )


def test_simulate_asymmetric_prior_concentrates_on_high_usage_topic():
    """With usage 10:1:1 and very low theta_alpha, most patients should be
    dominated by topic 1 — and across patients, ~Ũ_k = 10/12 of true_topic_id
    counts should fall on topic 1.
    """
    from simulate_lda_omop import simulate

    df = simulate(
        beta=_tiny_beta(),
        n_patients=400,
        theta_alpha=0.05,           # sparse mixtures
        visits_per_patient_mean=3,
        codes_per_visit_mean=8,
        seed=2026,
        topic_metadata=_tiny_topic_metadata(),
    )
    counts = df["true_topic_id"].value_counts(normalize=True).reindex(
        [0, 1, 2], fill_value=0.0,
    )
    # Expected proportions Ũ = [1/12, 10/12, 1/12]; allow generous slack
    # (one Dirichlet draw per patient → finite-sample noise).
    assert counts[1] > 0.6, f"topic 1 should dominate, got {counts.to_dict()}"
    assert counts[0] < 0.25
    assert counts[2] < 0.25


def test_simulate_symmetric_prior_unchanged_when_metadata_omitted():
    """With no metadata arg, output equals the pre-asymmetric path."""
    from simulate_lda_omop import simulate

    args = dict(
        beta=_tiny_beta(),
        n_patients=20,
        theta_alpha=0.1,
        visits_per_patient_mean=2,
        codes_per_visit_mean=3,
        seed=99,
    )
    a = simulate(**args)
    b = simulate(**args, topic_metadata=None)
    pd.testing.assert_frame_equal(a, b)
