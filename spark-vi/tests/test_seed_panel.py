"""Tests for spark_vi.eval.topic.seed_panel (seed-panel acceptance test for
the STM generative concentration scale c). Pure numpy, no Spark."""
from __future__ import annotations

import numpy as np

from spark_vi.eval.topic.seed_panel import (
    conditioned_theta, seed_panel_sweep, signature_seeds,
)
from spark_vi.models.topic.partition import TopicBlockPartition


def _tiny_two_topic():
    """K=2: topic 0 is the sole background topic (also the reference), topic 1
    is group 'g''s one foreground topic. Disjoint vocab: topic 0 concentrates
    on code 0, topic 1 on codes 1-3."""
    part = TopicBlockPartition(group_var="g", background_k=1, foreground=(("g", 1),))
    V = 4
    beta = np.array([
        [0.97, 0.01, 0.01, 0.01],   # topic 0 (background/reference)
        [0.02, 0.40, 0.33, 0.25],   # topic 1 (group g)
    ])
    R = np.eye(2)
    Gamma = np.zeros((1, 2))
    return part, beta, R, Gamma


def test_signature_seeds_returns_topic_top_codes():
    part = TopicBlockPartition(group_var="g", background_k=1, foreground=(("g", 2),))
    V = 6
    beta = np.zeros((3, V))
    beta[0] = [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]     # background (reference)
    beta[1] = [0.01, 0.01, 0.5, 0.3, 0.09, 0.09]      # group topic 1: top code 2
    beta[2] = [0.01, 0.01, 0.01, 0.01, 0.4, 0.56]     # group topic 2: top code 5

    seeds = signature_seeds(beta, part, group="g", n_codes=1, reference=0)
    by_topic = {topic_id: seed_indices for topic_id, seed_indices, _ in seeds}

    assert set(by_topic.keys()) == {1, 2}
    assert by_topic[1].tolist() == [np.argmax(beta[1])] == [2]
    assert by_topic[2].tolist() == [np.argmax(beta[2])] == [5]

    # reference (topic 0) is never seeded even if background isn't excluded.
    seeds_incl_bg = signature_seeds(
        beta, part, group="g", n_codes=1, reference=0, exclude_background=False,
    )
    assert 0 not in {topic_id for topic_id, _, _ in seeds_incl_bg}


def test_conditioned_theta_recovers_seed_topic():
    part, beta, R, Gamma = _tiny_two_topic()
    seed_indices = np.array([np.argmax(beta[1])])   # topic 1's own top code
    seed_counts = np.ones(1)

    theta = conditioned_theta(
        beta, Gamma, R, part,
        group="g", seed_indices=seed_indices, seed_counts=seed_counts,
        c=4.0, reference=0,
    )

    assert int(np.argmax(theta)) == 1
    assert theta[1] > 0.5


def test_higher_c_concentrates():
    part, beta, R, Gamma = _tiny_two_topic()
    seed_indices = np.array([np.argmax(beta[1])])
    seed_counts = np.ones(1)

    def top_mass_at(c):
        theta = conditioned_theta(
            beta, Gamma, R, part,
            group="g", seed_indices=seed_indices, seed_counts=seed_counts,
            c=c, reference=0,
        )
        return theta.max()

    assert top_mass_at(8.0) >= top_mass_at(2.0)


def test_seed_panel_sweep_row_shape():
    """Smoke test on the aggregate sweep helper: every row carries the
    expected keys and recovers_self is consistent with argmax(theta)."""
    part, beta, R, Gamma = _tiny_two_topic()
    rows = seed_panel_sweep(
        beta, Gamma, R, part, group="g", c_grid=[2.0, 4.0, 8.0],
        n_codes=1, reference=0,
    )
    assert len(rows) == 3   # 1 seeded topic (topic 1) x 3 c values
    for row in rows:
        assert set(row.keys()) == {
            "seed_topic", "c", "recovered_topic", "recovers_self",
            "top_mass", "eff_topics", "second_mass",
        }
        assert row["seed_topic"] == 1
        assert row["recovers_self"] == (row["recovered_topic"] == 1)
