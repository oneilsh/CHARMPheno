"""The 0111 GATE-OCCUPANCY probe (spec R5.14) — the empty-gate cost of a
uniform-random index, measured before a fit.

What is worth pinning is what would fail silently: the half-open gate boundary
(a first attestation exactly at index+90d must be OUTSIDE the gate, matching the
label window `conversion_analysis` pins), the episode-vs-random contrast the
whole experiment turns on (episode indices sit before their own codes and are
non-empty by construction; a uniform-random index in a quiet interval is empty),
the DETERMINISM of the salted random draw (same salt reproduces byte-for-byte,
never `F.rand()`), and that every random index provably passes the assembler's
own observation gates.

Spark tests use local Spark (`@slow`).
"""
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUD = REPO_ROOT / "analysis" / "cloud"
for _p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
                str(REPO_ROOT / "spark-vi"), os.environ.get("PYTHONPATH", ""))
    if p)

import diag_episode_probe as ep  # noqa: E402
import episode_index as ei  # noqa: E402


def _first(spark, rows):
    return spark.createDataFrame(
        rows, "person_id long, node_cid long, first_attested_date date")


def _obs(spark, rows):
    return spark.createDataFrame(
        rows, "person_id long, observation_period_start_date date, "
              "observation_period_end_date date")


def _idx(spark, rows):
    return spark.createDataFrame(rows, "person_id long, index_date date")


# --------------------------------------------------------------------------- #
# 1. The gate window (the shared occupancy core).                             #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_gate_window_is_half_open_at_index_plus_gate_days(spark):
    """index+90d is EXCLUDED, index+89d and the index day itself are INCLUDED.
    The gate is `[index, index+gate_days)` — half-open, matching the label
    window; flipping the upper bound to inclusive would silently over-count the
    gate and count as incident a code that belongs to the NEXT window."""
    idx = _idx(spark, [(1, date(2021, 1, 1))])
    first = _first(spark, [
        (1, 10, date(2021, 1, 1)),    # == index → included
        (1, 11, date(2021, 3, 31)),   # index+89 → included
        (1, 12, date(2021, 4, 1)),    # index+90 → EXCLUDED (half-open)
    ])
    occ = ep.gate_occupancy(idx, first, gate_days=90)
    assert occ["n_docs"] == 1
    assert occ["n_nonempty"] == 1
    assert occ["nonempty_fraction"] == 1.0
    assert occ["gate_size_mean"] == pytest.approx(2.0)   # nodes 10 and 11 only


@pytest.mark.slow
def test_episode_gate_nonempty_random_quiet_gate_empty(spark):
    """The contrast the experiment exists to measure, made deterministic. One
    person's activity is a single 2018 cluster. An episode index (episode_start
    - 1) sits just before it → gate non-empty, size = the cluster's nodes. A
    random index planted in a quiet 2013 interval sees nothing in its 90 days →
    gate EMPTY. Per-arm non-empty fractions and gate sizes come out exactly as
    constructed."""
    first = _first(spark, [
        (1, 10, date(2018, 6, 1)),
        (1, 11, date(2018, 6, 15)),   # one episode cluster
    ])
    episode_idx = _idx(spark, [(1, date(2018, 5, 31))])   # start-1: covers both
    random_idx = _idx(spark, [(1, date(2013, 1, 1))])     # quiet interval

    ep_occ = ep.gate_occupancy(episode_idx, first, gate_days=90)
    rnd_occ = ep.gate_occupancy(random_idx, first, gate_days=90)

    assert ep_occ["nonempty_fraction"] == 1.0
    assert ep_occ["gate_size_mean"] == pytest.approx(2.0)
    assert rnd_occ["nonempty_fraction"] == 0.0
    assert rnd_occ["n_nonempty"] == 0
    assert rnd_occ["gate_size_mean"] == pytest.approx(0.0)


@pytest.mark.slow
def test_gate_occupancy_empty_frame_reports_not_raises(spark):
    """An index frame with zero documents must report cleanly (None fields), not
    divide by zero — a degenerate corpus is the probe's job to SAY so."""
    idx = spark.createDataFrame([], "person_id long, index_date date")
    first = _first(spark, [(1, 10, date(2018, 6, 1))])
    occ = ep.gate_occupancy(idx, first, gate_days=90)
    assert occ["n_docs"] == 0 and occ["nonempty_fraction"] is None


# --------------------------------------------------------------------------- #
# 2. The random arm (the uniform draw).                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_random_draw_is_deterministic_and_salt_sensitive(spark):
    """Same salt on the same input reproduces the SAME indices byte for byte
    (resume-stable, never `F.rand()`); a different salt reshuffles them. The
    valid interval is 13 years wide, so an identical 3-day draw across two
    different salts is vanishingly unlikely."""
    obs = _obs(spark, [(1, date(2010, 1, 1), date(2025, 1, 1))])

    def _draw(salt):
        rows = ei.random_index_frame(
            obs, cap=3, salt=salt, prior_obs_days=365, window_days=365).collect()
        return sorted(r["index_date"] for r in rows)

    a1, a2 = _draw("A"), _draw("A")
    b = _draw("B")
    assert a1 == a2            # determinism: same salt, identical draw
    assert a1 != b             # salt-sensitivity: different salt, different draw
    assert 1 <= len(a1) <= 3   # up to cap distinct valid indices


@pytest.mark.slow
def test_random_indices_all_pass_the_observation_gates(spark):
    """Every drawn index sits in `[op_start+365, op_end-365]` — none in the
    excluded 365-day head or tail. Validity is the assembler's own
    `_window_observed_cohort`, so the draw cannot leak an under-observed index."""
    obs = _obs(spark, [(1, date(2010, 1, 1), date(2025, 1, 1))])
    rows = ei.random_index_frame(
        obs, cap=3, salt="X", prior_obs_days=365, window_days=365).collect()
    assert rows
    for r in rows:
        idx = r["index_date"]
        # prior-obs gate: index >= op_start + 365
        assert idx >= date(2010, 1, 1) + timedelta(days=365)
        # follow-up gate: index + 365 <= op_end
        assert idx + timedelta(days=365) <= date(2025, 1, 1)


@pytest.mark.slow
def test_random_draw_respects_person_restriction(spark):
    """`persons=` restricts the draw to a population (exp 0111 passes the episode
    arm's survivors so the arms compare on an identical person set). A person
    absent from that set draws no index."""
    obs = _obs(spark, [(1, date(2010, 1, 1), date(2025, 1, 1)),
                       (2, date(2010, 1, 1), date(2025, 1, 1))])
    keep = spark.createDataFrame([(1,)], "person_id long")
    rows = ei.random_index_frame(
        obs, cap=3, salt="X", prior_obs_days=365, window_days=365,
        persons=keep).collect()
    assert {r["person_id"] for r in rows} == {1}


@pytest.mark.slow
def test_random_draw_drops_persons_with_no_valid_interval(spark):
    """A person whose observation period is too short to admit ANY fully-observed
    365/365 index (here ~1.5 years, < 730 days of gates) draws nothing — the
    `_n_valid >= 1` guard and the gate re-check agree."""
    obs = _obs(spark, [(1, date(2020, 1, 1), date(2021, 6, 1))])
    rows = ei.random_index_frame(
        obs, cap=3, salt="X", prior_obs_days=365, window_days=365).collect()
    assert rows == []


# --------------------------------------------------------------------------- #
# 3. End to end.                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_run_gate_occupancy_episode_arm_is_fully_occupied(spark):
    """End to end on a one-episode person: the EPISODE arm's gate is non-empty by
    construction (index = episode_start - 1, the episode's own codes fall in the
    90 days), the RANDOM arm draws on the SAME person, and the formatter runs on
    the real output shape without raising."""
    first = _first(spark, [
        (1, 10, date(2018, 6, 1)),
        (1, 11, date(2018, 6, 15)),
    ])
    obs = _obs(spark, [(1, date(2010, 1, 1), date(2025, 1, 1))])
    res = ep.run_gate_occupancy(first, obs, gap_days=90, gate_days=90, cap=3,
                                salt="0111", prior_obs_days=365, window_days=365)
    assert res["episode_arm"]["nonempty_fraction"] == 1.0
    assert res["episode_arm"]["gate_size_mean"] == pytest.approx(2.0)
    # Same person population in both arms.
    assert res["random_arm"]["n_docs"] >= 1
    assert 0.0 <= res["random_arm"]["nonempty_fraction"] <= 1.0
    banner = ep.format_gate_occupancy_report(res)
    assert "EPISODE" in banner and "RANDOM" in banner and "gate=[index" in banner


# --------------------------------------------------------------------------- #
# 4. The formatter is total (pure).                                           #
# --------------------------------------------------------------------------- #
def test_gate_occupancy_formatter_survives_empty_arms():
    """A run over an empty corpus (both arms zero docs) must report, not crash."""
    empty = {"n_docs": 0, "n_nonempty": 0, "nonempty_fraction": None,
             "gate_size_mean": None, "gate_size_p50": None,
             "gate_size_p90": None, "gate_size_p99": None}
    res = {"gap_days": 90, "gate_days": 90, "cap": 3, "salt": "0111",
           "prior_obs_days": 365, "window_days": 365,
           "episode_arm": empty, "random_arm": empty}
    banner = ep.format_gate_occupancy_report(res)
    assert "gate=[index, index+90d)" in banner and "n/a" in banner
