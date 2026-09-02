"""The 0111 episode probes (plan WP8a) — clustering, gates, and pooled stats.

What is worth pinning is what would fail silently: the gap boundary (inclusive
vs exclusive decides whether two 60-day-apart diagnoses are one presentation or
two), the index convention (episode codes must land INSIDE the half-open label
window and OUTSIDE the lookback), the fact that survival is judged by the
assembler's own `_window_observed_cohort` semantics, and the first-vs-later
decomposition R5.10's claim rests on.

Spark tests use local Spark (`@slow`); the report formatter is pure.
"""
import os
import sys
from datetime import date
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


def _first(spark, rows):
    return spark.createDataFrame(
        rows, "person_id long, node_cid long, first_attested_date date")


def _obs(spark, rows):
    return spark.createDataFrame(
        rows, "person_id long, observation_period_start_date date, "
              "observation_period_end_date date")


# --------------------------------------------------------------------------- #
# 1. The clustering.                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_gap_boundary_is_inclusive(spark):
    """Exactly `gap_days` apart = SAME episode; one day more = the next. The
    break condition is `datediff > gap`, and flipping it to `>=` would split
    every dense workup into fragments."""
    first = _first(spark, [
        (1, 10, date(2020, 1, 1)),
        (1, 11, date(2020, 3, 1)),   # +60d from Jan 1 → same episode at gap=60
        (1, 12, date(2020, 5, 1)),   # +61d from Mar 1 → NEW episode
    ])
    eps = {r["episode_no"]: r for r in
           ep.build_episodes(first, gap_days=60).collect()}
    assert set(eps) == {1, 2}
    assert eps[1]["episode_start"] == date(2020, 1, 1)
    assert eps[1]["n_new_nodes"] == 2
    assert eps[2]["episode_start"] == date(2020, 5, 1)
    assert eps[2]["n_new_nodes"] == 1


@pytest.mark.slow
def test_index_is_the_day_before_episode_start(spark):
    """index = start − 1: with the half-open label window `[index, index+W)`
    the episode's own first codes land inside the window and outside the
    lookback — the whole anchoring idea in one subtraction."""
    first = _first(spark, [(1, 10, date(2020, 6, 15))])
    row = ep.build_episodes(first, gap_days=60).collect()[0]
    assert row["index_date"] == date(2020, 6, 14)


@pytest.mark.slow
def test_same_day_multi_node_is_one_episode_counting_both(spark):
    """Two nodes first-attested the same day: one episode, n_new_nodes=2 — the
    distinct-dates islands must not collapse the node count with the dates."""
    first = _first(spark, [
        (1, 10, date(2020, 1, 1)),
        (1, 11, date(2020, 1, 1)),
    ])
    rows = ep.build_episodes(first, gap_days=60).collect()
    assert len(rows) == 1 and rows[0]["n_new_nodes"] == 2


# --------------------------------------------------------------------------- #
# 2. The gates (the assembler's own).                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_gate_kills_early_and_late_episodes(spark):
    """Three episodes against one observation period [2015-01-01, 2022-01-01]:
    the record-start episode dies to the prior-obs gate, the record-end episode
    dies to the follow-up gate, the middle one survives both — exactly R5.10's
    claim in miniature."""
    first = _first(spark, [
        (1, 10, date(2015, 2, 1)),    # index 2015-01-31 < start+365 → prior kill
        (1, 11, date(2018, 6, 1)),    # comfortably inside → survives
        (1, 12, date(2021, 8, 1)),    # index+365 > 2022-01-01 → follow-up kill
    ])
    obs = _obs(spark, [(1, date(2015, 1, 1), date(2022, 1, 1))])
    eps = ep.build_episodes(first, gap_days=60)
    both = ep.gate_episodes(eps, obs, prior_obs_days=365, window_days=365)
    assert [r["episode_start"] for r in both.collect()] == [date(2018, 6, 1)]
    prior_only = ep.gate_episodes(eps, obs, prior_obs_days=365, window_days=0)
    assert {r["episode_start"] for r in prior_only.collect()} == {
        date(2018, 6, 1), date(2021, 8, 1)}
    follow_only = ep.gate_episodes(eps, obs, prior_obs_days=0, window_days=365)
    assert {r["episode_start"] for r in follow_only.collect()} == {
        date(2015, 2, 1), date(2018, 6, 1)}


@pytest.mark.slow
def test_first_vs_later_kill_decomposition(spark):
    """Person 1's first episode dies, second survives; person 2's single (first)
    episode survives. first: n=2, 1 killed; later: n=1, 0 killed."""
    first = _first(spark, [
        (1, 10, date(2015, 2, 1)),
        (1, 11, date(2018, 6, 1)),
        (2, 10, date(2018, 6, 1)),
    ])
    obs = _obs(spark, [(1, date(2015, 1, 1), date(2022, 1, 1)),
                       (2, date(2015, 1, 1), date(2022, 1, 1))])
    eps = ep.build_episodes(first, gap_days=60)
    both = ep.gate_episodes(eps, obs, prior_obs_days=365, window_days=365)
    fl = ep._first_vs_later_kill(eps, both)
    assert fl["first_episodes"] == {"n": 2, "surviving": 1, "kill_rate": 0.5}
    assert fl["later_episodes"] == {"n": 1, "surviving": 1, "kill_rate": 0.0}


# --------------------------------------------------------------------------- #
# 3. Pooled stats.                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_per_person_stats_bands_and_caps(spark):
    """Person 1 has 4 episodes, person 2 has 1: multiplier 2.5; bands count
    persons; cap 3 keeps min(4,3)+min(1,3)=4."""
    first = _first(spark, [
        (1, 10, date(2016, 1, 1)), (1, 11, date(2017, 1, 1)),
        (1, 12, date(2018, 1, 1)), (1, 13, date(2019, 1, 1)),
        (2, 10, date(2018, 6, 1)),
    ])
    eps = ep.build_episodes(first, gap_days=60)
    stats = ep._per_person_stats(eps, caps=(3, 5))
    assert stats["n_persons"] == 2 and stats["n_episodes"] == 5
    assert stats["episodes_per_person_mean"] == pytest.approx(2.5)
    assert stats["person_count_bands"]["1"] == 1
    assert stats["person_count_bands"]["4"] == 1
    assert stats["capped_totals"] == {"3": 4, "5": 5}
    assert stats["new_nodes_per_episode_mean"] == pytest.approx(1.0)


@pytest.mark.slow
def test_run_probe_decomposition_is_consistent(spark):
    """End to end on the three-episode fixture: both-gates survivors can never
    exceed either single-gate count, and the report formatter runs on the real
    output shape without raising."""
    first = _first(spark, [
        (1, 10, date(2015, 2, 1)),
        (1, 11, date(2018, 6, 1)),
        (1, 12, date(2021, 8, 1)),
    ])
    obs = _obs(spark, [(1, date(2015, 1, 1), date(2022, 1, 1))])
    res = ep.run_probe(first, obs, gap_days=60, window_days=365)
    dec = res["gate_decomposition"]
    assert dec["raw"] == 3 and dec["surviving_both"] == 1
    assert dec["surviving_both"] <= dec["surviving_prior_only"]
    assert dec["surviving_both"] <= dec["surviving_followup_only"]
    banner = ep.format_probe_report(res)
    assert "R5.9" in banner and "R5.10" in banner and "R5.8" in banner


@pytest.mark.slow
def test_node_yield_counts_gated_episodes_per_frontier_node(spark):
    """Node 10 first-attests in a killed episode and node 11 in a surviving one:
    only node 11 shows any gated yield. Counts are OF NODES (pooled), and the
    frontier grain makes them a lower bound — closure folding only adds."""
    first = _first(spark, [
        (1, 10, date(2015, 2, 1)),    # prior-gate kill
        (1, 11, date(2018, 6, 1)),    # survives
    ])
    obs = _obs(spark, [(1, date(2015, 1, 1), date(2022, 1, 1))])
    episodes, assignments = ep.build_episodes(first, gap_days=60,
                                              return_assignments=True)
    both = ep.gate_episodes(episodes, obs, prior_obs_days=365, window_days=365)
    got = ep.node_yield(assignments, both, bars=(1, 20))
    assert got == {"nodes_with_any_gated_episode": 1,
                   "nodes_ge_1": 1, "nodes_ge_20": 0}


# --------------------------------------------------------------------------- #
# 4. The formatter is total (pure).                                            #
# --------------------------------------------------------------------------- #
def test_formatter_survives_empty_results():
    """A corpus with zero episodes (or zero survivors) must report, not crash —
    the probe's job on a degenerate input is to SAY so."""
    res = {"gap_days": 60, "window_days": 365, "prior_obs_days": 365,
           "raw": {"n_persons": 0, "n_episodes": 0},
           "gated_both": {"n_persons": 0, "n_episodes": 0,
                          "new_nodes_per_episode_mean": float("nan")},
           "gate_decomposition": {"raw": 0, "surviving_both": 0,
                                  "surviving_prior_only": 0,
                                  "surviving_followup_only": 0},
           "first_vs_later_kill": {}}
    banner = ep.format_probe_report(res)
    assert "gap=60d" in banner
