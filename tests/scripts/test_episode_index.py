"""Exp 0111 WP-D1 — the episode index provider and EpisodeDocSpec (local Spark).

What is worth pinning here, beyond `tests/scripts/test_episode_probe.py`'s
clustering/gate coverage (which this module reuses via `build_episodes`,
moved to `episode_index.py` and imported back — see that module's docstring):
the CAP is deterministic and salt-sensitive (never `F.rand()`), the kept
`episode_no` is the ORIGINAL chronological ordinal and never a post-sample
re-rank, an episode the observation gates kill is never eligible for the cap
sample regardless of salt, and `EpisodeDocSpec` appends (never inserts) its
`index_date` component so `doc_id.split(":").getItem(0)` still recovers
`source_cohort` (audit R5.1).

Spark tests use local Spark (`@slow`).
"""
import os
import sys
from datetime import date

import pytest

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
CLOUD = REPO_ROOT / "analysis" / "cloud"
for _p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
                str(REPO_ROOT / "spark-vi"), os.environ.get("PYTHONPATH", ""))
    if p)

import episode_index as ei  # noqa: E402

from charmpheno.omop.doc_spec import EpisodeDocSpec  # noqa: E402


def _first(spark, rows):
    return spark.createDataFrame(
        rows, "person_id long, node_cid long, first_attested_date date")


def _obs(spark, rows):
    return spark.createDataFrame(
        rows, "person_id long, observation_period_start_date date, "
              "observation_period_end_date date")


# A person with 5 widely-spaced (>90d apart) episodes, all clearing both
# observation gates at prior_obs_days=365/window_days=365 against a wide
# [2005-01-01, 2020-01-01) observation period — cap=3 must bind (5 -> 3), and
# the deterministic sha2 rank must be exercised for real, not vacuously.
_FIVE_EPISODE_STARTS = [
    date(2010, 1, 1), date(2010, 7, 20), date(2011, 2, 5),
    date(2011, 8, 24), date(2012, 3, 11),
]


def _five_episode_fixture(spark, person_id=1):
    rows = [(person_id, 100 + i, d)
            for i, d in enumerate(_FIVE_EPISODE_STARTS, start=1)]
    first = _first(spark, rows)
    obs = _obs(spark, [(person_id, date(2005, 1, 1), date(2020, 1, 1))])
    return first, obs


# --------------------------------------------------------------------------- #
# 1. build_episodes / column constants still work through the re-export.       #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_build_episodes_reexported_and_column_names_stable(spark):
    """`episode_index.build_episodes` is the SAME function
    `diag_episode_probe` re-exports (lib<-tool direction) — pinned here so a
    future refactor that breaks the re-export fails at THIS module, not only
    `test_episode_probe.py`."""
    import diag_episode_probe as ep

    first, _ = _five_episode_fixture(spark)
    a = ei.build_episodes(first, gap_days=90).collect()
    b = ep.build_episodes(first, gap_days=90).collect()
    assert ep.build_episodes is ei.build_episodes
    assert sorted((r["episode_no"], r["episode_start"]) for r in a) == \
        sorted((r["episode_no"], r["episode_start"]) for r in b)
    assert (ei.PERSON_COL, ei.DATE_COL, ei.INDEX_COL, ei.EPISODE_COL,
            ei.START_COL, ei.NNODES_COL) == \
        (ep.PERSON_COL, ep.DATE_COL, ep.INDEX_COL, ep.EPISODE_COL,
         ep.START_COL, ep.NNODES_COL)


# --------------------------------------------------------------------------- #
# 2. episode_index_frame: determinism, cap, salt-sensitivity, ordinals.        #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_determinism_same_salt_identical_across_runs(spark):
    """Two independent calls with the SAME salt return byte-identical output —
    resume-stability, the whole point of the sha2 pick over `F.rand()`."""
    first, obs = _five_episode_fixture(spark)
    a = ei.episode_index_frame(first, obs, gap_days=90, cap=3, salt="seedA",
                               prior_obs_days=365, window_days=365).collect()
    b = ei.episode_index_frame(first, obs, gap_days=90, cap=3, salt="seedA",
                               prior_obs_days=365, window_days=365).collect()
    key = lambda rows: sorted(
        (r["person_id"], r["index_date"], r["episode_no"]) for r in rows)
    assert key(a) == key(b)


@pytest.mark.slow
def test_cap_enforcement_keeps_exactly_cap_of_five_survivors(spark):
    """5 surviving episodes, cap=3 -> exactly 3 kept, all for the one person."""
    first, obs = _five_episode_fixture(spark)
    kept = ei.episode_index_frame(first, obs, gap_days=90, cap=3, salt="seedA",
                                  prior_obs_days=365, window_days=365).collect()
    assert len(kept) == 3
    assert {r["person_id"] for r in kept} == {1}
    assert len({r["index_date"] for r in kept}) == 3  # no duplicate rows


@pytest.mark.slow
def test_selection_changes_with_a_different_salt(spark):
    """Same 5-episode input, different salt -> a DIFFERENT 3-of-5 sample.

    Expected episode_no sets are the ascending-sha2-hash rank computed
    independently in Python (hashlib.sha256 over "person_id|episode_start|salt"
    with '|' as the separator, matching F.sha2(F.concat_ws("|", ...), 256)) —
    pinned so a change to the hash inputs (column order, separator, cast) is
    caught here rather than discovered as a silent reshuffle."""
    first, obs = _five_episode_fixture(spark)
    kept_a = ei.episode_index_frame(first, obs, gap_days=90, cap=3,
                                    salt="seedA", prior_obs_days=365,
                                    window_days=365).collect()
    kept_b = ei.episode_index_frame(first, obs, gap_days=90, cap=3,
                                    salt="seedB", prior_obs_days=365,
                                    window_days=365).collect()
    got_a = sorted(r["episode_no"] for r in kept_a)
    got_b = sorted(r["episode_no"] for r in kept_b)
    assert got_a == [1, 3, 4]
    assert got_b == [1, 4, 5]
    assert got_a != got_b


@pytest.mark.slow
def test_gate_reuse_killed_episode_never_sampled(spark):
    """A person's first episode dies to the SAME prior-obs gate
    `diag_episode_probe.gate_episodes` uses (`test_gate_kills_early_and_late_
    episodes`'s fixture, reused): it must never appear in the capped output
    for ANY salt, and the survivors' episode_no values must be their ORIGINAL
    ordinals (2, 3, 4), never renumbered starting at 1 — cap=3 does not bind
    here (only 3 of 4 episodes survive the gate), so this isolates gate reuse
    and ordinal preservation from the cap-selection logic."""
    first = _first(spark, [
        (5, 10, date(2015, 2, 1)),    # episode 1: prior-obs gate kill
        (5, 11, date(2018, 6, 1)),    # episode 2: survives
        (5, 12, date(2019, 6, 1)),    # episode 3: survives
        (5, 13, date(2020, 6, 1)),    # episode 4: survives
    ])
    obs = _obs(spark, [(5, date(2015, 1, 1), date(2022, 1, 1))])
    for salt in ("seedA", "seedB", "a-third-salt"):
        kept = ei.episode_index_frame(first, obs, gap_days=60, cap=3,
                                      salt=salt, prior_obs_days=365,
                                      window_days=365).collect()
        got = sorted(r["episode_no"] for r in kept)
        assert got == [2, 3, 4], f"salt={salt!r} kept {got}"
        assert 1 not in got


# --------------------------------------------------------------------------- #
# 3. EpisodeDocSpec.                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_episode_doc_spec_doc_id_shape(spark):
    """doc_id = "{source_cohort}:{person_id}:{index_date}" — the append."""
    events = spark.createDataFrame(
        [(1, "cancer", date(2020, 6, 14), 42)],
        "person_id long, source_cohort string, index_date date, concept_id long")
    out = EpisodeDocSpec().derive_docs(events).collect()[0]
    assert out["doc_id"] == "cancer:1:2020-06-14"


@pytest.mark.slow
def test_episode_doc_spec_doc_id_append_survives_prefix_parse(spark):
    """`doc_id.split(":").getItem(0)` still recovers source_cohort — R5.1: a
    prefix parser written against PatientCohortDocSpec's "{cohort}:{person}"
    shape must not misparse the episode doc_id's appended third component."""
    from pyspark.sql import functions as F

    events = spark.createDataFrame(
        [(7, "dementia", date(2021, 1, 1), 99),
         (7, "dementia", date(2021, 9, 30), 99)],
        "person_id long, source_cohort string, index_date date, concept_id long")
    out = EpisodeDocSpec().derive_docs(events).withColumn(
        "_recovered_cohort", F.split(F.col("doc_id"), ":").getItem(0)
    ).collect()
    assert {r["_recovered_cohort"] for r in out} == {"dementia"}
    # And it stays a 3-part id (cohort, person, index) — the append landed at
    # the END, not merged into an existing component.
    assert all(r["doc_id"].count(":") == 2 for r in out)


def test_episode_doc_spec_missing_index_date_raises_named_error():
    """No Spark needed: the check runs before any column is touched, so a
    plain object with a `.columns` attribute exercises it. The message must
    name index_date AND the WP-C passthrough — a generic missing-column
    KeyError three frames later would send someone hunting in the wrong
    file."""
    class _FakeFrame:
        columns = ["person_id", "source_cohort"]

    with pytest.raises(ValueError) as exc:
        EpisodeDocSpec().derive_docs(_FakeFrame())
    msg = str(exc.value)
    assert "index_date" in msg
    assert "WP-C" in msg


@pytest.mark.slow
def test_episode_doc_spec_manifest_round_trip():
    spec = EpisodeDocSpec(min_doc_length=7)
    d = spec.manifest()
    assert d == {"name": "episode", "min_doc_length": 7}
    from charmpheno.omop.doc_spec import DocSpec
    restored = DocSpec.from_manifest(d)
    assert isinstance(restored, EpisodeDocSpec)
    assert restored.min_doc_length == 7


# --------------------------------------------------------------------------- #
# 4. R7.5 — min_doc_length drop-rate by episode ordinal.                       #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_r75_drop_rate_by_ordinal_table(spark):
    """Hand-built (person_id, episode_no, doc_length) fixture, min_doc_length=10:

    band 1 (episode_no=1): lengths [5, 15]   -> 1 kept, 1 dropped
    band 2 (episode_no=2): lengths [20, 3]   -> 1 kept, 1 dropped
    band 3+ (episode_no in {3,4,5}): [50,2,8]-> 1 kept, 2 dropped

    Concentration toward the incident (low-ordinal) end is exactly what R7.5
    exists to surface next to the 66.2% first-episode gate-kill."""
    doc_lengths = spark.createDataFrame(
        [(1, 1, 5), (2, 1, 15),
         (3, 2, 20), (4, 2, 3),
         (5, 3, 50), (6, 4, 2), (7, 5, 8)],
        "person_id long, episode_no int, doc_length int")
    table = ei.min_doc_length_drop_rate_by_ordinal(doc_lengths, min_doc_length=10)
    assert table["1"] == {"n": 2, "kept": 1, "dropped": 1, "drop_rate": 0.5}
    assert table["2"] == {"n": 2, "kept": 1, "dropped": 1, "drop_rate": 0.5}
    assert table["3+"]["n"] == 3 and table["3+"]["kept"] == 1
    assert table["3+"]["dropped"] == 2
    assert table["3+"]["drop_rate"] == pytest.approx(2 / 3)


@pytest.mark.slow
def test_r75_drop_rate_table_empty_band_reports_none_not_error(spark):
    """No episode_no=2 rows at all: band "2" reports zeros and drop_rate=None,
    never a division error — a corpus (or a fixture) can legitimately have no
    second episodes."""
    doc_lengths = spark.createDataFrame(
        [(1, 1, 15), (2, 3, 1)],
        "person_id long, episode_no int, doc_length int")
    table = ei.min_doc_length_drop_rate_by_ordinal(doc_lengths, min_doc_length=10)
    assert table["2"] == {"n": 0, "kept": 0, "dropped": 0, "drop_rate": None}
    assert set(table) == {"1", "2", "3+"}
