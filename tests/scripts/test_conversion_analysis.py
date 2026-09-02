"""Future-conversion analysis (spec E4 / plan WP6, unit ii) — the pure core.

Everything that decides a NUMBER here is pure numpy between two Spark reads, and
all of it is testable off-Spark. Four groups, one per thing that would be silently
wrong:

  1. **The horizon arithmetic.** A conversion is strictly AFTER the label window
     and inside the horizon; "never attested" is a non-conversion, not a missing
     value.
  2. **RIGHT-CENSORING (R4.4).** The denominator is gated on
     `observation_period_end_date` at EACH horizon, so denominators shrink
     monotonically as the horizon grows. Without the gate the "conversion rate" is
     a coverage artifact — this is the acceptance item the spec names.
  3. **The decile split (R4.5/R4.8).** Rank-based, decile 9 = top, and a planted
     score-conversion gradient comes back as a positive top-minus-bottom — which is
     the case-finding claim, measured on the model's own negatives.
  4. **The closure fold and the framing.** "First attestation of closure(c)" is the
     min over c's SUBTREE, including a diamond; and the PU lower-bound language is
     attached to every table (R4.6) — mandatory, not decorative.
"""
import os
import sys
from pathlib import Path

import numpy as np
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

import conversion_analysis as ca  # noqa: E402

LW = 365          # label window days
H = (365, 730, 1095)


# --------------------------------------------------------------------------- #
# 1. The horizon arithmetic.                                                   #
# --------------------------------------------------------------------------- #
def test_a_conversion_is_strictly_after_the_label_window():
    """Five documents, one index date, one long observation period, every case by
    hand. Day 0 is the index; the label window ends on day 365."""
    index = np.zeros(5)
    obs = np.full(5, 10_000.0)              # everyone observed far past 3y
    first = np.array([
        200.0,      # inside the label window: NOT a conversion (it is a positive)
        365.0,      # the last day of the window: still not a conversion
        400.0,      # 1y horizon (365 < 400 <= 730)
        900.0,      # 2y horizon
        1500.0,     # past 3y (end+1095 = 1460): never converts
    ])
    got = ca.conversion_counts(first, index, obs, LW, horizons=H)
    assert list(got[365][0]) == [False, False, True, False, False]
    assert list(got[730][0]) == [False, False, True, True, False]
    assert list(got[1095][0]) == [False, False, True, True, False]
    for h in H:
        assert got[h][1].all()               # all observed


def test_never_attested_is_a_non_conversion_not_a_missing_value():
    """A person with no row in the sidecar has `nan`, which means "never coded at
    this node anywhere in the record" — a non-converter at every horizon."""
    got = ca.conversion_counts(np.array([np.nan, 500.0]), np.zeros(2),
                               np.full(2, 10_000.0), LW, horizons=H)
    assert list(got[365][0]) == [False, True]    # 365 < 500 <= 365+365
    assert list(got[730][0]) == [False, True]
    assert got[730][1].all()                 # both still in the DENOMINATOR


def test_the_horizon_is_measured_from_the_end_of_the_label_window():
    """`end = index + label_window_days`, not from the index — the label window is
    the period the negative label already covers."""
    got = ca.conversion_counts(np.array([700.0]), np.array([0.0]),
                               np.array([10_000.0]), LW, horizons=(365,))
    assert got[365][0][0]                    # 700 <= 365 + 365
    got2 = ca.conversion_counts(np.array([700.0]), np.array([0.0]),
                                np.array([10_000.0]), 0, horizons=(365,))
    assert not got2[365][0][0]               # with no label window, 700 > 365


# --------------------------------------------------------------------------- #
# 2. Right-censoring (R4.4) — the spec's own acceptance item.                  #
# --------------------------------------------------------------------------- #
def test_denominators_shrink_monotonically_with_the_horizon():
    """Staggered observation ends: as the horizon grows, fewer people are still in
    the record to be seen converting, so the denominator can only fall. Without
    this gate a conversion rate is a censoring artifact, not a contamination
    estimate."""
    index = np.zeros(4)
    obs = np.array([700.0, 800.0, 1_200.0, 5_000.0])     # end days
    first = np.full(4, np.nan)
    got = ca.conversion_counts(first, index, obs, LW, horizons=H)
    dens = [int(got[h][1].sum()) for h in H]
    assert dens == [3, 2, 1]
    assert dens == sorted(dens, reverse=True)


def test_a_person_who_leaves_the_record_is_EXCLUDED_not_counted_as_a_negative():
    """The distinction the whole gate exists for: dropping out is 'we cannot see',
    not 'it did not happen'."""
    index = np.zeros(2)
    obs = np.array([400.0, 5_000.0])         # first person gone before 1y horizon
    first = np.full(2, np.nan)
    got = ca.conversion_counts(first, index, obs, LW, horizons=(365,))
    converted, observed = got[365]
    assert list(observed) == [False, True]
    assert not converted.any()
    table = ca.node_conversion_table(got, min_count=1)
    assert table["horizons"][365]["n_observed"] == 1     # not 2


def test_a_conversion_after_the_observation_end_cannot_be_counted():
    """A first-attestation date past the person's own observation end is a data
    inconsistency; the gate makes it a non-event rather than a phantom conversion
    in a denominator that excludes it."""
    got = ca.conversion_counts(np.array([600.0]), np.array([0.0]),
                               np.array([500.0]), LW, horizons=(365,))
    converted, observed = got[365]
    assert not observed.any() and not converted.any()


# --------------------------------------------------------------------------- #
# 3. Deciles (R4.5 / R4.8).                                                    #
# --------------------------------------------------------------------------- #
def test_decile_of_is_rank_based_and_puts_the_top_scores_in_decile_nine():
    d = ca.decile_of(np.arange(100.0))
    assert d[0] == 0 and d[-1] == 9
    assert sorted(np.bincount(d, minlength=10).tolist()) == [10] * 10
    # rank-based, so a pile-up on one value does not collapse the split
    piled = ca.decile_of(np.array([0.5] * 50 + [0.9] * 50))
    assert set(piled.tolist()) == set(range(10))
    assert ca.decile_of(np.zeros(0)).size == 0


def test_a_planted_score_gradient_comes_back_as_decile_enrichment():
    """R4.8: if the top decile converts materially above the bottom, the model is
    finding future cases among its own negatives. Here the gradient is planted, so
    the sign is known in advance."""
    n = 1000
    rng = np.random.default_rng(1)
    scores = np.linspace(0, 1, n)
    index = np.zeros(n)
    obs = np.full(n, 10_000.0)
    # conversion probability rises with the score
    converts = rng.random(n) < scores
    first = np.where(converts, 500.0, np.nan)
    counts = ca.conversion_counts(first, index, obs, LW, horizons=(365,))
    table = ca.node_conversion_table(counts, scores, min_count=1)
    dec = table["horizons"][365]["deciles"]
    assert dec[9]["rate"] > dec[0]["rate"] + 0.5
    pooled = ca.pool_tables({0: table}, min_count=1)
    assert pooled[365]["top_minus_bottom"] > 0.5
    assert pooled[365]["n_nodes"] == 1


def test_deciles_are_absent_when_no_scores_are_supplied():
    """`--deciles off` is a real mode (it skips the readout solve), so the table
    has to be well-formed without them rather than carrying ten empty cells."""
    counts = ca.conversion_counts(np.array([500.0, np.nan]), np.zeros(2),
                                  np.full(2, 10_000.0), LW, horizons=(365,))
    table = ca.node_conversion_table(counts, None, min_count=1)
    assert "deciles" not in table["horizons"][365]
    pooled = ca.pool_tables({0: table}, min_count=1)
    assert all(c["n_observed"] == 0 for c in pooled[365]["deciles"])
    assert pooled[365]["rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 4. The closure fold, egress, and the mandatory framing.                      #
# --------------------------------------------------------------------------- #
class _Lay:
    """A hand-built closure: 0 <- 1,2 <- 3 (a diamond: 3's closure is {3,1,2,0})."""

    _CL = {0: (0,), 1: (1, 0), 2: (2, 0), 3: (3, 1, 2, 0)}

    def closure(self, c):
        return self._CL[int(c)]


def _cols(*triples):
    """The three parallel sidecar COLUMNS the fold reads (Arrow, not Rows)."""
    return (np.array([t[0] for t in triples]),
            np.array([t[1] for t in triples]),
            np.array([float(t[2]) for t in triples]))


def test_the_closure_fold_pushes_a_date_UP_to_every_ancestor_incl_a_diamond():
    """"First attestation of closure(c)" is the min over every node whose closure
    CONTAINS c. Node 3 sits under both 1 and 2, so attesting it dates 1, 2 and the
    root — once each, not twice — and never dates its siblings."""
    got = ca.first_attestation_matrix(
        *_cols((7, 103, 500)), [7], {103: 3}, _Lay(), 4)
    assert list(got[0]) == [500.0, 500.0, 500.0, 500.0]

    # attesting node 1 dates 1 and the root, but NOT 2 or 3
    got = ca.first_attestation_matrix(
        *_cols((7, 101, 500)), [7], {101: 1}, _Lay(), 4)
    assert got[0][0] == 500.0 and got[0][1] == 500.0
    assert got[0][2] == np.inf and got[0][3] == np.inf


def test_the_fold_keeps_the_EARLIEST_date_per_ancestor():
    """Two frontier nodes sharing an ancestor: the ancestor takes the earlier."""
    got = ca.first_attestation_matrix(
        *_cols((7, 101, 900), (7, 102, 400)), [7], {101: 1, 102: 2}, _Lay(), 4)
    assert got[0][0] == 400.0                # the root gets the earlier of the two
    assert got[0][1] == 900.0 and got[0][2] == 400.0


def test_unknown_persons_and_pruned_nodes_are_dropped_not_crashed_on():
    """The sidecar covers the whole sampled population and the DAG is pruned, so
    both kinds of extra row are normal and must be ignored."""
    got = ca.first_attestation_matrix(
        *_cols((999, 101, 500), (7, 555, 500)), [7], {101: 1}, _Lay(), 4)
    assert np.isinf(got).all()


def test_a_null_date_is_skipped_rather_than_folded_as_a_zero():
    """A NULL/NaT date must not become day 0, which would look like the earliest
    attestation there has ever been."""
    got = ca.first_attestation_matrix(
        np.array([7]), np.array([101]), np.array([np.nan]), [7], {101: 1},
        _Lay(), 4)
    assert np.isinf(got).all()


def test_a_duplicate_person_in_the_scored_split_raises_by_name():
    """R4.3's grain wall: the per-person horizon frame has ONE index date, so a
    multi-document person (exp 0111's doc unit) must stop this analysis loudly
    rather than silently attaching one person's index to two documents."""
    with pytest.raises(ValueError, match="PER PERSON"):
        ca.first_attestation_matrix(*_cols((7, 101, 500)), [7, 7], {101: 1},
                                    _Lay(), 4)


def test_cells_under_the_floor_are_marked_undisclosable_and_stay_out_of_pooled():
    """The per-node file is workspace-internal, so the arithmetic still happens —
    but a cell under 20 is flagged and never enters a published pooled figure."""
    small = ca.conversion_counts(np.array([500.0] * 5), np.zeros(5),
                                 np.full(5, 10_000.0), LW, horizons=(365,))
    big = ca.conversion_counts(np.array([500.0] * 40), np.zeros(40),
                               np.full(40, 10_000.0), LW, horizons=(365,))
    per_node = {0: ca.node_conversion_table(small), 1: ca.node_conversion_table(big)}
    assert per_node[0]["horizons"][365]["disclosable"] is False
    assert per_node[1]["horizons"][365]["disclosable"] is True
    pooled = ca.pool_tables(per_node)
    assert pooled[365]["n_nodes"] == 1 and pooled[365]["n_observed"] == 40


def test_every_table_carries_the_PU_LOWER_BOUND_language():
    """R4.6 is mandatory language, not a nicety: a conversion number quoted without
    it becomes "the contamination rate" within one copy-paste."""
    counts = ca.conversion_counts(np.array([500.0] * 40), np.zeros(40),
                                  np.full(40, 10_000.0), LW, horizons=H)
    pooled = ca.pool_tables({0: ca.node_conversion_table(counts)})
    text = ca.format_conversion_report(pooled, {"label_window_days": LW})
    assert "LOWER BOUND on PU channel 1" in text
    assert "UNMEASURED" in text and "care-fragmentation" in text
    assert "at least" in text
    assert "CASE-FINDING VALIDATION" in text
    assert "right-censoring" in text
    assert "EGRESS" in text
    assert "prospective" in ca.LOWER_BOUND_NOTE or "not a prospective" in text
