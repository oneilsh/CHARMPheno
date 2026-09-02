"""Incident-local cells + P-strata (spec E3 / plan WP5) — `conditional_readout`.

The cell was already built; what E3 adds is an eligibility array and a stratum key.
That makes the FIRST test here the important one: with eligibility all-ones and the
stratum collapsed, the function must return NUMERICALLY THE SAME conditional block
it returned before, because that block is what exps 0104/0109/0110 are compared on.
Everything else is new behaviour that only exists on the incident path.

Four groups:

  1. **The identity regression.** eligibility=None is the old function; eligibility
     all-ones changes no metric on any edge.
  2. **The incident filter (D5).** A prior carrier of the CHILD leaves both classes
     of that edge's cell; a prior carrier of the PARENT does not leave anything —
     it is a stratum, never a gate (D6, and the plan's "intersect at the cohort" is
     read as the child, deliberately: see `conditional_readout`'s docstring).
  3. **The strata.** Two strata, both labelled, with materially different AUCs on a
     planted fixture, and a shared-edge-set comparison beside the two averages.
  4. **EGRESS (R3.5).** No published cell — pooled or stratified, printed or in the
     results dict — carries either class under 20, and the suppressed ones say so
     without saying by how much.

All pure numpy; no Spark.
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

import gated_pc_cloud as gpc  # noqa: E402
from gated_pc_cloud import STRATUM_P_KNOWN, STRATUM_P_UNKNOWN  # noqa: E402

# One parent (node 1) under the root, with three children (2, 3, 4).
PARENT_INT = {0: [], 1: [0], 2: [1], 3: [1], 4: [1]}
C = 5


def _corpus(D=1200, seed=0):
    """A parent cohort with three children and a real (if modest) signal.

    Every child gets enough positives and negatives that the 20/20 egress floor is
    cleared BOTH pooled and per-stratum — otherwise the strata tests would be
    testing the suppression path rather than the metric."""
    rng = np.random.default_rng(seed)
    y = np.zeros((D, C), dtype=np.float64)
    y[:, 0] = 1.0
    y[:, 1] = 1.0                                  # everyone is in P's cohort
    proba = rng.random((D, C))
    for j, c in enumerate((2, 3, 4)):
        # a third of the cohort is positive for each child, ranked by a noisy score
        take = slice(j * D // 3, (j + 1) * D // 3)
        y[take, c] = 1.0
        proba[take, c] += 0.6                      # signal: positives score higher
    mask = np.ones((D, C), dtype=np.float64)
    return proba, y, mask


def _strip_strata(cond):
    """The block minus everything E3 adds, for the identity comparison."""
    edges = [{k: v for k, v in e.items() if k not in ("strata", "n_neg")}
             for e in cond["edges"]]
    return {"edges": edges, "parents": cond["parents"], "ece": cond["ece"],
            "node_ece": cond["node_ece"]}


# --------------------------------------------------------------------------- #
# 1. The identity regression — the whole safety net.                           #
# --------------------------------------------------------------------------- #
def test_all_ones_eligibility_reproduces_todays_conditional_block_exactly():
    """Spec E3 acceptance, verbatim: "with eligibility all-ones and the stratum
    collapsed, output is numerically identical to today's conditional block"."""
    proba, y, mask = _corpus()
    base = gpc.conditional_readout(proba, y, mask, PARENT_INT, C, min_count=20)
    inc = gpc.conditional_readout(proba, y, mask, PARENT_INT, C, min_count=20,
                                  eligibility=np.ones_like(y, dtype=bool))
    assert _strip_strata(inc) == base
    assert len(base["edges"]) == 3
    # ...and the prevalent block gained NO keys at all
    assert set(base) == {"edges", "parents", "ece", "node_ece"}
    assert "incident" in inc


def test_the_collapsed_stratum_suppresses_rather_than_inventing_a_number():
    """All-ones eligibility means nobody carried P pre-index, so the P-known
    stratum is empty — which must read as SUPPRESSED, never as an AUC over zero
    rows."""
    proba, y, mask = _corpus()
    inc = gpc.conditional_readout(proba, y, mask, PARENT_INT, C, min_count=20,
                                  eligibility=np.ones_like(y, dtype=bool))
    for e in inc["edges"]:
        assert e["strata"][STRATUM_P_KNOWN]["suppressed"] == "<20"
        assert "n_pos" not in e["strata"][STRATUM_P_KNOWN]
        assert e["strata"][STRATUM_P_UNKNOWN]["cond_auc"] is not None
    assert inc["incident"]["strata"][STRATUM_P_KNOWN]["n_edges"] == 0
    assert inc["incident"]["strata"][STRATUM_P_UNKNOWN]["n_edges"] == 3


# --------------------------------------------------------------------------- #
# 2. The incident filter (D5) — child gates, parent stratifies.                #
# --------------------------------------------------------------------------- #
def test_a_prior_carrier_of_the_CHILD_leaves_that_edges_cell():
    """D5 clause (i) intersected with D2: eligibility for c removes rows from the
    edge P->c, and from BOTH classes of it."""
    proba, y, mask = _corpus()
    elig = np.ones_like(y, dtype=bool)
    # rows 0-399 are c=2's positives and 400-1199 its negatives, so this strips
    # 200 of each class — the symmetry D2 demands, in one fixture.
    elig[:200, 2] = False
    elig[400:600, 2] = False
    inc = gpc.conditional_readout(proba, y, mask, PARENT_INT, C, min_count=20,
                                  eligibility=elig)
    base = gpc.conditional_readout(proba, y, mask, PARENT_INT, C, min_count=20)
    got = {e["child"]: e for e in inc["edges"]}
    want = {e["child"]: e for e in base["edges"]}
    assert want[2]["n_pos"] == 400 and got[2]["n_pos"] == 200      # positives dropped
    # the prevalent edge records `cohort` and `n_pos`; 1200 - 400 = 800 negatives
    assert want[2]["cohort"] - want[2]["n_pos"] == 800
    assert got[2]["n_neg"] == 600                                  # negatives too
    # the OTHER children's cells are untouched: eligibility is per (doc, node)
    for c in (3, 4):
        assert got[c]["n_pos"] == want[c]["n_pos"]
        assert got[c]["cond_auc"] == pytest.approx(want[c]["cond_auc"])


def test_the_parent_stratum_is_NEVER_a_gate():
    """D6 in one assertion: making every document a prior carrier of the PARENT
    must not remove a single row from any cell. Requiring pre-index P would starve
    the cells and delete the de-novo positives — the harder, more interesting half."""
    proba, y, mask = _corpus()
    elig = np.ones_like(y, dtype=bool)
    elig[:, 1] = False                          # everyone already carried P
    inc = gpc.conditional_readout(proba, y, mask, PARENT_INT, C, min_count=20,
                                  eligibility=elig)
    base = gpc.conditional_readout(proba, y, mask, PARENT_INT, C, min_count=20)
    got = {e["child"]: e for e in inc["edges"]}
    want = {e["child"]: e for e in base["edges"]}
    for c in (2, 3, 4):
        assert got[c]["n_pos"] == want[c]["n_pos"]
        assert got[c]["cond_auc"] == pytest.approx(want[c]["cond_auc"])
    # ...it only moved every row into the OTHER stratum
    for e in inc["edges"]:
        assert e["strata"][STRATUM_P_KNOWN]["cond_auc"] is not None
        assert e["strata"][STRATUM_P_UNKNOWN]["suppressed"] == "<20"


# --------------------------------------------------------------------------- #
# 3. The two strata.                                                           #
# --------------------------------------------------------------------------- #
def test_two_strata_with_different_AUCs_are_both_labelled_and_both_reported():
    """The two strata routinely have materially different AUCs and must not be
    mushed into one unlabeled number (D6). Here the signal is PLANTED only in the
    P-known half, so the difference is known in advance."""
    D = 1200
    rng = np.random.default_rng(4)
    y = np.zeros((D, C), dtype=np.float64)
    y[:, 0] = 1.0
    y[:, 1] = 1.0
    proba = rng.random((D, C))
    elig = np.ones((D, C), dtype=bool)
    elig[: D // 2, 1] = False                   # the first half carried P pre-index
    for c in (2, 3):
        pos = np.zeros(D, dtype=bool)
        pos[np.arange(D) % 4 == (c - 2)] = True
        y[pos, c] = 1.0
        # signal only among the P-known half
        boost = pos & (~elig[:, 1])
        proba[boost, c] += 1.0
    mask = np.ones((D, C), dtype=np.float64)
    inc = gpc.conditional_readout(proba, y, mask, PARENT_INT, C, min_count=20,
                                  eligibility=elig)
    for e in inc["edges"]:
        known = e["strata"][STRATUM_P_KNOWN]
        unknown = e["strata"][STRATUM_P_UNKNOWN]
        assert known["cond_auc"] > 0.9          # planted signal
        assert unknown["cond_auc"] < 0.6        # noise
    s = inc["incident"]["strata"]
    assert s[STRATUM_P_KNOWN]["cond_auc"] > s[STRATUM_P_UNKNOWN]["cond_auc"]
    # the shared edge set is where the two are actually a comparison (R2.2/R3.3)
    sh = inc["incident"]["strata_shared_edge_set"]
    assert sh["n_edges"] == 2
    assert sh[STRATUM_P_KNOWN] > sh[STRATUM_P_UNKNOWN]
    assert sorted(sh["edge_set"]) == [[1, 2], [1, 3]]


def test_pooled_is_reported_first_and_carries_the_D7_naming_rule():
    """R3.2 (pooled primary) and R2.4/D7 (every incident output says what it is)."""
    proba, y, mask = _corpus()
    elig = np.ones_like(y, dtype=bool)
    elig[: len(y) // 2, 1] = False
    inc = gpc.conditional_readout(proba, y, mask, PARENT_INT, C, min_count=20,
                                  eligibility=elig)["incident"]
    assert "PREVALENT-FIT" in inc["naming"]
    assert inc["tags"]["claim_type"] == "discrimination"
    assert "NEVER GATING" in inc["stratum_definition"]
    assert inc["pooled"]["n_edges"] == 3
    text = gpc.format_incident_conditional({"incident": inc})
    lines = [ln for ln in text.splitlines() if "cond_AUC=" in ln]
    assert "pooled" in lines[0]                 # pooled FIRST, strata second
    assert STRATUM_P_KNOWN in lines[1] and STRATUM_P_UNKNOWN in lines[2]
    assert "PREVALENT-FIT" in text and "DISCRIMINATION" in text
    assert gpc.format_incident_conditional({}).startswith("[incident conditional]")


# --------------------------------------------------------------------------- #
# 4. EGRESS (R3.5).                                                            #
# --------------------------------------------------------------------------- #
def _all_counts(obj):
    """Every n_pos/n_neg anywhere in the emitted structure."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("n_pos", "n_neg") and isinstance(v, int):
                out.append((k, v))
            else:
                out.extend(_all_counts(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_all_counts(v))
    return out


def test_no_published_cell_carries_either_class_under_the_egress_floor():
    """R3.5: stratified cell tables are a disclosure surface. Nothing under 20
    leaves the workspace — in the results dict OR in the printed table."""
    D = 300
    rng = np.random.default_rng(9)
    y = np.zeros((D, C), dtype=np.float64)
    y[:, 0] = 1.0
    y[:, 1] = 1.0
    proba = rng.random((D, C))
    elig = np.ones((D, C), dtype=bool)
    # a lopsided split: the P-known stratum is deliberately tiny
    elig[:25, 1] = False
    for c in (2, 3, 4):
        y[np.arange(D) % 3 == (c - 2), c] = 1.0
    mask = np.ones((D, C), dtype=np.float64)
    inc = gpc.conditional_readout(proba, y, mask, PARENT_INT, C, min_count=5,
                                  eligibility=elig)
    for kind, n in _all_counts(inc):
        assert n >= gpc.EGRESS_MIN_COUNT, (kind, n)
    # the suppressed cells are COUNTED, not shown
    assert inc["incident"]["n_edges_suppressed"] >= 0
    for e in inc["edges"]:
        for stratum in e.get("strata", {}).values():
            if "suppressed" in stratum:
                assert set(stratum) == {"suppressed", "reason"}


def test_a_pooled_cell_under_the_floor_is_suppressed_whole():
    """min_count is the statistical dial; the egress floor is separate and higher.
    A cell computed at min_count=5 must still not be PUBLISHED at n_pos=6."""
    D = 200
    rng = np.random.default_rng(2)
    y = np.zeros((D, C), dtype=np.float64)
    y[:, 0] = 1.0
    y[:, 1] = 1.0
    y[:6, 2] = 1.0                              # six positives: computable, not
    y[:80, 3] = 1.0                             # publishable
    proba = rng.random((D, C))
    mask = np.ones((D, C), dtype=np.float64)
    elig = np.ones((D, C), dtype=bool)
    inc = gpc.conditional_readout(proba, y, mask, PARENT_INT, C, min_count=5,
                                  eligibility=elig)
    by_child = {e["child"]: e for e in inc["edges"]}
    assert by_child[2]["suppressed"] == "<20"
    assert "cond_auc" not in by_child[2] and "n_pos" not in by_child[2]
    assert by_child[3]["cond_auc"] is not None
    # a suppressed edge is excluded from the pooled average, not averaged as zero
    assert inc["incident"]["pooled"]["n_edges"] == 1
    assert inc["incident"]["n_edges_suppressed"] == 1
    # the printed table survives suppressed edges without a KeyError
    text = gpc.format_conditional_readout(inc, {}, {})
    assert "depth" in text
