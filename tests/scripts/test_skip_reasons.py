"""The readout's three skip reasons (`analysis.pc.evaluate`, incident-eval spec R2.1).

What remains of the dual prevalent/incident readout's tests (spec E2 / plan WP4)
after the pre-index closure was cut in the 2026-10 cleanup: the eligibility mask and
the lean collect's fourth (pre-index) run went with it; new-diagnosis evaluation is
now `prevalence_check.py` over the whole-history sidecar. The skip reasons are live:
a constant prediction column is skipped with its own reason (off by default, so
recorded macros do not move), and the three reasons are counted separately.

All pure; no Spark, no cluster.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.pc.evaluate import (SKIP_CONSTANT, SKIP_DEGENERATE, SKIP_SMALL,  # noqa: E402
                                  _macro, _score_label)


def test_a_constant_column_with_both_classes_is_skipped_not_scored_at_half():
    """The C2.1 population, in one column: a train-degenerate node's constant
    fallback column that has ACQUIRED negatives. `roc_auc_score` is happy to
    return exactly 0.5 for it, `skipped=None`, straight into the macro."""
    y = np.array([1, 1, 0, 0, 1, 0], dtype=float)
    p = np.full(6, 0.83)                       # the head's constant fallback
    off = _score_label(y, p)
    assert off["skipped"] is None and off["auc"] == pytest.approx(0.5)
    on = _score_label(y, p, skip_constant=True)
    assert on["auc"] is None and on["skip_code"] == SKIP_CONSTANT
    assert "constant prediction column" in on["skipped"]


def test_the_guard_is_off_by_default_so_prevalent_macros_do_not_move():
    """Every recorded prevalent number in the repo was computed without it; the
    incident arm turns it on and reports the count instead."""
    y = np.array([1, 0, 1, 0], dtype=float)
    p = np.full(4, 0.5)
    assert _score_label(y, p)["skipped"] is None
    informative = np.array([0.9, 0.1, 0.8, 0.2])
    assert _score_label(y, informative, skip_constant=True)["skipped"] is None


def test_the_three_skip_reasons_are_counted_separately_and_never_summed():
    """Spec §8, standing rule 5. Three reasons, three counts."""
    per_label = {
        0: _score_label(np.array([1.0, 1.0, 1.0]), np.array([.1, .2, .3])),
        1: _score_label(np.array([1.0, 0.0, 1.0]), np.array([.1, .2, .3]),
                        min_count=20),
        2: _score_label(np.array([1.0, 0.0, 1.0]), np.full(3, 0.7),
                        skip_constant=True),
        3: _score_label(np.array([1.0, 0.0, 1.0]), np.array([.9, .1, .8])),
    }
    macro = _macro(per_label)
    assert macro["n_labels_scored"] == 1
    assert macro["skipped_by_reason"] == {SKIP_DEGENERATE: 1, SKIP_SMALL: 1,
                                          SKIP_CONSTANT: 1}
    # the lumped total is still there and still equals the sum, but the block is
    # what gets reported (never the sum of the three)
    assert macro["n_labels_skipped"] == 3
