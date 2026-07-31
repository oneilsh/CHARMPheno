"""Tests for the --fixed-only fast readout path (no spark_vi / pyspark).

build_fixed_result / render_fixed_markdown are pure aggregation over synthetic
evaluation dicts. The worker-dispatch test stubs multidomain_weighting (as
test_weighting_parallel does) to prove --fixed-only routes to
evaluate_anchor_fixed, not the slow nested evaluator.
"""
import sys
import types
from pathlib import Path
from statistics import median

_CLOUD = str(Path(__file__).resolve().parent.parent)
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)

_ORDER = ["condition", "drug", "measurement", "fixed:inclusive"]
_RK = ("0.1", "0.25", "0.5", "0.8")


def _metric(ap):
    return {"ap": ap, "precision_at_recall": {rk: ap for rk in _RK}}


def _evaluation(anchor, aps_by_repeat, n_positive=10):
    # aps_by_repeat: list over repeats of {strategy: ap}
    return {
        "anchor": anchor, "n_docs": 1000, "n_positive": n_positive,
        "prevalence": n_positive / 1000.0,
        "strategy_order": list(_ORDER),
        "domain_names": ["condition", "drug", "measurement"],
        "repeats": [
            {"repeat": i, "strategies": {s: _metric(aps[s]) for s in _ORDER}}
            for i, aps in enumerate(aps_by_repeat)
        ],
    }


def test_build_fixed_result_medians_over_repeats_then_anchors():
    from multidomain_weighting_readout import build_fixed_result

    # anchor A: two repeats -> per-strategy median over repeats
    evA = _evaluation(1, [
        {"condition": 0.20, "drug": 0.05, "measurement": 0.10, "fixed:inclusive": 0.30},
        {"condition": 0.30, "drug": 0.15, "measurement": 0.20, "fixed:inclusive": 0.40},
    ], n_positive=100)  # >= floor -> count shown
    evB = _evaluation(2, [
        {"condition": 0.10, "drug": 0.02, "measurement": 0.50, "fixed:inclusive": 0.55},
    ], n_positive=6)    # < floor -> count suppressed
    artifact = {"targets": [
        {"anchor": 1, "concept_id": 11, "name": "A"},
        {"anchor": 2, "concept_id": 22, "name": "B"},
    ]}
    result = build_fixed_result(artifact, [evA, evB], min_cell=20)

    assert result["mode"] == "fixed"
    assert result["strategy_order"] == _ORDER
    # anchor A condition AP = median(0.20, 0.30) = 0.25
    a = next(x for x in result["anchors"] if x["anchor"] == 1)
    assert a["strategies"]["condition"]["ap"] == 0.25
    assert a["strategies"]["fixed:inclusive"]["ap"] == median([0.30, 0.40])
    assert a["dominant_domain"] == "condition"          # highest domain-only AP
    assert a["count_suppressed"] is False and a["n_positive"] == 100
    # macro measurement AP = median over anchors of their medians = median(0.15, 0.50)
    assert result["macro"]["measurement"]["ap"] == median([median([0.10, 0.20]), 0.50])
    assert set(result["macro"]["condition"]["median_precision_at_recall"]) == set(_RK)

    # sub-floor anchor B: count + prevalence suppressed; AP still present
    b = next(x for x in result["anchors"] if x["anchor"] == 2)
    assert b["count_suppressed"] is True
    assert b["n_positive"] is None and b["prevalence"] is None
    assert b["dominant_domain"] == "measurement"        # 0.50 is highest domain AP
    assert b["strategies"]["measurement"]["ap"] == 0.50  # aggregate stat kept
    assert result["n_count_suppressed"] == 1


def test_render_fixed_markdown_has_macro_and_per_anchor_sections():
    from multidomain_weighting_readout import build_fixed_result, render_fixed_markdown

    evA = _evaluation(1, [
        {"condition": 0.2, "drug": 0.05, "measurement": 0.4, "fixed:inclusive": 0.45}],
        n_positive=8)  # < floor -> suppressed in the rendered table
    artifact = {"targets": [{"anchor": 1, "concept_id": 11, "name": "Long QT"}]}
    md = render_fixed_markdown(build_fixed_result(artifact, [evA], min_cell=20))
    assert "Macro median AP" in md
    assert "Per-anchor AP" in md
    assert "Long QT" in md
    assert "fixed:inclusive" in md
    assert "<20" in md          # suppressed count shown as the floor marker
    assert "| 8 |" not in md    # raw sub-floor count never rendered


def _install_stub(calls):
    fake = types.ModuleType("multidomain_weighting")

    def evaluate_anchor_fixed(bows, lam_dict, lay, frontiers, *, anchor,
                              parent_int, domain_labels, **cfg):
        calls.append(("fixed", anchor, tuple(sorted(cfg))))
        return _evaluation(anchor, [{s: 0.1 for s in _ORDER}])

    def evaluate_anchor_nested(*a, **k):
        calls.append(("nested", k.get("anchor")))
        return {}

    fake.evaluate_anchor_fixed = evaluate_anchor_fixed
    fake.evaluate_anchor_nested = evaluate_anchor_nested
    prior = sys.modules.get("multidomain_weighting")
    sys.modules["multidomain_weighting"] = fake
    return prior


def test_worker_dispatches_to_fixed_when_mode_is_fixed():
    from multidomain_weighting_readout import _worker_init, _worker_eval

    calls = []
    prior = _install_stub(calls)
    try:
        cv = {"outer_folds": 5, "repeats": 3, "seed": 0, "mode": "fixed"}
        _worker_init({}, {}, None, [], {}, {}, cv)
        target, evaluation, error = _worker_eval(
            {"anchor": 7, "concept_id": 70, "name": "X"})
    finally:
        if prior is None:
            sys.modules.pop("multidomain_weighting", None)
        else:
            sys.modules["multidomain_weighting"] = prior

    assert error is None
    assert calls == [("fixed", 7, ("outer_folds", "repeats", "seed"))]  # no 'mode'
    assert evaluation["anchor"] == 7
