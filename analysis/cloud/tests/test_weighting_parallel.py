"""Tests for the per-anchor parallelism plumbing in the weighting readout.

Stubs ``multidomain_weighting.evaluate_anchor_nested`` via sys.modules so this
runs without the cluster stack. The fork-based pool workers inherit the parent's
sys.modules, so they see the same stub — letting us exercise the real parallel
path and assert it matches the serial path exactly.
"""
import sys
import types


def _install_stub_evaluator():
    fake = types.ModuleType("multidomain_weighting")

    def evaluate_anchor_nested(bows, lam_dict, lay, frontiers, *, anchor,
                               parent_int, domain_labels, **cv_config):
        if anchor == 999:
            raise ValueError("boom")
        return {"anchor": anchor, "seed": cv_config.get("seed")}

    fake.evaluate_anchor_nested = evaluate_anchor_nested
    prior = sys.modules.get("multidomain_weighting")
    sys.modules["multidomain_weighting"] = fake
    return prior


def _restore(prior):
    if prior is None:
        sys.modules.pop("multidomain_weighting", None)
    else:
        sys.modules["multidomain_weighting"] = prior


def _artifact():
    return {
        "bows": {}, "lam_dict": {}, "lay": None, "frontiers": [],
        "parent_int": {}, "domain_labels": {},
        "targets": [
            {"anchor": 1, "concept_id": 10, "name": "A"},
            {"anchor": 2, "concept_id": 20, "name": "B"},
            {"anchor": 999, "concept_id": 30, "name": "C"},  # stub raises for 999
        ],
    }


_CV = {"outer_folds": 5, "inner_folds": 4, "repeats": 5, "grid_step": 0.05, "seed": 0}


def test_serial_evaluation_preserves_order_and_captures_failures():
    from multidomain_weighting_readout import _evaluate_targets, _partition_results

    prior = _install_stub_evaluator()
    try:
        results = _evaluate_targets(_artifact(), _CV, jobs=1)
    finally:
        _restore(prior)

    assert [t["anchor"] for t, _, _ in results] == [1, 2, 999]
    assert results[0][1] == {"anchor": 1, "seed": 0} and results[0][2] is None
    assert results[2][1] is None and results[2][2] == "boom"

    evals, evaluated, failures = _partition_results(results)
    assert [t["anchor"] for t in evaluated] == [1, 2]
    assert len(evals) == 2
    assert len(failures) == 1 and failures[0]["anchor"] == 999
    assert "boom" in failures[0]["reason"]


def test_parallel_matches_serial_exactly():
    from multidomain_weighting_readout import _evaluate_targets

    prior = _install_stub_evaluator()
    try:
        serial = _evaluate_targets(_artifact(), _CV, jobs=1)
        parallel = _evaluate_targets(_artifact(), _CV, jobs=2)
    finally:
        _restore(prior)

    # identical ordered (target, evaluation, error) triples regardless of --jobs
    assert parallel == serial
