"""Review 2026-11-13 D5: `_base.yaml`'s plain-LDA values must not reach a DAG-gated
fit unannounced. `GATED_BASE_GUARDED` is exactly the knobs where `_base.yaml` and the
estimator's recommended preset differ, and `base_gated_leaks` names any a run would
take from `_base.yaml`."""
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = ROOT / "experiments" / "defaults"
_spec = importlib.util.spec_from_file_location("run_experiment",
                                               ROOT / "scripts" / "run_experiment.py")
rx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rx)

# yaml key -> (preset Param, how the yaml value maps to the Param's value)
_PARAM = {"n_bg": "gateNBg", "tpn": "gateTpn", "max_iter": "maxIter",
          "subsampling_rate": "subsamplingRate", "tau0": "learningOffset",
          "kappa": "learningDecay", "optimize_doc_concentration":
          "optimizeDocConcentration", "doc_concentration": "docConcentration",
          "gamma_shape": "gammaShape", "cavi_max_iter": "caviMaxIter",
          "cavi_tol": "caviTol", "gate_warm": "gateWarm",
          "gate_seed_leaves": "gateSeedLeaves", "gate_open_seed": "gateOpenSeed",
          "gate_seed_mode": "gateSeedMode"}


def test_guarded_knobs_are_exactly_where_base_and_the_preset_differ():
    pytest.importorskip("pyspark")
    from spark_vi.mllib.topic.dag_gated_lda import RECOMMENDED_PARAMS
    base = yaml.safe_load((DEFAULTS / "_base.yaml").read_text())
    differ = set()
    for key, param in _PARAM.items():
        if key not in base or param not in RECOMMENDED_PARAMS:
            continue
        pv = RECOMMENDED_PARAMS[param]
        pv = pv[0] if isinstance(pv, list) and len(pv) == 1 else pv
        if base[key] != pv:
            differ.add(key)
    assert differ == set(rx.GATED_BASE_GUARDED)


def test_leaks_named_unless_the_cohort_or_front_matter_sets_them(tmp_path):
    (tmp_path / "_base.yaml").write_text("n_bg: 20\nkappa: 0.7\ntau0: 64\n")
    (tmp_path / "c.yaml").write_text("n_bg: 200\n")
    assert rx.base_gated_leaks("c", {}, tmp_path) == {"kappa": 0.7}
    assert rx.base_gated_leaks("c", {"kappa": 0.51}, tmp_path) == {}


def test_the_mondo_cohort_sets_every_guarded_knob():
    assert rx.base_gated_leaks("population_mondo_all", {}, DEFAULTS) == {}
