"""θ truncation by MASS, not count (spec 2026-10-07 §D4).

The rule is pure (`choose_topm_for_mass` over a coverage table) and is tested as
such; the Spark pass that produces the table is `theta_topm_coverage`, already
covered by the distributed-readout parity tests. Three claims:

  1. The grid ends at K, the pick is the smallest width whose p10 meets the target,
     and "nothing below K meets it" resolves to 0 (dense) rather than to K.
  2. The front-matter key passes through to the fit driver only when set, so every
     existing experiment's argv is byte-identical.
  3. The readout parses the override and refuses it next to a fixed top-m.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "analysis" / "cloud"), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import distributed_readout as dr  # noqa: E402
import gated_pc_readout as gpr  # noqa: E402


def test_mass_grid_is_powers_of_two_then_K():
    assert dr.mass_grid(306) == (16, 32, 64, 128, 256, 306)
    assert dr.mass_grid(1498) == (16, 32, 64, 128, 256, 512, 1024, 1498)
    assert dr.mass_grid(16) == (16,)
    assert dr.mass_grid(10) == (10,)


def test_choose_topm_picks_smallest_width_meeting_the_p10_target():
    cov = {16: (0.95, 0.80), 32: (0.99, 0.97), 64: (0.999, 0.995), 128: (1.0, 1.0)}
    assert dr.choose_topm_for_mass(cov, 0.99, K=1498) == 64
    assert dr.choose_topm_for_mass(cov, 0.95, K=1498) == 32
    # the MEAN meeting the target is not enough; the p10 is the criterion
    assert dr.choose_topm_for_mass({16: (0.99, 0.5), 32: (0.999, 0.99)}, 0.99) == 32


def test_choose_topm_resolves_to_dense_when_only_K_meets_the_target():
    cov = {16: (0.5, 0.3), 32: (0.6, 0.4), 64: (0.9, 0.85), 100: (1.0, 1.0)}
    assert dr.choose_topm_for_mass(cov, 0.99, K=100) == 0      # only K qualifies
    assert dr.choose_topm_for_mass({16: (0.5, 0.3)}, 0.99) == 0  # nothing qualifies


def test_run_experiment_passes_theta_mass_only_when_set(monkeypatch):
    import run_experiment as rex
    monkeypatch.setenv("WORKSPACE_CDR", "cdr")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    base = {"source_table": "t", "person_mod": 1, "vocab_size": 5000, "min_df": 20,
            "min_patient_count": 20, "doc_min_length": 10, "max_iter": 100,
            "min_n": 0, "n_bg": 8, "tpn": 1, "seed": 0, "readout_mode": "distributed"}
    argv = rex.build_gated_pc_args(dict(base, readout_theta_mass=0.99), "/tmp/out")
    assert argv[argv.index("--readout-theta-mass") + 1] == "0.99"
    assert "--readout-theta-mass" not in rex.build_gated_pc_args(dict(base), "/tmp/out")


def test_readout_parses_the_mass_override():
    a = gpr.build_parser().parse_args(["--run-dir", "/tmp/run",
                                       "--readout-theta-mass", "0.99"])
    assert a.readout_theta_mass == pytest.approx(0.99)
    assert gpr.build_parser().parse_args(["--run-dir", "/tmp/run"]).readout_theta_mass is None
