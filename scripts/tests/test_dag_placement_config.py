"""The dag_placement A/B experiment files parse, merge, and build valid argv."""
import importlib
from pathlib import Path

# Verified run_experiment APIs (scripts/run_experiment.py):
#   read_frontmatter(path: Path) -> dict
#   load_defaults(cohort: str, defaults_dir: Path) -> dict   (merges _base + <cohort>.yaml)
#   merge_config(base: dict, override: dict) -> dict
_REPO = Path(__file__).resolve().parents[2]        # scripts/tests/ -> repo root
_DEFAULTS = _REPO / "experiments" / "defaults"


def _load_effective(mod, exp_path):
    fm = mod.read_frontmatter(_REPO / exp_path)
    defaults = mod.load_defaults(fm["cohort"], _DEFAULTS)
    return mod.merge_config(defaults, fm)


def test_diabetes_experiments_parse_and_build(monkeypatch):
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("p.d", "bill"))
    for exp, init in [("docs/experiments/0052-dag-placement-diabetes-random.md", "random"),
                      ("docs/experiments/0053-dag-placement-diabetes-spectral.md", "spectral")]:
        eff = _load_effective(mod, exp)
        assert eff["model_class"] == "dag_placement"
        assert eff["init"] == init
        assert eff["anchor"] == 201820
        mod.validate_frontmatter(eff)                 # must not exit
        args = mod.build_dag_placement_args(eff, "/out")
        assert args[args.index("--init") + 1] == init
