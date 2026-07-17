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
        assert eff["disease"] == "diabetes"
        mod.validate_frontmatter(eff)                 # must not exit
        args = mod.build_dag_placement_args(eff, "/out")
        assert args[args.index("--init") + 1] == init


def test_rare6_forest_experiment_parses_and_builds(monkeypatch):
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("p.d", "bill"))
    eff = _load_effective(mod, "docs/experiments/0055-dag-placement-rare6-forest.md")
    assert eff["model_class"] == "dag_placement"
    assert eff["disease"] == "rare6"
    assert eff["min_n"] == 20 and eff["n_bg"] == 40 and eff["person_mod"] == 1
    assert eff["init"] == "spectral"
    mod.validate_frontmatter(eff)                     # must not exit
    args = mod.build_dag_placement_args(eff, "/out")
    assert args[args.index("--disease") + 1] == "rare6"
    assert args[args.index("--min-n") + 1] == "20"
    assert "--K" not in args                          # K is emergent
    assert args[args.index("--node-alpha-scale") + 1] == "0.1"   # asymmetric (52+)


def test_dag_placement_batch_is_asymmetric_alpha(monkeypatch):
    # 0052-0055 all carry the block-asymmetric prior (node_alpha_scale: 0.1);
    # the engine default stays symmetric (1.0) via _base for any other exp.
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("p.d", "bill"))
    for exp in [
        "docs/experiments/0052-dag-placement-diabetes-random.md",
        "docs/experiments/0053-dag-placement-diabetes-spectral.md",
        "docs/experiments/0054-dag-placement-diabetes-strip-both.md",
        "docs/experiments/0055-dag-placement-rare6-forest.md",
    ]:
        eff = _load_effective(mod, exp)
        assert eff["node_alpha_scale"] == 0.1
        mod.validate_frontmatter(eff)
        args = mod.build_dag_placement_args(eff, "/out")
        assert args[args.index("--node-alpha-scale") + 1] == "0.1"
