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
    assert eff["init"] == "random"                    # dense spectral too slow at K=180/V=10k
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


def test_dag_placement_svi_schedule_defaults(monkeypatch):
    # _base sets mini-batch SVI (0.1) + a gentler slow-start (tau0 10, kappa 0.7)
    # + max_iter 200; all dag_placement exps inherit it and emit the CLI flags.
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("p.d", "bill"))
    eff = _load_effective(mod, "docs/experiments/0055-dag-placement-rare6-forest.md")
    assert eff["mini_batch_fraction"] == 0.1
    assert eff["learning_rate_tau0"] == 10.0
    assert eff["learning_rate_kappa"] == 0.7
    assert eff["max_iter"] == 200
    args = mod.build_dag_placement_args(eff, "/out")
    assert args[args.index("--mini-batch-fraction") + 1] == "0.1"
    assert args[args.index("--learning-rate-tau0") + 1] == "10.0"
    assert args[args.index("--learning-rate-kappa") + 1] == "0.7"
    assert args[args.index("--max-iter") + 1] == "200"


def test_rare6_spectral_init_diagnostic_parses_and_builds(monkeypatch):
    # exp 0059 = the best 2x2 cell (0058: sym alpha + strip both) with init flipped
    # random->spectral, to isolate the init axis of the 0.709->0.585 under-training
    # confound. Same corpus/schedule as 0058; only `init` differs.
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("p.d", "bill"))
    eff = _load_effective(
        mod, "docs/experiments/0059-dag-placement-rare6-sym-stripboth-spectral.md")
    assert eff["disease"] == "rare6"
    assert eff["init"] == "spectral"
    assert eff["node_alpha_scale"] == 1.0          # symmetric (matches 0058)
    assert eff["strip_mode"] == "both"             # strip_both (matches 0058)
    assert eff["spectral_max_vocab"] == 12000      # dense-path guard above V~10000
    mod.validate_frontmatter(eff)                  # must not exit
    args = mod.build_dag_placement_args(eff, "/out")
    assert args[args.index("--init") + 1] == "spectral"
    assert args[args.index("--strip-mode") + 1] == "both"
    assert args[args.index("--node-alpha-scale") + 1] == "1.0"


def test_rare6_alpha_by_strip_2x2_grid(monkeypatch):
    # The rare6 2x2: strip_mode (test_only|both) x node_alpha_scale (0.1|1.0).
    #   0055 asym/test_only  0056 sym/test_only
    #   0057 asym/both       0058 sym/both
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("p.d", "bill"))
    grid = {
        "0055-dag-placement-rare6-forest": (0.1, "test_only"),
        "0056-dag-placement-rare6-symmetric": (1.0, "test_only"),
        "0057-dag-placement-rare6-asym-stripboth": (0.1, "both"),
        "0058-dag-placement-rare6-sym-stripboth": (1.0, "both"),
    }
    for slug, (scale, strip) in grid.items():
        eff = _load_effective(mod, f"docs/experiments/{slug}.md")
        assert eff["disease"] == "rare6"
        assert eff["node_alpha_scale"] == scale
        assert eff["strip_mode"] == strip
        mod.validate_frontmatter(eff)
        args = mod.build_dag_placement_args(eff, "/out")
        assert args[args.index("--node-alpha-scale") + 1] == str(scale)
        assert args[args.index("--strip-mode") + 1] == strip
