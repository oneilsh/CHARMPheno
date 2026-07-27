import importlib
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent.parent / "docs" / "experiments"


def _fm(name):
    mod = importlib.import_module("run_experiment")
    return mod, mod.read_frontmatter(EXP / name)


def test_exp_0071_is_valid_lookback_three_domain_multidomain():
    mod, fm = _fm("0071-multidomain-rare6-cond-drug-obs.md")
    mod.validate_frontmatter(fm)                       # must not sys.exit
    assert fm["model_class"] == "multidomain"
    assert fm["disease"] == "rare6"
    assert fm["window_mode"] == "lookback"
    assert fm["domains"] == "drug_era,observation"


def test_exp_0070_migrated_to_explicit_forward_two_domain():
    mod, fm = _fm("0070-multidomain-diabetes-drug-condition.md")
    mod.validate_frontmatter(fm)
    assert fm["domains"] == "drug_era"
    assert fm["window_mode"] == "forward"
