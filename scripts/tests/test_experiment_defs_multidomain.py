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


def test_exps_0070_0071_are_pinned_full_batch():
    # Both baselines pin mini_batch_fraction 0.0 so they don't inherit _base's 0.1
    # once the multidomain driver wires the mini-batch knob.
    for name in ("0070-multidomain-diabetes-drug-condition.md",
                 "0071-multidomain-rare6-cond-drug-obs.md"):
        _, fm = _fm(name)
        assert fm["mini_batch_fraction"] == 0.0, name


def test_exp_0072_is_the_minibatch_ab_of_0071():
    mod, fm = _fm("0072-multidomain-rare6-cond-drug-obs-minibatch.md")
    mod.validate_frontmatter(fm)
    assert fm["model_class"] == "multidomain"
    assert fm["disease"] == "rare6" and fm["window_mode"] == "lookback"
    assert fm["domains"] == "drug_era,observation"
    # the A/B distinction: mini-batch schedule (vs 0071's pinned 0.0)
    assert fm["mini_batch_fraction"] == 0.1
    assert fm["learning_rate_tau0"] == 10.0 and fm["learning_rate_kappa"] == 0.7


def test_exps_0071_0072_strip_the_ppi_vocabulary():
    # insight 0071: observation is net-negative; strip the AoU survey vocabulary.
    for name in ("0071-multidomain-rare6-cond-drug-obs.md",
                 "0072-multidomain-rare6-cond-drug-obs-minibatch.md"):
        mod, fm = _fm(name)
        mod.validate_frontmatter(fm)
        assert fm["obs_exclude_vocab"] == "PPI", name
