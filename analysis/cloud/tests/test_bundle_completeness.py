"""Fail-loud guards for the STM dashboard export.

Two silent-degradation modes the cloud build used to swallow:

1. A STALE covariate sidecar (content-blind cache key) whose design width no
   longer matches the re-fit model's Gamma — a dimension mismatch that crashed
   corpus_prevalence, got caught, and fell back to the intercept stand-in while
   also dropping gating.json (stm_gc became None). Observed on the exp-0028
   re-fit: known_sex_only dropped the Unknown sex level (P 4 -> 3) but the
   cached sidecar was still P=4.
2. A covariate-cache MISS, which silently skipped gating.json /
   covariate_schema.json, producing a bundle that *looks* complete but renders
   ungated with no covariate panel.
"""
import sys
from pathlib import Path

import pytest

_CLOUD = str(Path(__file__).resolve().parent.parent)
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)

import build_dashboard_cloud as bdc  # noqa: E402


def test_sidecar_matches_model_ok_when_names_equal():
    # identical design columns -> no raise
    bdc.assert_covariate_sidecar_matches_model(
        sidecar_names=["Intercept", "C(sex)[T.M]", "age"],
        model_covariate_names=["Intercept", "C(sex)[T.M]", "age"],
    )


def test_sidecar_mismatch_raises_systemexit_with_rebuild_hint():
    # the exp-0028 case: stale P=4 sidecar vs re-fit P=3 model
    with pytest.raises(SystemExit) as ei:
        bdc.assert_covariate_sidecar_matches_model(
            sidecar_names=["Intercept", "C(sex)[T.M]", "C(sex)[T.Unknown]", "age"],
            model_covariate_names=["Intercept", "C(sex)[T.M]", "age"],
        )
    msg = str(ei.value)
    assert "stale" in msg.lower()
    assert "build-covariates" in msg          # tells the operator how to fix it
    assert "4" in msg and "3" in msg          # names the two widths


def test_sidecar_mismatch_on_reordering_raises():
    # same width, different columns -> still a mismatch
    with pytest.raises(SystemExit):
        bdc.assert_covariate_sidecar_matches_model(
            sidecar_names=["Intercept", "age", "C(sex)[T.M]"],
            model_covariate_names=["Intercept", "C(sex)[T.M]", "age"],
        )


def _touch(d: Path, *names):
    for n in names:
        (d / n).write_text("{}")


def test_bundle_complete_gated_all_present_ok(tmp_path):
    _touch(tmp_path, "gating.json", "covariate_schema.json", "covariate_effects.json")
    bdc.assert_stm_bundle_complete(tmp_path, gated=True, allow_incomplete=False)


def test_bundle_complete_gated_missing_gating_raises(tmp_path):
    _touch(tmp_path, "covariate_schema.json", "covariate_effects.json")  # no gating.json
    with pytest.raises(SystemExit) as ei:
        bdc.assert_stm_bundle_complete(tmp_path, gated=True, allow_incomplete=False)
    msg = str(ei.value)
    assert "gating.json" in msg
    assert "allow-incomplete-bundle" in msg   # names the escape hatch


def test_bundle_complete_nongated_requires_covariate_schema(tmp_path):
    _touch(tmp_path, "covariate_effects.json")  # schema missing
    with pytest.raises(SystemExit):
        bdc.assert_stm_bundle_complete(tmp_path, gated=False, allow_incomplete=False)


def test_bundle_complete_allow_incomplete_does_not_raise(tmp_path):
    _touch(tmp_path, "covariate_effects.json")  # gating + schema missing
    # escape hatch: warn-and-continue instead of abort
    bdc.assert_stm_bundle_complete(tmp_path, gated=True, allow_incomplete=True)


# --- Early sidecar gate: fail loud at the covariate-cache miss, before the
#     expensive phases and immune to a reused out_dir's stale files (the
#     failure mode that shipped a stale/mixed bundle on 2026-07-06). ---

def test_sidecar_present_gated_ok():
    # sidecar loaded -> no raise, build proceeds gated
    bdc.assert_covariate_sidecar_present(
        is_stm=True, gated=True, sidecar_present=True, allow_incomplete=False)


def test_sidecar_missing_gated_raises_with_build_covariates_hint(capsys):
    # Exit code is now the distinct COVARIATE_CACHE_MISS_EXIT sentinel (see
    # test_covariate_miss_exit_code.py); the guidance message is printed
    # (or logged) before the raise rather than carried in the SystemExit
    # payload, so run_experiment.py can match on the code alone.
    with pytest.raises(SystemExit) as ei:
        bdc.assert_covariate_sidecar_present(
            is_stm=True, gated=True, sidecar_present=False,
            allow_incomplete=False, exp_hint="0028")
    assert ei.value.code == bdc.COVARIATE_CACHE_MISS_EXIT
    msg = capsys.readouterr().out
    assert "build-covariates EXP=0028" in msg     # actionable, names the exp
    assert "predictive_gain" in msg               # names what gets skipped
    assert "allow-incomplete-bundle" in msg       # names the escape hatch


def test_sidecar_missing_gated_allow_incomplete_does_not_raise():
    # escape hatch: operator explicitly wants a degraded ungated bundle
    bdc.assert_covariate_sidecar_present(
        is_stm=True, gated=True, sidecar_present=False, allow_incomplete=True)


def test_sidecar_missing_nongated_does_not_raise():
    # non-gated STM never needs the sidecar for gating; no abort
    bdc.assert_covariate_sidecar_present(
        is_stm=True, gated=False, sidecar_present=False, allow_incomplete=False)


def test_sidecar_missing_non_stm_does_not_raise():
    # LDA / HDP builds have no covariate sidecar concept
    bdc.assert_covariate_sidecar_present(
        is_stm=False, gated=False, sidecar_present=False, allow_incomplete=False)


def test_sidecar_missing_hint_absent_still_actionable(capsys):
    with pytest.raises(SystemExit) as ei:
        bdc.assert_covariate_sidecar_present(
            is_stm=True, gated=True, sidecar_present=False,
            allow_incomplete=False, exp_hint=None)
    assert ei.value.code == bdc.COVARIATE_CACHE_MISS_EXIT
    assert "build-covariates EXP=<id>" in capsys.readouterr().out
