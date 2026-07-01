"""Regression guard: the dashboard build's covariate-cache key must include
prior_obs_days, so it matches the key the fit/build-covariates producers wrote.

The bug this guards: build_dashboard_cloud derived the key WITHOUT prior_obs_days,
so it defaulted to 365 and missed the cache for any experiment with a non-default
lookback (e.g. exp 0027, prior_obs_days=0) — silently dropping gating.json /
covariate_schema.json and forcing the intercept stand-in for corpus_prevalence.
"""
import sys
from pathlib import Path

_CLOUD = str(Path(__file__).resolve().parent.parent)
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)

from _covariates_cache import compute_cache_key  # noqa: E402
import build_dashboard_cloud as bdc  # noqa: E402


_CORPUS = {"person_mod": 4, "cdr": "cdr.R2024", "prior_obs_days": 0}
_COV_MANIFEST = {"covariate_formula": "~ C(sex) + age"}


def _fit_key(prior_obs_days: int) -> str:
    """The key a producer (fit / build-covariates) writes."""
    return compute_cache_key(
        covariate_formula=_COV_MANIFEST["covariate_formula"],
        person_mod=_CORPUS["person_mod"], cdr=_CORPUS["cdr"],
        source_table="condition_era", cohort="cancer_or_dementia",
        prior_obs_days=prior_obs_days,
    )


def test_build_key_matches_fit_key_for_nondefault_prior_obs_days():
    """The build's key equals the producer's key when prior_obs_days != 365."""
    build_key = bdc._covariate_cache_key(
        corpus=_CORPUS, cov_manifest=_COV_MANIFEST,
        source_table="condition_era", cohort="cancer_or_dementia",
    )
    assert build_key == _fit_key(0)          # matches the fit (prior_obs_days=0)
    assert build_key != _fit_key(365)        # and is NOT the old buggy default


def test_missing_prior_obs_days_falls_back_to_365():
    """Pre-record checkpoints (no prior_obs_days stamped) keep the old default."""
    corpus_no_pod = {"person_mod": 4, "cdr": "cdr.R2024"}
    build_key = bdc._covariate_cache_key(
        corpus=corpus_no_pod, cov_manifest=_COV_MANIFEST,
        source_table="condition_era", cohort="cancer_or_dementia",
    )
    assert build_key == _fit_key(365)
