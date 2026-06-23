"""Unit tests for the analysis/cloud covariate-cache key derivation.

Only the pure-Python hashing logic is tested here; the Spark-backed
try_load / save round-trip lives in test_covariates_load.py.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "cloud"))

from _covariates_cache import compute_cache_key  # noqa: E402


_BASE = dict(
    covariate_formula="~ C(source_cohort) + C(sex) + age",
    person_mod=10,
    cdr="proj.ds",
    source_table="condition_era",
    cohort="cancer_or_dementia",
    prior_obs_days=365,
)


def test_key_is_stable_across_calls():
    assert compute_cache_key(**_BASE) == compute_cache_key(**_BASE)


def test_key_changes_with_formula():
    other = dict(_BASE, covariate_formula="~ C(sex) + age")
    assert compute_cache_key(**_BASE) != compute_cache_key(**other)


def test_key_changes_with_cohort():
    other = dict(_BASE, cohort="first_cancer_year")
    assert compute_cache_key(**_BASE) != compute_cache_key(**other)


def test_key_changes_with_prior_obs_days():
    """In composite mode the covariate person set is the corpus's persons,
    so the prior-observation lookback (which sets corpus membership) must key
    the covariate cache too -- otherwise a widened cohort silently reloads
    the narrower covariate set and the inner join drops the new patients."""
    other = dict(_BASE, prior_obs_days=0)
    assert compute_cache_key(**_BASE) != compute_cache_key(**other)
