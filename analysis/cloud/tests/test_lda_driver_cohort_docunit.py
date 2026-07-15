# analysis/cloud/tests/test_lda_driver_cohort_docunit.py  (new; mirrors the
# dir of other cloud tests, e.g. test_stm_driver_partition.py)
"""LDA driver argparse: --cohort / --doc-unit widened to admit the shared
population_cancer cohort + patient_cohort doc-unit (Task 2.5).

lda_bigquery_cloud.py has no separate parse_args()/build_arg_parser() seam
(unlike stm_bigquery_cloud.py) — argparse construction lives inline in
main(), which also builds a Spark session and hits BigQuery once past the
env-var check. So instead of extracting a seam (out of scope: additive-only
task, no driver refactor), this test calls main() directly with
WORKSPACE_CDR/GOOGLE_CLOUD_PROJECT unset. main() validates argv (choices +
the source-table compatibility guard) BEFORE touching those env vars, so
reaching the env-var error message — rather than argparse's SystemExit(2)
or the guard's own stderr message — proves both gates accepted the new
combo for real. No mocks.
"""
import sys
from pathlib import Path

import pytest

_CLOUD = str(Path(__file__).resolve().parents[1])
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)

# lda_bigquery_cloud.main() imports charmpheno.omop.cohorts.SUPPORTED_COHORTS
# lazily (see the driver's own comment) so the *module* stays importable
# without charmpheno on the path; calling main() itself still needs it, so
# make the sibling charmpheno package importable the same way the charmpheno
# poetry project's own venv would already have it.
_CHARMPHENO_PKG_ROOT = str(Path(__file__).resolve().parents[3] / "charmpheno")
if _CHARMPHENO_PKG_ROOT not in sys.path:
    sys.path.insert(0, _CHARMPHENO_PKG_ROOT)


def test_lda_accepts_population_cancer_and_patient_cohort(monkeypatch, capsys):
    monkeypatch.delenv("WORKSPACE_CDR", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    try:
        import lda_bigquery_cloud
    except ImportError:
        pytest.skip("driver imports unavailable (PySpark/charmpheno deps not installed)")

    rc = lda_bigquery_cloud.main([
        "--cohort", "population_cancer",
        "--doc-unit", "patient_cohort",
        "--source-table", "condition_era",
    ])
    captured = capsys.readouterr()
    # RC 1 + the WORKSPACE_CDR message (not argparse's exit(2), not the
    # source-table guard's own error) proves both gates passed.
    assert rc == 1
    assert "WORKSPACE_CDR" in captured.err


def test_lda_rejects_unregistered_cohort_via_argparse_choices():
    """Sanity control: an unregistered name still fails fast at argparse,
    proving --cohort choices is bound to the registry, not wide open."""
    try:
        import lda_bigquery_cloud
    except ImportError:
        pytest.skip("driver imports unavailable (PySpark/charmpheno deps not installed)")

    with pytest.raises(SystemExit) as exc_info:
        lda_bigquery_cloud.main(["--cohort", "not_a_real_cohort"])
    assert exc_info.value.code == 2


def test_lda_registry_widening_is_general():
    """The --cohort choices source from SUPPORTED_COHORTS, so population_cancer
    (added as part of this task's motivating comparison) and every other
    registered cohort are admitted without per-cohort re-widening."""
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS
    assert "population_cancer" in SUPPORTED_COHORTS

    from charmpheno.omop.doc_spec import PatientCohortDocSpec, doc_spec_from_cli
    assert isinstance(doc_spec_from_cli("patient_cohort"), PatientCohortDocSpec)
