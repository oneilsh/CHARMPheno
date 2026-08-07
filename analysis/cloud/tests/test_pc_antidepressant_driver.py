"""argparse / env-validation gate for the PC antidepressant driver.

Mirrors test_lda_driver_cohort_docunit.py: pc_antidepressant_cloud.main()
validates the CLI + environment BEFORE importing charmpheno/analysis or opening
a Spark session, so calling main() with the workspace env unset reaches the
--cdr/--billing error (RC 1) rather than touching BigQuery. No mocks.
"""
import sys
from pathlib import Path

import pytest

_CLOUD = str(Path(__file__).resolve().parents[1])
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)

import pc_antidepressant_cloud as drv  # noqa: E402


def test_env_unset_validates_before_touching_bq(monkeypatch, capsys):
    monkeypatch.delenv("WORKSPACE_CDR", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    rc = drv.main([])                       # no --cdr/--billing, no env
    captured = capsys.readouterr()
    assert rc == 1
    assert "--cdr/--billing" in captured.err


def test_window_must_be_at_least_stability(monkeypatch, capsys):
    # With cdr/billing supplied, the window-vs-stability guard fires before any
    # BQ import/read, so RC 1 + its message proves the guard (not a BQ error).
    monkeypatch.delenv("WORKSPACE_CDR", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    rc = drv.main([
        "--cdr", "proj.ds", "--billing", "bill",
        "--window-days", "30", "--stability-days", "90",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "must be >=" in captured.err


def test_cdr_billing_default_from_env(monkeypatch):
    # The parser sources --cdr/--billing defaults from the workspace env vars.
    monkeypatch.setenv("WORKSPACE_CDR", "myproj.mydataset")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "mybilling")
    ns = drv._build_parser().parse_args([])
    assert ns.cdr == "myproj.mydataset"
    assert ns.billing == "mybilling"


def test_parser_surface_has_all_flags():
    ns = drv._build_parser().parse_args([
        "--cdr", "p.d", "--billing", "b",
        "--K", "12", "--weight-y", "2.5", "--alpha", "1.2", "--tau", "1.3",
        "--pi-iters", "40", "--max-iter", "200", "--lookback-days", "180",
        "--window-days", "200", "--stability-days", "120", "--grace-gap-days", "45",
        "--vocab-size", "500", "--min-df", "5", "--min-patient-count", "7",
        "--person-mod", "10", "--test-frac", "0.3", "--seed", "13",
        "--cache-uri", "gs://x/y", "--out", "/tmp/r.json",
    ])
    assert ns.K == 12 and ns.weight_y == 2.5 and ns.alpha == 1.2 and ns.tau == 1.3
    assert ns.pi_iters == 40 and ns.max_iter == 200
    assert ns.lookback_days == 180 and ns.window_days == 200
    assert ns.stability_days == 120 and ns.grace_gap_days == 45
    assert ns.vocab_size == 500 and ns.min_df == 5 and ns.min_patient_count == 7
    assert ns.person_mod == 10 and ns.test_frac == 0.3 and ns.seed == 13
    assert ns.cache_uri == "gs://x/y" and ns.out == "/tmp/r.json"
