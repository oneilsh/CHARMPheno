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


# --- checkpoint / resume / eval flags (VI backend only) ----------------------

def test_parser_has_persistence_flags():
    ns = drv._build_parser().parse_args([
        "--cdr", "p.d", "--billing", "b", "--backend", "vi",
        "--save-dir", "/runs/0071", "--save-interval", "5",
        "--resume-from", "/runs/0071", "--eval-only",
    ])
    assert ns.save_dir == "/runs/0071"
    assert ns.save_interval == 5
    assert ns.resume_from == "/runs/0071"
    assert ns.eval_only is True


def test_persistence_flag_defaults():
    ns = drv._build_parser().parse_args(["--cdr", "p.d", "--billing", "b"])
    assert ns.save_dir == "" and ns.save_interval == -1
    assert ns.resume_from == "" and ns.eval_only is False


def test_save_dir_rejected_for_inmem_backend(monkeypatch, capsys):
    # --save-dir is VI-only: with the default inmem backend it must fail fast
    # (before any BQ import), since L-BFGS has no interim state to checkpoint.
    rc = drv.main([
        "--cdr", "proj.ds", "--billing", "bill",
        "--save-dir", "/runs/0071",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "--save-dir is VI-only" in captured.err


def test_eval_only_requires_save_dir(monkeypatch, capsys):
    rc = drv.main([
        "--cdr", "proj.ds", "--billing", "bill",
        "--backend", "vi", "--eval-only",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "--eval-only requires --save-dir" in captured.err


def test_eval_only_errors_cleanly_without_checkpoint(tmp_path, capsys):
    # --eval-only pointed at a dir with no manifest.json errors before touching BQ.
    empty = tmp_path / "no_ckpt"
    empty.mkdir()
    rc = drv.main([
        "--cdr", "proj.ds", "--billing", "bill",
        "--backend", "vi", "--eval-only", "--save-dir", str(empty),
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "no checkpoint (manifest.json)" in captured.err


def test_eval_only_rejected_for_inmem_backend(capsys):
    # --eval-only with the default inmem backend (no --save-dir, so the save-dir
    # VI-only gate doesn't pre-empt) reaches the eval-only VI-only guard.
    rc = drv.main([
        "--cdr", "proj.ds", "--billing", "bill", "--eval-only",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "--eval-only is VI-only" in captured.err


# --- unsupervised warm-start flag (VI backend only) --------------------------

def test_parser_has_warm_start_flag_default_zero():
    ns = drv._build_parser().parse_args(["--cdr", "p.d", "--billing", "b"])
    assert ns.warm_start_unsup_iters == 0
    ns2 = drv._build_parser().parse_args([
        "--cdr", "p.d", "--billing", "b", "--backend", "vi",
        "--warm-start-unsup-iters", "50",
    ])
    assert ns2.warm_start_unsup_iters == 50


def test_warm_start_rejected_for_inmem_backend(capsys):
    # --warm-start-unsup-iters is VI-only: with the default inmem backend it must
    # fail fast (before any BQ import) — L-BFGS has no SVI phase-1 to warm from.
    rc = drv.main([
        "--cdr", "proj.ds", "--billing", "bill",
        "--warm-start-unsup-iters", "50",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "--warm-start-unsup-iters is VI-only" in captured.err
