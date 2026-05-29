"""Unit tests for scripts/run_experiment.py."""
from __future__ import annotations

from pathlib import Path

import pytest

# scripts/tests/conftest.py already inserts scripts/ into sys.path.
import run_experiment as rx

FIXTURES = Path(__file__).parent / "fixtures"


def test_read_frontmatter_parses_yaml_block():
    path = FIXTURES / "sample_experiment.md"
    fm = rx.read_frontmatter(path)
    assert fm["id"] == 42
    assert fm["slug"] == "try-k60-dementia"
    assert fm["status"] == "pending"
    assert fm["model_class"] == "lda"
    assert fm["cohort"] == "dementia"
    assert fm["K"] == 60


def test_read_frontmatter_raises_on_missing_block(tmp_path):
    path = tmp_path / "no_frontmatter.md"
    path.write_text("# No frontmatter\nJust a body.\n")
    with pytest.raises(ValueError, match="frontmatter"):
        rx.read_frontmatter(path)


def test_read_frontmatter_raises_on_unterminated_block(tmp_path):
    path = tmp_path / "bad.md"
    path.write_text("---\nid: 1\n# never closed\n")
    with pytest.raises(ValueError, match="frontmatter"):
        rx.read_frontmatter(path)


def test_merge_config_later_wins():
    base = {"a": 1, "b": 2, "c": 3}
    override = {"b": 20, "d": 4}
    merged = rx.merge_config(base, override)
    assert merged == {"a": 1, "b": 20, "c": 3, "d": 4}
    # Inputs not mutated
    assert base == {"a": 1, "b": 2, "c": 3}
    assert override == {"b": 20, "d": 4}


def test_load_defaults_three_way_merge():
    fixtures = FIXTURES / "sample_defaults"
    effective = rx.load_defaults("dementia", fixtures)
    # base provides model_class, max_iter, vocab_size; dementia overrides K + adds cohort
    assert effective["model_class"] == "lda"
    assert effective["max_iter"] == 20
    assert effective["vocab_size"] == 10000
    assert effective["K"] == 50          # dementia override beats base
    assert effective["cohort"] == "dementia"


def test_load_defaults_missing_cohort_file_raises(tmp_path):
    (tmp_path / "_base.yaml").write_text("K: 40\n")
    with pytest.raises(FileNotFoundError, match="bogus"):
        rx.load_defaults("bogus", tmp_path)


def test_load_defaults_missing_base_raises(tmp_path):
    (tmp_path / "dementia.yaml").write_text("cohort: dementia\n")
    with pytest.raises(FileNotFoundError, match="_base.yaml"):
        rx.load_defaults("dementia", tmp_path)


def _write_experiment(dir_path: Path, *, id: int, slug: str, status: str) -> Path:
    """Test helper: writes a minimal experiment record file."""
    path = dir_path / f"{id:04d}-{slug}.md"
    path.write_text(
        f"---\n"
        f"id: {id}\n"
        f"slug: {slug}\n"
        f"status: {status}\n"
        f"model_class: lda\n"
        f"cohort: dementia\n"
        f"---\n\n# {slug}\n"
    )
    return path


def test_find_next_pending_picks_lowest_id(tmp_path):
    _write_experiment(tmp_path, id=3, slug="c", status="pending")
    _write_experiment(tmp_path, id=1, slug="a", status="done")
    _write_experiment(tmp_path, id=2, slug="b", status="pending")
    result = rx.find_next_pending(tmp_path)
    assert result is not None
    assert result.name == "0002-b.md"


def test_find_next_pending_returns_none_when_no_pending(tmp_path):
    _write_experiment(tmp_path, id=1, slug="a", status="done")
    _write_experiment(tmp_path, id=2, slug="b", status="archived")
    assert rx.find_next_pending(tmp_path) is None


def test_find_next_pending_empty_dir_returns_none(tmp_path):
    assert rx.find_next_pending(tmp_path) is None


def test_find_next_pending_ignores_non_md_files(tmp_path):
    _write_experiment(tmp_path, id=1, slug="a", status="pending")
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "0099-draft.md.bak").write_text("ignore me too")
    result = rx.find_next_pending(tmp_path)
    assert result is not None
    assert result.name == "0001-a.md"


def test_find_by_id_returns_matching_path(tmp_path):
    _write_experiment(tmp_path, id=42, slug="try-k60", status="pending")
    _write_experiment(tmp_path, id=43, slug="other", status="pending")
    result = rx.find_by_id(tmp_path, 42)
    assert result.name == "0042-try-k60.md"


def test_find_by_id_raises_when_missing(tmp_path):
    _write_experiment(tmp_path, id=1, slug="a", status="pending")
    with pytest.raises(FileNotFoundError, match="0042"):
        rx.find_by_id(tmp_path, 42)


def test_sanitize_line_passes_normal_lines():
    line = "[iter 5] ELBO=-1.234e9  time=180s\n"
    assert rx.sanitize_line(line, rx.PATIENT_PATTERNS) == line


def test_sanitize_line_drops_person_hash_show_output():
    # PySpark .show() output format
    line = "|a1b2c3d4e5f6  |(60,[0,1],[0.5,0.5])  |\n"
    # The first column is 12 hex chars; without context this is fine, but the
    # surrounding .show()-header makes it identifiable as the transform-sample.
    # Hard to safely match by hash alone -- match by header context instead.
    assert rx.sanitize_line("|person_hash|topicDistribution|\n", rx.PATIENT_PATTERNS) is None


def test_sanitize_line_drops_lines_with_person_id_equals():
    line = "[debug] person_id=12345 has 17 tokens\n"
    assert rx.sanitize_line(line, rx.PATIENT_PATTERNS) is None


def test_sanitize_line_drops_lines_with_explicit_hash_prefix():
    line = "    hash:a1b2c3d4e5f6 -> topic 3\n"
    assert rx.sanitize_line(line, rx.PATIENT_PATTERNS) is None


def test_sanitize_line_drops_transform_sample_phase_marker():
    line = "[driver] >>> transform sample\n"
    assert rx.sanitize_line(line, rx.PATIENT_PATTERNS) is None


def test_sanitize_line_keeps_aggregate_person_count():
    line = "[driver]   OMOP: 1234567 rows, 89012 distinct persons\n"
    assert rx.sanitize_line(line, rx.PATIENT_PATTERNS) == line


def test_sanitize_line_keeps_iter_topic_prints():
    line = "  Topic 5:    192671  Type 2 diabetes mellitus  0.0234\n"
    assert rx.sanitize_line(line, rx.PATIENT_PATTERNS) == line


def test_build_lda_args_required_fields(tmp_path):
    effective = {
        "model_class": "lda",
        "source_table": "condition_era",
        "doc_unit": "patient_year",
        "doc_min_length": 20,
        "K": 60,
        "max_iter": 20,
        "vocab_size": 10000,
        "min_df": 20,
        "min_patient_count": 20,
        "subsampling_rate": 0.2,
        "tau0": 64,
        "kappa": 0.7,
        "save_interval": 5,
        "print_topics_every": 1,
        "person_mod": 10,
        "top_n_tokens": 6,
        "seed": 42,
        "optimize_doc_concentration": True,
        "optimize_topic_concentration": False,
        "cohort": "dementia",
        "cohort_def": "first_dementia_year",
    }
    save_dir = tmp_path / "0042-try-k60-dementia"
    args = rx.build_lda_args(effective, save_dir, resume_from=None)
    # Required overrides
    assert "--save-dir" in args
    assert str(save_dir) in args
    assert "--K" in args
    assert "60" in args
    assert "--source-table" in args
    assert "condition_era" in args
    assert "--cohort" in args
    # --cohort takes the cohort_def driver value, not the display id
    assert "first_dementia_year" in args
    assert "dementia" not in args  # the display id is NOT passed to --cohort
    # Resume not set when None
    assert "--resume-from" not in args


def test_build_lda_args_threads_resume_from(tmp_path):
    effective = {
        "model_class": "lda", "source_table": "condition_era",
        "doc_unit": "patient_year", "doc_min_length": 20,
        "K": 60, "max_iter": 20, "vocab_size": 10000,
        "min_df": 20, "min_patient_count": 20,
        "subsampling_rate": 0.2, "tau0": 64, "kappa": 0.7,
        "save_interval": 5, "print_topics_every": 1,
        "person_mod": 10, "top_n_tokens": 6, "seed": 42,
        "optimize_doc_concentration": True,
        "optimize_topic_concentration": False,
        "cohort": "dementia",
        "cohort_def": "first_dementia_year",
    }
    save_dir = tmp_path / "0042-try-k60"
    resume = tmp_path / "0042-try-k60"
    args = rx.build_lda_args(effective, save_dir, resume_from=resume)
    assert "--resume-from" in args
    idx = args.index("--resume-from")
    assert args[idx + 1] == str(resume)


def test_build_lda_args_general_cohort_maps_to_none():
    """General-population cohort: cohort=general, cohort_def=none."""
    effective = {
        "model_class": "lda", "source_table": "condition_era",
        "doc_unit": "patient", "doc_min_length": 20,
        "K": 40, "max_iter": 20, "vocab_size": 10000,
        "min_df": 20, "min_patient_count": 20,
        "subsampling_rate": 0.2, "tau0": 64, "kappa": 0.7,
        "save_interval": 5, "print_topics_every": 1,
        "person_mod": 10, "top_n_tokens": 6, "seed": 42,
        "optimize_doc_concentration": True,
        "optimize_topic_concentration": False,
        "cohort": "general",
        "cohort_def": "none",
    }
    args = rx.build_lda_args(effective, Path("/tmp/foo"), resume_from=None)
    assert "--cohort" in args
    idx = args.index("--cohort")
    assert args[idx + 1] == "none"


def test_build_eval_args_minimum(tmp_path):
    checkpoint = tmp_path / "0042-try-k60"
    args = rx.build_eval_args(checkpoint, {"model_class": "lda"})
    assert "--checkpoint" in args
    assert str(checkpoint) in args
    assert "--model-class" in args
    assert "lda" in args


def test_build_spark_submit_cmd_structure(tmp_path):
    script = "/repo/analysis/cloud/lda_bigquery_cloud.py"
    script_args = ["--K", "60", "--save-dir", "/tmp/foo"]
    repo_root = tmp_path
    (repo_root / "spark-vi" / "dist").mkdir(parents=True)
    (repo_root / "spark-vi" / "dist" / "spark_vi.zip").touch()
    (repo_root / "charmpheno" / "dist").mkdir(parents=True)
    (repo_root / "charmpheno" / "dist" / "charmpheno.zip").touch()

    cmd = rx.build_spark_submit_cmd(script, script_args, repo_root)
    assert cmd[0] == "spark-submit"
    assert "--master" in cmd and "yarn" in cmd
    assert "--deploy-mode" in cmd and "client" in cmd
    assert "--py-files" in cmd
    # py-files value contains both zips comma-joined
    py_files_idx = cmd.index("--py-files")
    py_files_val = cmd[py_files_idx + 1]
    assert "spark_vi.zip" in py_files_val
    assert "charmpheno.zip" in py_files_val
    # Script + script args at the end, in order
    assert cmd[-len(script_args) - 1] == script
    assert cmd[-len(script_args):] == script_args


def test_write_summary_header_creates_file(tmp_path):
    summary_path = tmp_path / "summary.md"
    effective = {"model_class": "lda", "cohort": "dementia", "K": 60}
    rx.write_summary_header(summary_path, exp_id=42, slug="try-k60", effective=effective)
    text = summary_path.read_text()
    assert "# Experiment 0042" in text
    assert "try-k60" in text
    assert "## Effective config" in text
    assert "K: 60" in text
    assert "cohort: dementia" in text
    assert "model_class: lda" in text


def test_write_summary_header_appends_session_marker(tmp_path):
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# Existing summary\n\n## Fit session 1\n... (old) ...\n")
    effective = {"model_class": "lda", "cohort": "dementia", "K": 60}
    rx.write_summary_header(summary_path, exp_id=42, slug="try-k60", effective=effective)
    text = summary_path.read_text()
    # Old content preserved
    assert "(old)" in text
    # New session marker appended
    assert "## Fit session 2" in text


def test_run_subprocess_tee_sanitize_writes_filtered_output(tmp_path, capsys):
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# header\n")  # pre-existing
    # Use a tiny shell command that prints 3 lines, two of which should drop
    cmd = [
        "sh", "-c",
        "echo '[iter 1] ELBO=-1.23'; echo '|person_hash|topicDistribution|'; echo '[driver] done'",
    ]
    exit_code = rx.run_subprocess_tee_sanitize(cmd, summary_path, rx.PATIENT_PATTERNS)
    assert exit_code == 0
    summary_text = summary_path.read_text()
    # Sanitized lines NOT in summary
    assert "person_hash" not in summary_text
    # Clean lines in summary
    assert "[iter 1] ELBO=-1.23" in summary_text
    assert "[driver] done" in summary_text
    # Pre-existing header preserved
    assert "# header" in summary_text
    # All three lines (including dropped) appear on stdout for live debugging
    captured = capsys.readouterr()
    assert "person_hash" in captured.out
    assert "[iter 1]" in captured.out


def test_run_subprocess_tee_sanitize_propagates_exit_code(tmp_path):
    summary_path = tmp_path / "summary.md"
    cmd = ["sh", "-c", "echo 'starting'; exit 7"]
    exit_code = rx.run_subprocess_tee_sanitize(cmd, summary_path, rx.PATIENT_PATTERNS)
    assert exit_code == 7
    # Partial output captured
    assert "starting" in summary_path.read_text()


def test_append_eval_section_marker_and_body(tmp_path):
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# header\n\n## Fit session 1\n... fit output ...\n")
    eval_stdout = "mean NPMI: 0.224  median: 0.198\n  topic 0: 0.31\n"
    rx.append_eval_section(summary_path, eval_stdout, exit_code=0)
    text = summary_path.read_text()
    # Header includes timestamp prefix
    assert "## Eval (NPMI)" in text
    # Content present
    assert "mean NPMI: 0.224" in text
    # Original content preserved
    assert "## Fit session 1" in text
    # Completion marker
    assert "### Eval complete (exit 0)" in text


def test_append_eval_section_sanitizes_body(tmp_path):
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# header\n")
    eval_stdout = (
        "mean NPMI: 0.2\n"
        "|person_hash|topicDistribution|\n"   # patient info — dropped
        "26/05/28 20:46:09 INFO YarnClientImpl: ...\n"  # cluster noise — dropped
        "  topic 0: 0.31\n"
    )
    rx.append_eval_section(summary_path, eval_stdout, exit_code=0)
    text = summary_path.read_text()
    assert "mean NPMI: 0.2" in text
    assert "topic 0: 0.31" in text
    assert "person_hash" not in text
    assert "YarnClientImpl" not in text


def test_noise_patterns_drops_hadoop_info_log():
    line = "26/05/28 20:46:09 INFO Configuration: resource-types.xml not found\n"
    assert rx.sanitize_line(line, rx.DROP_PATTERNS) is None


def test_noise_patterns_drops_yarn_log():
    line = "26/05/28 20:46:10 INFO YarnClientImpl: Submitted application application_1779908374279_0012\n"
    assert rx.sanitize_line(line, rx.DROP_PATTERNS) is None


def test_noise_patterns_drops_gcs_chatter():
    line = "26/05/28 20:46:13 INFO GoogleHadoopOutputStream: hflush(): No-op due to rate limit ...\n"
    assert rx.sanitize_line(line, rx.DROP_PATTERNS) is None


class TestModelClassDispatch:
    def test_stm_passes_validation(self):
        fm = {
            "id": "0099-test", "slug": "test", "cohort": "dementia",
            "model_class": "stm", "covariate_formula": "~ C(sex) + age",
            "categorical_cols": ["sex"], "continuous_cols": ["age"],
        }
        # Should not raise (LDA gate previously rejected anything != "lda").
        from run_experiment import validate_frontmatter
        validate_frontmatter(fm)  # idempotent — passes for stm with required keys

    def test_stm_requires_covariate_formula(self):
        fm = {
            "id": "0099-test", "slug": "test", "cohort": "dementia",
            "model_class": "stm",
            # covariate_formula missing
        }
        from run_experiment import validate_frontmatter
        with pytest.raises(SystemExit):
            validate_frontmatter(fm)

    def test_build_fit_args_dispatches_to_stm_driver(self):
        fm = {
            "id": "0099", "slug": "test", "cohort": "dementia",
            "model_class": "stm", "covariate_formula": "~ C(sex)",
            "categorical_cols": ["sex"], "continuous_cols": [],
        }
        effective = {**fm, "K": 40, "max_iter": 20}
        from run_experiment import build_fit_driver_path
        path = build_fit_driver_path(effective)
        assert path.endswith("stm_bigquery_cloud.py")

    def test_build_fit_driver_path_lda(self):
        from run_experiment import build_fit_driver_path
        path = build_fit_driver_path({"model_class": "lda"})
        assert path.endswith("lda_bigquery_cloud.py")


def test_noise_patterns_keep_driver_lines():
    lines = [
        "[driver] Spark 3.5.3, master=yarn, defaultParallelism=2\n",
        "[driver]   iter 1/2: ELBO=-97881.5970, batch=245, rho=0.0538, 86.4s\n",
        "[driver]   --- topics @ iter 1 ---\n",
        "[driver]    topic  3  α=0.1969  E[β]=0.2405  Σλ=1.58e+03  ...\n",
        "[driver] fit complete\n",
    ]
    for ln in lines:
        assert rx.sanitize_line(ln, rx.DROP_PATTERNS) == ln, f"unexpectedly dropped: {ln!r}"


def test_noise_patterns_keep_traceback_lines():
    """Python tracebacks are signal, not noise — they explain failures."""
    lines = [
        "Traceback (most recent call last):\n",
        '  File "/some/path.py", line 42, in main\n',
        "FileNotFoundError: No manifest.json at resumeFrom path: ...\n",
    ]
    for ln in lines:
        assert rx.sanitize_line(ln, rx.DROP_PATTERNS) == ln


def test_drop_patterns_still_drops_patient_info():
    """DROP_PATTERNS must be a superset of PATIENT_PATTERNS."""
    line = "|person_hash|topicDistribution|\n"
    assert rx.sanitize_line(line, rx.DROP_PATTERNS) is None


def test_parse_iter_marker_extracts_iter_number():
    line = "[driver]   iter 1/2: ELBO=-97881.5970, batch=245, rho=0.0538, 86.4s\n"
    assert rx.parse_iter_marker(line) == 1


def test_parse_iter_marker_extracts_higher_iter():
    line = "[driver]   iter 17/40: ELBO=-1.234e9, batch=512, time=183s\n"
    assert rx.parse_iter_marker(line) == 17


def test_parse_iter_marker_returns_none_for_non_iter_lines():
    lines = [
        "[driver]   --- topics @ iter 1 ---\n",   # not an iter-start line
        "[driver]    topic  3  α=0.1969  ...\n",
        "[driver] fit complete\n",
        "26/05/28 20:46:09 INFO Configuration: ...\n",
        "\n",
    ]
    for ln in lines:
        assert rx.parse_iter_marker(ln) is None, f"unexpectedly matched: {ln!r}"


def test_parse_iter_marker_tolerates_no_newline():
    """In case the wrapper passes a line without trailing newline."""
    assert rx.parse_iter_marker("[driver]   iter 5/10: ELBO=-1.0") == 5


import os
import signal as _signal
import threading


def test_run_subprocess_writes_killed_marker_on_sigterm(tmp_path):
    """SIGTERM during the tee loop writes a killed marker referencing last seen iter."""
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# header\n")

    # Subprocess prints two iter lines fast, then hangs for 60s.
    cmd = [
        "sh", "-c",
        "echo '[driver]   iter 1/10: ELBO=-1.0 time=1s'; "
        "sleep 0.2; "
        "echo '[driver]   iter 2/10: ELBO=-2.0 time=1s'; "
        "sleep 60",
    ]

    # From a daemon thread, SIGTERM the current process ~0.5s in.
    def send_sigterm():
        import time
        time.sleep(0.5)
        os.kill(os.getpid(), _signal.SIGTERM)
    threading.Thread(target=send_sigterm, daemon=True).start()

    exit_code = rx.run_subprocess_tee_sanitize(cmd, summary_path, rx.DROP_PATTERNS)

    assert exit_code == 130, f"expected signal-exit 130, got {exit_code}"
    text = summary_path.read_text()
    assert "Killed at iter 2" in text, f"missing iter marker in: {text!r}"
    # Original header preserved
    assert "# header" in text
    # Both iter lines tee'd to summary before kill
    assert "iter 1/10" in text
    assert "iter 2/10" in text


def test_append_eval_section_records_nonzero_exit(tmp_path):
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# header\n")
    rx.append_eval_section(summary_path, "partial output\n", exit_code=1)
    text = summary_path.read_text()
    assert "### Eval complete (exit 1)" in text


def test_append_eval_section_timestamp_in_header(tmp_path):
    """Header includes a UTC timestamp so multiple evals are distinguishable."""
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# header\n")
    rx.append_eval_section(summary_path, "body\n", exit_code=0)
    text = summary_path.read_text()
    import re
    # Match `## Eval (NPMI) — YYYY-MM-DD HH:MM:SS UTC`
    assert re.search(r"## Eval \(NPMI\) — \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC", text), \
        f"missing timestamped eval header in: {text!r}"


def test_find_most_recent_fit_returns_none_for_empty_dir(tmp_path):
    assert rx.find_most_recent_fit(tmp_path) is None


def test_find_most_recent_fit_picks_latest_mtime(tmp_path):
    """Three experiment dirs, only the second has the most recent manifest mtime."""
    # 0010-old/manifest.json — oldest
    (tmp_path / "0010-old").mkdir()
    older = tmp_path / "0010-old" / "manifest.json"
    older.write_text("{}")
    import os
    os.utime(older, (1_700_000_000, 1_700_000_000))

    # 0042-target/manifest.json — newest
    (tmp_path / "0042-target").mkdir()
    target = tmp_path / "0042-target" / "manifest.json"
    target.write_text("{}")
    os.utime(target, (1_700_000_200, 1_700_000_200))

    # 0050-middle/manifest.json — between
    (tmp_path / "0050-middle").mkdir()
    middle = tmp_path / "0050-middle" / "manifest.json"
    middle.write_text("{}")
    os.utime(middle, (1_700_000_100, 1_700_000_100))

    assert rx.find_most_recent_fit(tmp_path) == 42


def test_find_most_recent_fit_ignores_dirs_without_manifest(tmp_path):
    """A save_dir without manifest.json is in-progress / failed, not eligible."""
    (tmp_path / "0001-pending").mkdir()  # no manifest.json — ignored
    (tmp_path / "0002-done").mkdir()
    (tmp_path / "0002-done" / "manifest.json").write_text("{}")
    assert rx.find_most_recent_fit(tmp_path) == 2


def test_find_most_recent_fit_skips_malformed_dir_names(tmp_path):
    """Dirs not matching NNNN-* are ignored even if they have a manifest."""
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch" / "manifest.json").write_text("{}")
    (tmp_path / "0007-real").mkdir()
    (tmp_path / "0007-real" / "manifest.json").write_text("{}")
    assert rx.find_most_recent_fit(tmp_path) == 7


def test_find_most_recent_fit_missing_runs_dir_returns_none(tmp_path):
    """Nonexistent runs_dir is treated as empty."""
    assert rx.find_most_recent_fit(tmp_path / "does-not-exist") is None


class TestFindMostRecentFitNeedingBuild:
    """find_most_recent_fit_needing_build returns the freshest fit whose
    dashboard bundle is missing or stale."""

    def test_returns_none_when_runs_dir_missing(self, tmp_path):
        assert rx.find_most_recent_fit_needing_build(tmp_path / "nope") is None

    def test_returns_none_when_no_fits_exist(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        assert rx.find_most_recent_fit_needing_build(runs) is None

    def test_picks_never_built_fit(self, tmp_path):
        runs = tmp_path / "runs"
        (runs / "0001-pilot").mkdir(parents=True)
        (runs / "0001-pilot" / "manifest.json").write_text("{}")
        assert rx.find_most_recent_fit_needing_build(runs) == 1

    def test_skips_current_bundles(self, tmp_path):
        import os
        runs = tmp_path / "runs"
        d = runs / "0001-pilot"
        d.mkdir(parents=True)
        manifest = d / "manifest.json"
        manifest.write_text("{}")
        bundle = d / "dashboard_bundle"
        bundle.mkdir()
        marker = bundle / "corpus_stats.json"
        marker.write_text("{}")
        # Force marker mtime strictly after manifest mtime
        later = manifest.stat().st_mtime + 100
        os.utime(marker, (later, later))
        assert rx.find_most_recent_fit_needing_build(runs) is None

    def test_treats_equal_mtime_as_current(self, tmp_path):
        """Ties go to 'current' — `>=` not `>` in the staleness check.

        Pins the design choice from the function's comment: when marker mtime
        equals manifest mtime exactly, the bundle is treated as current and
        the fit is skipped. A future regression that changed `>=` to `>` would
        be caught here.
        """
        import os
        runs = tmp_path / "runs"
        d = runs / "0001-pilot"
        d.mkdir(parents=True)
        manifest = d / "manifest.json"
        manifest.write_text("{}")
        bundle = d / "dashboard_bundle"
        bundle.mkdir()
        marker = bundle / "corpus_stats.json"
        marker.write_text("{}")
        # Force both files to exactly the same mtime
        t = manifest.stat().st_mtime
        os.utime(marker, (t, t))
        os.utime(manifest, (t, t))
        assert rx.find_most_recent_fit_needing_build(runs) is None

    def test_picks_stale_bundle(self, tmp_path):
        import os
        runs = tmp_path / "runs"
        d = runs / "0001-pilot"
        d.mkdir(parents=True)
        manifest = d / "manifest.json"
        manifest.write_text("{}")
        bundle = d / "dashboard_bundle"
        bundle.mkdir()
        marker = bundle / "corpus_stats.json"
        marker.write_text("{}")
        # Force manifest mtime strictly after marker mtime
        later = marker.stat().st_mtime + 100
        os.utime(manifest, (later, later))
        assert rx.find_most_recent_fit_needing_build(runs) == 1

    def test_multiple_candidates_picks_newest_fit(self, tmp_path):
        import os
        runs = tmp_path / "runs"
        # 0001-pilot: never-built, older fit
        d1 = runs / "0001-pilot"
        d1.mkdir(parents=True)
        m1 = d1 / "manifest.json"
        m1.write_text("{}")
        # 0002-bigger: never-built, newer fit
        d2 = runs / "0002-bigger"
        d2.mkdir(parents=True)
        m2 = d2 / "manifest.json"
        m2.write_text("{}")
        os.utime(m2, (m1.stat().st_mtime + 100, m1.stat().st_mtime + 100))
        assert rx.find_most_recent_fit_needing_build(runs) == 2

    def test_skips_dirs_without_manifest(self, tmp_path):
        runs = tmp_path / "runs"
        (runs / "0001-empty").mkdir(parents=True)
        # No manifest.json — not yet fit, nothing to build
        assert rx.find_most_recent_fit_needing_build(runs) is None


class TestBuildDashboardArgs:
    """build_dashboard_args produces the argv for build_dashboard_cloud.py."""

    def test_minimal_lda(self, tmp_path):
        effective = {
            "model_class": "lda",
            "vocab_top_n": 5000,
            "top_n_codes_for_npmi": 20,
        }
        args = rx.build_dashboard_args(
            effective, tmp_path / "ck", "0001-pilot-dashboard.zip",
        )
        assert args[:2] == ["--checkpoint", str(tmp_path / "ck")]
        assert "--model-class" in args
        idx = args.index("--model-class")
        assert args[idx + 1] == "lda"
        assert "--zip-name" in args
        assert args[args.index("--zip-name") + 1] == "0001-pilot-dashboard.zip"
        assert "--vocab-top-n" in args
        assert args[args.index("--vocab-top-n") + 1] == "5000"
        assert "--top-n-codes-for-npmi" in args
        assert args[args.index("--top-n-codes-for-npmi") + 1] == "20"

    def test_vocab_top_n_override(self, tmp_path):
        effective = {
            "model_class": "lda",
            "vocab_top_n": 1000,
            "top_n_codes_for_npmi": 10,
        }
        args = rx.build_dashboard_args(effective, tmp_path / "ck", "z.zip")
        assert args[args.index("--vocab-top-n") + 1] == "1000"
        assert args[args.index("--top-n-codes-for-npmi") + 1] == "10"

    def test_missing_required_keys_raises(self, tmp_path):
        import pytest
        # Missing vocab_top_n
        effective = {"model_class": "lda", "top_n_codes_for_npmi": 20}
        with pytest.raises(KeyError):
            rx.build_dashboard_args(effective, tmp_path / "ck", "z.zip")

    def test_missing_top_n_codes_for_npmi_raises(self, tmp_path):
        """Symmetric to test_missing_required_keys_raises — both required
        keys must trigger KeyError when absent, not just vocab_top_n.
        """
        import pytest
        # Missing top_n_codes_for_npmi
        effective = {"model_class": "lda", "vocab_top_n": 5000}
        with pytest.raises(KeyError):
            rx.build_dashboard_args(effective, tmp_path / "ck", "z.zip")


class TestWriteBuildSectionHeader:
    def test_appends_header_with_timestamp(self, tmp_path):
        import re
        summary = tmp_path / "summary.md"
        summary.write_text("# Existing content\n\n")
        rx.write_build_section_header(summary)
        text = summary.read_text()
        # Existing content preserved
        assert "# Existing content" in text
        # Timestamped header appended
        assert re.search(
            r"^## Dashboard build — \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$",
            text, re.MULTILINE,
        )

    def test_multiple_calls_append_distinct_sections(self, tmp_path):
        summary = tmp_path / "summary.md"
        summary.write_text("")
        rx.write_build_section_header(summary)
        rx.write_build_section_header(summary)
        text = summary.read_text()
        assert text.count("## Dashboard build —") == 2


class TestBuildOnlyMain:
    """--build-only branch in main(): auto-discovery, mutual-exclusion."""

    def test_build_only_and_eval_only_mutually_exclusive(self, tmp_path, capsys):
        rc = rx.main(["--eval-only", "--build-only", "--runs-dir", str(tmp_path)])
        assert rc == 2
        captured = capsys.readouterr()
        out_lower = captured.out.lower()
        assert "contradictory" in out_lower or "exclusive" in out_lower

    def test_build_only_and_no_eval_mutually_exclusive(self, tmp_path, capsys):
        rc = rx.main(["--no-eval", "--build-only", "--runs-dir", str(tmp_path)])
        assert rc == 2

    def test_build_only_auto_discover_none_found_exits_zero(
        self, tmp_path, capsys,
    ):
        """When no fits need building, --build-only without --id should
        print a friendly message and exit 0 (not error)."""
        runs = tmp_path / "runs"
        runs.mkdir()
        # No checkpoints under runs_dir -> nothing to build
        rc = rx.main([
            "--build-only",
            "--runs-dir", str(runs),
            "--experiments-dir", str(tmp_path / "exp_dir_unused"),
            "--defaults-dir", str(tmp_path / "defaults_unused"),
        ])
        assert rc == 0
        out_lower = capsys.readouterr().out.lower()
        assert "no fits need building" in out_lower \
               or "nothing to build" in out_lower
