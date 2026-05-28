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
    assert "dementia" in args
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
    }
    save_dir = tmp_path / "0042-try-k60"
    resume = tmp_path / "0042-try-k60"
    args = rx.build_lda_args(effective, save_dir, resume_from=resume)
    assert "--resume-from" in args
    idx = args.index("--resume-from")
    assert args[idx + 1] == str(resume)


def test_build_lda_args_omits_cohort_when_none():
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
        "cohort": "none",
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
