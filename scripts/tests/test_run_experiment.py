"""Unit tests for scripts/run_experiment.py."""
from __future__ import annotations

from pathlib import Path

import pytest

# scripts/tests/conftest.py already inserts scripts/ into sys.path.
import run_experiment as rx

FIXTURES = Path(__file__).parent / "fixtures"


class TestResumeCorpusMismatches:
    """The resume guard compares a checkpoint's corpus_manifest to the current
    effective config and refuses to warm-start onto a different corpus."""

    BASE_MANIFEST = {
        "person_mod": 4,
        "source_table": "condition_era",
        "cohort": "cancer_or_dementia",
        "prior_obs_days": 0,
        "vocab_size": 10000,
        "doc_spec": {"name": "patient_cohort", "min_doc_length": 20},
    }
    BASE_EFFECTIVE = {
        "person_mod": 4,
        "source_table": "condition_era",
        "cohort_def": "cancer_or_dementia",
        "prior_obs_days": 0,
        "vocab_size": 10000,
        "doc_unit": "patient_cohort",
    }

    def test_matching_config_has_no_mismatches(self):
        assert rx._resume_corpus_mismatches(self.BASE_MANIFEST, self.BASE_EFFECTIVE) == []

    def test_person_mod_change_is_flagged(self):
        eff = dict(self.BASE_EFFECTIVE, person_mod=1)
        out = rx._resume_corpus_mismatches(self.BASE_MANIFEST, eff)
        assert any("person_mod" in m for m in out)

    def test_prior_obs_days_change_is_flagged(self):
        eff = dict(self.BASE_EFFECTIVE, prior_obs_days=365)
        out = rx._resume_corpus_mismatches(self.BASE_MANIFEST, eff)
        assert any("prior_obs_days" in m for m in out)

    def test_doc_spec_name_change_is_flagged(self):
        eff = dict(self.BASE_EFFECTIVE, doc_unit="patient_year")
        out = rx._resume_corpus_mismatches(self.BASE_MANIFEST, eff)
        assert any("doc_spec" in m for m in out)

    def test_vocab_size_difference_is_not_flagged(self):
        """vocab_size is NOT a guard field: STM stores the realized vocab count
        (post-pruning) while config carries the CountVectorizer cap, so they
        legitimately differ for the same corpus. A true vocab-dimension change
        fails loudly at warm-start (K x V shape), not silently."""
        manifest = dict(self.BASE_MANIFEST, vocab_size=4422)
        eff = dict(self.BASE_EFFECTIVE, vocab_size=10000)
        assert rx._resume_corpus_mismatches(manifest, eff) == []

    def test_missing_checkpoint_field_is_not_flagged(self):
        """A checkpoint predating a field (e.g. prior_obs_days) must not block
        resume — we can't verify it, so we don't penalize it."""
        manifest = {k: v for k, v in self.BASE_MANIFEST.items()
                    if k != "prior_obs_days"}
        eff = dict(self.BASE_EFFECTIVE, prior_obs_days=365)
        assert rx._resume_corpus_mismatches(manifest, eff) == []

    def test_cohort_none_sentinel_matches_python_none(self):
        manifest = dict(self.BASE_MANIFEST, cohort=None)
        eff = dict(self.BASE_EFFECTIVE, cohort_def="none")
        assert rx._resume_corpus_mismatches(manifest, eff) == []


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


def _make_zips(repo_root):
    (repo_root / "spark-vi" / "dist").mkdir(parents=True)
    (repo_root / "spark-vi" / "dist" / "spark_vi.zip").touch()
    (repo_root / "charmpheno" / "dist").mkdir(parents=True)
    (repo_root / "charmpheno" / "dist" / "charmpheno.zip").touch()


def test_build_spark_submit_cmd_appends_overlay_when_present(tmp_path):
    # The dependency-overlay zip carries the pure-Python deps the image lacks
    # (formulaic & friends). It rides on --py-files alongside the source zips,
    # on the image's own python -- no interpreter override, no --files. See
    # build_spark_submit_cmd / Makefile `cluster-overlay`.
    repo_root = tmp_path
    _make_zips(repo_root)
    dist = repo_root / "analysis" / "cloud" / "dist"
    dist.mkdir(parents=True)
    (dist / "formulaic_overlay.zip").touch()

    cmd = rx.build_spark_submit_cmd(
        "/repo/analysis/cloud/stm_bigquery_cloud.py", ["--K", "40"], repo_root
    )

    # Overlay appended to --py-files alongside the source zips.
    py_files_val = cmd[cmd.index("--py-files") + 1]
    assert "formulaic_overlay.zip" in py_files_val
    assert "spark_vi.zip" in py_files_val
    assert "charmpheno.zip" in py_files_val
    # No PEX-era interpreter machinery: the image's python runs untouched.
    assert "--files" not in cmd
    assert not any(c.startswith("spark.pyspark.python=") for c in cmd)
    assert not any(c.startswith("spark.pyspark.driver.python=") for c in cmd)


def test_build_spark_submit_cmd_omits_overlay_when_absent(tmp_path):
    # No overlay (e.g. LDA/HDP runs that never import formulaic) => only the
    # source zips ride; the job uses the image's python untouched.
    repo_root = tmp_path
    _make_zips(repo_root)

    cmd = rx.build_spark_submit_cmd(
        "/repo/analysis/cloud/lda_bigquery_cloud.py", [], repo_root
    )
    py_files_val = cmd[cmd.index("--py-files") + 1]
    assert "formulaic_overlay.zip" not in py_files_val
    assert "spark_vi.zip" in py_files_val and "charmpheno.zip" in py_files_val
    assert "--files" not in cmd


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

    def test_build_stm_args_required_flags_present(self, tmp_path, monkeypatch):
        """build_stm_args must emit all flags required by the STM driver argparse."""
        monkeypatch.setenv("WORKSPACE_CDR", "myproject.mydataset")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-billing-project")
        effective = {
            "model_class": "stm",
            "source_table": "condition_era",
            "doc_unit": "patient_year",
            "doc_min_length": 20,
            "K": 40,
            "max_iter": 20,
            "vocab_size": 10000,
            "min_df": 20,
            "min_patient_count": 20,
            "subsampling_rate": 0.2,
            "tau0": 64.0,
            "kappa": 0.7,
            "save_interval": 5,
            "person_mod": 10,
            "cohort": "dementia",
            "cohort_def": "first_dementia_year",
            "covariate_formula": "~ C(sex) + age",
            "categorical_cols": ["sex"],
            "continuous_cols": ["age"],
        }
        args = rx.build_stm_args(effective, str(tmp_path / "out"))
        # Required driver flags must be present.
        assert "--cdr" in args and "myproject.mydataset" in args
        assert "--billing" in args and "my-billing-project" in args
        assert "--out-dir" in args and str(tmp_path / "out") in args
        assert "--covariate-formula" in args
        # Renamed flags — old names must not appear.
        assert "--doc-spec" in args
        assert "--doc-unit" not in args
        assert "--save-dir" not in args
        assert "--seed" not in args
        # No resume by default.
        assert "--resume-from" not in args

    def test_build_stm_args_threads_resume_from(self, tmp_path, monkeypatch):
        """resume_from -> --resume-from on the STM driver, via build_stm_args
        and the build_fit_args dispatch (so re-runs continue a checkpoint)."""
        monkeypatch.setenv("WORKSPACE_CDR", "p.d")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "bill")
        effective = {
            "model_class": "stm", "source_table": "condition_era",
            "doc_unit": "patient_cohort", "doc_min_length": 20, "K": 40,
            "max_iter": 20, "vocab_size": 10000, "min_df": 20,
            "min_patient_count": 20, "subsampling_rate": 0.2, "tau0": 64.0,
            "kappa": 0.7, "save_interval": 5, "person_mod": 4,
            "cohort_def": "cancer_or_dementia",
            "covariate_formula": "~ C(source_cohort)",
            "categorical_cols": ["source_cohort"], "continuous_cols": [],
        }
        ckpt = tmp_path / "ckpt"
        direct = rx.build_stm_args(effective, str(tmp_path / "out"), ckpt)
        assert "--resume-from" in direct
        assert str(ckpt) in direct
        # The dispatch threads it too.
        viafit = rx.build_fit_args(effective, str(tmp_path / "out"), ckpt)
        assert "--resume-from" in viafit and str(ckpt) in viafit

    def test_build_stm_args_sources_cdr_billing_from_env(self, tmp_path, monkeypatch):
        """Regression: cdr/billing come from the workspace env, NOT the merged
        config (which never carries them). Previously build_stm_args read
        effective['cdr'] and KeyError'd on the cluster."""
        monkeypatch.setenv("WORKSPACE_CDR", "env.cdr")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-billing")
        effective = {  # deliberately NO 'cdr'/'billing' keys
            "model_class": "stm", "source_table": "condition_era",
            "doc_min_length": 20, "K": 4, "max_iter": 2, "vocab_size": 100,
            "min_df": 2, "min_patient_count": 20, "subsampling_rate": 1.0,
            "tau0": 64.0, "kappa": 0.7, "save_interval": 5, "person_mod": 10,
            "cohort": "cancer_or_dementia", "cohort_def": "cancer_or_dementia",
            "covariate_formula": "~ C(sex) + age",
            "categorical_cols": ["sex"], "continuous_cols": ["age"],
        }
        args = rx.build_stm_args(effective, str(tmp_path / "out"))
        assert args[args.index("--cdr") + 1] == "env.cdr"
        assert args[args.index("--billing") + 1] == "env-billing"

    def test_build_stm_args_missing_env_exits_cleanly(self, tmp_path, monkeypatch):
        """No WORKSPACE_CDR/GOOGLE_CLOUD_PROJECT -> clean exit(2), not KeyError."""
        import pytest
        monkeypatch.delenv("WORKSPACE_CDR", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        effective = {
            "model_class": "stm", "source_table": "condition_era",
            "doc_min_length": 20, "K": 4, "max_iter": 2, "vocab_size": 100,
            "min_df": 2, "min_patient_count": 20, "subsampling_rate": 1.0,
            "tau0": 64.0, "kappa": 0.7, "save_interval": 5, "person_mod": 10,
            "cohort": "general", "cohort_def": "none",
            "covariate_formula": "~ age",
            "categorical_cols": [], "continuous_cols": ["age"],
        }
        with pytest.raises(SystemExit):
            rx.build_stm_args(effective, str(tmp_path / "out"))

    def test_build_stm_args_parses_against_driver_argparse(self, tmp_path, monkeypatch):
        """argv from build_stm_args must parse cleanly via the driver's own argparse."""
        import argparse
        import importlib.util

        monkeypatch.setenv("WORKSPACE_CDR", "proj.ds")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
        effective = {
            "model_class": "stm",
            "source_table": "condition_era",
            "doc_unit": "patient_year",
            "doc_min_length": 20,
            "K": 10,
            "max_iter": 5,
            "vocab_size": 1000,
            "min_df": 5,
            "min_patient_count": 5,
            "subsampling_rate": 0.1,
            "tau0": 64.0,
            "kappa": 0.7,
            "save_interval": 2,
            "person_mod": 10,
            "cohort": "dementia",
            "cohort_def": "first_dementia_year",
            "covariate_formula": "~ C(sex)",
            "categorical_cols": ["sex"],
            "continuous_cols": [],
        }
        argv = rx.build_stm_args(effective, str(tmp_path / "out"))

        # Load the driver module without executing __main__ so we can call
        # parse_args directly with a supplied argv list.
        # The driver does `from _driver_common import ...` so its directory
        # must be on sys.path during import.
        import sys
        driver_dir = Path(__file__).parents[2] / "analysis" / "cloud"
        driver_path = driver_dir / "stm_bigquery_cloud.py"
        sys.path.insert(0, str(driver_dir))
        try:
            spec_obj = importlib.util.spec_from_file_location("stm_driver_under_test", driver_path)
            mod = importlib.util.module_from_spec(spec_obj)
            spec_obj.loader.exec_module(mod)
        except ImportError:
            pytest.skip("driver imports unavailable (PySpark/cloud deps not installed)")
        finally:
            sys.path.pop(0)

        # Patch error() to raise ValueError instead of calling sys.exit(2),
        # so a bad argv produces a test failure rather than a process exit.
        _orig_error = argparse.ArgumentParser.error

        def _raise(self, message):
            raise ValueError(f"argparse error: {message}")

        argparse.ArgumentParser.error = _raise
        try:
            # parse_args() in the driver calls p.parse_args() with no args —
            # we monkey-patch sys.argv momentarily.
            import sys
            _orig_argv = sys.argv
            sys.argv = ["stm_bigquery_cloud.py"] + argv
            ns = mod.parse_args()
        finally:
            sys.argv = _orig_argv
            argparse.ArgumentParser.error = _orig_error

        assert ns.cdr == "proj.ds"
        assert ns.billing == "proj"
        assert ns.out_dir == str(tmp_path / "out")
        assert ns.covariate_formula == "~ C(sex)"

    def test_build_fit_driver_path_lda(self):
        from run_experiment import build_fit_driver_path
        path = build_fit_driver_path({"model_class": "lda"})
        assert path.endswith("lda_bigquery_cloud.py")

    def _base_stm_effective(self, monkeypatch):
        """Minimal STM effective config for build_stm_args tests."""
        monkeypatch.setenv("WORKSPACE_CDR", "p.d")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "bill")
        return {
            "model_class": "stm",
            "source_table": "condition_era",
            "doc_min_length": 20,
            "K": 4,
            "max_iter": 2,
            "vocab_size": 100,
            "min_df": 2,
            "min_patient_count": 20,
            "subsampling_rate": 1.0,
            "tau0": 64.0,
            "kappa": 0.7,
            "save_interval": 5,
            "person_mod": 10,
            "cohort": "dementia",
            "cohort_def": "first_dementia_year",
            "covariate_formula": "~ C(sex)",
            "categorical_cols": ["sex"],
            "continuous_cols": [],
        }

    def test_build_stm_args_spectral_method_scalable_emitted(
            self, tmp_path, monkeypatch):
        """spectral_method='scalable' + spectral_d + spectral_min_doc_freq emit
        the three flags into the arg list."""
        effective = self._base_stm_effective(monkeypatch)
        effective["spectral_method"] = "scalable"
        effective["spectral_d"] = 256
        effective["spectral_min_doc_freq"] = 3
        args = rx.build_stm_args(effective, str(tmp_path / "out"))
        assert "--spectral-method" in args
        assert args[args.index("--spectral-method") + 1] == "scalable"
        assert "--spectral-d" in args
        assert args[args.index("--spectral-d") + 1] == "256"
        assert "--spectral-min-doc-freq" in args
        assert args[args.index("--spectral-min-doc-freq") + 1] == "3"

    def test_build_stm_args_spectral_method_default_dense_omitted(
            self, tmp_path, monkeypatch):
        """An effective config with no spectral_method key emits neither
        --spectral-method nor --spectral-d (dense default path stays clean)."""
        effective = self._base_stm_effective(monkeypatch)
        # Deliberately do NOT set spectral_method / spectral_d / spectral_min_doc_freq.
        args = rx.build_stm_args(effective, str(tmp_path / "out"))
        assert "--spectral-method" not in args
        assert "--spectral-d" not in args
        assert "--spectral-min-doc-freq" not in args


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

    def test_stm_with_cache_uri_passes_cache_uri(self, tmp_path):
        effective = {
            "model_class": "stm",
            "vocab_top_n": 5000,
            "top_n_codes_for_npmi": 20,
            "cache_uri": "gs://bucket/cache",
        }
        args = rx.build_dashboard_args(effective, tmp_path / "ck", "z.zip")
        assert "--cache-uri" in args
        assert args[args.index("--cache-uri") + 1] == "gs://bucket/cache"

    def test_no_cache_uri_omits_flag(self, tmp_path):
        # Without cache_uri configured (e.g. LDA, or STM before the cache is
        # built), the flag must be absent so the driver falls back cleanly.
        effective = {
            "model_class": "lda",
            "vocab_top_n": 5000,
            "top_n_codes_for_npmi": 20,
        }
        args = rx.build_dashboard_args(effective, tmp_path / "ck", "z.zip")
        assert "--cache-uri" not in args


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


class TestBuildCovariatesOnly:
    """--build-covariates-only flag: arg parsing, validation, mutual exclusion."""

    def _write_stm_experiment(self, dir_path: Path, *, id: int, slug: str) -> Path:
        path = dir_path / f"{id:04d}-{slug}.md"
        path.write_text(
            f"---\n"
            f"id: {id}\n"
            f"slug: {slug}\n"
            f"status: pending\n"
            f"model_class: stm\n"
            f"cohort: dementia\n"
            f"covariate_formula: \"~ C(sex) + age\"\n"
            f"categorical_cols: [sex]\n"
            f"continuous_cols: [age]\n"
            f"---\n\n# {slug}\n"
        )
        return path

    def test_flag_is_parseable(self, tmp_path, capsys):
        """--build-covariates-only is accepted by the parser (not an unknown arg)."""
        runs = tmp_path / "runs"
        runs.mkdir()
        # Without --id this should fail cleanly (not with an argparse error).
        rc = rx.main([
            "--build-covariates-only",
            "--runs-dir", str(runs),
        ])
        assert rc == 2
        out = capsys.readouterr().out
        assert "build-covariates-only" in out

    def test_requires_id(self, tmp_path, capsys):
        """--build-covariates-only without --id exits 2 with a clear message."""
        rc = rx.main([
            "--build-covariates-only",
            "--runs-dir", str(tmp_path / "runs"),
        ])
        assert rc == 2
        assert "id" in capsys.readouterr().out.lower()

    def test_mutually_exclusive_with_eval_only(self, tmp_path, capsys):
        rc = rx.main([
            "--build-covariates-only", "--eval-only",
            "--id", "1",
            "--runs-dir", str(tmp_path),
        ])
        assert rc == 2
        assert "contradictory" in capsys.readouterr().out.lower()

    def test_mutually_exclusive_with_no_eval(self, tmp_path, capsys):
        rc = rx.main([
            "--build-covariates-only", "--no-eval",
            "--id", "1",
            "--runs-dir", str(tmp_path),
        ])
        assert rc == 2

    def test_mutually_exclusive_with_build_only(self, tmp_path, capsys):
        rc = rx.main([
            "--build-covariates-only", "--build-only",
            "--id", "1",
            "--runs-dir", str(tmp_path),
        ])
        assert rc == 2

    def test_force_covariates_requires_build_covariates_only(self, tmp_path, capsys):
        """--force-covariates without --build-covariates-only exits 2."""
        rc = rx.main([
            "--force-covariates", "--id", "1",
            "--runs-dir", str(tmp_path),
        ])
        assert rc == 2
        assert "force-covariates" in capsys.readouterr().out.lower()

    def test_rejects_non_stm_model_class(self, tmp_path, capsys):
        """--build-covariates-only on an LDA experiment exits 2."""
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        defaults_dir = tmp_path / "defaults"
        defaults_dir.mkdir()
        (defaults_dir / "_base.yaml").write_text(
            "source_table: condition_era\nperson_mod: 10\n"
            "cache_uri: gs://fake/cache\n"
        )
        (defaults_dir / "dementia.yaml").write_text("cohort: dementia\n")
        p = exp_dir / "0042-lda-test.md"
        p.write_text(
            "---\n"
            "id: 42\nslug: lda-test\ncohort: dementia\nmodel_class: lda\n"
            "status: pending\n"
            "---\n"
        )
        runs = tmp_path / "runs"
        runs.mkdir()
        rc = rx.main([
            "--id", "42",
            "--build-covariates-only",
            "--runs-dir", str(runs),
            "--experiments-dir", str(exp_dir),
            "--defaults-dir", str(defaults_dir),
        ])
        assert rc == 2
        out = capsys.readouterr().out.lower()
        assert "model_class" in out or "stm" in out

    def test_build_covariates_args_minimal(self, monkeypatch):
        """build_covariates_args produces expected CLI flags from effective config."""
        monkeypatch.setenv("WORKSPACE_CDR", "my_cdr")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my_billing")
        effective = {
            "source_table": "condition_era",
            "person_mod": 10,
            "cache_uri": "gs://bucket/cache",
            "covariate_formula": "~ C(sex) + age",
            "categorical_cols": ["sex"],
            "continuous_cols": ["age"],
            "cohort": "dementia",
            "cohort_def": "first_dementia_year",
        }
        args = rx.build_covariates_args(effective)
        assert "--cdr" in args and "my_cdr" in args
        assert "--billing" in args and "my_billing" in args
        assert "--cache-uri" in args and "gs://bucket/cache" in args
        assert "--covariate-formula" in args
        assert "--categorical-cols" in args
        assert "--continuous-cols" in args
        assert "--cohort" in args and "first_dementia_year" in args

    def test_build_covariates_args_no_cohort_def(self, monkeypatch):
        """When cohort_def is 'none', --cohort is omitted."""
        monkeypatch.setenv("WORKSPACE_CDR", "c")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "b")
        effective = {
            "source_table": "condition_era",
            "person_mod": 10,
            "cache_uri": "gs://bucket/cache",
            "covariate_formula": "~ age",
            "categorical_cols": [],
            "continuous_cols": ["age"],
            "cohort": "general",
            "cohort_def": "none",
        }
        args = rx.build_covariates_args(effective)
        assert "--cohort" not in args


def test_build_stm_args_includes_gating_flags(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CDR", "proj.ds")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    from run_experiment import build_stm_args
    eff = {
        "source_table": "condition_era", "doc_min_length": 20, "K": 50,
        "max_iter": 20, "vocab_size": 10000, "min_df": 20,
        "min_patient_count": 20, "subsampling_rate": 0.2, "tau0": 64.0,
        "kappa": 0.7, "save_interval": 5, "person_mod": 4,
        "covariate_formula": "~ C(sex) + age", "categorical_cols": ["sex"],
        "continuous_cols": ["age"],
        "background_k": 30, "foreground": "cancer:10,dementia:10",
        "group_var": "source_cohort",
    }
    argv = build_stm_args(eff, "/tmp/out")
    assert "--background-k" in argv and "30" in argv
    assert "--foreground" in argv and "cancer:10,dementia:10" in argv
    assert "--group-var" in argv and "source_cohort" in argv


def test_build_stm_args_omits_gating_when_absent(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CDR", "proj.ds")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    from run_experiment import build_stm_args
    eff = {
        "source_table": "condition_era", "doc_min_length": 20, "K": 40,
        "max_iter": 20, "vocab_size": 10000, "min_df": 20,
        "min_patient_count": 20, "subsampling_rate": 0.2, "tau0": 64.0,
        "kappa": 0.7, "save_interval": 5, "person_mod": 4,
        "covariate_formula": "~ C(sex) + age", "categorical_cols": ["sex"],
        "continuous_cols": ["age"],
    }
    argv = build_stm_args(eff, "/tmp/out")
    assert "--background-k" not in argv and "--foreground" not in argv


def test_resume_mismatch_on_changed_partition():
    from run_experiment import _resume_corpus_mismatches
    ck = {"person_mod": 4, "source_table": "condition_era",
          "topic_block_spec": {"group_var": "source_cohort", "background_k": 30,
                               "foreground": [["cancer", 10], ["dementia", 10]]}}
    eff = {"person_mod": 4, "source_table": "condition_era",
           "background_k": 20, "foreground": "cancer:10,dementia:10",
           "group_var": "source_cohort", "K": 40}
    out = _resume_corpus_mismatches(ck, eff)
    assert any("topic_block_spec" in m for m in out)


def test_build_stm_args_threads_hardening_flags(monkeypatch):
    import run_experiment
    monkeypatch.setattr(run_experiment, "_require_workspace_env",
                        lambda: ("proj.ds", "billing"))
    effective = {
        "source_table": "condition_era", "doc_unit": "patient",
        "doc_min_length": 1, "K": 40, "max_iter": 300, "vocab_size": 3000,
        "min_df": 5, "min_patient_count": 20, "subsampling_rate": 1.0,
        "tau0": 64.0, "kappa": 0.7, "save_interval": 50, "person_mod": 4,
        "covariate_formula": "~ C(sex) + age", "categorical_cols": ["sex"],
        "continuous_cols": ["age"],
        "reference_topic": True,
        "sigma_prior_scale": 2.0, "sigma_prior_count": 500.0,
        "spectral_init": True,
    }
    args = run_experiment.build_stm_args(effective, out_dir="/tmp/out")
    assert "--reference-topic" in args
    assert "--spectral-init" in args
    i = args.index("--sigma-prior-scale"); assert args[i + 1] == "2.0"
    j = args.index("--sigma-prior-count"); assert args[j + 1] == "500.0"


def test_build_stm_args_hardening_flags_default_on(monkeypatch):
    import run_experiment
    monkeypatch.setattr(run_experiment, "_require_workspace_env",
                        lambda: ("proj.ds", "billing"))
    effective = {
        "source_table": "condition_era", "doc_unit": "patient",
        "doc_min_length": 1, "K": 40, "max_iter": 300, "vocab_size": 3000,
        "min_df": 5, "min_patient_count": 20, "subsampling_rate": 1.0,
        "tau0": 64.0, "kappa": 0.7, "save_interval": 50, "person_mod": 4,
        "covariate_formula": "~ C(sex) + age", "categorical_cols": ["sex"],
        "continuous_cols": ["age"],
    }
    args = run_experiment.build_stm_args(effective, out_dir="/tmp/out")
    assert "--reference-topic" in args
    assert "--spectral-init" in args
    assert "--sigma-prior-scale" not in args
    assert "--sigma-prior-count" not in args


def test_build_stm_args_hardening_flags_disabled(monkeypatch):
    import run_experiment
    monkeypatch.setattr(run_experiment, "_require_workspace_env",
                        lambda: ("proj.ds", "billing"))
    effective = {
        "source_table": "condition_era", "doc_unit": "patient",
        "doc_min_length": 1, "K": 40, "max_iter": 300, "vocab_size": 3000,
        "min_df": 5, "min_patient_count": 20, "subsampling_rate": 1.0,
        "tau0": 64.0, "kappa": 0.7, "save_interval": 50, "person_mod": 4,
        "covariate_formula": "~ C(sex) + age", "categorical_cols": ["sex"],
        "continuous_cols": ["age"],
        "reference_topic": False,
        "spectral_init": False,
    }
    args = run_experiment.build_stm_args(effective, out_dir="/tmp/out")
    assert "--no-reference-topic" in args
    assert "--no-spectral-init" in args


def test_build_stm_args_emits_full_sigma_knobs(monkeypatch):
    """sigma_diag_shrink + min_pair_support are emitted when set in effective."""
    import run_experiment
    monkeypatch.setattr(run_experiment, "_require_workspace_env",
                        lambda: ("proj.ds", "billing"))
    eff = {
        "source_table": "condition_era", "doc_unit": "patient",
        "doc_min_length": 1, "K": 40, "max_iter": 2, "vocab_size": 100,
        "min_df": 2, "min_patient_count": 20, "subsampling_rate": 1.0,
        "tau0": 64.0, "kappa": 0.7, "save_interval": 5, "person_mod": 4,
        "covariate_formula": "~ C(sex) + age", "categorical_cols": ["sex"],
        "continuous_cols": ["age"],
        "sigma_diag_shrink": 0.25, "min_pair_support": 30,
    }
    args = run_experiment.build_stm_args(eff, out_dir="/tmp/out")
    assert "--sigma-diag-shrink" in args and "0.25" in args
    assert "--min-pair-support" in args and "30" in args


def test_build_stm_args_omits_full_sigma_knobs_when_absent(monkeypatch):
    """Default path: no sigma_diag_shrink / min_pair_support in effective -> neither flag emitted."""
    import run_experiment
    monkeypatch.setattr(run_experiment, "_require_workspace_env",
                        lambda: ("proj.ds", "billing"))
    eff = {
        "source_table": "condition_era", "doc_unit": "patient",
        "doc_min_length": 1, "K": 40, "max_iter": 2, "vocab_size": 100,
        "min_df": 2, "min_patient_count": 20, "subsampling_rate": 1.0,
        "tau0": 64.0, "kappa": 0.7, "save_interval": 5, "person_mod": 4,
        "covariate_formula": "~ C(sex) + age", "categorical_cols": ["sex"],
        "continuous_cols": ["age"],
    }
    args = run_experiment.build_stm_args(eff, out_dir="/tmp/out")
    assert "--sigma-diag-shrink" not in args
    assert "--min-pair-support" not in args
