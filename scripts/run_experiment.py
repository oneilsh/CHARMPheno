"""Experiment-tracking runner: reads a docs/experiments/NNNN-slug.md
record, merges defaults, dispatches lda_bigquery_cloud.py via spark-submit,
captures sanitized stdout to summary.md in the run dir, then runs eval.

See docs/superpowers/specs/2026-05-28-experiment-tracking-design.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

import datetime as _dt
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "docs" / "experiments"
DEFAULTS_DIR = REPO_ROOT / "experiments" / "defaults"


def read_frontmatter(path: Path) -> dict:
    """Parse YAML frontmatter from a markdown file.

    Frontmatter is delimited by leading and trailing `---` lines on their own.
    Returns the parsed dict. Raises ValueError if absent or unterminated.
    """
    text = path.read_text()
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing frontmatter block (expected leading '---')")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError(f"{path}: unterminated frontmatter block (no trailing '---')")
    yaml_text = text[4:end]
    return yaml.safe_load(yaml_text) or {}


def merge_config(base: dict, override: dict) -> dict:
    """Shallow merge: returns a new dict with override taking precedence over base."""
    out = dict(base)
    out.update(override)
    return out


def load_defaults(cohort: str, defaults_dir: Path) -> dict:
    """Load _base.yaml then <cohort>.yaml and merge.

    Raises FileNotFoundError if either file is missing.
    """
    base_path = defaults_dir / "_base.yaml"
    cohort_path = defaults_dir / f"{cohort}.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"missing defaults file: {base_path}")
    if not cohort_path.exists():
        raise FileNotFoundError(f"missing defaults file: {cohort_path}")
    base = yaml.safe_load(base_path.read_text()) or {}
    cohort_overrides = yaml.safe_load(cohort_path.read_text()) or {}
    return merge_config(base, cohort_overrides)


def _list_experiment_files(experiments_dir: Path) -> list[Path]:
    """All files matching NNNN-*.md in experiments_dir, sorted by id."""
    pattern = re.compile(r"^(\d{4})-.+\.md$")
    out = []
    for p in experiments_dir.iterdir():
        if pattern.match(p.name):
            out.append(p)
    out.sort(key=lambda p: p.name)
    return out


def find_next_pending(experiments_dir: Path) -> Path | None:
    """Return the lowest-id experiment file with status: pending, or None."""
    for p in _list_experiment_files(experiments_dir):
        try:
            fm = read_frontmatter(p)
        except ValueError:
            continue
        if fm.get("status") == "pending":
            return p
    return None


def find_by_id(experiments_dir: Path, exp_id: int) -> Path:
    """Return the experiment file with the given id. Raises FileNotFoundError if absent."""
    prefix = f"{exp_id:04d}-"
    for p in _list_experiment_files(experiments_dir):
        if p.name.startswith(prefix):
            return p
    raise FileNotFoundError(f"no experiment with id {exp_id:04d} in {experiments_dir}")


# Patterns that indicate per-patient row-level info. Belt-and-suspenders for
# driver-side stripping; if a new driver path re-introduces patient prints,
# these catch them at the wrapper boundary before they reach summary.md.
PATIENT_PATTERNS: list[re.Pattern] = [
    re.compile(r"person_hash", re.IGNORECASE),
    re.compile(r"person_id\s*=\s*\S+"),
    re.compile(r"\bhash:[0-9a-f]{6,}", re.IGNORECASE),
    re.compile(r"transform sample", re.IGNORECASE),  # the phase marker bracketing it
]


def sanitize_line(line: str, patterns: list[re.Pattern]) -> str | None:
    """Return the line if safe to commit, or None to drop.

    Drops any line matching any patient-info pattern. Aggregate counts and
    topic-level prints (no per-patient identifiers) pass through.
    """
    for pat in patterns:
        if pat.search(line):
            return None
    return line


def build_lda_args(
    effective: dict, save_dir: Path, resume_from: Path | None,
) -> list[str]:
    """Build the CLI arg list for lda_bigquery_cloud.py from an effective config.

    The driver's argparse defaults handle anything not set here; we explicitly
    pass everything in `effective` to keep the cluster's behavior reproducible
    from the config alone.
    """
    args: list[str] = [
        "--save-dir", str(save_dir),
        "--save-interval", str(effective["save_interval"]),
        "--source-table", str(effective["source_table"]),
        "--doc-unit", str(effective["doc_unit"]),
        "--doc-min-length", str(effective["doc_min_length"]),
        "--K", str(effective["K"]),
        "--max-iter", str(effective["max_iter"]),
        "--vocab-size", str(effective["vocab_size"]),
        "--min-df", str(effective["min_df"]),
        "--min-patient-count", str(effective["min_patient_count"]),
        "--subsampling-rate", str(effective["subsampling_rate"]),
        "--tau0", str(effective["tau0"]),
        "--kappa", str(effective["kappa"]),
        "--print-topics-every", str(effective["print_topics_every"]),
        "--person-mod", str(effective["person_mod"]),
        "--top-n-tokens", str(effective["top_n_tokens"]),
        "--seed", str(effective["seed"]),
        "--cohort", str(effective["cohort"]),
    ]
    # BooleanOptionalAction in the driver: --optimize-doc-concentration / --no-...
    if effective.get("optimize_doc_concentration", True):
        args.append("--optimize-doc-concentration")
    else:
        args.append("--no-optimize-doc-concentration")
    if effective.get("optimize_topic_concentration", False):
        args.append("--optimize-topic-concentration")
    else:
        args.append("--no-optimize-topic-concentration")
    if resume_from is not None:
        args += ["--resume-from", str(resume_from)]
    return args


def build_eval_args(checkpoint_dir: Path, effective: dict) -> list[str]:
    """Build the CLI arg list for eval_coherence_cloud.py."""
    return [
        "--checkpoint", str(checkpoint_dir),
        "--model-class", str(effective.get("model_class", "lda")),
    ]


# Spark configuration constants. Match the existing per-cohort Makefile targets.
# If these need to vary per-experiment, lift them into _base.yaml in a future
# increment.
SPARK_SUBMIT_FLAGS = [
    "--master", "yarn",
    "--deploy-mode", "client",
    "--driver-memory", "4g",
    "--conf", "spark.executor.cores=4",
    "--conf", "spark.executor.memory=6g",
    "--conf", "spark.executor.memoryOverhead=2g",
    "--conf", "spark.python.worker.memory=2g",
]


def build_spark_submit_cmd(
    script: str, script_args: list[str], repo_root: Path,
) -> list[str]:
    """Build the full spark-submit command line."""
    spark_vi_zip = repo_root / "spark-vi" / "dist" / "spark_vi.zip"
    charmpheno_zip = repo_root / "charmpheno" / "dist" / "charmpheno.zip"
    py_files_val = f"{spark_vi_zip},{charmpheno_zip}"
    return (
        ["spark-submit"]
        + SPARK_SUBMIT_FLAGS
        + ["--py-files", py_files_val, script]
        + script_args
    )


def _count_existing_fit_sessions(summary_text: str) -> int:
    """Count occurrences of '## Fit session N' headers in existing summary."""
    return len(re.findall(r"^## Fit session \d+", summary_text, re.MULTILINE))


def write_summary_header(
    summary_path: Path, *, exp_id: int, slug: str, effective: dict,
) -> None:
    """Write or append the per-session header to summary.md.

    If the file doesn't exist: writes the experiment-level header (title + effective
    config block) plus '## Fit session 1' header.
    If the file exists: appends '## Fit session N' header where N is (count+1).
    """
    started = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if not summary_path.exists():
        lines = [
            f"# Experiment {exp_id:04d} — {slug}",
            "",
            "## Effective config",
        ]
        for k in sorted(effective.keys()):
            lines.append(f"{k}: {effective[k]}")
        lines += ["", "## Fit session 1", f"Started: {started}", ""]
        summary_path.write_text("\n".join(lines) + "\n")
    else:
        existing = summary_path.read_text()
        n = _count_existing_fit_sessions(existing) + 1
        with summary_path.open("a") as f:
            f.write(f"\n## Fit session {n}\nStarted: {started}\n\n")
