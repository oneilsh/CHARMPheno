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
