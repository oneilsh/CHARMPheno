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
