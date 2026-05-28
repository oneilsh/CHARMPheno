"""Tests for scripts.build_dashboard_manifest."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add scripts/ to sys.path so we can import the script as a module.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import build_dashboard_manifest as bdm  # noqa: E402


def _make_bundle(dir_: Path, *, with_cohort: bool, cohort_id: str = "x", label: str = "L", description: str = "D"):
    dir_.mkdir(parents=True)
    payload: dict = {"corpus_size_docs": 100, "mean_codes_per_doc": 5.0, "k": 10, "v": 50, "v_full": 100}
    if with_cohort:
        payload["cohort"] = {"id": cohort_id, "label": label, "description": description}
    (dir_ / "corpus_stats.json").write_text(json.dumps(payload))


def test_discover_cohort_entries_picks_up_inline_metadata(tmp_path: Path):
    _make_bundle(tmp_path / "cancer", with_cohort=True, label="Cancer label", description="Cancer desc")
    _make_bundle(tmp_path / "dementia", with_cohort=True, label="Dementia label", description="Dementia desc")
    entries = bdm.discover_cohort_entries(tmp_path)
    assert entries == [
        {"id": "cancer", "label": "Cancer label", "description": "Cancer desc"},
        {"id": "dementia", "label": "Dementia label", "description": "Dementia desc"},
    ]


def test_discover_cohort_entries_fallback_for_missing_cohort_field(tmp_path: Path):
    _make_bundle(tmp_path / "local-test", with_cohort=False)
    entries = bdm.discover_cohort_entries(tmp_path)
    assert entries == [{"id": "local-test", "label": "local-test", "description": ""}]


def test_discover_cohort_entries_skips_subdirs_without_corpus_stats(tmp_path: Path):
    (tmp_path / "not-a-cohort").mkdir()
    _make_bundle(tmp_path / "cancer", with_cohort=True, label="C")
    entries = bdm.discover_cohort_entries(tmp_path)
    assert [e["id"] for e in entries] == ["cancer"]


def test_resolve_default_prefers_requested(tmp_path: Path):
    entries = [{"id": "cancer", "label": "C", "description": ""}, {"id": "dementia", "label": "D", "description": ""}]
    assert bdm.resolve_default(entries, requested="dementia", existing_manifest_path=tmp_path / "missing.json") == "dementia"


def test_resolve_default_rejects_unknown_requested(tmp_path: Path):
    entries = [{"id": "cancer", "label": "C", "description": ""}]
    with pytest.raises(SystemExit):
        bdm.resolve_default(entries, requested="foo", existing_manifest_path=tmp_path / "missing.json")


def test_resolve_default_preserves_existing_when_valid(tmp_path: Path):
    entries = [{"id": "cancer", "label": "C", "description": ""}, {"id": "dementia", "label": "D", "description": ""}]
    existing = tmp_path / "manifest.json"
    existing.write_text(json.dumps({"default": "dementia", "cohorts": []}))
    assert bdm.resolve_default(entries, requested=None, existing_manifest_path=existing) == "dementia"


def test_resolve_default_falls_back_to_first_alphabetical_when_existing_invalid(tmp_path: Path):
    entries = [{"id": "cancer", "label": "C", "description": ""}, {"id": "dementia", "label": "D", "description": ""}]
    existing = tmp_path / "manifest.json"
    existing.write_text(json.dumps({"default": "gone", "cohorts": []}))
    assert bdm.resolve_default(entries, requested=None, existing_manifest_path=existing) == "cancer"


def test_main_writes_manifest_with_correct_shape(tmp_path: Path, capsys):
    _make_bundle(tmp_path / "cancer", with_cohort=True, label="Cancer label", description="Cancer desc")
    _make_bundle(tmp_path / "dementia", with_cohort=True, label="Dementia label", description="Dementia desc")
    rc = bdm.main(["--data-dir", str(tmp_path), "--default", "dementia"])
    assert rc == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["default"] == "dementia"
    assert [c["id"] for c in manifest["cohorts"]] == ["cancer", "dementia"]
