"""Unit tests for analysis/cloud/resolve_vocab_names.py -- the off-YARN vocab
name resolver. The BigQuery call is not exercised (no cluster); the pure
decode/assemble logic and the --names-json path are.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

CLOUD = Path(__file__).resolve().parents[2] / "analysis" / "cloud"
sys.path.insert(0, str(CLOUD))
import resolve_vocab_names as rv  # noqa: E402


def test_token_tools_decode_and_labels():
    BASE, decode, label = rv._token_tools()
    assert BASE == 100
    # token = concept * 100 + state
    assert decode(30003) == (300, 3)
    assert decode(30001) == (300, 1)
    assert label(3) in ("high",) and label(1) in ("low",)


def test_collect_features_decodes_measurement_only():
    vocab_maps = [{"200": 0, "201": 1},                 # condition: real ids
                  {"30003": 0, "30001": 1}]             # measurement: tokens
    features, real_ids = rv.collect_features(vocab_maps, ["condition", "measurement"])
    fmap = {fid: (dom, real, st) for fid, dom, real, st in features}
    # condition feature id IS the real concept, no state
    assert fmap[200] == ("condition", 200, None)
    # measurement token decodes to real concept 300 + state
    assert fmap[30003] == ("measurement", 300, 3)
    assert fmap[30001] == ("measurement", 300, 1)
    assert real_ids == {200, 201, 300}


def test_build_rows_appends_measurement_state():
    vocab_maps = [{"200": 0}, {"30003": 0, "30001": 1}]
    features, _ = rv.collect_features(vocab_maps, ["condition", "measurement"])
    _, _, label = rv._token_tools()
    rows = dict(rv.build_rows(features, {200: "hypertension", 300: "creatinine"},
                              label))
    assert rows[200] == "hypertension"
    assert rows[30003] == "creatinine [high]"
    assert rows[30001] == "creatinine [low]"


def test_build_rows_skips_unnamed_real_ids():
    vocab_maps = [{"200": 0, "999": 1}]
    features, _ = rv.collect_features(vocab_maps, ["condition"])
    _, _, label = rv._token_tools()
    rows = dict(rv.build_rows(features, {200: "hypertension"}, label))  # 999 absent
    assert 200 in rows and 999 not in rows


def test_main_names_json_writes_csv(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "domain_names": ["condition", "measurement"],
        "corpus_manifest": {"cdr": "proj.ds", "billing": "bill"}}))
    (tmp_path / "meta.json").write_text(json.dumps({
        "vocab_maps": [{"200": 0}, {"30003": 0}]}))
    (tmp_path / "names.json").write_text(json.dumps({"200": "htn", "300": "creat"}))
    out = tmp_path / "concept_names.csv"
    r = subprocess.run(
        [sys.executable, str(CLOUD / "resolve_vocab_names.py"),
         "--run-dir", str(tmp_path), "--bundle-meta", str(tmp_path / "meta.json"),
         "--names-json", str(tmp_path / "names.json"), "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = {row["concept_id"]: row["concept_name"]
           for row in csv.DictReader(out.open())}
    assert got["200"] == "htn"
    assert got["30003"] == "creat [high]"


def test_dry_run_needs_no_bq(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "domain_names": ["condition"], "corpus_manifest": {"cdr": "p.d", "billing": "b"}}))
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_maps": [{"200": 0}]}))
    r = subprocess.run(
        [sys.executable, str(CLOUD / "resolve_vocab_names.py"),
         "--run-dir", str(tmp_path), "--bundle-meta", str(tmp_path / "meta.json"),
         "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "first-batch SQL" in r.stdout and "p.d.concept" in r.stdout
