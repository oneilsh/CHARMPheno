from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path


def test_make_dev_bundle_emits_four_conformant_files(tmp_path: Path):
    out = tmp_path / "data"
    subprocess.check_call([
        sys.executable, "scripts/make_dev_bundle.py",
        "--out-dir", str(out), "--k", "5", "--v", "20", "--seed", "1",
    ])
    assert {p.name for p in out.iterdir()} == {
        "model.json", "vocab.json", "phenotypes.json", "corpus_stats.json"
    }
    model = json.loads((out / "model.json").read_text())
    assert model["K"] == 5 and model["V"] == 20
    phenos = json.loads((out / "phenotypes.json").read_text())["phenotypes"]
    assert all("original_topic_id" in p for p in phenos)
