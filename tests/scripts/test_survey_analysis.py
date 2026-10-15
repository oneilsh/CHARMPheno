"""Exp 0150 follow-up: survey_analysis.py (dedupe, kappa, reasons parse)."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "cloud"))

import survey_analysis as sa  # noqa: E402


def _metrics(b3, b5, b6):
    mk = lambda b: {"bal_top1": b, "top1": b, "top3": 0.9, "logloss": 1.0, "ece": 0.1}
    return {"prior": mk(0.25), "E3 multiclass head": mk(b3), "E5 code head": mk(b5),
            "E6 θ + codes": mk(b6)}


def test_kappa_dedupe_and_reasons(tmp_path, capsys):
    (tmp_path / "readout_dag.json").write_text(json.dumps({
        "C_fit": 5, "int2cid": {"0": -1, "1": 11, "2": 12, "3": 13, "4": 14, "5": 15,
                                 "6": 16},
        "name_by_id": {"15": "group A", "16": "group A alias"},
        "pos_sources": [[0], [1], [2], [3], [4], [1, 2, 3, 4], [1, 2, 3, 4]]}))
    nq = {"visible": {"p_true": 0.2}, "full": {"p_true": 0.4},
          "by": {"VOI (codes)@5": {"p_true": 0.25}, "population table@5": {"p_true": 0.24}}}
    js = {"group A": {"metrics": _metrics(0.4, 0.5, 0.55), "n_eval": 100,
                      "next_question": nq},
          "group A alias": {"metrics": _metrics(0.4, 0.5, 0.55), "n_eval": 100}}
    (tmp_path / "differential_s.json").write_text(json.dumps(js))
    (tmp_path / "differential_s.md").write_text(
        "## group A — 4 members · 100 scored\n\nE5 reasons (top)\n- m1: aspirin · x\n\n")
    assert sa.main(["x", str(tmp_path), "s"]) == 0
    out = capsys.readouterr().out
    assert "1 distinct member sets (1 duplicate names dropped)" in out
    assert abs(sa.kappa(0.55, 4) - 0.4) < 1e-9
    assert "| group A (+1) | 4 | 100 | E6 | 0.40 | 0.25 | +0.010 |" in out
    assert "- m1: aspirin · x" in out
