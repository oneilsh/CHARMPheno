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


def _nested_run(d, b6):
    # 1 is the parent of 2 (nested): group 5 = {1, 2, 3, 4} -> most specific {2, 3, 4}
    (d / "readout_dag.json").write_text(json.dumps({
        "C_fit": 5, "int2cid": {"0": -1, "1": 11, "2": 12, "3": 13, "4": 14, "5": 15},
        "name_by_id": {"15": "group A"},
        "parent_int": {"1": [5], "2": [1], "3": [5], "4": [5], "5": [0]},
        "pos_sources": [[0], [1], [2], [3], [4], [1, 2, 3, 4]]}))
    (d / "manifest.json").write_text(json.dumps({"corpus_manifest": {}}))
    (d / "differential_s.json").write_text(json.dumps(
        {"group A": {"metrics": _metrics(0.4, 0.5, b6), "n_eval": 100}}))
    (d / "differential_s.md").write_text("")


def test_nested_label_space_uses_most_specific_members(tmp_path):
    _nested_run(tmp_path, 0.6)
    assert sa.group_members(tmp_path)["group A"] == (2, 3, 4)
    assert abs(sa.kappa_by_group(tmp_path, "s")["group A"][1] - sa.kappa(0.6, 3)) < 1e-9


def test_names_fall_back_to_the_fit_manifest(tmp_path):
    _nested_run(tmp_path, 0.6)
    rd = json.loads((tmp_path / "readout_dag.json").read_text())
    rd["name_by_id"] = {}
    (tmp_path / "readout_dag.json").write_text(json.dumps(rd))
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"corpus_manifest": {"name_by_id": {"15": "group A"}}}))
    assert "group A" in sa.group_members(tmp_path)


def test_paired_by_group_name(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _nested_run(a, 0.6)
    _nested_run(b, 0.5)
    assert sa.main(["x", str(a), "s", str(b), "s"]) == 0
    out = capsys.readouterr().out
    assert "1 shared groups" in out and "A better in 1 of 1" in out


def test_select_groups_with_identical_top_level_members(tmp_path):
    import select_survey_groups as ssg
    nested, flat = tmp_path / "n", tmp_path / "f"
    nested.mkdir(); flat.mkdir()
    # nested: group 5 -> top members {1, 4}; 1 has subtypes 2, 3
    (nested / "readout_dag.json").write_text(json.dumps({
        "C_fit": 5, "int2cid": {"0": -1, "1": 11, "2": 12, "3": 13, "4": 14, "5": 15},
        "name_by_id": {"15": "group A"},
        "parent_int": {"1": [5], "2": [1], "3": [1], "4": [5], "5": [0]},
        "pos_sources": [[0], [1, 2, 3], [2], [3], [4], [1, 2, 3, 4]]}))
    (nested / "manifest.json").write_text(json.dumps({"corpus_manifest": {}}))
    # flat: group with members cid 11, 14 (same terms)
    (flat / "readout_dag.json").write_text(json.dumps({
        "C_fit": 3, "int2cid": {"0": -1, "1": 11, "2": 14, "3": 15},
        "name_by_id": {"15": "group A"},
        "pos_sources": [[0], [1], [2], [1, 2]]}))
    (flat / "manifest.json").write_text(json.dumps({"corpus_manifest": {"label_set": "x"}}))
    (flat / "differential_s.json").write_text(json.dumps(
        {"group A": {"metrics": _metrics(0.4, 0.5, 0.6), "n_eval": 100}}))
    picked, n_id, n_sz, n_all = ssg.select(nested, flat, "s")
    assert picked == ["group A"] and n_id == 1


def test_framing3_sources_finds_the_record_check_and_skips_resolves_and_raw(tmp_path):
    import json
    from prevalence_check import framing3_sources
    for n, v in [("prevalence_check_stacked_nodes_mass99_warm_it400.json", "warm"),
                 ("prevalence_check_stacked_nodes_rawm2000.json", "rawm"),
                 ("prevalence_check_stacked_nodes_raw2000.json", "raw")]:
        (tmp_path / n).write_text(json.dumps({"v": v}))
    th, rw = framing3_sources(tmp_path)
    assert th is None and rw == {"v": "raw"}          # no θ record yet; raw found
    (tmp_path / "prevalence_check_stacked_nodes_mass99.json").write_text('{"v": "theta"}')
    assert framing3_sources(tmp_path)[0] == {"v": "theta"}
    (tmp_path / "prevalence_check_history.json").write_text('{"v": "history"}')
    assert framing3_sources(tmp_path)[0] == {"v": "history"}
