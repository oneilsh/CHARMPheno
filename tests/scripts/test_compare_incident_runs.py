"""compare_incident_runs.py: nodes matched by Mondo id across runs."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "cloud"))
import compare_incident_runs as cr  # noqa: E402


def _run(d, i2c, aucs):
    d.mkdir()
    (d / "readout_dag.json").write_text(json.dumps({"int2cid": i2c, "name_by_id": {}}))
    pn = {str(k): {"auc_incident": a, "n_pos_incident": 30} for k, a in aucs.items()}
    (d / "p.json").write_text(json.dumps({"history_stacked": {"per_node": pn},
                                          "history_flat": {"per_node": pn}}))


def test_match_by_mondo_id(tmp_path, capsys):
    _run(tmp_path / "a", {"0": -1, "1": 100, "2": 200}, {1: 0.6, 2: 0.7})
    _run(tmp_path / "b", {"0": -1, "5": 200, "7": 100, "8": 300}, {5: 0.75, 7: 0.7, 8: 0.9})
    A = cr.by_cid(tmp_path / "a", "p.json", "history_stacked")
    B = cr.by_cid(tmp_path / "b", "p.json", "history_stacked")
    c = cr.compare(A, B)
    assert c["n"] == 2 and abs(c["B_minus_A_median"] - 0.075) < 1e-9 and c["B_wins"] == 2
