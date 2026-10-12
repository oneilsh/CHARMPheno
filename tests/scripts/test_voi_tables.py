"""WP-D part 1: code-level VOI over Mondo candidate sets (voi_tables.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "analysis" / "cloud"), str(REPO_ROOT / "tests" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import voi_tables as vt  # noqa: E402


def test_eig_is_zero_for_a_code_no_member_differs_on_and_max_for_a_separator():
    prior = np.array([0.5, 0.5])
    P = np.array([[0.9, 0.3, 0.5],      # disease A
                  [0.1, 0.3, 0.5]])     # disease B
    eig, post1 = vt.eig_table(prior, P)
    assert eig[1] == pytest.approx(0.0, abs=1e-12) and eig[2] == pytest.approx(0.0, abs=1e-12)
    assert eig[0] > 0.5                                  # a strong separator
    assert post1[0, 0] == pytest.approx(0.9)             # present -> favours A
    assert eig[0] <= 1.0 + 1e-12                         # never more than H(prior)


def test_presence_prob_keeps_absence_weak_through_the_background():
    p = vt.presence_prob(np.array([0.0, 0.1]), np.array([0.01, 0.01]), L=40, rho=0.3)
    assert 0 < p[0] < p[1] < 1                            # never a perfect negative


def test_candidate_sets_take_label_members_of_matching_readout_nodes():
    rdag = {"C_fit": 4, "pos_sources": [[0], [1], [2], [3], [1, 2], [3]]}
    names = {4: "cardiomyopathy", 5: "connective tissue disorder", 1: "DCM"}
    sets = vt.candidate_sets(rdag, "cardio", names=names)
    assert sets == [(4, "cardiomyopathy", [1, 2])]
    assert vt.candidate_sets(rdag, "connective", names=names) == []   # 1 member only


def test_voi_report_on_a_saved_run_ranks_the_separating_code_first(tmp_path):
    from test_inspect_topics import _make_census_run
    _make_census_run(tmp_path)        # node1 signature on idx 0; node2 on idx 2 and 3
    (tmp_path / "vocab.json").write_text(json.dumps([{str(500 + i): i for i in range(12)}]))
    (tmp_path / "readout_dag.json").write_text(json.dumps({
        "C_fit": 4, "C": 5, "int2cid": {"4": 4994},
        "name_by_id": {"4994": "cardiomyopathy"},
        "pos_sources": [[0], [1], [2], [3], [1, 2]]}))
    rep = vt.build_voi_report(tmp_path, "cardio", vocab_path=str(tmp_path / "vocab.json"),
                              top=3, thin_frac=0)
    assert "## cardiomyopathy — 2 member(s)" in rep
    rows = [l for l in rep.splitlines() if l.startswith("| cid:")]
    assert rows and rows[0].startswith("| cid:500 ") or rows[0].startswith("| cid:502 ")
    assert "cid:511" not in "\n".join(rows[:2])          # the background code is uninformative


def test_voi_cli_dispatch_writes_voi_tables_md(tmp_path, monkeypatch):
    import inspect_topics as it
    from test_inspect_topics import _make_census_run
    _make_census_run(tmp_path)
    (tmp_path / "vocab.json").write_text(json.dumps([{str(500 + i): i for i in range(12)}]))
    (tmp_path / "readout_dag.json").write_text(json.dumps({
        "C_fit": 4, "C": 5, "int2cid": {"4": 4994}, "name_by_id": {"4994": "cardiomyopathy"},
        "pos_sources": [[0], [1], [2], [3], [1, 2]]}))
    monkeypatch.setattr(sys, "argv", ["inspect_topics", str(tmp_path), "--voi", "cardio",
                                      "--vocab-map", str(tmp_path / "vocab.json")])
    it.main()
    assert "code-level VOI" in (tmp_path / "voi_tables.md").read_text()
