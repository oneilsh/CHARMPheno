"""Tests for the post-fit summary card (numpy-only, synthetic run dir)."""
import json
from pathlib import Path

import numpy as np


def test_layout_matches_daglayout_block_assignment():
    from summarize_fit import layout

    lay = layout({1: [0], 2: [0], 3: [1]}, n_bg=2, tpn=2)
    assert lay["nodes"] == [1, 2, 3]
    assert lay["block"] == {1: [2, 3], 2: [4, 5], 3: [6, 7]}
    assert lay["K"] == 8
    assert lay["anchors"] == [1, 2]  # children of root 0 (node 3 is under 1)


def test_select_sample_nodes_evenly_spread_and_capped():
    from summarize_fit import select_sample_nodes

    assert select_sample_nodes([10, 20, 30], 8) == [10, 20, 30]  # fewer than cap
    assert select_sample_nodes(list(range(10)), 0) == list(range(10))  # 0 = all
    picked = select_sample_nodes(list(range(100)), 5)
    assert len(picked) == 5 and picked[0] == 0 and picked == sorted(picked)


def test_top_tokens_ranks_and_drops_zeros():
    from summarize_fit import top_tokens

    row = np.array([0.0, 5.0, 1.0, 0.0])
    names = {1: "b", 2: "c"}
    assert top_tokens(row, names, top_n=6) == ["b", "c"]  # zeros excluded


def test_upsert_is_idempotent(tmp_path):
    from summarize_fit import upsert_summary_card, CARD_START

    sm = tmp_path / "summary.md"
    sm.write_text("# run stdout\nsome fit logs\n")
    card = f"{CARD_START}\n## Fit card\nbody\n<!-- FIT-CARD END -->\n"
    upsert_summary_card(sm, card)
    upsert_summary_card(sm, card)  # second call must not duplicate
    assert sm.read_text().count(CARD_START) == 1
    assert "some fit logs" in sm.read_text()  # prior content preserved


def _synthetic_run(tmp_path: Path) -> Path:
    run = tmp_path / "0076-run"
    (run / "params").mkdir(parents=True)
    K, V = 8, 3
    lam0 = np.full((K, V), 0.01)
    lam1 = np.full((K, V), 0.01)
    # node 1 block = topics [2,3]; make condition idx 0 ("cond_a") dominate it
    lam0[2, 0] = 9.0
    lam1[2, 1] = 8.0  # drug idx 1 ("drug_b")
    np.save(run / "params" / "lambda_0.npy", lam0)
    np.save(run / "params" / "lambda_1.npy", lam1)
    manifest = {
        "disease": "rare_priority",
        "domains": ["condition", "drug"],
        "n_bg": 2, "tpn": 2,
        "init": "spectral", "seed": 42, "mini_batch_fraction": 0.0,
        "dead_nodes": [], "starved_topics": [7],
        "corpus_stats": {"n_train": 100, "n_test": 25,
                         "by_source_cohort": {"rare_priority": 30, "general": 95}},
        "corpus_manifest": {
            "parent_int": {"1": [0], "2": [0], "3": [1]},
            "int2cid": {"0": 100, "1": 200, "2": 300, "3": 400},
            "name_by_id": {"100": "root", "200": "Alpha disease",
                           "300": "Beta disease", "400": "Alpha subtype"},
            "vocab_condition": {"10": 0, "11": 1, "12": 2},
            "vocab_names_condition": {"10": "cond_a", "11": "cond_b", "12": "cond_c"},
            "vocab_drug": {"20": 0, "21": 1, "22": 2},
            "vocab_names_drug": {"20": "drug_a", "21": "drug_b", "22": "drug_c"},
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "summary.md").write_text("# exp 0076 stdout\nfit logs here\n")
    return run


def test_build_card_end_to_end(tmp_path):
    from summarize_fit import build_card, CARD_START, CARD_END

    run = _synthetic_run(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    lam = {0: np.load(run / "params" / "lambda_0.npy"),
           1: np.load(run / "params" / "lambda_1.npy")}
    card = build_card(manifest, lam, top_n=6, sample=8)

    assert card.startswith(CARD_START) and card.rstrip().endswith(CARD_END)
    assert "K=8 = 2 bg + 3 nodes × 2 tpn · 2 anchors" in card
    assert "condition V=3 · drug V=3" in card
    assert "Alpha disease" in card and "Beta disease" in card
    # node 1 (Alpha disease) top tokens reflect the planted mass
    assert "cond_a" in card and "drug_b" in card
    # node 3's block has engine topic 7 which is starved
    assert "starved" in card


def test_main_writes_card_and_upserts_summary(tmp_path):
    import summarize_fit

    run = _synthetic_run(tmp_path)
    rc = summarize_fit.main(["--run-dir", str(run)])
    assert rc == 0
    assert (run / "fit_card.md").exists()
    summary = (run / "summary.md").read_text()
    assert summary.count(summarize_fit.CARD_START) == 1
    assert "fit logs here" in summary  # original stdout preserved
    # idempotent re-run
    summarize_fit.main(["--run-dir", str(run)])
    assert (run / "summary.md").read_text().count(summarize_fit.CARD_START) == 1
