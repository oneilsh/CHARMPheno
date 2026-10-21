"""Leaves-first profile seed (exp 0157/0158): profile_eta.build_profile_seed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "cloud"))
import profile_eta as pe  # noqa: E402


def test_profile_seed_idf_neg_and_vocab_rules():
    df = pd.DataFrame({
        "mondo_id": ["MONDO:1", "MONDO:1", "MONDO:1", "MONDO:2", "MONDO:2", "MONDO:3",
                     "MONDO:2"],
        "concept_id": [100, 101, 102, 100, 103, 100, 103],
        "neg": [0, 0, 0, 0, 0, 0, 1]})
    vocab = {100: 0, 101: 1, 103: 3}                  # 102 not in vocab
    seed, st = pe.build_profile_seed(df, eid_by_mondo={"MONDO:1": 5, "MONDO:2": 6,
                                                        "MONDO:3": 7},
                                     vocab_index_by_concept=vocab)
    # 100 is in every profiled node -> weight 0 -> dropped; 103 contested for node 6;
    # node 7's only concept is the shared one -> not seeded
    assert seed == {5: {1: float(np.log(3 / 1))}}
    assert st["n_nodes_with_vocab_profile"] == 3 and st["n_nodes_seeded"] == 1
    assert st["n_rows_concept_not_in_vocab"] == 1
