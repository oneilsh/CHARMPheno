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


def test_fit_profile_seed_reads_a_marks_table_with_hp_id(tmp_path):
    """`dag_gated_fit._profile_seed` on the marks table as `hpo_profile_marks.py` writes
    it now (mondo_id, hp_id, concept_id, neg; one row per term and concept): the same
    seed and stats as the old three-column table, a concept reached by two terms once."""
    import dag_gated_fit as fit
    rows = [("MONDO:1", "HP:1", 201, 0), ("MONDO:1", "HP:2", 201, 0),
            ("MONDO:1", "HP:3", 202, 0), ("MONDO:1", "HP:4", 999, 0),
            ("MONDO:1", "HP:5", 999, 0), ("MONDO:2", "HP:1", 201, 0),
            ("MONDO:2", "HP:9", 203, 0), ("MONDO:3", "HP:1", 201, 0)]
    new, old = tmp_path / "new.tsv", tmp_path / "old.tsv"
    new.write_text("mondo_id\thp_id\tconcept_id\tneg\n"
                   + "".join(f"{m}\t{h}\t{c}\t{n}\n" for m, h, c, n in rows))
    old.write_text("mondo_id\tconcept_id\tneg\n"
                   + "".join(f"{m}\t{c}\t{n}\n" for m, c, n in sorted(
                       {(m, c, n) for m, _h, c, n in rows})))
    dag, vocab = {"MONDO:1": [], "MONDO:2": []}, {201: 0, 202: 1, 203: 2}
    seed, st = fit._profile_seed(str(new), dag, vocab)
    assert (seed, st) == fit._profile_seed(str(old), dag, vocab)
    assert seed == {"MONDO:1": {("features_0", 202): float(np.log(2))},
                    "MONDO:2": {("features_0", 203): float(np.log(2))}}
    assert st["n_rows_concept_not_in_vocab"] == 1
