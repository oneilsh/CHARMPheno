"""The prenatal-subtree exclusion on emitted profile codes (exp 0124 → 0125):
terms under HP:0001197 must not reach the guide; maternal terms must."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "cloud"))
import hpoa_profile_survey as hs  # noqa: E402


def _toy_parents():
    # HP:0000001 root; HP:0001197 prenatal/birth; HP:0001558 decreased fetal
    # movement (under 1197 via HP:0001197->HP:0001558); HP:0001789 hydrops
    # fetalis under an intermediate; HP:0100602 pre-eclampsia elsewhere.
    return {
        "HP:0001197": ["HP:0000001"],
        "HP:0001558": ["HP:0001197"],
        "HP:0034241": ["HP:0001197"],          # intermediate
        "HP:0001789": ["HP:0034241"],
        "HP:0100602": ["HP:0000118"],          # pre-eclampsia, NOT prenatal
        "HP:0000118": ["HP:0000001"],
        "HP:0001635": ["HP:0000118"],          # congestive heart failure
    }


def test_subtree_includes_root_and_all_descendants_only():
    sub = hs.hpo_subtree(_toy_parents(), "HP:0001197")
    assert sub == {"HP:0001197", "HP:0001558", "HP:0034241", "HP:0001789"}


def test_exclusion_drops_fetal_terms_keeps_maternal_and_counts():
    prof = pd.DataFrame({
        "mondo_id": ["MONDO:0005021", "MONDO:0005021", "MONDO:0005021",
                     "MONDO:0005081", "MONDO:0005081"],
        "hpo_id":   ["HP:0001558", "HP:0001789", "HP:0001635",
                     "HP:0100602", "HP:0001558"],
        "neg":      [False] * 5,
        "freq":     [0.5] * 5,
        "inherited": [True, True, False, False, True],
    })
    kept, st = hs.exclude_hpo_subtree(prof, _toy_parents())
    assert set(kept["hpo_id"]) == {"HP:0001635", "HP:0100602"}
    assert st["n_rows_dropped"] == 3 and st["n_terms_dropped"] == 2
    assert st["n_nodes_touched"] == 2 and st["n_subtree_terms"] == 4
    # columns preserved (inherited rides through to profile_code_rows)
    assert list(kept.columns) == list(prof.columns)


def test_exclusion_is_identity_when_nothing_is_under_the_root():
    prof = pd.DataFrame({"mondo_id": ["M"], "hpo_id": ["HP:0001635"],
                         "neg": [False], "freq": [1.0]})
    kept, st = hs.exclude_hpo_subtree(prof, _toy_parents())
    assert kept.equals(prof) and st["n_rows_dropped"] == 0


def test_name_spectral_anchors_renders_names_flags_and_grep():
    import name_spectral_anchors as nsa
    doc = {"vocab_domain": 0, "nodes": {
        "147": {"cid": 5021, "name": "dilated cardiomyopathy", "anchors": [3, 9, 77],
                "from_profile": [True, False, True], "n_preferred": 2, "n_eligible": 2},
        "19": {"cid": 4994, "name": "cardiomyopathy", "anchors": [1],
               "from_profile": [False], "n_preferred": 0, "n_eligible": None}}}
    vm0 = {"77619": 3, "433736": 9, "4150384": 1}          # idx 77 is NOT in the map
    names = {77619: "Reduced fetal movement", 433736: "Obesity", 4150384: "Urine cytology abnormal"}
    lines = nsa.name_anchors(doc, vm0, names)
    assert lines == ["cardiomyopathy | Urine cytology abnormal",
                     "dilated cardiomyopathy | Reduced fetal movement* · Obesity · idx:77*"]
    assert nsa.name_anchors(doc, vm0, names, grep="^dilated") == lines[1:]
