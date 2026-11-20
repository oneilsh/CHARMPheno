import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "cloud"))
sys.path.insert(0, str(ROOT))

import hpo_profile_marks as hpm  # noqa: E402
import inspect_topics as it  # noqa: E402


def _write_marks(tmp_path):
    """Stage-1 codes -> a stubbed bq mapping -> the marks table, as `main` writes it."""
    codes = tmp_path / "codes.tsv"
    codes.write_text("mondo_id\thp_id\tneg\tfreq\tvocab\tcode\n"
                     "MONDO:0000005\tHP:1\tFalse\t1.0\tSNOMED\t111\n"
                     "MONDO:0000005\tHP:2\tTrue\t1.0\tSNOMED\t222\n"
                     "MONDO:0000005\tHP:3\tFalse\t1.0\tUMLS\tC9\n"
                     "MONDO:0000005\tHP:4\tFalse\t\tSNOMED\t111\n"     # same code
                     "MONDO:0000005\tHP:5\tFalse\t\tSNOMED\t333\n")    # unmapped
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps(
        {"corpus_manifest": {"cdr": "p.d", "billing": "b"}}))
    stub = tmp_path / "bq"                 # a fake `bq`: SNOMED code -> standard ids
    stub.write_text("#!/bin/sh\nprintf 'code,std_id\\n111,201\\n111,202\\n222,203\\n'\n")
    stub.chmod(0o755)
    out = tmp_path / "marks.tsv"
    assert hpm.main(["--run-dir", str(run), "--profile-codes", str(codes),
                     "--out", str(out), "--bq-bin", str(stub)]) == 0
    return codes, out


def test_marks_table_keeps_the_hp_term_and_feeds_the_digest_profile_sets(tmp_path):
    codes, out = _write_marks(tmp_path)
    prof = hpm.read_profile_codes(codes)
    assert prof[:2] == [("MONDO:0000005", "HP:1", "111", False),
                        ("MONDO:0000005", "HP:2", "222", True)]
    lines = out.read_text().splitlines()
    assert lines[0] == "mondo_id\thp_id\tconcept_id\tneg"
    # one row per (node, term, concept); HP:5's code maps to nothing -> no row
    assert lines[1:] == ["MONDO:0000005\tHP:1\t201\t0", "MONDO:0000005\tHP:1\t202\t0",
                         "MONDO:0000005\tHP:2\t203\t1", "MONDO:0000005\tHP:4\t201\t0",
                         "MONDO:0000005\tHP:4\t202\t0"]
    manifest = {"corpus_manifest": {"int2cid": {"0": 1, "1": 5}}}
    sets = it._profile_vocab_sets(str(out), manifest,
                                  {"201": 0, "202": 1, "203": 2})
    assert sets == {1: frozenset({0, 1})}      # the NOT term's concept is dropped


def test_marks_table_with_hp_id_feeds_the_profile_seed_and_the_exporter(tmp_path):
    """The fit's seed (`profile_eta.build_profile_seed`, read as `_profile_seed` reads
    it) and the exporter's reader take the new table: the extra `hp_id` column is
    ignored and a concept reached by two terms counts once."""
    import pandas as pd
    import profile_eta as pe
    import export_dag_gated as drv
    _codes, out = _write_marks(tmp_path)
    df = pd.read_csv(out, sep="\t")
    old = df.drop(columns=["hp_id"]).drop_duplicates()
    kw = dict(eid_by_mondo={"MONDO:0000005": 5, "MONDO:0000009": 9},
              vocab_index_by_concept={201: 0, 203: 2})
    seed_new, st_new = pe.build_profile_seed(df, **kw)
    seed_old, st_old = pe.build_profile_seed(old, **kw)
    assert (seed_new, st_new) == (seed_old, st_old)
    assert st_new["n_rows_concept_not_in_vocab"] == 1      # 202 once, not per term
    assert drv.read_profile_marks(out) == [("MONDO:0000005", 201, False),
                                           ("MONDO:0000005", 202, False),
                                           ("MONDO:0000005", 203, True)]
    assert drv.read_profile_mark_terms(out)[0] == ("MONDO:0000005", "HP:1", 201, False)


def test_mapping_sql_is_one_join_on_the_cdr():
    sql = hpm.mapping_sql("p.d", ["1", "2"])
    assert "`p.d.concept_relationship`" in sql and "'Maps to'" in sql
    assert "IN ('1','2')" in sql and "domain_id = 'Condition'" in sql
