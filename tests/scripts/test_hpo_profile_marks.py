import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "cloud"))
sys.path.insert(0, str(ROOT))

import hpo_profile_marks as hpm  # noqa: E402
import inspect_topics as it  # noqa: E402


def test_marks_table_feeds_the_digest_profile_sets(tmp_path):
    codes = tmp_path / "codes.tsv"
    codes.write_text("mondo_id\thp_id\tneg\tfreq\tvocab\tcode\n"
                     "MONDO:0000005\tHP:1\tFalse\t1.0\tSNOMED\t111\n"
                     "MONDO:0000005\tHP:2\tTrue\t1.0\tSNOMED\t222\n"
                     "MONDO:0000005\tHP:3\tFalse\t1.0\tUMLS\tC9\n")
    prof = hpm.read_profile_codes(codes)
    assert prof == [("MONDO:0000005", "111", False), ("MONDO:0000005", "222", True)]
    rows = hpm.mark_rows(prof, {"111": {201, 202}, "222": {203}})
    out = tmp_path / "marks.tsv"
    out.write_text("mondo_id\tconcept_id\tneg\n"
                   + "".join(f"{m}\t{c}\t{int(n)}\n" for m, c, n in rows))
    manifest = {"corpus_manifest": {"int2cid": {"0": 1, "1": 5}}}
    sets = it._profile_vocab_sets(str(out), manifest,
                                  {"201": 0, "202": 1, "203": 2})
    assert sets == {1: frozenset({0, 1})}      # the NOT term's concept is dropped


def test_mapping_sql_is_one_join_on_the_cdr():
    sql = hpm.mapping_sql("p.d", ["1", "2"])
    assert "`p.d.concept_relationship`" in sql and "'Maps to'" in sql
    assert "IN ('1','2')" in sql and "domain_id = 'Condition'" in sql
