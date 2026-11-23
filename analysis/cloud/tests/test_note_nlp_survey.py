"""note_nlp_survey: the egress floor in the report, schema adaptation, and SQL that
parses as BigQuery (when sqlglot is installed)."""
import pytest

from analysis.cloud import note_nlp_survey as s

CDR = "proj.ds"
NOTE_JOIN = {"note_nlp": {"note_id", "note_nlp_concept_id", "term_exists", "nlp_system"},
             "note": {"note_id", "person_id", "note_date"}}


def test_cell_floor():
    assert s._cell("0") == "0"
    assert s._cell("1") == "<20" and s._cell("19") == "<20"
    assert s._cell("20") == "20" and s._cell("123456") == "123,456"
    assert s._cell("3.14159") == "3.14" and s._cell("Asthma") == "Asthma"
    assert s._cell(None) == ""


@pytest.mark.parametrize("cols,person,year,n", [
    (NOTE_JOIN, True, True, 15),
    ({"note_nlp": {"note_id", "note_nlp_concept_id", "person_id"}}, True, False, 14),
    ({"note_nlp": {"note_id", "note_nlp_concept_id"}}, False, False, 7),
])
def test_schema_adaptation(cols, person, year, n):
    cte, hp, hy = s.nlp_cte(CDR, cols)
    assert (hp, hy) == (person, year)
    assert len(s.queries(CDR, cte, hp, hy)) == n
    if not person:
        assert "CAST(NULL AS INT64) AS person_id" in cte


def test_sql_parses():
    sqlglot = pytest.importorskip("sqlglot")
    cte, hp, hy = s.nlp_cte(CDR, NOTE_JOIN)
    for _, _, q in s.queries(CDR, cte, hp, hy):
        sqlglot.parse_one(q, read="bigquery")


def test_survey_with_fake_bq():
    def run(sql):
        if "INFORMATION_SCHEMA" in sql:
            return [{"table_name": t, "column_name": c}
                    for t, cs in NOTE_JOIN.items() for c in cs]
        return [{"n_rows": "7", "patients": "500"}]
    out = []
    s.survey(CDR, run, out.append)
    text = "\n".join(out)
    assert "| 500 | <20 |" in text and "## (patient, concept) pair overlap" in text


def test_offcycle_nlp_dataset():
    def run(sql):
        if "proj.nlp`.INFORMATION_SCHEMA" in sql:
            return [{"table_name": "note_nlp", "column_name": c}
                    for c in ("note_id", "note_nlp_concept_id", "person_id")]
        if "INFORMATION_SCHEMA" in sql:
            return [{"table_name": "condition_occurrence", "column_name": "person_id"},
                    {"table_name": "note", "column_name": "note_id"}]
        assert "`proj.nlp.note_nlp`" in sql and "`proj.ds.condition_occurrence`" in sql
        return [{"n_rows": "1000"}]
    out = []
    s.survey(CDR, run, out.append, nlp_cdr="proj.nlp")
    text = "\n".join(out)
    assert "`note_nlp` (`proj.nlp`)" in text and "## Per-patient rates" in text


def test_column_order_and_verbatim_ids():
    t = s.table([{"patients": "500", "concept_id": "4329041", "source": "x",
                  "note_year": "2019"}])
    assert t.splitlines()[0] == "| source | concept_id | note_year | patients |"
    assert "| x | 4329041 | 2019 | 500 |" in t
