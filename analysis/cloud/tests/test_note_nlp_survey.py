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
    (NOTE_JOIN, True, True, 12),
    ({"note_nlp": {"note_id", "note_nlp_concept_id", "person_id"}}, True, False, 11),
    ({"note_nlp": {"note_id", "note_nlp_concept_id"}}, False, False, 4),
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
    assert "| <20 | 500 |" in text and "## (patient, concept) pair overlap" in text
