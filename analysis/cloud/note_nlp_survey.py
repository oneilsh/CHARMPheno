"""One-off survey: what `note_nlp` holds, next to `condition_occurrence`.

Read-only `bq query` jobs against the workspace CDR (no Spark, no YARN). It reports:

* the shape of `note_nlp`: rows, notes, patients, distinct concepts, concept 0, and the
  split by `nlp_system` and `term_exists` (negation);
* what the extracted concepts are: by domain, vocabulary and standard flag;
* per-patient rates (rows and distinct concepts per patient: mean, median, p90, p99)
  for NLP all, NLP Condition-domain, NLP Condition-domain affirmed, and
  `condition_occurrence`;
* patient overlap (NLP condition patients vs condition_occurrence patients);
* concept diversity and overlap (distinct Condition concepts in each, shared, NLP rows
  whose concept never appears in condition_occurrence);
* (patient, concept) pair overlap, exact concept match, both directions;
* the top Condition concepts by NLP patients, overall and among those absent from
  condition_occurrence, with both sources' patient counts;
* NLP rows and patients by note year (when `note` is joinable).

`note_nlp` in OMOP has no `person_id`; patients come through `note` (`note_id`). The
script reads the columns first (INFORMATION_SCHEMA) and adapts: it uses
`note_nlp.person_id` if the CDR has one, the `note` join otherwise, and skips the
patient-level sections when neither is available. Missing `term_exists` /
`nlp_system` columns read as 'NA'.

"Affirmed" = `term_exists` is not 'N' (NULL counts as affirmed; the split shows how many
are NULL).

Egress: the output is workspace-internal. Every count from 1 to 19 prints as "<20"
(`EGRESS_FLOOR`), and the concept tables list only concepts with >= 20 NLP patients,
so the report is safe to read and to quote pooled figures from; per-concept rows still
stay in the workspace.

    python analysis/cloud/note_nlp_survey.py --out <file.md> \
        [--nlp-cdr <project>.C_V9_R6_offcycle_nlp]
      (reads WORKSPACE_CDR and GOOGLE_CLOUD_PROJECT from the environment;
       `. analysis/cloud/.workspace_env` first)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EGRESS_FLOOR = 20


# --------------------------------------------------------------------------- #
# bq                                                                           #
# --------------------------------------------------------------------------- #
def make_runner(billing: str, bq_bin: str = "bq"):
    """A function `sql -> list[dict]` running one query with the bq CLI."""
    def run(sql: str):
        cmd = [bq_bin, "--project_id", billing, "--format", "json", "query",
               "--use_legacy_sql=false", "--max_rows", "100000", "--quiet", sql]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError((res.stderr or res.stdout).strip()[-2000:])
        out = res.stdout.strip()
        return json.loads(out) if out else []
    return run


# --------------------------------------------------------------------------- #
# formatting                                                                   #
# --------------------------------------------------------------------------- #
def _cell(v):
    """bq json gives every value as a string; integers 1..19 print as '<20'."""
    if v is None:
        return ""
    s = str(v)
    try:
        i = int(s)
    except ValueError:
        try:
            f = float(s)
        except ValueError:
            return s.replace("|", "/")
        return f"{f:,.3g}" if abs(f) < 1000 else f"{f:,.0f}"
    if 0 < i < EGRESS_FLOOR:
        return f"<{EGRESS_FLOOR}"
    return f"{i:,}"


def table(rows, cols=None) -> str:
    if not rows:
        return "_(no rows)_\n"
    cols = cols or list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(_cell(r.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# SQL                                                                          #
# --------------------------------------------------------------------------- #
def schema_sql(cdr: str, all_tables: bool = False) -> str:
    where = "" if all_tables else (
        " WHERE table_name IN ('note_nlp', 'note', 'person', 'condition_occurrence')")
    return (f"SELECT table_name, column_name FROM `{cdr}`.INFORMATION_SCHEMA.COLUMNS"
            + where)


def nlp_cte(cdr: str, cols: dict, nlp_cdr: str | None = None,
            note_cdr: str | None = None) -> tuple[str, bool, bool]:
    """`WITH nlp AS (...)` with person_id, note_id, cid, te, sys, yr; and whether the
    patient id and the note year are available. `note_nlp` is read from `nlp_cdr` and
    `note` from `note_cdr` (both default to `cdr`: All of Us ships NLP output in an
    off-cycle dataset, e.g. `C_V9_R6_offcycle_nlp`); concept, person and
    condition_occurrence always come from `cdr`."""
    nlp_cdr, note_cdr = nlp_cdr or cdr, note_cdr or cdr
    nn = cols.get("note_nlp", set())
    note = cols.get("note", set())
    te = "CAST(n.term_exists AS STRING)" if "term_exists" in nn else "'NA'"
    sys_ = "CAST(n.nlp_system AS STRING)" if "nlp_system" in nn else "'NA'"
    join, has_person, has_year = "", True, False
    if "person_id" in nn:
        pid = "n.person_id"
        if {"note_id", "note_date"} <= note:
            join = f"LEFT JOIN `{note_cdr}.note` t ON t.note_id = n.note_id"
            yr, has_year = "EXTRACT(YEAR FROM t.note_date)", True
        else:
            yr = "CAST(NULL AS INT64)"
    elif {"note_id", "person_id"} <= note:
        pid = "t.person_id"
        join = f"LEFT JOIN `{note_cdr}.note` t ON t.note_id = n.note_id"
        if "note_date" in note:
            yr, has_year = "EXTRACT(YEAR FROM t.note_date)", True
        else:
            yr = "CAST(NULL AS INT64)"
    else:
        pid, has_person, yr = "CAST(NULL AS INT64)", False, "CAST(NULL AS INT64)"
    cte = f"""WITH nlp AS (
  SELECT {pid} AS person_id, n.note_id AS note_id, n.note_nlp_concept_id AS cid,
         {te} AS te, {sys_} AS sys, {yr} AS yr
  FROM `{nlp_cdr}.note_nlp` n {join}
),
nlpc AS (
  SELECT nlp.* FROM nlp JOIN `{cdr}.concept` c ON c.concept_id = nlp.cid
  WHERE c.domain_id = 'Condition'
),
nlpa AS (SELECT * FROM nlpc WHERE te IS NULL OR te != 'N'),
co AS (
  SELECT person_id, condition_concept_id AS cid FROM `{cdr}.condition_occurrence`
  WHERE condition_concept_id != 0
)
"""
    return cte, has_person, has_year


def queries(cdr: str, cte: str, has_person: bool, has_year: bool) -> list:
    """`[(title, note, sql)]` in report order."""
    q = []
    q.append(("note_nlp overview", "", cte + """
SELECT COUNT(*) AS n_rows, COUNT(DISTINCT note_id) AS notes,
       COUNT(DISTINCT person_id) AS patients, COUNT(DISTINCT cid) AS concepts,
       COUNTIF(cid = 0) AS rows_concept_0, COUNTIF(person_id IS NULL) AS rows_no_patient
FROM nlp"""))
    q.append(("By nlp_system and term_exists", "", cte + """
SELECT sys AS nlp_system, te AS term_exists, COUNT(*) AS n_rows,
       COUNT(DISTINCT person_id) AS patients, COUNT(DISTINCT cid) AS concepts
FROM nlp GROUP BY 1, 2 ORDER BY n_rows DESC LIMIT 40"""))
    q.append(("Extracted concepts by domain, vocabulary, standard flag", "top 40 groups by rows",
              cte + f"""
SELECT c.domain_id, c.vocabulary_id, IFNULL(c.standard_concept, '-') AS standard,
       COUNT(*) AS n_rows, COUNT(DISTINCT nlp.cid) AS concepts,
       COUNT(DISTINCT nlp.person_id) AS patients
FROM nlp LEFT JOIN `{cdr}.concept` c ON c.concept_id = nlp.cid
GROUP BY 1, 2, 3 ORDER BY n_rows DESC LIMIT 40"""))
    q.append(("Totals side by side", "Condition domain only for NLP rows below this line",
              cte + f"""
SELECT 'note_nlp (all)' AS source, COUNT(*) AS n_rows, COUNT(DISTINCT person_id) AS patients,
       COUNT(DISTINCT cid) AS concepts FROM nlp
UNION ALL SELECT 'note_nlp Condition', COUNT(*), COUNT(DISTINCT person_id), COUNT(DISTINCT cid) FROM nlpc
UNION ALL SELECT 'note_nlp Condition affirmed', COUNT(*), COUNT(DISTINCT person_id), COUNT(DISTINCT cid) FROM nlpa
UNION ALL SELECT 'condition_occurrence', COUNT(*), COUNT(DISTINCT person_id), COUNT(DISTINCT cid) FROM co
UNION ALL SELECT 'person table', COUNT(*), COUNT(DISTINCT person_id), NULL FROM `{cdr}.person`"""))
    if not has_person:
        return q
    q.append(("Per-patient rates", "over patients with >= 1 row in that source", cte + """
, per AS (
  SELECT 'note_nlp (all)' AS source, person_id, COUNT(*) AS n, COUNT(DISTINCT cid) AS k
    FROM nlp WHERE person_id IS NOT NULL GROUP BY person_id
  UNION ALL SELECT 'note_nlp Condition', person_id, COUNT(*), COUNT(DISTINCT cid)
    FROM nlpc WHERE person_id IS NOT NULL GROUP BY person_id
  UNION ALL SELECT 'note_nlp Condition affirmed', person_id, COUNT(*), COUNT(DISTINCT cid)
    FROM nlpa WHERE person_id IS NOT NULL GROUP BY person_id
  UNION ALL SELECT 'condition_occurrence', person_id, COUNT(*), COUNT(DISTINCT cid)
    FROM co GROUP BY person_id
)
SELECT source, COUNT(*) AS patients,
       AVG(n) AS rows_mean, APPROX_QUANTILES(n, 100)[OFFSET(50)] AS rows_p50,
       APPROX_QUANTILES(n, 100)[OFFSET(90)] AS rows_p90,
       APPROX_QUANTILES(n, 100)[OFFSET(99)] AS rows_p99,
       AVG(k) AS concepts_mean, APPROX_QUANTILES(k, 100)[OFFSET(50)] AS concepts_p50,
       APPROX_QUANTILES(k, 100)[OFFSET(90)] AS concepts_p90
FROM per GROUP BY source ORDER BY source"""))
    q.append(("Patient overlap", "NLP = Condition-domain affirmed", cte + """
, a AS (SELECT DISTINCT person_id FROM nlpa WHERE person_id IS NOT NULL),
  b AS (SELECT DISTINCT person_id FROM co)
SELECT COUNTIF(a.person_id IS NOT NULL AND b.person_id IS NOT NULL) AS in_both,
       COUNTIF(b.person_id IS NULL) AS nlp_only,
       COUNTIF(a.person_id IS NULL) AS co_only
FROM a FULL OUTER JOIN b ON a.person_id = b.person_id"""))
    q.append(("Condition concept overlap",
              "distinct concepts; nlp_rows_concept_not_in_co = NLP affirmed rows whose "
              "concept never appears in condition_occurrence", cte + """
, a AS (SELECT DISTINCT cid FROM nlpa), b AS (SELECT DISTINCT cid FROM co)
SELECT (SELECT COUNT(*) FROM a) AS nlp_concepts, (SELECT COUNT(*) FROM b) AS co_concepts,
       (SELECT COUNT(*) FROM a JOIN b USING (cid)) AS shared_concepts,
       (SELECT COUNT(*) FROM nlpa WHERE cid NOT IN (SELECT cid FROM b))
         AS nlp_rows_concept_not_in_co,
       (SELECT COUNT(*) FROM nlpa) AS nlp_rows"""))
    q.append(("(patient, concept) pair overlap",
              "exact concept match, among patients present in both sources; NLP = "
              "Condition-domain affirmed", cte + """
, bothp AS (SELECT DISTINCT person_id FROM nlpa WHERE person_id IS NOT NULL
           INTERSECT DISTINCT SELECT DISTINCT person_id FROM co),
  a AS (SELECT DISTINCT person_id, cid FROM nlpa JOIN bothp USING (person_id)),
  b AS (SELECT DISTINCT person_id, cid FROM co JOIN bothp USING (person_id))
SELECT (SELECT COUNT(*) FROM bothp) AS patients_in_both,
       (SELECT COUNT(*) FROM a) AS nlp_pairs, (SELECT COUNT(*) FROM b) AS co_pairs,
       (SELECT COUNT(*) FROM a JOIN b USING (person_id, cid)) AS shared_pairs,
       SAFE_DIVIDE((SELECT COUNT(*) FROM a JOIN b USING (person_id, cid)),
                   (SELECT COUNT(*) FROM a)) AS frac_nlp_pairs_in_co,
       SAFE_DIVIDE((SELECT COUNT(*) FROM a JOIN b USING (person_id, cid)),
                   (SELECT COUNT(*) FROM b)) AS frac_co_pairs_in_nlp"""))
    top = f"""
, pn AS (SELECT cid, COUNT(DISTINCT person_id) AS nlp_patients, COUNT(*) AS nlp_rows
         FROM nlpa GROUP BY cid HAVING COUNT(DISTINCT person_id) >= {EGRESS_FLOOR}),
  pc AS (SELECT cid, COUNT(DISTINCT person_id) AS co_patients FROM co GROUP BY cid)
SELECT pn.cid AS concept_id, c.concept_name, pn.nlp_patients, pn.nlp_rows,
       IFNULL(pc.co_patients, 0) AS co_patients
FROM pn LEFT JOIN pc USING (cid) LEFT JOIN `{cdr}.concept` c ON c.concept_id = pn.cid
"""
    q.append(("Top 30 Condition concepts by NLP patients", "affirmed", cte + top +
              "ORDER BY nlp_patients DESC LIMIT 30"))
    q.append(("Top 30 NLP Condition concepts absent from condition_occurrence",
              "affirmed; concepts with no condition_occurrence row at all", cte + top +
              "WHERE pc.cid IS NULL ORDER BY nlp_patients DESC LIMIT 30"))
    q.append(("NLP gain per concept, distribution",
              "per shared Condition concept with >= 20 NLP patients: NLP patients / "
              "condition_occurrence patients", cte + top.replace(
                  "SELECT pn.cid AS concept_id",
                  ", r AS (SELECT pn.cid AS concept_id") + f"""
WHERE pc.cid IS NOT NULL)
SELECT COUNT(*) AS concepts, APPROX_QUANTILES(SAFE_DIVIDE(nlp_patients, co_patients), 100)[OFFSET(10)] AS ratio_p10,
       APPROX_QUANTILES(SAFE_DIVIDE(nlp_patients, co_patients), 100)[OFFSET(50)] AS ratio_p50,
       APPROX_QUANTILES(SAFE_DIVIDE(nlp_patients, co_patients), 100)[OFFSET(90)] AS ratio_p90
FROM r"""))
    if has_year:
        q.append(("NLP rows by note year", "all NLP rows", cte + """
SELECT yr AS note_year, COUNT(*) AS n_rows, COUNT(DISTINCT person_id) AS patients
FROM nlp GROUP BY 1 ORDER BY 1"""))
    return q


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def survey(cdr: str, run, emit=print, nlp_cdr: str | None = None) -> None:
    nlp_cdr = nlp_cdr or cdr
    emit(f"# note_nlp vs condition_occurrence survey\n\nCDR `{cdr}` · NLP dataset "
         f"`{nlp_cdr}` · "
         f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} · counts 1-19 shown as "
         f"<{EGRESS_FLOOR} · workspace-internal\n")
    cols: dict = {}
    for r in run(schema_sql(cdr)):
        cols.setdefault(r["table_name"], set()).add(r["column_name"])
    if not cols:
        emit(f"INFORMATION_SCHEMA of `{cdr}` returned no rows; stopping.")
        return
    where = {t: cdr for t in cols}
    if nlp_cdr != cdr:
        ncols: dict = {}
        for r in run(schema_sql(nlp_cdr, all_tables=True)):
            ncols.setdefault(r["table_name"], set()).add(r["column_name"])
        emit(f"## Tables in `{nlp_cdr}`\n")
        emit(", ".join(f"`{t}` ({len(c)} columns)" for t, c in sorted(ncols.items()))
             or "**none visible**")
        emit("")
        for t in ("note_nlp", "note"):
            if t in ncols:
                cols[t], where[t] = ncols[t], nlp_cdr
    emit("## Columns\n")
    for t in ("note_nlp", "note"):
        emit(f"- `{t}`" + (f" (`{where[t]}`): " + ", ".join(sorted(cols[t]))
                           if t in cols else ": **not present**"))
    emit("")
    if "note_nlp" not in cols:
        emit(f"No `note_nlp` table in `{nlp_cdr}`; stopping.")
        return
    cte, has_person, has_year = nlp_cte(cdr, cols, where["note_nlp"], where.get("note"))
    if not has_person:
        emit("No patient id for note_nlp rows (no `note_nlp.person_id`, no joinable "
             "`note`): patient-level sections skipped.\n")
    for title, note, sql in queries(cdr, cte, has_person, has_year):
        t0 = time.time()
        emit(f"## {title}\n")
        if note:
            emit(f"_{note}_\n")
        try:
            emit(table(run(sql)))
        except RuntimeError as e:
            emit(f"**query failed:** `{str(e).splitlines()[-1] if str(e) else e}`\n")
        print(f"[note_nlp_survey] {title}: {time.time() - t0:.0f}s", file=sys.stderr,
              flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdr", default=os.environ.get("WORKSPACE_CDR"))
    ap.add_argument("--nlp-cdr", default=None,
                    help="<project>.<dataset> holding note_nlp (and note, if there) "
                         "when it is not the CDR, e.g. <project>.C_V9_R6_offcycle_nlp")
    ap.add_argument("--billing", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    ap.add_argument("--out", default=None, help="also write the report here (markdown)")
    ap.add_argument("--bq-bin", default="bq")
    a = ap.parse_args(argv)
    if not a.cdr or not a.billing:
        raise SystemExit("need WORKSPACE_CDR and GOOGLE_CLOUD_PROJECT "
                         "(source analysis/cloud/.workspace_env) or --cdr / --billing")
    lines = []

    def emit(s):
        print(s, flush=True)
        lines.append(s)
    survey(a.cdr, make_runner(a.billing, a.bq_bin), emit, nlp_cdr=a.nlp_cdr)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(lines) + "\n")
        print(f"[note_nlp_survey] wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
