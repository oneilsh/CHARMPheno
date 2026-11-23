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

"Affirmed" = not negated: `term_exists` is not false / 'N' / 0 (the AoU off-cycle
release stores 'true' / 'false'; NULL counts as affirmed). `condition_occurrence` holds
standard concepts only, while CLAMP emits some non-standard ones (ICD9CM, CIEL, Nebraska
Lexicon, non-standard SNOMED), so every comparison maps the extracted concept to its
standard 'Maps to' target first; the mapping section counts what maps and what does not.
`term_temporal` and `term_modifiers` print their top values (family history, hypothetical
and historical mentions live there).

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

# `--context`: keyword flags over the lowercased `snippet` (the text around the match).
# Only shares and counts of flagged rows are printed, never snippet text. Patterns are
# deliberately broad (a flag marks a row as possibly not about the patient's own
# current condition); `son` and `fh` can over-match, which errs toward filtering more.
CONTEXT_FLAGS = (
    ("fam", r"\b(family|famhx|fhx|fh|mother|father|mom|dad|sister|brother|sibling|"
            r"aunt|uncle|grandmother|grandfather|grandma|grandpa|cousin|son|daughter|"
            r"maternal|paternal)\b"),
    ("scr", r"(screen|colonoscop|mammogra|vaccin|immuniz|\bshot\b)"),
    ("hedge", r"(rule out|\br/o\b|possible|probable|suspect|concern for|risk of|"
              r"at risk|\?)"),
    ("hist", r"(history of|\bh/o\b|\bhx\b|\bs/p\b|status post|\bprior\b|previous)"),
)
# Concepts the first survey flagged as suspicious (colon carcinoma, influenza,
# 'Disease', infectious disease, traumatic injury, discomfort) and controls (asthma,
# hypertensive disorder, pain).
CONTEXT_CONCEPTS = (4149847, 4266367, 4274025, 432250, 440921, 4090431,
                    317009, 316866, 4329041)


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


# bq's json output sorts keys; tables print label columns first, then counts in this
# order, then anything else alphabetically. Identifier columns print verbatim.
_ORDER = ["source", "nlp_system", "term_exists", "mapping", "section_id", "section",
          "domain_id",
          "vocabulary_id", "standard", "concept_id", "concept_name", "value", "note_year",
          "patients", "n_rows", "notes", "concepts", "rows_concept_0", "rows_no_patient",
          "rows_mean", "rows_p50", "rows_p90", "rows_p99", "concepts_mean",
          "concepts_p50", "concepts_p90", "in_both", "nlp_only", "co_only",
          "nlp_concepts", "co_concepts", "shared_concepts", "nlp_rows",
          "nlp_rows_concept_not_in_co", "patients_in_both", "nlp_pairs", "co_pairs",
          "shared_pairs", "frac_nlp_pairs_in_co", "frac_co_pairs_in_nlp",
          "nlp_patients", "clean_patients", "no_fam_patients", "co_patients",
          "negated", "share_fam", "share_scr", "share_hedge", "share_hist", "share_any",
          "share"]
_VERBATIM = {"concept_id", "note_year", "section_id"}


def _order(keys):
    rank = {c: i for i, c in enumerate(_ORDER)}
    return sorted(keys, key=lambda c: (rank.get(c, len(_ORDER)), c))


def table(rows, cols=None) -> str:
    if not rows:
        return "_(no rows)_\n"
    cols = cols or _order(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(
            ("" if r.get(c) is None else str(r.get(c))) if c in _VERBATIM
            else _cell(r.get(c)) for c in cols) + " |")
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
    tmod = "CAST(n.term_modifiers AS STRING)" if "term_modifiers" in nn else "'NA'"
    ttmp = "CAST(n.term_temporal AS STRING)" if "term_temporal" in nn else "'NA'"
    sec = "n.section_concept_id" if "section_concept_id" in nn else "CAST(NULL AS INT64)"
    lex = "CAST(n.lexical_variant AS STRING)" if "lexical_variant" in nn else "'NA'"
    snip = "LOWER(CAST(n.snippet AS STRING))" if "snippet" in nn else "''"
    flags = ",\n         ".join(f"REGEXP_CONTAINS({snip}, r'{rx}') AS {name}"
                                for name, rx in CONTEXT_FLAGS)
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
    # rcid = the concept as extracted; cid = its standard concept (itself when
    # standard, else its 'Maps to' target, the smallest id when one-to-many; NULL when
    # unmapped): condition_occurrence holds standard concepts only, so the overlap
    # sections compare cid. Negated = term_exists false / N / 0 (the AoU off-cycle
    # release stores 'true' / 'false').
    cte = f"""WITH mp AS (
  SELECT cr.concept_id_1 AS src, MIN(cr.concept_id_2) AS tgt
  FROM `{cdr}.concept_relationship` cr
  JOIN `{cdr}.concept` c2 ON c2.concept_id = cr.concept_id_2 AND c2.standard_concept = 'S'
  WHERE cr.relationship_id = 'Maps to' AND cr.invalid_reason IS NULL
  GROUP BY 1
),
nlp AS (
  SELECT {pid} AS person_id, n.note_id AS note_id, n.note_nlp_concept_id AS rcid,
         CASE WHEN c.standard_concept = 'S' THEN n.note_nlp_concept_id ELSE mp.tgt END AS cid,
         CASE WHEN c.standard_concept = 'S' THEN 'standard'
              WHEN mp.tgt IS NOT NULL THEN 'mapped to standard' ELSE 'unmapped' END AS mapst,
         {te} AS te, LOWER({te}) IN ('false', 'n', '0', 'no', 'f') AS neg,
         {sys_} AS sys, {tmod} AS tmod, {ttmp} AS ttmp, {yr} AS yr,
         {sec} AS sec, {lex} AS lex,
         {flags}
  FROM `{nlp_cdr}.note_nlp` n {join}
  LEFT JOIN `{cdr}.concept` c ON c.concept_id = n.note_nlp_concept_id
  LEFT JOIN mp ON mp.src = n.note_nlp_concept_id
),
nlpc AS (
  SELECT nlp.* FROM nlp JOIN `{cdr}.concept` c ON c.concept_id = nlp.cid
  WHERE c.domain_id = 'Condition'
),
nlpa AS (SELECT * FROM nlpc WHERE NOT IFNULL(neg, FALSE)),
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
       COUNT(DISTINCT person_id) AS patients, COUNT(DISTINCT rcid) AS concepts,
       COUNTIF(rcid = 0) AS rows_concept_0, COUNTIF(person_id IS NULL) AS rows_no_patient
FROM nlp"""))
    q.append(("By nlp_system and term_exists", "", cte + """
SELECT sys AS nlp_system, te AS term_exists, COUNT(*) AS n_rows,
       COUNT(DISTINCT person_id) AS patients, COUNT(DISTINCT rcid) AS concepts
FROM nlp GROUP BY 1, 2 ORDER BY n_rows DESC LIMIT 40"""))
    for col, lab in (("ttmp", "term_temporal"), ("tmod", "term_modifiers")):
        q.append((f"Top 25 {lab} values", "all rows; negated = share of the value's rows "
                  "with term_exists false", cte + f"""
SELECT {col} AS value, COUNT(*) AS n_rows, COUNT(DISTINCT person_id) AS patients,
       AVG(IF(neg, 1, 0)) AS negated
FROM nlp GROUP BY 1 ORDER BY n_rows DESC LIMIT 25"""))
    q.append(("Mapping the extracted concepts to standard",
              "condition_occurrence holds standard concepts; the overlap sections below "
              "use the mapped standard concept", cte + """
SELECT mapst AS mapping, COUNT(*) AS n_rows, COUNT(DISTINCT rcid) AS concepts,
       COUNT(DISTINCT person_id) AS patients
FROM nlp GROUP BY 1 ORDER BY n_rows DESC"""))
    q.append(("Extracted concepts by domain, vocabulary, standard flag", "top 40 groups by rows",
              cte + f"""
SELECT c.domain_id, c.vocabulary_id, IFNULL(c.standard_concept, '-') AS standard,
       COUNT(*) AS n_rows, COUNT(DISTINCT nlp.rcid) AS concepts,
       COUNT(DISTINCT nlp.person_id) AS patients
FROM nlp LEFT JOIN `{cdr}.concept` c ON c.concept_id = nlp.rcid
GROUP BY 1, 2, 3 ORDER BY n_rows DESC LIMIT 40"""))
    q.append(("Totals side by side", "below the first row: NLP rows by their standard "
              "concept, Condition domain; affirmed = not negated",
              cte + f"""
SELECT 'note_nlp (all, as extracted)' AS source, COUNT(*) AS n_rows,
       COUNT(DISTINCT person_id) AS patients, COUNT(DISTINCT rcid) AS concepts FROM nlp
UNION ALL SELECT 'note_nlp Condition', COUNT(*), COUNT(DISTINCT person_id), COUNT(DISTINCT cid) FROM nlpc
UNION ALL SELECT 'note_nlp Condition affirmed', COUNT(*), COUNT(DISTINCT person_id), COUNT(DISTINCT cid) FROM nlpa
UNION ALL SELECT 'condition_occurrence', COUNT(*), COUNT(DISTINCT person_id), COUNT(DISTINCT cid) FROM co
UNION ALL SELECT 'condition_occurrence, NLP patients', COUNT(*), COUNT(DISTINCT person_id),
       COUNT(DISTINCT cid) FROM co WHERE person_id IN (SELECT person_id FROM nlp)
UNION ALL SELECT 'person table', COUNT(*), COUNT(DISTINCT person_id), NULL FROM `{cdr}.person`"""))
    if not has_person:
        return q
    q.append(("Per-patient rates", "over patients with >= 1 row in that source", cte + """
, per AS (
  SELECT 'note_nlp (all, as extracted)' AS source, person_id, COUNT(*) AS n,
    COUNT(DISTINCT rcid) AS k FROM nlp WHERE person_id IS NOT NULL GROUP BY person_id
  UNION ALL SELECT 'note_nlp Condition', person_id, COUNT(*), COUNT(DISTINCT cid)
    FROM nlpc WHERE person_id IS NOT NULL GROUP BY person_id
  UNION ALL SELECT 'note_nlp Condition affirmed', person_id, COUNT(*), COUNT(DISTINCT cid)
    FROM nlpa WHERE person_id IS NOT NULL GROUP BY person_id
  UNION ALL SELECT 'condition_occurrence', person_id, COUNT(*), COUNT(DISTINCT cid)
    FROM co GROUP BY person_id
  UNION ALL SELECT 'condition_occurrence, NLP patients', person_id, COUNT(*),
    COUNT(DISTINCT cid) FROM co WHERE person_id IN (SELECT person_id FROM nlp)
    GROUP BY person_id
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


def context_queries(cdr: str, cte: str, concepts=CONTEXT_CONCEPTS) -> list:
    """`[(title, note, sql)]`: can family history, screening and hedged mentions be
    filtered? Sections, snippet keyword flags overall and per concept, and the top
    concepts once flagged rows are dropped."""
    ids = ", ".join(str(int(c)) for c in concepts)
    shares = ", ".join(f"AVG(IF({n}, 1, 0)) AS share_{n}" for n, _ in CONTEXT_FLAGS)
    anyflag = " OR ".join(n for n, _ in CONTEXT_FLAGS)
    q = []
    q.append(("Note sections", "all rows; top 30 sections by rows; share_* = rows whose "
              "snippet matches the flag", cte + f"""
SELECT CAST(sec AS STRING) AS section_id, IFNULL(c.concept_name, '(none)') AS section,
       COUNT(*) AS n_rows, COUNT(DISTINCT person_id) AS patients,
       AVG(IF(neg, 1, 0)) AS negated, {shares}
FROM nlp LEFT JOIN `{cdr}.concept` c ON c.concept_id = nlp.sec
GROUP BY 1, 2 ORDER BY n_rows DESC LIMIT 30"""))
    q.append(("Snippet flags, affirmed Condition rows", "any = at least one flag", cte + f"""
SELECT COUNT(*) AS n_rows, {shares}, AVG(IF({anyflag}, 1, 0)) AS share_any,
       COUNT(DISTINCT IF(fam, person_id, NULL)) AS patients_with_fam_row
FROM nlpa"""))
    q.append(("Suspicious and control concepts: flags and what survives filtering",
              "affirmed; clean = no flag; co_patients = condition_occurrence", cte + f"""
, pc AS (SELECT cid, COUNT(DISTINCT person_id) AS co_patients FROM co
         WHERE cid IN ({ids}) GROUP BY cid)
SELECT CAST(a.cid AS STRING) AS concept_id, ANY_VALUE(c.concept_name) AS concept_name,
       COUNT(DISTINCT a.person_id) AS nlp_patients,
       COUNT(DISTINCT IF(NOT ({anyflag}), a.person_id, NULL)) AS clean_patients,
       COUNT(DISTINCT IF(NOT fam, a.person_id, NULL)) AS no_fam_patients,
       ANY_VALUE(IFNULL(pc.co_patients, 0)) AS co_patients, COUNT(*) AS n_rows, {shares}
FROM nlpa a LEFT JOIN pc USING (cid) LEFT JOIN `{cdr}.concept` c ON c.concept_id = a.cid
WHERE a.cid IN ({ids}) GROUP BY 1 ORDER BY nlp_patients DESC"""))
    q.append(("Top 3 sections per concept", "affirmed rows", cte + f"""
, s AS (SELECT cid, sec, COUNT(*) AS n FROM nlpa WHERE cid IN ({ids}) GROUP BY 1, 2)
SELECT CAST(s.cid AS STRING) AS concept_id, c1.concept_name,
       IFNULL(c2.concept_name, '(none)') AS section, s.n AS n_rows,
       s.n / SUM(s.n) OVER (PARTITION BY s.cid) AS share
FROM s LEFT JOIN `{cdr}.concept` c1 ON c1.concept_id = s.cid
LEFT JOIN `{cdr}.concept` c2 ON c2.concept_id = s.sec
QUALIFY ROW_NUMBER() OVER (PARTITION BY s.cid ORDER BY s.n DESC) <= 3
ORDER BY concept_id, n_rows DESC"""))
    q.append(("Top 5 matched terms (lexical_variant) per concept",
              f"affirmed rows; terms with >= {EGRESS_FLOOR} rows", cte + f"""
, v AS (SELECT cid, LOWER(lex) AS term, COUNT(*) AS n FROM nlpa WHERE cid IN ({ids})
        GROUP BY 1, 2 HAVING COUNT(*) >= {EGRESS_FLOOR})
SELECT CAST(v.cid AS STRING) AS concept_id, c.concept_name, v.term AS value, v.n AS n_rows
FROM v LEFT JOIN `{cdr}.concept` c ON c.concept_id = v.cid
QUALIFY ROW_NUMBER() OVER (PARTITION BY v.cid ORDER BY v.n DESC) <= 5
ORDER BY concept_id, n_rows DESC"""))
    q.append(("Top 30 concepts by NLP patients after dropping flagged rows",
              "affirmed, no flag; compare the first survey's top 30", cte + f"""
, pn AS (SELECT cid, COUNT(DISTINCT person_id) AS clean_patients FROM nlpa
         WHERE NOT ({anyflag}) GROUP BY cid HAVING COUNT(DISTINCT person_id) >= {EGRESS_FLOOR}),
  pa AS (SELECT cid, COUNT(DISTINCT person_id) AS nlp_patients FROM nlpa GROUP BY cid),
  pc AS (SELECT cid, COUNT(DISTINCT person_id) AS co_patients FROM co GROUP BY cid)
SELECT CAST(pn.cid AS STRING) AS concept_id, c.concept_name, pn.clean_patients,
       pa.nlp_patients, IFNULL(pc.co_patients, 0) AS co_patients
FROM pn JOIN pa USING (cid) LEFT JOIN pc USING (cid)
LEFT JOIN `{cdr}.concept` c ON c.concept_id = pn.cid
ORDER BY clean_patients DESC LIMIT 30"""))
    return q


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def survey(cdr: str, run, emit=print, nlp_cdr: str | None = None,
           context: bool = False) -> None:
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
    qs = (context_queries(cdr, cte) if context
          else queries(cdr, cte, has_person, has_year))
    for title, note, sql in qs:
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
    ap.add_argument("--context", action="store_true",
                    help="run only the context check: sections, snippet flags (family "
                         "history, screening, hedging, past history) and what survives "
                         "filtering")
    a = ap.parse_args(argv)
    if not a.cdr or not a.billing:
        raise SystemExit("need WORKSPACE_CDR and GOOGLE_CLOUD_PROJECT "
                         "(source analysis/cloud/.workspace_env) or --cdr / --billing")
    lines = []

    def emit(s):
        print(s, flush=True)
        lines.append(s)
    survey(a.cdr, make_runner(a.billing, a.bq_bin), emit, nlp_cdr=a.nlp_cdr,
           context=a.context)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(lines) + "\n")
        print(f"[note_nlp_survey] wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
