"""Parse the Monarch dismech priority rare-disease list into an anchor seed.

Stage 1 of the expanded-SNOMED anchor-selection pipeline (see
docs/superpowers/specs/2026-07-31-expanded-snomed-anchor-selection-design.md).

Input is the markdown of dismech issue #1079 ("Priority diseases to curate"): a
few hundred rare diseases grouped under ``## <Category>`` / ``### <Category>``
headers, each a task-list line carrying a MONDO id and a curation checkbox, e.g.

    ## Neuroimmune (12 diseases, 10 curated in dismech)
    - [x] [myasthenia gravis](https://monarchinitiative.org/MONDO:0009688)
    - [ ] [neuromyelitis optica](https://monarchinitiative.org/MONDO:0019100)

A minority of lines link to a dismech issue instead of monarchinitiative.org and
carry the MONDO id in a trailing parenthesis, e.g.

    - [x] [episodic ataxia type 2](https://github.com/.../issues/1678) (MONDO:0007163)

Both forms are handled: the MONDO id is read from anywhere on the line. A disease
may appear under more than one category (the issue notes this); one seed row is
emitted per (mondo_id, category) so downstream de-duplication of the *anchor* set
is an explicit, later choice rather than silently done here.

Pure stdlib, no Spark: this only turns markdown into a clean seed table. Mapping
MONDO->OMOP, subtree counts, and neighborhood assembly are later on-cluster
stages.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

# A markdown ATX header: one-or-more '#', then the title. We keep the leaf title
# text (before any trailing "(N diseases, ...)" annotation) as the category.
_HEADER_RE = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$")
# A task-list item: "- [x] ..." or "- [ ] ..." (x may be upper/lower case).
_TASK_RE = re.compile(r"^\s*[-*]\s+\[(?P<mark>[ xX])\]\s+(?P<rest>.*)$")
# A MONDO curie anywhere in a line.
_MONDO_RE = re.compile(r"MONDO:\d{5,7}")
# The visible label is the first markdown link text "[label](...)" on the line.
_LABEL_RE = re.compile(r"\[(?P<label>[^\]]+)\]\(")
# Strip a trailing "(123 diseases, 45 curated ...)" style count annotation from
# a category header so the stored category is just the name.
_HEADER_COUNT_RE = re.compile(r"\s*\(\s*\d[\d,]*\s+disease.*\)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class SeedRow:
    """One (disease, category) membership parsed from the priority list."""

    mondo_id: str
    label: str
    category: str
    curated: bool


def _clean_category(title: str) -> str:
    return _HEADER_COUNT_RE.sub("", title).strip()


def parse_priority_seed(markdown: str) -> list[SeedRow]:
    """Parse issue-#1079 markdown into seed rows, in document order.

    The active category is the most recent ATX header seen. Task lines with no
    MONDO id, and any line before the first header, are skipped — so the intro /
    methodology prose (which lives under its own headers) simply yields no rows.
    """
    rows: list[SeedRow] = []
    category: str | None = None
    for line in markdown.splitlines():
        task = _TASK_RE.match(line)
        if task is None:
            header = _HEADER_RE.match(line)
            if header is not None:
                category = _clean_category(header.group("title"))
            continue
        rest = task.group("rest")
        mondo = _MONDO_RE.search(rest)
        if mondo is None:
            continue
        label_m = _LABEL_RE.search(rest)
        label = label_m.group("label").strip() if label_m else ""
        rows.append(
            SeedRow(
                mondo_id=mondo.group(0),
                label=label,
                category=category or "",
                curated=task.group("mark").lower() == "x",
            )
        )
    return rows


def unique_diseases(rows: list[SeedRow]) -> dict[str, list[str]]:
    """Map each MONDO id to the sorted list of categories it appears under.

    The anchor set is over distinct diseases; this exposes multi-category
    membership so a later stage can decide how to place a disease that seeds more
    than one neighborhood.
    """
    out: dict[str, set[str]] = {}
    for r in rows:
        out.setdefault(r.mondo_id, set()).add(r.category)
    return {m: sorted(c) for m, c in sorted(out.items())}


def to_tsv(rows: list[SeedRow]) -> str:
    """Serialize seed rows as a TSV with a header line."""
    lines = ["mondo_id\tlabel\tcategory\tcurated"]
    for r in rows:
        lines.append(f"{r.mondo_id}\t{r.label}\t{r.category}\t{int(r.curated)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# YAML-source categorization: reproduce the dismech #1079 grouping directly from
# the authoritative prioritised-rare-disease-list.yml, so the seed does not
# depend on hand-transcribing a rendered issue. Keyword rules are verbatim from
# the issue's "Grouping methodology" section; applied to the current YAML they
# reproduce its category counts (Neurodevelopmental 311, Neurodegenerative 164,
# Neuroimmune 12, Cardiac 306).
# ---------------------------------------------------------------------------

# A disease joins a category if any keyword is a substring of its lowercased
# searchable metadata blob (label + synonyms + category labels + HPO categories).
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Neurodevelopmental": (
        "neurodevelopmental", "intellectual disability", "epileptic encephalopathy",
        "developmental and epileptic encephalopathy", "autism", "rett syndrome",
        "angelman", "fragile x", "tuberous sclerosis", "lissencephaly",
        "microcephaly", "holoprosencephaly",
    ),
    "Neurodegenerative": (
        "neurodegenerative", "neuronal ceroid", "amyotrophic lateral", "huntington",
        "parkinson", "alzheimer", "ataxia", "spinal muscular atrophy",
        "leukodystrophy", "neurodegenerat", "frontotemporal dementia", "prion",
        "batten",
    ),
    "Neuroimmune": (
        "multiple sclerosis", "autoimmune encephalitis", "myasthenia gravis",
        "guillain-barr", "neuromyelitis optica", "neuroimmune", "neuroinflam",
        "anti-nmda", "transverse myelitis", "chronic inflammatory demyelinating",
    ),
    "Cardiac": (
        "cardiomyopath", "arrhythm", "long qt", "brugada", "cardiac", "heart defect",
        "congenital heart", "catecholaminergic polymorphic", "marfan",
        "loeys-dietz", "pulmonary arterial hypertension", "familial hypercholesterol",
    ),
}
# Extra Cardiac rule: any disease whose mondo_categories / mondo_category_body_
# system labels contain one of these is Cardiac, regardless of the keyword blob.
_CARDIAC_CATEGORY_LABEL_KEYWORDS = ("cardiovascular", "cardiac", "heart")
# Metadata fields (besides mondo_label + mondo_synonyms) whose label text feeds
# the searchable blob, per the #1079 methodology.
_META_LABEL_FIELDS = (
    "mondo_categories", "mondo_category_body_system", "mondo_category_developmental",
    "mondo_category_etiologic", "mondo_category_genetic", "mondo_category_extrinsic",
    "mondo_category_molecular", "hpo_high_level_categories",
)


def _labels(value) -> list[str]:
    """Label strings from a YAML field: a str, a list of str, or a list of
    ``{id, label}`` dicts. Anything else yields nothing."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    out: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                out.append(str(item.get("label", "")))
            else:
                out.append(str(item))
    return out


def _search_blob(record: dict) -> str:
    parts = [str(record.get("mondo_label", ""))]
    parts += _labels(record.get("mondo_synonyms"))
    for field in _META_LABEL_FIELDS:
        parts += _labels(record.get(field))
    return " ~ ".join(parts).lower()


def _category_label_text(record: dict) -> str:
    parts: list[str] = []
    for field in ("mondo_categories", "mondo_category_body_system"):
        parts += _labels(record.get(field))
    return " ~ ".join(parts).lower()


def categorize(record: dict) -> set[str]:
    """Categories a disease record joins under the #1079 keyword rules."""
    blob = _search_blob(record)
    cats = {cat for cat, kws in CATEGORY_KEYWORDS.items() if any(k in blob for k in kws)}
    cat_text = _category_label_text(record)
    if any(k in cat_text for k in _CARDIAC_CATEGORY_LABEL_KEYWORDS):
        cats.add("Cardiac")
    return cats


def seed_rows_from_yaml(diseases: list[dict]) -> list[dict]:
    """One row per (disease, category) for every categorized disease, preserving
    the priorities/prevalence priors useful to the later power filter."""
    rows: list[dict] = []
    for d in diseases:
        cats = categorize(d)
        if not cats:
            continue
        for cat in sorted(cats):
            rows.append(
                {
                    "mondo_id": d.get("mondo_id", ""),
                    "label": d.get("mondo_label", ""),
                    "category": cat,
                    "prevalence_per_100k_us": d.get("prevalence_per_100k_us"),
                    "prioritization_category": d.get("prioritization_category", ""),
                }
            )
    return rows


_YAML_TSV_COLS = (
    "mondo_id", "label", "category", "prevalence_per_100k_us",
    "prioritization_category",
)


def yaml_seed_to_tsv(rows: list[dict]) -> str:
    lines = ["\t".join(_YAML_TSV_COLS)]
    for r in rows:
        vals = []
        for c in _YAML_TSV_COLS:
            v = r.get(c)
            vals.append("" if v is None else str(v))
        lines.append("\t".join(vals))
    return "\n".join(lines) + "\n"


def _main(argv: list[str]) -> int:
    usage = (
        "usage:\n"
        "  python anchor_selection.py parse-md <issue_1079.md>       > seed.tsv\n"
        "  python anchor_selection.py from-yaml <priority_list.yml>  > seed.tsv\n"
    )
    if len(argv) != 3 or argv[1] not in ("parse-md", "from-yaml"):
        sys.stderr.write(usage)
        return 2

    if argv[1] == "parse-md":
        with open(argv[2], encoding="utf-8") as fh:
            rows = parse_priority_seed(fh.read())
        uniq = unique_diseases(rows)
        sys.stderr.write(
            f"parsed {len(rows)} (disease, category) rows; "
            f"{len(uniq)} distinct MONDO ids\n"
        )
        sys.stdout.write(to_tsv(rows))
        return 0

    import yaml  # local import: only the YAML path needs PyYAML

    with open(argv[2], encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    rows = seed_rows_from_yaml(data["diseases"])
    distinct = {r["mondo_id"] for r in rows}
    from collections import Counter

    per_cat = Counter(r["category"] for r in rows)
    sys.stderr.write(
        f"categorized {len(rows)} (disease, category) rows; "
        f"{len(distinct)} distinct MONDO ids; per-category {dict(per_cat)}\n"
    )
    sys.stdout.write(yaml_seed_to_tsv(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
