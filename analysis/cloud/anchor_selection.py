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


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write(
            "usage: python anchor_selection.py <priority_list.md> > seed.tsv\n"
        )
        return 2
    with open(argv[1], encoding="utf-8") as fh:
        rows = parse_priority_seed(fh.read())
    uniq = unique_diseases(rows)
    sys.stderr.write(
        f"parsed {len(rows)} (disease, category) rows; "
        f"{len(uniq)} distinct MONDO ids\n"
    )
    sys.stdout.write(to_tsv(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
