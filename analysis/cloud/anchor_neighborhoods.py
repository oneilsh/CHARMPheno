"""Neighborhood assembly + nesting for the expanded-SNOMED anchor set.

Pure graph logic over the anchor set + the anchor<->anchor `concept_ancestor`
edges emitted by anchor_selection_cloud.py. No Spark, no I/O — unit-tested.

Stage 6-7 of the pipeline
(docs/superpowers/specs/2026-07-31-expanded-snomed-anchor-selection-design.md).
This first increment provides the nesting rule; SNOMED-neighborhood grouping and
positive-overlap collapse land as the candidate structure is understood.
"""
from __future__ import annotations

import csv
import sys
from collections.abc import Iterable

# rare6 anchors, always pinned (ADR 0039). Concept ids + labels mirror
# charmpheno.omop.cohorts._RARE6_ANCESTORS; labels kept here so the freeze stage
# has no Spark/charmpheno import dependency.
RARE6_LABELS: dict[int, str] = {
    79145: "Ehlers-Danlos syndrome",
    438688: "Sarcoidosis",
    257628: "Systemic lupus erythematosus",
    40352976: "Scleroderma / systemic sclerosis",
    76685: "Myasthenia gravis",
    432595: "Amyloidosis",
}


def maximal_anchors(
    clearing_ids: Iterable[int],
    ancestry_pairs: Iterable[tuple[int, int]],
) -> set[int]:
    """The nesting rule: keep only the most-specific anchors that clear the floor.

    ``clearing_ids`` are the anchors that pass the positives floor.
    ``ancestry_pairs`` are ``(ancestor, descendant)`` is-a relations *among
    anchors* (as read from `concept_ancestor`, self-pairs excluded). An anchor is
    dropped when it is a proper ancestor of another clearing anchor — so of a
    nested chain we retain the deepest members that still clear the floor. Anchors
    incomparable to all others are always kept.
    """
    clearing = {int(x) for x in clearing_ids}
    ancestors_of_clearing: set[int] = set()
    for ancestor, descendant in ancestry_pairs:
        a, d = int(ancestor), int(descendant)
        if a != d and a in clearing and d in clearing:
            ancestors_of_clearing.add(a)
    return clearing - ancestors_of_clearing


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def select_frozen_anchors(
    candidate_rows: Iterable[dict],
    *,
    floor: int,
    ceiling: int,
    rare6: dict[int, str] | None = None,
    exclude_ids: Iterable[int] = (),
) -> list[dict]:
    """The frozen anchor set (ADR 0039): rare6 pinned, plus prioritised anchors in
    the [floor, ceiling] band that survive nesting, minus explicit exclusions.

    ``candidate_rows`` are dict rows from candidates_with_counts.tsv (need
    standard_concept_id, standard_concept_name, positive_count, is_maximal).
    Returns dict rows: concept_id, label, positive_count, source ('rare6' |
    'prioritised'), rare6 first then by descending count.
    """
    rare6 = dict(RARE6_LABELS if rare6 is None else rare6)
    rare6_ids = {int(k) for k in rare6}
    exclude = {int(x) for x in exclude_ids}

    rows = list(candidate_rows)  # consumed twice; tolerate a generator
    counts = {int(r["standard_concept_id"]): int(r["positive_count"]) for r in rows}

    frozen: dict[int, dict] = {}
    for cid, label in rare6.items():
        frozen[int(cid)] = {
            "concept_id": int(cid), "label": label,
            "positive_count": counts.get(int(cid)), "source": "rare6",
        }
    for r in rows:
        cid = int(r["standard_concept_id"])
        if cid in rare6_ids or cid in exclude:
            continue
        pc = int(r["positive_count"])
        if pc < floor or pc > ceiling or not _as_bool(r["is_maximal"]):
            continue
        frozen[cid] = {
            "concept_id": cid, "label": r["standard_concept_name"],
            "positive_count": pc, "source": "prioritised",
        }
    return sorted(
        frozen.values(),
        key=lambda d: (d["source"] != "rare6", -(d["positive_count"] or 0)),
    )


def _main(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Freeze the expanded anchor set (ADR 0039).")
    p.add_argument("--candidates", required=True, help="candidates_with_counts.tsv")
    p.add_argument("--out", required=True, help="frozen anchor TSV path")
    p.add_argument("--floor", type=int, default=50)
    p.add_argument("--ceiling", type=int, default=10000)
    p.add_argument("--exclude", default="",
                   help="comma-separated concept_ids to drop (over-broad / rare6-redundant)")
    args = p.parse_args(argv)

    with open(args.candidates, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    exclude = [int(x) for x in args.exclude.split(",") if x.strip()]
    frozen = select_frozen_anchors(
        rows, floor=args.floor, ceiling=args.ceiling, exclude_ids=exclude
    )

    with open(args.out, "w", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["concept_id", "label", "positive_count", "source"],
                           delimiter="\t")
        w.writeheader()
        for r in frozen:
            w.writerow(r)

    ids = tuple(r["concept_id"] for r in frozen)
    n_rare6 = sum(1 for r in frozen if r["source"] == "rare6")
    sys.stderr.write(
        f"[freeze] {len(frozen)} anchors ({n_rare6} rare6 + {len(frozen) - n_rare6} "
        f"prioritised), floor={args.floor} ceiling={args.ceiling} "
        f"excluded={len(exclude)}. wrote {args.out}\n"
    )
    sys.stderr.write(f"[freeze] registry tuple: _RAREN_ANCESTORS = {ids}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
