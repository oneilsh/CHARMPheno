"""Neighborhood assembly + nesting for the expanded-SNOMED anchor set.

Pure graph logic over the anchor set + the anchor<->anchor `concept_ancestor`
edges emitted by anchor_selection_cloud.py. No Spark, no I/O — unit-tested.

Stage 6-7 of the pipeline
(docs/superpowers/specs/2026-07-31-expanded-snomed-anchor-selection-design.md).
This first increment provides the nesting rule; SNOMED-neighborhood grouping and
positive-overlap collapse land as the candidate structure is understood.
"""
from __future__ import annotations

from collections.abc import Iterable


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
