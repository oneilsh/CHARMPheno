"""Correlation bundle export: logistic-normal topic correlation R + identified
mask, over the dashboard's kept topics in block order. Unidentified cells (no
joint document support) serialize as null so the frontend can grey them.

The identifiability floor is the model's min_pair_support: a cell is identified
iff the two topics were co-realized in >= min_pair_support documents (Blei &
Lafferty 2007 logistic-normal correlation; identifiability by support).
"""
from __future__ import annotations

import math


def _cell(x):
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)


def build_correlation_json(R, identified, support, partition, kept_topic_ids,
                            reference_id=None):
    """correlation.json over kept topics in block order; null for unidentified R.

    reference_id: if given, this topic id is dropped from topic_order (and
    every row/col it participates in). The STM reference topic is pinned at
    eta=0 (unit-variance, ~zero off-diagonal in Sigma) so it is inert; its
    n_pairs entries still mark it identified, which would otherwise render a
    spurious zero-correlation band in the dashboard (Component 3,
    docs/superpowers/specs/2026-07-01-gated-ctm-correlation-reporting-design.md).
    """
    labels = partition.topic_labels()                 # length K, by original id
    order = [i for i in kept_topic_ids if i != reference_id]  # already block-ordered upstream
    block_labels = [labels[i] for i in order]
    R_out, id_out, sup_out = [], [], []
    for i in order:
        R_out.append([_cell(R[i][j]) for j in order])
        id_out.append([bool(identified[i][j]) for j in order])
        sup_out.append([int(support[i][j]) for j in order])
    return {
        "topic_order": [int(i) for i in order],
        "block_labels": block_labels,
        "R": R_out,
        "identified": id_out,
        "support": sup_out,
        "reference_topic": (int(reference_id) if reference_id is not None else None),
    }
