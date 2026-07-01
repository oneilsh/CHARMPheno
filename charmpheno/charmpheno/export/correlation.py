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


def build_correlation_json(R, identified, support, partition, kept_topic_ids):
    """correlation.json over kept topics in block order; null for unidentified R."""
    labels = partition.topic_labels()                 # length K, by original id
    order = [i for i in kept_topic_ids]               # already block-ordered upstream
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
    }
