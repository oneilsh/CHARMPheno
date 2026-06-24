"""Gating bundle export: per-topic block labels + block-level k-anon.

A foreground group whose patient count is below the small-cell threshold k is
fully suppressed: dropped from the gating groups list AND its foreground topics
are excluded from the bundle (the dashboard never reveals a sub-k group). This
is the honest information floor for the rare-disease case.
"""
from __future__ import annotations


def suppressed_topic_ids(partition, group_counts, k):
    """Original topic ids of foreground blocks whose group count < k.

    Background topics are never suppressed. group_counts maps group label ->
    patient count.
    """
    supp = set()
    for g in partition.groups:
        if int(group_counts.get(g, 0)) < int(k):
            supp.update(int(i) for i in partition.block_indices(g))
    return supp


def build_gating_json(partition, group_counts, k, kept_topic_ids):
    """gating.json: kept groups (count >= k) + per-kept-topic block label."""
    kept_groups = [g for g in partition.groups
                   if int(group_counts.get(g, 0)) >= int(k)]
    labels = partition.topic_labels()                 # length K, by original id
    topic_blocks = [labels[i] for i in kept_topic_ids]
    return {
        "group_var": partition.group_var,
        "groups": kept_groups,
        "topic_blocks": topic_blocks,
    }
