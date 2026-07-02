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


def _humanize(raw: str) -> str:
    """Humanize a raw id for display: underscores to spaces, sentence-case.
    A reasonable default label so no authoring is required (source_cohort ->
    'Source cohort'); overridable at the call site for a real name."""
    s = str(raw).replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else s


def build_gating_json(partition, group_counts, k, kept_topic_ids,
                      group_label_overrides=None):
    """gating.json: kept groups (count >= k) + per-kept-topic block label +
    humanized group labels + a k-anon-safe group_proportions map over kept
    groups (fractions summing to 1; sub-k groups already excluded)."""
    overrides = group_label_overrides or {}
    kept_groups = [g for g in partition.groups
                   if int(group_counts.get(g, 0)) >= int(k)]
    labels = partition.topic_labels()                 # length K, by original id
    topic_blocks = [labels[i] for i in kept_topic_ids]
    kept_counts = {g: int(group_counts.get(g, 0)) for g in kept_groups}
    total = sum(kept_counts.values()) or 1
    return {
        "group_var": partition.group_var,
        "group_var_label": _humanize(partition.group_var),
        "groups": kept_groups,
        "group_labels": {g: overrides.get(g, _humanize(g)) for g in kept_groups},
        "group_proportions": {g: kept_counts[g] / total for g in kept_groups},
        "topic_blocks": topic_blocks,
    }
