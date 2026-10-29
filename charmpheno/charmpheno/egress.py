"""The All of Us disclosure floor, in one place.

Any aggregate cell computed from fewer than `EGRESS_FLOOR` persons is not
disclosable: per-cell tables stay workspace-internal and committed reports carry
pooled figures and counts-of-nodes only (AGENTS.md, "Egress floor"). Every driver
that suppresses, filters or pools by that rule reads this constant, so the floor
cannot drift between them (ADR 0051 amendment, 2026-10-29).

Clinical policy, not framework: spark-vi functions that need a minimum count take
it as an argument and the drivers pass this value.
"""
EGRESS_FLOOR = 20
