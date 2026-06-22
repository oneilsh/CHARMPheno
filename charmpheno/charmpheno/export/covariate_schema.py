"""Build the scrubbed covariate schema the Atlas uses to render STM controls.

Pure: takes pre-computed level counts + continuous percentiles (the caller does
the in-enclave Spark aggregation) and the design-column names, and emits the
`covariate_schema.json` payload — controls (one per variable) + per-design-column
recipes (index-aligned with Gamma) + unsupported. Categorical levels under `k`
patients are suppressed here; continuous ranges come from percentiles (never
min/max), so nothing single-patient can be reconstructed.
"""
from __future__ import annotations

import re

_DUMMY_RE = re.compile(r"^C\((?P<var>[^)]+)\)\[T\.(?P<level>.+)\]$")


def _recipe_for(name: str, continuous_cols: list[str]):
    """Return a recipe dict for one design-column name, or None if unparseable."""
    if name == "Intercept":
        return {"kind": "intercept"}
    if name in continuous_cols:
        return {"kind": "main", "var": name}
    m = _DUMMY_RE.match(name)
    if m:
        return {"kind": "dummy", "var": m.group("var"), "level": m.group("level")}
    if ":" in name:
        parts = name.split(":")
        factors = [_recipe_for(p, continuous_cols) for p in parts]
        if all(f is not None for f in factors):
            return {"kind": "interaction", "factors": factors}
    return None


def build_covariate_schema(
    *,
    covariate_names: list[str],
    continuous_cols: list[str],
    categorical_levels: dict[str, dict],
    level_counts: dict[str, int],
    continuous_stats: dict[str, tuple[float, float, float]],
    k: int,
) -> dict:
    design_columns = []
    unsupported = []
    for name in covariate_names:
        recipe = _recipe_for(name, continuous_cols)
        if recipe is None:
            unsupported.append(name)
        else:
            design_columns.append({"name": name, "recipe": recipe})

    # Which categorical levels survive the k-anon guard. A level is kept if it
    # is the reference (always selectable) or its dummy count >= k.
    kept_levels: dict[str, set] = {}
    for name, cnt in level_counts.items():
        m = _DUMMY_RE.match(name)
        if m and cnt >= k:
            kept_levels.setdefault(m.group("var"), set()).add(m.group("level"))

    controls = []
    for var in continuous_cols:
        p5, p50, p95 = continuous_stats[var]
        controls.append({
            "name": var, "type": "continuous",
            "range": [p5, p95], "default": p50,
        })
    for var, info in categorical_levels.items():
        ref = info["reference"]
        surviving = [
            lvl for lvl in info["levels"]
            if lvl == ref or lvl in kept_levels.get(var, set())
        ]
        controls.append({
            "name": var, "type": "categorical",
            "reference": ref, "levels": surviving,
        })

    return {
        "k": k,
        "controls": controls,
        "design_columns": design_columns,
        "unsupported": unsupported,
    }
