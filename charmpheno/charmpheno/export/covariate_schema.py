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


def _split_top_level(name: str, sep: str = ":") -> list[str]:
    """Split *name* on *sep* characters that are NOT inside parentheses or brackets.

    This prevents a naive split from breaking a factor like ``C(dx)[T.A:B]``
    whose level string contains a colon inside the bracket group.
    """
    parts, depth, cur = [], 0, []
    for ch in name:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _recipe_for(name: str, continuous_cols: list[str]) -> dict | None:
    """Return a recipe dict for one design-column name, or None if unparseable."""
    if name == "Intercept":
        return {"kind": "intercept"}
    if name in continuous_cols:
        return {"kind": "main", "var": name}
    m = _DUMMY_RE.match(name)
    if m:
        return {"kind": "dummy", "var": m.group("var"), "level": m.group("level")}
    parts = _split_top_level(name)
    if len(parts) > 1:
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
    # design_columns lists only parseable columns in covariate_names order.
    # When unsupported is empty this is exactly 1:1 with covariate_names (and
    # with the Gamma rows), which is the only case the client evaluates recipes
    # — the client disables the covariate reader when unsupported is non-empty.
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
