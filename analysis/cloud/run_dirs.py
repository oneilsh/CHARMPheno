"""Run-dir resolution shared by the drivers (dag_gated_readout, inspect_topics, the
sidecar builders). Pure stdlib; no Spark import."""
from __future__ import annotations

import glob
from pathlib import Path

RESULT_FILE = "gated_pc_result.npz"


def resolve_run_dir(pattern, *, who="readout"):
    """Resolve a run dir from an exact path or a glob (e.g. `.../0111-*`).

    An exact dir that holds gated_pc_result.npz is used as-is. Otherwise the value
    is treated as a glob (the Makefile passes `.../<id>-*` QUOTED so the shell does
    not pre-expand it) and filtered to matched dirs that contain gated_pc_result.npz
    — so a runs/ numbering COLLISION with another experiment of the same id (e.g. a
    stale `0076-multidomain-...` from another branch) is discarded automatically.
    Requires exactly one match; a clear error (prefixed `[who]`) otherwise."""
    p = Path(pattern)
    if p.is_dir() and (p / RESULT_FILE).exists():
        return p
    matches = [Path(m) for m in glob.glob(str(pattern))]
    hits = [m for m in matches if m.is_dir() and (m / RESULT_FILE).exists()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(
            f"[{who}] no run dir with {RESULT_FILE} matches {pattern!r} "
            f"(matched: {[m.name for m in matches]}). Has the fit finished + saved? "
            f"Pass an exact run dir.")
    raise SystemExit(
        f"[{who}] {len(hits)} run dirs match {pattern!r}: "
        f"{[m.name for m in hits]}. Pass an exact run dir.")
