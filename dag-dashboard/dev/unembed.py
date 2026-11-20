"""Split a self-contained `dag_view.html` back into its four export files.

`analysis/cloud/export_dag_gated.py` writes `<run>/dag_view.html`: the browser page with
the `dag-gated-export/1` files embedded in a `<script id="dag-data"
type="application/json">` block (`charmpheno.export.dag_gated.embed_in_page`). That one
file is what leaves the workspace. For GitHub Pages the repo keeps the page
(`dag-dashboard/index.html`) and the data separately, so a page fix does not mean
re-exporting; this script recovers the data files from the reviewed HTML.

    python dag-dashboard/dev/unembed.py dag_view.html dag-dashboard/published/0174/

The files are written exactly as the exporter writes them (compact separators, UTF-8,
no ASCII escaping, no NaN), so they are byte-identical to the run's `dag_export/`.
Standalone: standard library only, no charmpheno import.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

FILES = ("manifest.json", "nodes.json", "vocab.json", "topics.json")
_TAG = re.compile(r'<script id="dag-data" type="application/json">(.*?)</script>', re.S)


def dumps(obj) -> bytes:
    """The exporter's serialization (`charmpheno.export.dag_gated.dumps`)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def unembed(page: str) -> dict:
    """`{file name: parsed file}` from the page's embedded block. The embed escapes
    `</` as `<\\/`, a valid JSON escape, so `json.loads` reads it back unchanged."""
    # The embed puts the block just before </head>; search only there, since the page's
    # own script mentions the tag in a comment.
    m = _TAG.search(page.split("</head>", 1)[0])
    if m is None:
        raise SystemExit('no <script id="dag-data"> block: not a self-contained export page')
    blob = json.loads(m.group(1))
    missing = [f for f in FILES if f not in blob]
    if missing:
        raise SystemExit(f"embedded block lacks {missing}")
    fmt = blob["manifest.json"].get("schema")
    if fmt != "dag-gated-export/1":
        raise SystemExit(f"unexpected export format {fmt!r}")
    return {f: blob[f] for f in FILES}


def main(argv) -> None:
    if len(argv) != 3:
        raise SystemExit("usage: unembed.py DAG_VIEW_HTML OUT_DIR")
    files = unembed(Path(argv[1]).read_text(encoding="utf-8"))
    out = Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in files.items():
        b = dumps(obj)
        (out / name).write_bytes(b)
        print(f"{name}: {len(b):,} bytes")


if __name__ == "__main__":
    main(sys.argv)
