"""Fetch a Mondo release file (version-keyed, cached on local disk) and read a CDR table.

The two I/O helpers every Mondo label build starts from. Moved with ADR 0051 from
`analysis/cloud/anchor_selection_cloud.py` (the expanded-SNOMED anchor-selection
driver, removed), where the native build had been importing them from.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MONDO_RELEASE = (
    "https://github.com/monarch-initiative/mondo/releases/download/v{v}/{f}"
)


def download_cached(version: str, filename: str, cache_dir: Path) -> Path:
    """`<cache_dir>/<version>_<filename>`, downloaded from the Mondo release once.

    Keyed by the release version, so a cached file can never stand in for another
    release; a zero-byte file (an interrupted download) is fetched again."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{version}_{filename}"
    if dest.exists() and dest.stat().st_size > 0:
        sys.stderr.write(f"[mondo] cache hit {dest}\n")
        return dest
    url = MONDO_RELEASE.format(v=version, f=filename)
    sys.stderr.write(f"[mondo] downloading {url}\n")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted Monarch release URL)
    return dest


def read_bq(spark, cdr_dataset: str, billing_project: str, table: str):
    """One CDR table through the spark-bigquery connector, billed to `billing_project`."""
    return (
        spark.read.format("bigquery")
        .option("table", f"{cdr_dataset}.{table}")
        .option("parentProject", billing_project)
        .load()
    )
