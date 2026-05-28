"""Build dashboard/public/data/manifest.json from per-cohort corpus_stats.json files.

Single source of truth: each cohort's corpus_stats.json carries inline
{id, label, description} metadata (written by
charmpheno.export.corpus_stats.write_corpus_stats_sidecar). This script
aggregates them into the top-level manifest the dashboard needs to
populate its cohort selector.

Directory name = frontend cohort id (the mapping from
internal-cohort-key to frontend-id lives implicitly in how the build
drivers are invoked — e.g. cloud driver runs with `--out-dir <id>` where
<id> is the frontend id, even though the cohort's internal name is
something else like `first_dementia_year`).

Bundles lacking inline cohort metadata (e.g. dev bundles built without
a real OMOP cohort) are included but with a placeholder label
derived from the directory name.

Usage:
    python scripts/build_dashboard_manifest.py
    python scripts/build_dashboard_manifest.py --data-dir dashboard/public/data
    python scripts/build_dashboard_manifest.py --default cancer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def discover_cohort_entries(data_dir: Path) -> list[dict[str, str]]:
    """Walk subdirectories of data_dir and return cohort manifest entries.

    Each subdirectory containing a corpus_stats.json contributes one entry.
    Sorted by directory name for stable output.
    """
    entries: list[dict[str, str]] = []
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        stats_path = sub / "corpus_stats.json"
        if not stats_path.exists():
            continue
        try:
            stats = json.loads(stats_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(
                f"warning: skipping {sub.name}: cannot read corpus_stats.json ({e})",
                file=sys.stderr,
            )
            continue
        cohort = stats.get("cohort") or {}
        frontend_id = sub.name
        label = cohort.get("label") or frontend_id
        description = cohort.get("description") or ""
        entries.append({
            "id": frontend_id,
            "label": label,
            "description": description,
        })
    return entries


def resolve_default(
    entries: list[dict[str, str]],
    *,
    requested: str | None,
    existing_manifest_path: Path,
) -> str:
    """Resolve the manifest's `default` field.

    Priority:
      1. Explicit --default flag (must match a discovered cohort).
      2. Existing manifest's default, if still a discovered cohort.
      3. First discovered cohort alphabetically.
    """
    ids = {e["id"] for e in entries}
    if requested is not None:
        if requested not in ids:
            raise SystemExit(
                f"--default {requested!r} not in discovered cohorts: "
                f"{sorted(ids)}"
            )
        return requested
    if existing_manifest_path.exists():
        try:
            existing = json.loads(existing_manifest_path.read_text())
            existing_default = existing.get("default")
            if isinstance(existing_default, str) and existing_default in ids:
                return existing_default
        except (OSError, json.JSONDecodeError):
            pass
    return entries[0]["id"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build dashboard/public/data/manifest.json from per-cohort "
                    "corpus_stats.json files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dashboard/public/data"),
        help="Directory containing per-cohort subdirectories. Default: %(default)s",
    )
    parser.add_argument(
        "--default",
        dest="default_id",
        type=str,
        default=None,
        help="Frontend id of the default cohort. If omitted, preserves the existing "
             "manifest's default when still valid; otherwise picks the "
             "alphabetically-first discovered cohort.",
    )
    args = parser.parse_args(argv)

    entries = discover_cohort_entries(args.data_dir)
    if not entries:
        raise SystemExit(f"No cohorts with corpus_stats.json found in {args.data_dir}")

    manifest_path = args.data_dir / "manifest.json"
    default_id = resolve_default(
        entries,
        requested=args.default_id,
        existing_manifest_path=manifest_path,
    )

    manifest = {"default": default_id, "cohorts": entries}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Wrote {manifest_path} with {len(entries)} cohort(s); "
        f"default={default_id}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
