"""WORKUP sidecar: every code first recorded in the year after each person's index
(spec 2026-10-28 workup-voi-design; exp 0167).

WHAT IT IS
----------
One row per `(person_id, dom, token)`: the token's FIRST date in
`[index_date, index_date + label_window_days)`, for every person of the run's sample
(train and test) and every corpus domain — condition_era concepts (dom 0) and each
extra domain in the corpus's order (drug concepts, measurement result tokens), read
with exactly the loaders the assembler uses, so the tokens are the ones the bundle's
vocabularies hold.

Framing 2 asks which fact recorded during the WORKUP (the index up to the first
diagnosis of a group member) would most move a patient's differential. The bundle
cannot answer that: the assembler keeps the post-index window for conditions only,
and only as label nodes; post-index drugs and measurements are discarded at windowing
(`multi_domain.lookback_feature_frames`). So the year after the index is captured here,
where the full-history frames still exist, and a group's window `[index, t_dx)` is
applied at analysis time (`dag_gated_readout.workup_frame`) from the conversion
sidecar's first-attestation dates.

WHY ITS OWN KEY
---------------
Driver-owned, like the conversion sidecar, and for the same reason: building it
inside the assembler would edit a source-hashed module and move every bundle key. Its
key folds what decides its content: the conversion sidecar's key (which fixes the
persons, the data release and, through the index horizon, every index date), the
domains, the label window, and this module's source. Not the bundle key.
"""
from __future__ import annotations

import hashlib
import json
import sys

WORKUP_SIDECAR_VERSION = "1"
WORKUP_FILE = "workup_first.parquet"
WORKUP_COLUMNS = ("person_id", "dom", "token", "first_date")


def _module_source_hash(module) -> str:
    """Duplicated from `conversion_sidecar` on purpose (see its note)."""
    import inspect
    try:
        return hashlib.sha256(inspect.getsource(module).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return "src-unavailable"


def workup_sidecar_key(*, conversion_key, domains, label_window_days) -> str:
    payload = {"artifact": "workup_sidecar", "version": WORKUP_SIDECAR_VERSION,
               "conversion_key": str(conversion_key), "domains": list(domains),
               "label_window_days": int(label_window_days),
               "src": _module_source_hash(sys.modules[__name__])}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def workup_path(uri, key) -> str:
    return f"{str(uri).rstrip('/')}/{key}/{WORKUP_FILE}"


def first_in_window(events, horizon, *, dom, date_col, token_col="concept_id",
                    label_window_days=365):
    """`(person_id, dom, token, first_date)`: each token's first date inside
    `[index_date, index_date + label_window_days)`. `horizon` carries
    `(person_id, index_date)`; persons without an index are dropped."""
    from pyspark.sql import functions as F
    hz = horizon.select("person_id", "index_date")
    return (events.join(hz, "person_id")
            .where(F.col(date_col) >= F.col("index_date"))
            .where(F.col(date_col) < F.date_add(F.col("index_date"),
                                                int(label_window_days)))
            .groupBy("person_id", F.col(token_col).cast("long").alias("token"))
            .agg(F.min(F.col(date_col)).alias("first_date"))
            .withColumn("dom", F.lit(int(dom)))
            .select(*WORKUP_COLUMNS))


def try_load_workup(spark, uri, key):
    """The cached frame, or None on a miss (a wrong-column parquet raises)."""
    try:
        df = spark.read.parquet(workup_path(uri, key))
    except Exception:
        return None
    missing = [c for c in WORKUP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{workup_path(uri, key)} is not a workup sidecar "
                         f"(missing {missing})")
    return df.select(*WORKUP_COLUMNS)


def main(argv=None) -> int:
    """Build the workup sidecar for a finished run. Cluster-only (BigQuery); needs
    the run's conversion sidecar (its index horizon) on this cluster."""
    import argparse
    from pathlib import Path

    from _driver_common import _phase, configure_logging, make_spark_session
    from conversion_sidecar import try_load_index_horizon
    from dag_gated_readout import resolve_run_dir

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--sidecar-uri", default=None,
                   help="Root for the artifact. Defaults to <bundle cache_uri>/"
                        "workup_sidecar (a sibling of the bundle cache).")
    p.add_argument("--force", action="store_true", help="Rebuild even on a HIT.")
    args = p.parse_args(argv)
    configure_logging()

    run_dir = resolve_run_dir(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cm = manifest.get("corpus_manifest") or {}
    wp = Path(run_dir) / "conversion_sidecar.json"
    if not cm or not wp.exists():
        print("[workup] ERROR: needs the run's corpus_manifest and "
              "<run>/conversion_sidecar.json (make build-conversion-sidecar ID=N "
              "first).", flush=True)
        return 2
    wit = json.loads(wp.read_text())
    extra = list(cm.get("extra_domains") or [])
    domains = ["condition", *extra]
    window = int(cm.get("label_window_days") or 365)
    key = workup_sidecar_key(conversion_key=wit["key"], domains=domains,
                             label_window_days=window)
    uri = args.sidecar_uri or f"{str(cm.get('cache_uri')).rstrip('/')}/workup_sidecar"
    print(f"[workup] key={key} uri={uri} domains={domains} window={window}d",
          flush=True)

    with make_spark_session(app_name="build-workup-sidecar") as spark:
        hit = None if args.force else try_load_workup(spark, uri, key)
        if hit is not None:
            print("[workup] HIT — nothing to build (pass --force to rebuild).",
                  flush=True)
            df = hit
        else:
            hz = try_load_index_horizon(spark, wit["sidecar_uri"], wit["key"])
            if hz is None:
                print(f"[workup] ERROR: conversion sidecar MISS at "
                      f"{wit['sidecar_uri']}/{wit['key']} — build it on this cluster "
                      "first (make build-conversion-sidecar ID=N).", flush=True)
                return 2
            from charmpheno.omop import load_omop_bigquery
            pm = int(cm.get("person_mod") or 0) or None
            hz = hz.select("person_id", "index_date").cache()
            parts = []
            with _phase("load the domains and keep the year after each index"):
                cond = load_omop_bigquery(
                    spark=spark, cdr_dataset=cm["cdr"], billing_project=cm["billing"],
                    person_sample_mod=pm, source_table="condition_era")
                parts.append(first_in_window(cond, hz, dom=0,
                                             date_col="condition_era_start_date",
                                             label_window_days=window))
                for i, dom in enumerate(extra, start=1):
                    d = load_omop_bigquery(
                        spark=spark, cdr_dataset=cm["cdr"],
                        billing_project=cm["billing"], person_sample_mod=pm,
                        concept_types=(dom,))
                    parts.append(first_in_window(d, hz, dom=i, date_col="event_date",
                                                 label_window_days=window))
            df = parts[0]
            for x in parts[1:]:
                df = df.unionByName(x)
            with _phase("write"):
                df.write.mode("overwrite").parquet(workup_path(uri, key))
                df = try_load_workup(spark, uri, key)
            hz.unpersist()
        from pyspark.sql import functions as F
        stats = {int(r["dom"]): (int(r["n"]), int(r["p"])) for r in
                 df.groupBy("dom").agg(F.count("*").alias("n"),
                                       F.countDistinct("person_id").alias("p"))
                 .collect()}
        for d, name in enumerate(domains):
            n, ppl = stats.get(d, (0, 0))
            print(f"[workup]   {name}: {n:,} (person, token) rows over {ppl:,} persons",
                  flush=True)
        out = {"version": WORKUP_SIDECAR_VERSION, "key": key, "sidecar_uri": uri,
               "path": workup_path(uri, key), "domains": domains,
               "label_window_days": window, "conversion_key": wit["key"],
               "rows_by_domain": {domains[d]: v[0] for d, v in stats.items()
                                  if d < len(domains)}}
        (Path(run_dir) / "workup_sidecar.json").write_text(json.dumps(out, indent=2))
        print(f"[workup] wrote {Path(run_dir) / 'workup_sidecar.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
