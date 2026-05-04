"""Cluster smoke test for BigQuery read via the spark-bigquery-connector.

Reads a small slice of an OMOP fact table from the workspace CDR, prints
the schema and row count, exits. No LDA, no charmpheno imports — the goal
is to isolate connector + auth wiring before threading BQ data through
the full pipeline.

Reads two environment variables (set by the workspace setup notebook;
both end up exported in ~/.bashrc on the Dataproc master):
    WORKSPACE_CDR          - "<data-project>.<dataset>", read-only
    GOOGLE_CLOUD_PROJECT   - billing/compute project for the BQ job

Submit (from this directory on the Dataproc master):
    make bq-smoke
"""
from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession

# Smallest fact table in OMOP CDM that fits the canonical (person_id,
# visit_occurrence_id, concept_id) shape. condition_era won't work here —
# era tables aggregate across visits and drop visit_occurrence_id.
TABLE = "condition_occurrence"
LIMIT_ROWS = 1000


def main() -> int:
    cdr = os.environ.get("WORKSPACE_CDR")
    billing_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not cdr or not billing_project:
        print("ERROR: WORKSPACE_CDR and GOOGLE_CLOUD_PROJECT must be set in env. "
              "Run the workspace setup notebook (or `source ~/.bashrc`) first.",
              file=sys.stderr)
        return 1

    fq_table = f"{cdr}.{TABLE}"
    print(f"[driver] reading gs://bigquery/{fq_table} (limit {LIMIT_ROWS}), "
          f"billing project={billing_project}", flush=True)

    spark = SparkSession.builder.appName("bq_smoke_cloud").getOrCreate()
    sc = spark.sparkContext
    print(f"[driver] Spark {sc.version}, master={sc.master}, "
          f"defaultParallelism={sc.defaultParallelism}", flush=True)

    # Table-mode read — no materializationDataset needed. parentProject
    # routes BQ-job billing to compute project rather than the read-only
    # CDR data project.
    df = (
        spark.read.format("bigquery")
        .option("table", fq_table)
        .option("parentProject", billing_project)
        .load()
        .limit(LIMIT_ROWS)
    )

    print("[driver] schema:", flush=True)
    df.printSchema()

    n = df.count()
    print(f"[driver] row count (capped at {LIMIT_ROWS}): {n}", flush=True)
    if n == 0:
        print("[driver] WARNING: 0 rows returned — check table name / permissions",
              flush=True)
        spark.stop()
        return 2

    print("[driver] BQ SMOKE TEST PASSED", flush=True)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
