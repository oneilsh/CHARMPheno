"""Top-level analysis package: drivers and shared eval/fit utilities.

Submodules:
    analysis.local   - local (parquet-backed) fit / eval drivers
    analysis.cloud   - Dataproc/BigQuery fit / eval drivers
    analysis._eval_common - bits both driver families share (split-contract check, etc.)

The package is intended to be importable when the repo root is on sys.path
(which is the case for the local poetry run, and on the Dataproc master
when this directory is staged alongside spark-submit).
"""
