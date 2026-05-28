"""Shared driver utilities for the cloud spark-submit drivers.

Three helpers extracted from the per-driver copies in
``lda_bigquery_cloud.py``, ``hdp_bigquery_cloud.py``,
``eval_coherence_cloud.py``, and ``build_dashboard_cloud.py``:

- ``_phase``: bracket a driver phase with start/end markers and elapsed
  wall time. Use as ``with _phase("phase name"): ...``.
- ``configure_logging``: route ``spark_vi.core.runner`` per-iter INFO
  output through Python logging with a ``[driver]`` prefix so cluster
  log capture sees the same lines a notebook user would.
- ``make_spark_session``: build a SparkSession with the standard cluster
  config, quiet the executor-loss noise via
  ``_log_utils.quiet_spot_reclamation``, and print a one-line driver
  banner with Spark version + master + defaultParallelism.

The drivers retain their model-specific bodies; only the boilerplate
moves here.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

from pyspark.sql import SparkSession

from _log_utils import quiet_spot_reclamation


@contextmanager
def _phase(name: str) -> Iterator[None]:
    """Bracket a driver phase with start/end markers and elapsed wall time."""
    print(f"[driver] >>> {name}", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[driver] <<< {name}: {time.perf_counter() - t0:.1f}s", flush=True)


def configure_logging(extra_loggers: dict[str, int] | None = None) -> None:
    """Surface spark_vi.core.runner per-iter INFO lines with [driver] prefix.

    Root stays at WARNING so PySpark / numpy / etc don't spam. spark_vi is
    bumped to INFO so the runner's iteration progress lines come through.
    ``force=True`` overrides any handler PySpark may have installed.

    Args:
        extra_loggers: optional mapping of logger name -> level to set after
            the base configuration. Drivers with additional verbose packages
            (e.g. ``{"charmpheno": logging.INFO}``) pass them here.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="[driver]   %(message)s",
        stream=__import__("sys").stdout,
        force=True,
    )
    logging.getLogger("spark_vi").setLevel(logging.INFO)
    if extra_loggers:
        for name, level in extra_loggers.items():
            logging.getLogger(name).setLevel(level)


def make_spark_session(app_name: str) -> SparkSession:
    """Build the standard cluster SparkSession, quiet executor-loss noise,
    and print a one-line banner. Returns the session for caller use."""
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    # Silence the GCS connector chatter (RequestTracker / hflush rate-limit
    # noise from event-log writes). Set BEFORE any actions.
    spark.sparkContext.setLogLevel("WARN")
    # Additionally silence the spot-reclamation flood (BlockManager cascades,
    # FetchFailed stack traces from TaskSetManager, etc.) without losing
    # other WARN messages.
    quiet_spot_reclamation(spark)
    sc = spark.sparkContext
    print(
        f"[driver] Spark {sc.version}, master={sc.master}, "
        f"defaultParallelism={sc.defaultParallelism}",
        flush=True,
    )
    return spark
