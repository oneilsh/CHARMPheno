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
    """Bracket a driver phase with start/end markers, a wall-clock timestamp,
    and elapsed wall time. The HH:MM:SS timestamp makes separate runs
    distinguishable in a captured log (two runs never share it) and shows
    real-time progress -- so a slow phase is not mistaken for a hang.

    The whole banner is bold cyan (term_colors; identity when color is off)
    -- a phase boundary is the first thing to scan for in a long run, so it
    gets the strongest, least-ambiguous treatment in the palette. The import
    is LOCAL, not module-level: several drivers that import `_phase` (e.g.
    `hpoa_stage2_probe.py`) ride Spark's `--py-files` for their own
    mapPartitions kernels, and term_colors must never become a transitive
    top-level dependency an executor could be asked to import (see
    term_colors.py's module docstring)."""
    import term_colors
    print(term_colors.bold_cyan(
        f"[driver] [{time.strftime('%H:%M:%S')}] >>> {name}"), flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(term_colors.bold_cyan(
            f"[driver] [{time.strftime('%H:%M:%S')}] <<< {name}: "
            f"{time.perf_counter() - t0:.1f}s"), flush=True)


class _ColorLogFormatter(logging.Formatter):
    """Bold the `iter N/M` marker + dim the trailing per-iter timing on
    spark_vi's progress line, and green-highlight an `η_boost[...]` wiring
    sentinel wherever it appears (gated_lda's per-iter model summary,
    `spark_vi.models.topic.gated_lda`) -- WITHOUT editing spark-vi: both
    rules live in `term_colors` and are applied here, at the print site, to
    the fully-formatted "[driver]   ..." line. Identity on any other line,
    and identity (via term_colors.c) when color is disabled.

    `term_colors` is imported LOCALLY inside `format` (see `_phase` above
    for why) -- this class is only ever instantiated by `configure_logging`,
    driver-side, but stays importable with zero cost even where it isn't."""

    def format(self, record: logging.LogRecord) -> str:
        import term_colors
        msg = super().format(record)
        return term_colors.colorize_eta_boost(term_colors.colorize_iter_line(msg))


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
    # basicConfig(force=True) tears down and rebuilds the root handler with
    # the format string above; swap in the color-aware Formatter on that same
    # handler rather than duplicating the "[driver]   " format string here.
    for handler in logging.getLogger().handlers:
        handler.setFormatter(_ColorLogFormatter("[driver]   %(message)s"))
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


# --------------------------------------------------------------------------- #
# Durable driver log                                                           #
# --------------------------------------------------------------------------- #
# Mirror of scripts/run_experiment.py's PATIENT_PATTERNS + NOISE_PATTERNS (the
# wrapper's sanitize boundary). Duplicated by design: this tee runs INSIDE the
# spark-submit driver, which cannot import the wrapper, and the whole point is
# to not depend on the wrapper being alive. Keep the two lists in sync.
import re as _re

_TEE_DROP_PATTERNS = [
    _re.compile(r"person_hash", _re.IGNORECASE),
    _re.compile(r"person_id\s*=\s*\S+"),
    _re.compile(r"\bhash:[0-9a-f]{6,}", _re.IGNORECASE),
    _re.compile(r"transform sample", _re.IGNORECASE),
    _re.compile(r"^\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} (INFO|WARN|DEBUG) "),
    _re.compile(r"\[CONTEXT ratelimit_period="),
]

# `driver_log.md` is a markdown artifact (AGENTS.md: never colorized), but the
# live terminal this tee mirrors may legitimately be running in color
# (term_colors.enabled(), e.g. a real tty or CHARM_COLOR=1 forcing it through
# scripts/run_experiment.py's relay). Strip ANSI SGR escapes before a line is
# batched to disk -- ONLY the persisted copy is affected; `self._real.write`
# below stays a raw, uncolored-or-colored-as-is passthrough to whoever is
# actually watching (a terminal, or run_experiment.py's own relay, which
# strips again before its OWN summary.md).
_ANSI_RE = _re.compile(r"\x1b\[[0-9;]*m")


class _StdoutTee:
    """Forward every write to the real stdout AND append sanitized complete
    lines to a file in TIME-BATCHED open-append-close flushes.

    Two filesystem realities shaped this, in sequence:
      1. A single long-lived append handle is NOT durable on the AoU runs dir —
         it is a gcsfuse mount, where writes land in a local staging file and
         upload to GCS only on CLOSE. Exp 0103's smoke held one handle for 4h,
         the cluster died, and GCS kept only the last-closed content (the
         session header). Hence: open-append-CLOSE per flush.
      2. But per-LINE open-append-close is fatal on the same mount: each close
         is a FULL-OBJECT rewrite, GCS caps object mutations at ~1/s, and a
         bursty phase (readout heartbeats + Spark executor-loss stack traces)
         exceeds it — gcsfuse's staged temp files pile up behind the throttle
         until ENOSPC kills the run (exp 0104 smokes, twice, with the local
         disk 80% free). Hence: BATCH lines and close once per
         `flush_every_s` seconds (default 20 — a few mutations/min/object,
         bounded staging, and a crash loses at most one batch instead of
         causing the crash).
    Sanitization mirrors the wrapper's (patient rows and log4j chatter never
    reach disk)."""

    def __init__(self, path, real, flush_every_s=20.0):
        self._path = path
        self._real = real
        self._buf = ""
        self._pending: list[str] = []
        self._flush_every_s = float(flush_every_s)
        self._last_flush = time.monotonic()

    def _flush_pending(self):
        if not self._pending:
            self._last_flush = time.monotonic()
            return
        try:
            with open(self._path, "a") as f:
                f.write("\n".join(self._pending) + "\n")
            self._pending.clear()
        except OSError:
            # A failing tee must never take down the run it protects. Drop the
            # batch rather than let it grow without bound behind a dead mount.
            self._pending.clear()
        self._last_flush = time.monotonic()

    def write(self, s):
        n = self._real.write(s)
        self._buf += s
        *done, self._buf = self._buf.split("\n")
        for ln in done:
            plain = _ANSI_RE.sub("", ln)
            if not any(p.search(plain) for p in _TEE_DROP_PATTERNS):
                self._pending.append(plain)
        if (time.monotonic() - self._last_flush) >= self._flush_every_s:
            self._flush_pending()
        return n

    def flush(self):
        self._real.flush()

    def __getattr__(self, name):  # fileno/isatty/encoding for libraries that ask
        return getattr(self._real, name)


def install_stdout_tee(path) -> None:
    """Tee sys.stdout to `path` for the REST OF THE PROCESS (no restore: the
    drivers exit after main, and a context manager would force a whole-main
    re-indent for a lifetime that is the process anyway). Call once, right
    after the run dir exists, BEFORE the fit starts. Idempotent per path."""
    import atexit
    import sys
    if isinstance(sys.stdout, _StdoutTee):
        return
    print(f"[driver] durable log: {path}", flush=True)
    tee = _StdoutTee(path, sys.stdout)
    sys.stdout = tee
    # Clean exits upload the tail batch; crashes lose at most flush_every_s of
    # lines — the acceptable cost of not mutation-storming the gcsfuse object.
    atexit.register(tee._flush_pending)


def flush_stdout_tee() -> None:
    """Force the durable tee's pending batch to disk NOW, if one is installed.

    A client-mode `spark-submit` driver can fail to exit after `main` returns:
    the SparkSession is stopped (its `__exit__` unregisters the YARN app), but
    a lingering non-daemon py4j/gateway thread keeps the Python process alive,
    which wedges any CHAINED sweep waiting for the process to exit before it
    launches the next step. The drivers dodge that with `os._exit`, which is a
    hard teardown that SKIPS the `atexit` above — so the run-dir log would lose
    its tail batch (the very lines a reader copies out) unless the pending
    batch is flushed first. Call this immediately before `os._exit`."""
    import sys
    if isinstance(sys.stdout, _StdoutTee):
        sys.stdout._flush_pending()
