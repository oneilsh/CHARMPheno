"""Retry a Spark ACTION across spot-VM preemption waves (driver side).

A multi-hour sequence of Spark actions (the readout's L-BFGS makes hundreds of
data passes) on a preemptible cluster sees jobs die for reasons that have nothing
to do with the work: a preemption wave starves the scheduler and Spark aborts the
job. `retry_spark_action` re-runs an idempotent action a bounded number of times
with backoff, and fails FAST on the failures a retry can never fix (an analysis
error in the plan, a full-disk node cascade, a dead SparkContext). Nothing here is
readout-specific; the readout's passes are its callers today.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Driver-side retry around a Spark ACTION (spot-VM survivability).            #
# --------------------------------------------------------------------------- #
def _spark_job_failure_types():
    """The exception types a failed Spark JOB surfaces on the driver.

    Resolved lazily (and defensively) rather than imported at module scope, for
    the same reason every other pyspark import here is function-local: this module
    is importable on a worker that has no gateway. If py4j is somehow absent the
    tuple comes back EMPTY, which makes `except ():` match nothing — a missing
    dependency degrades to "no retry", never to "retry every exception, including
    our own bugs".

    `Py4JJavaError` is what an RDD action raises when the JVM job dies (a job
    abort, an unrecoverable stage failure, a lost driver-side call);
    `PySparkException` is the pyspark-3.x wrapper the SQL/DataFrame paths raise
    for the same underlying JVM error, and costs nothing to include.
    """
    types = []
    try:
        from py4j.protocol import Py4JJavaError
        types.append(Py4JJavaError)
    except Exception:                            # pragma: no cover - no py4j
        pass
    try:
        from pyspark.errors import PySparkException
        types.append(PySparkException)
    except Exception:                            # pragma: no cover - old pyspark
        pass
    return tuple(types)


def _spark_deterministic_failure_types():
    """The Spark failure types a retry can NEVER fix, so they must fail FAST.

    `AnalysisException` is raised while the query is being ANALYZED, on the
    driver, before a single task is submitted: an unresolved column, a type
    mismatch, a missing table. It is a pure function of the plan and the schema,
    both of which a backoff sleep leaves untouched — so the sleep is guaranteed
    waste and the eventual re-raise is the same exception (e.g. a column the
    transform never produced would otherwise surface only after 60+120+240s).

    It is a SUBCLASS of `PySparkException` (pyspark 3.4+), which is exactly why
    the retryable tuple above catches it, and why this exclusion has to be named
    explicitly rather than left to the "retry any Spark job failure" default —
    that default is right about JOB failures (which are about the cluster) and
    wrong about ANALYSIS failures (which are about our code). Resolved lazily and
    defensively like everything else here: an empty tuple degrades to the old
    "retry it" behaviour, never to a crash on import.
    """
    types = []
    try:
        from pyspark.errors import AnalysisException
        types.append(AnalysisException)
    except Exception:                            # pragma: no cover - old pyspark
        try:
            from pyspark.sql.utils import AnalysisException  # pyspark < 3.4
            types.append(AnalysisException)
        except Exception:                        # pragma: no cover - no pyspark
            pass
    return tuple(types)


def _error_first_line(exc):
    """First line of a Spark exception, safely.

    `str(Py4JJavaError)` round-trips to the JVM to fetch the stack trace, which
    can itself raise if the gateway is the thing that died — and this runs on the
    failure path, where a second exception would replace a recoverable job abort
    with an unrecoverable driver crash. So: try the real message, fall back to
    the py4j-side `errmsg`, then to `repr`.
    """
    try:
        text = str(exc)
    except Exception:                            # pragma: no cover - dead gateway
        try:
            text = getattr(exc, "errmsg", "") or repr(exc)
        except Exception:
            text = exc.__class__.__name__
    for line in str(text).splitlines():
        if line.strip():
            return line.strip()
    return exc.__class__.__name__


def _is_bad_node_failure(exc):
    """True when a Spark job failure carries the YARN NODE-HEALTH / bad-node
    cascade signature — a DISK/health failure a backoff sleep cannot heal.

    The failure this names: a full worker local disk makes YARN mark the
    NodeManager UNHEALTHY and SIGTERM its containers, which surface to the driver
    as `ExecutorLostFailure ... Container from a bad node` (exit 143). Unlike a
    spot-preemption `ExecutorLostFailure` — which self-heals on the backoff
    because YARN replaces the container and a fresh job resets the per-taskset
    exclude lists — this one does NOT self-heal: the disk is still full, and
    re-running a shuffle-heavy pass into a disk-pressured cluster spreads the
    failure to more nodes. So it fails fast (like the schema and dead-context
    cases) rather than feeding the cascade through the backoff.

    HEURISTIC ON AN UNTRUSTED EXTERNAL STRING, so matched CONSERVATIVELY: only
    two high-precision needles, `"bad node"` (the quoted container message) and
    `"unhealthy"` (YARN's node-health verdict). A plain preemption
    `ExecutorLostFailure` carries neither — its wording is "Killed by external
    signal" / "Container released on a lost node" / "preempted", and "lost node"
    is deliberately NOT matched. Anything this does not recognize defaults to the
    OLD full-retry behaviour.

    Scans the FULL message (the bad-node detail is deep in the JVM stack trace,
    not the first line) across the exception's cause/context chain, each
    `str(...)` guarded the same way `_error_first_line` is — the gateway may be
    the thing that died, and a second exception here must not replace a
    recoverable job abort with a driver crash."""
    needles = ("bad node", "unhealthy")
    seen = set()
    cur = exc
    for _ in range(8):                               # bound the chain walk
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        try:
            text = str(cur)
        except Exception:                            # pragma: no cover - dead gateway
            text = getattr(cur, "errmsg", "") or ""
        low = str(text).lower()
        if any(n in low for n in needles):
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False


def retry_spark_action(fn, *, attempts=4, base_sleep_s=60, label=""):
    """Run one Spark ACTION, retrying a failed job after a backoff sleep.

    **The caller's contract: `fn` must be idempotent.** It should be a pure
    function of persisted RDDs (partitions lost to a preemption recompute from
    lineage, so a re-run recomputes the same number), and it must build any
    broadcast it uses INSIDE itself: a broadcast made outside may have lost its
    blocks with the executors, and a retry must not reuse it.

    **Why a retry fixes the failure it is aimed at.** Spot-VM preemption waves
    accumulate Spark `excludeOnFailure` state until a task retry has no
    schedulable executor left and Spark aborts the TaskSet ("cannot run anywhere
    due to node and executor excludeOnFailure"). During the sleep YARN replaces
    the killed containers within a couple of minutes, and — decisively — the
    per-taskset exclusion lists are per-JOB, so a NEW job starts with a clean
    slate regardless of how poisoned the aborted one's were. The sleeps are
    sized for that: 60s, 120s, 240s.

    Retries on ANY Spark JOB failure rather than pattern-matching the abort
    message on purpose: the phrasings that matter (excludeOnFailure aborts,
    `FetchFailed`, `ExecutorLostFailure`, "Killed by external signal") vary by
    Spark version and by which layer noticed first, and a missed pattern costs
    hours. Non-Spark exceptions (our own `ValueError`s, `KeyboardInterrupt`)
    propagate on the first raise.

    Three cases fail FAST instead, because a sleep cannot change them:
    `AnalysisException` (by type; `_spark_deterministic_failure_types`), the YARN
    bad-node / unhealthy-node cascade (by message; `_is_bad_node_failure`,
    matched narrowly so ordinary preemption keeps its full budget), and a
    SparkContext that stopped during the action (the application died; a rerun
    of the command is the recovery).

    `label` is prefixed to the log lines so a failure in a multi-hour solve says
    WHICH pass died.
    """
    import time

    # A retry only helps while the APPLICATION is alive: a job abort leaves the
    # SparkContext healthy, but when YARN fails the whole application (e.g. an
    # unmanaged-AM heartbeat timeout) the client backend stops the context, and
    # every further attempt fails at `broadcast` before any work is submitted.
    # Sleeping through 3 more attempts then only delays the exit — a rerun of the
    # same command is the actual recovery.
    # "Dead" is judged RELATIVE TO ENTRY: `sc.stop()` nulls
    # `_active_spark_context`, so "a context existed when the action started and
    # is now gone/stopped" is the app-death signature — while "no context ever"
    # (a unit test driving the helper with fakes) is not evidence of anything and
    # keeps the normal retry budget.
    try:
        from pyspark import SparkContext as _SC
        _entry_ctx = _SC._active_spark_context
    except Exception:                            # pragma: no cover - no pyspark
        _SC, _entry_ctx = None, None

    def _context_is_dead():
        if _SC is None or _entry_ctx is None:
            return False
        try:
            sc = _SC._active_spark_context
            return sc is None or sc._jsc is None or sc._jsc.sc().isStopped()
        except Exception:
            return False

    attempts = max(1, int(attempts))
    retryable = _spark_job_failure_types()
    deterministic = _spark_deterministic_failure_types()
    tag = f"{label}: " if label else ""
    for k in range(attempts):
        try:
            return fn()
        except retryable as exc:
            if deterministic and isinstance(exc, deterministic):
                # Decided on the driver before any task ran; identical every time.
                log.warning("%sspark action FAILED with a SCHEMA/ANALYSIS error, "
                            "which is DETERMINISTIC — no retry can fix it, so "
                            "failing fast: %s", tag, _error_first_line(exc))
                raise
            if _is_bad_node_failure(exc):
                # A full-disk node-health cascade, not preemption: a backoff
                # cannot heal it and a re-run would feed it.
                log.warning("%sspark action FAILED on a BAD NODE / node-unhealthy "
                            "signature (a DISK/health cascade, not spot "
                            "preemption). A backoff cannot heal a full disk and a "
                            "retry would feed the cascade, so failing fast: rerun "
                            "the command once YARN has recovered the nodes. "
                            "Original error: %s", tag, _error_first_line(exc))
                raise
            if _context_is_dead():
                log.warning("%sspark action FAILED and the SparkContext has STOPPED "
                            "(application-level failure — YARN failed or killed "
                            "the app). Retrying cannot help; rerun the same command "
                            "to resume from the last checkpoint. Original error: "
                            "%s", tag, _error_first_line(exc))
                raise
            if k + 1 >= attempts:
                log.warning("%sspark action FAILED on attempt %d/%d, giving up: %s",
                            tag, k + 1, attempts, _error_first_line(exc))
                raise
            sleep_s = float(base_sleep_s) * (2 ** k)
            log.warning("%sspark action FAILED on attempt %d/%d: %s",
                        tag, k + 1, attempts, _error_first_line(exc))
            log.warning("%sretrying in %.0fs (a fresh job resets per-taskset "
                        "excludeOnFailure state; YARN replaces preempted "
                        "containers meanwhile)", tag, sleep_s)
            time.sleep(sleep_s)
    raise AssertionError("unreachable")          # pragma: no cover


