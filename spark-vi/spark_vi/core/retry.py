"""Retry a Spark ACTION across spot-VM preemption waves (driver side).

A multi-hour sequence of Spark actions (the readout's L-BFGS makes hundreds of
data passes) on a preemptible cluster sees jobs die for reasons that have nothing
to do with the work: a preemption wave starves the scheduler and Spark aborts the
job. `retry_spark_action` re-runs an idempotent action a bounded number of times
with backoff, and fails FAST on the failures a retry can never fix (an analysis
error in the plan, a dead SparkContext). Moved from the readout driver
(`analysis/cloud/distributed_readout.py`) with ADR 0051; nothing here is
readout-specific.
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
    is importable on a worker that has no gateway. If py4j is somehow absent the tuple comes back EMPTY, which
    makes `except ():` match nothing — a missing dependency degrades to "no
    retry", never to "retry every exception, including our own bugs".

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
    waste and the eventual re-raise is the same exception. Exp 0110 paid for
    this: a `collect_lean_proba(..., score_col="probability")` against an
    UNSUPERVISED transform (no such column) burned 60+120+240s of backoff before
    surfacing the schema error that was true from the first millisecond.

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

    The failure this names (exp 0110): a full worker local disk makes YARN mark
    the NodeManager UNHEALTHY and SIGTERM its containers, which surface to the
    driver as `ExecutorLostFailure ... Container from a bad node` (exit 143).
    Unlike a spot-preemption `ExecutorLostFailure` — which self-heals on the 60s
    backoff because YARN replaces the container and a fresh job resets the
    per-taskset exclude lists — this one does NOT self-heal: the disk is still
    full, and re-running a shuffle-heavy pass into a disk-pressured cluster is
    what turned one bad node into ~12 dead ones over an hour. So it must fail
    fast (like the schema and dead-context cases), not burn 60+120+240s of
    backoff feeding the cascade.

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

    **Why this is sound here, and not a general "retry everything" wrapper.**
    Every kernel in this module is a PURE FUNCTION of a persisted RDD: partitions
    lost to a preemption recompute from lineage, and the aggregate is a sum over
    partitions, so re-running the action recomputes exactly the same number. A
    retried action is therefore idempotent — not "probably fine", but identical
    by construction. (The one thing a retry must NOT reuse is a broadcast whose
    blocks may have died with the executors, which is why the call sites below
    build their broadcast INSIDE the retried closure.)

    **Why a retry actually fixes the failure it is aimed at** (exp 0104,
    2026-08-28): spot-VM preemption waves accumulate Spark `excludeOnFailure`
    state until a task retry has no schedulable executor left and Spark aborts
    the TaskSet — "task 0 (partition 0) cannot run anywhere due to node and
    executor excludeOnFailure". Two independent clocks then work in our favour
    during the sleep: YARN replaces the killed containers within a couple of
    minutes, and — decisively — the per-taskset exclusion lists are per-JOB, so a
    NEW job starts with a clean slate regardless of how poisoned the aborted
    one's were. (App-level node exclusions age out on
    `spark.excludeOnFailure.timeout`, which is why this run's doc pins it to
    10m instead of the 1h default.) The sleeps are sized for that: 60s, 120s,
    240s.

    Retries on ANY Spark JOB failure rather than pattern-matching the abort
    message on purpose. The phrasings that matter here (excludeOnFailure aborts,
    `FetchFailed`, `ExecutorLostFailure`, "Killed by external signal") vary by
    Spark version and by which layer noticed first, and a missed pattern costs
    hours of solve. Non-Spark exceptions (our own `ValueError`s,
    `KeyboardInterrupt`) propagate on the first raise.

    The ONE message pattern that is excluded is the YARN node-health / bad-node
    cascade (`_is_bad_node_failure`): a full worker local disk marks the node
    UNHEALTHY and its containers surface as "Container from a bad node". That is
    the OPPOSITE of spot preemption — it does NOT self-heal on a backoff (the
    disk is still full), and re-running a shuffle-heavy pass into the
    disk-pressured cluster is what cascaded ~12 nodes dead on exp 0110. It fails
    fast. The match is deliberately narrow (two high-precision needles, "lost
    node" NOT among them) so ordinary preemption keeps its full retry budget; see
    `_is_bad_node_failure`.

    The ONE Spark failure that is excluded by TYPE is `AnalysisException` (see
    `_spark_deterministic_failure_types`): it is decided during query analysis on
    the driver, before any task is submitted, so it cannot become a success on
    attempt 2 — the sleeps are pure waste and the log lines actively mislead by
    dressing a schema bug as cluster trouble. Exp 0110 spent 60+120+240s that way
    on a column that did not exist. Every other Spark job failure keeps the full
    retry budget.

    `label` is prefixed to the log lines so a failure in a multi-hour solve says
    WHICH pass died.
    """
    import time

    # A retry only helps while the APPLICATION is alive: a job abort leaves the
    # SparkContext healthy, but when YARN fails the whole application (exp 0104,
    # 2026-08-29: unmanaged-AM heartbeat timeout) the client backend stops the
    # context, and every further attempt fails at `broadcast` before any work is
    # even submitted. Sleeping through 3 more attempts then only delays the exit
    # the checkpoint needs — a rerun of the same command is the actual recovery.
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
                # Schema/analysis error: decided on the driver before any task
                # ran, and identical on every attempt. Fail on the first raise
                # instead of sleeping 60+120+240s to re-learn it (exp 0110).
                log.warning(f"{tag}spark action FAILED with a SCHEMA/ANALYSIS "
                      "error, which is DETERMINISTIC — no retry can fix it, so "
                      f"failing fast: {_error_first_line(exc)}")
                raise
            if _is_bad_node_failure(exc):
                # A YARN node-health / bad-node cascade (a full local disk marking
                # the node unhealthy), NOT spot preemption. It does not self-heal
                # on a backoff — the disk is still full — and re-running this
                # shuffle-heavy pass feeds the cascade (exp 0110: ~12 nodes dead
                # over an hour). Fail fast like the dead-context case; the recovery
                # is to rerun the command once the cluster is healthy again.
                log.warning(f"{tag}spark action FAILED on a BAD NODE / "
                      "node-unhealthy signature (a DISK/health cascade, not spot "
                      "preemption). A backoff cannot heal a full disk and a retry "
                      "would feed the cascade, so failing fast: rerun the command "
                      "once YARN has recovered the nodes. Original error: "
                      f"{_error_first_line(exc)}")
                raise
            if _context_is_dead():
                log.warning(f"{tag}spark action FAILED and the SparkContext "
                      "has STOPPED (application-level failure — YARN failed or "
                      "killed the app). Retrying cannot help; rerun the same "
                      "command to resume from the last checkpoint. Original "
                      f"error: {_error_first_line(exc)}")
                raise
            if k + 1 >= attempts:
                log.warning(f"{tag}spark action FAILED on attempt "
                      f"{k + 1}/{attempts}, giving up: {_error_first_line(exc)}")
                raise
            sleep_s = float(base_sleep_s) * (2 ** k)
            log.warning(f"{tag}spark action FAILED on attempt "
                  f"{k + 1}/{attempts}: {_error_first_line(exc)}")
            log.warning(f"{tag}retrying in {sleep_s:.0f}s (a fresh job resets "
                  "per-taskset excludeOnFailure state; YARN replaces preempted "
                  "containers meanwhile)")
            time.sleep(sleep_s)
    raise AssertionError("unreachable")          # pragma: no cover


