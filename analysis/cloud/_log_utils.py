"""Driver-side log4j helpers for cloud Spark jobs.

quiet_spot_reclamation downgrades the loggers that flood the console when a
spot/preemptible YARN executor disappears — block-replica cascades, scheduler
"lost executor" messages, and the FetchFailedException stack traces emitted
by TaskSetManager. Without this, a single spot reclamation can dump dozens
of kilobytes of stack trace per affected task into stdout.

The targeted approach (instead of a blanket setLogLevel("ERROR")) preserves
WARN-level messages from the BQ connector, our own driver, and codepaths
where a warning still represents something worth seeing. Real executor
problems still surface at ERROR level ("Task N in stage M failed M times;
aborting job") which we do not suppress.

Works against both log4j 1 (Spark <= 3.2) and log4j 2 (Spark 3.3+); tries
the log4j 1 API first since the Dataproc image historically exposes it as
a compatibility shim, falls back to log4j 2 LogManager if absent.
"""
from __future__ import annotations


_NOISY_LOGGERS_ERROR = (
    # "No more replicas available for rdd_..." cascade when executor dies.
    "org.apache.spark.storage.BlockManagerMasterEndpoint",
    # "Lost executor N on ...: Container released on a *lost* node"
    "org.apache.spark.scheduler.YarnScheduler",
    "org.apache.spark.scheduler.cluster.YarnSchedulerBackend",
    # Shuffle/block fetch retry chatter while the node is unreachable.
    "org.apache.spark.shuffle.RetryingBlockFetcher",
    "org.apache.spark.network.shuffle.RetryingBlockTransferor",
    "org.apache.spark.network.client.TransportClient",
    "org.apache.spark.network.client.TransportClientFactory",
    "org.apache.spark.network.server.TransportRequestHandler",
)

# TaskSetManager emits FetchFailed stack traces at WARN; bumping to ERROR
# keeps the "Task N in stage M failed M times" aggregate but suppresses the
# per-task multi-kilobyte stack trace. The stage-failed ERROR is logged by
# DAGScheduler, which we leave at WARN.
_NOISY_LOGGERS_FATAL = (
    "org.apache.spark.scheduler.TaskSetManager",
)


def _set_log4j1(spark, name, level_name) -> bool:
    """Try setting via log4j 1 API. Returns True on success."""
    try:
        log4j = spark._jvm.org.apache.log4j
        level = getattr(log4j.Level, level_name)
        log4j.Logger.getLogger(name).setLevel(level)
        return True
    except Exception:
        return False


def _set_log4j2(spark, name, level_name) -> bool:
    """Try setting via log4j 2 API. Returns True on success."""
    try:
        log_mgr = spark._jvm.org.apache.logging.log4j.LogManager
        level_cls = spark._jvm.org.apache.logging.log4j.Level
        level = level_cls.valueOf(level_name)
        logger = log_mgr.getLogger(name)
        # log4j 2: Configurator.setLevel for runtime override.
        configurator = spark._jvm.org.apache.logging.log4j.core.config.Configurator
        configurator.setLevel(name, level)
        return True
    except Exception:
        return False


def quiet_spot_reclamation(spark) -> None:
    """Silence the loggers that flood the console on spot executor death.

    Best-effort: any logger that can't be set on this Spark/log4j combo is
    skipped silently. The set is hand-picked from Spark 3.x source for the
    classes that emit the cascades described in the module docstring.
    """
    for name in _NOISY_LOGGERS_ERROR:
        _set_log4j1(spark, name, "ERROR") or _set_log4j2(spark, name, "ERROR")
    for name in _NOISY_LOGGERS_FATAL:
        _set_log4j1(spark, name, "FATAL") or _set_log4j2(spark, name, "FATAL")
