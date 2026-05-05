"""Strict precondition check that an RDD/DataFrame is actually cached.

The motivating bug: `df.storageLevel` reports the *requested* persist level,
not whether blocks materialized. A "forgot the action after .persist()"
pattern looks fine to a storageLevel check while leaving the cache empty,
which then forces every iteration of a loop-heavy training job to re-run
the entire upstream lineage (e.g. a BigQuery scan). Multi-minute regression
per iter, silent.

The asymmetry argues for raising rather than logging: cost of a false
negative (silent re-execution) is huge; cost of a false positive (loud
failure on a healthy cache) is one fit attempt. We promote to a hard
precondition.

Spot-cluster note: cache state is point-in-time, not a permanent guarantee.
The check requires `numCachedPartitions > 0` (not `== numPartitions`) so
partial preemption loss between persist and fit-entry doesn't false-
positive — Spark transparently recomputes lost partitions on next access.
The check fires once at fit entry; mid-fit eviction is not detected here
(deferred — see project_strict_persist_check.md history).
"""
from __future__ import annotations

from pyspark import RDD
from pyspark.sql import DataFrame


def assert_persisted(target: RDD | DataFrame, name: str = "input") -> None:
    """Raise RuntimeError if `target` has no cached partitions in the block manager.

    Polymorphic over RDD and DataFrame:

    * RDD: queries `sc.getRDDStorageInfo()` keyed by `target.id()` and
      requires at least one cached partition. This is the rigorous truth
      check — block-manager state, not the requested storage level.
    * DataFrame: checks `target.storageLevel != NONE`. PySpark exposes no
      public way to verify materialized blocks for DataFrames; this catches
      "forgot persist" and "accidentally unpersisted" but not "registered
      but no blocks materialized." Adequate in practice — the missing-
      persist case is the common failure, and downstream (e.g. the
      VanillaLDAEstimator shim's bow_rdd persist+count) typically
      materializes the upstream cache via lineage anyway.

    `name` is included in the error message for actionable diagnostics
    (e.g. "persist(omop): ...").
    """
    if isinstance(target, RDD):
        _assert_rdd_persisted(target, name)
    elif isinstance(target, DataFrame):
        _assert_df_persisted(target, name)
    else:
        raise TypeError(
            f"assert_persisted: unsupported type {type(target).__name__} "
            "(expected pyspark.RDD or pyspark.sql.DataFrame)"
        )


def _assert_rdd_persisted(rdd: RDD, name: str) -> None:
    sl = rdd.getStorageLevel()
    if not (sl.useMemory or sl.useDisk):
        raise RuntimeError(
            f"persist({name}): no persist level set on RDD "
            f"(getStorageLevel() == NONE). Call .persist(...) before fit."
        )

    rdd_id = rdd.id()
    sc = rdd.context
    # `_jsc` is PySpark's private Java-bridge handle — we reach through it
    # because `getRDDStorageInfo()` (a public Scala method on SparkContext)
    # has no public Python equivalent. The data shape is stable across
    # Spark 3.x; only the access path is unblessed. If a Spark major-version
    # bump renames `_jsc` or `getRDDStorageInfo`, this line breaks loudly
    # (AttributeError at fit entry) rather than silently — so the failure
    # mode is itself a signal, not a hidden regression.
    infos = sc._jsc.sc().getRDDStorageInfo()
    for i in range(len(infos)):
        info = infos[i]
        if info.id() == rdd_id:
            cached = info.numCachedPartitions()
            total = info.numPartitions()
            if cached == 0:
                raise RuntimeError(
                    f"persist({name}): 0/{total} partitions cached at fit entry. "
                    f".persist() was called but no action materialized blocks. "
                    f"Trigger an action (e.g. .count()) after .persist()."
                )
            return

    raise RuntimeError(
        f"persist({name}): RDD id={rdd_id} not in Spark's storage registry. "
        f".persist() was not called, or no action has run since. "
        f"Trigger an action (e.g. .count()) after .persist()."
    )


def _assert_df_persisted(df: DataFrame, name: str) -> None:
    # `df.storageLevel` is itself implemented via the JVM cache manager
    # (Dataset.storageLevel reads from `sharedState.cacheManager.lookupCachedData`
    # and falls back to NONE), so checking `!= NONE` already covers
    # "forgot persist" and "accidentally unpersisted." A separate cache-
    # manager probe would be redundant — and would reach through `private[sql]`
    # internals — so we stick with the public surface.
    sl = df.storageLevel
    if not (sl.useMemory or sl.useDisk):
        raise RuntimeError(
            f"persist({name}): DataFrame has no persist level "
            f"(storageLevel == NONE). Call .persist(...) and an action "
            f"before fit."
        )
