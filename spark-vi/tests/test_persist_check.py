"""Tests for spark_vi.diagnostics.persist.assert_persisted.

Pin down the four states the strict precondition cares about:

  RDD case:
    1. Persisted + materialized (action ran) → passes silently.
    2. Persisted but no action → raises (caught via storage registry empty).
    3. No persist call at all → raises (no persist level).

  DataFrame case:
    4. Persisted + materialized → passes silently.
    5. No persist call → raises.

Spot-cluster nuance ("partial loss after preemption stays >0 cached") is
not exercised here — would require simulating executor death, which the
local-mode test fixture cannot do. The check's `>0` rather than
`==numPartitions` is documented in the module docstring.
"""
from __future__ import annotations

import pytest

from spark_vi.diagnostics.persist import assert_persisted


def test_rdd_persisted_and_materialized_passes(spark):
    rdd = spark.sparkContext.parallelize([1, 2, 3, 4], numSlices=2).persist()
    rdd.count()  # materialize
    assert_persisted(rdd, name="cached_rdd")  # should not raise


def test_rdd_with_persist_but_no_action_raises(spark):
    """`.persist()` without an action: the RDD has a storage level set but
    no blocks are in the manager. The check looks up by RDD id in
    getRDDStorageInfo, which only includes RDDs with materialized blocks,
    so the lookup misses and we raise."""
    rdd = spark.sparkContext.parallelize([1, 2, 3, 4], numSlices=2).persist()
    # No action triggered.
    with pytest.raises(RuntimeError, match=r"persist\(forgot_action\)"):
        assert_persisted(rdd, name="forgot_action")


def test_rdd_without_persist_call_raises(spark):
    rdd = spark.sparkContext.parallelize([1, 2, 3, 4], numSlices=2)
    with pytest.raises(RuntimeError, match=r"no persist level set"):
        assert_persisted(rdd, name="never_persisted")


def test_dataframe_persisted_and_materialized_passes(spark):
    df = spark.createDataFrame([(1,), (2,), (3,)], schema=["x"]).persist()
    df.count()  # materialize
    assert_persisted(df, name="cached_df")  # should not raise


def test_dataframe_without_persist_call_raises(spark):
    df = spark.createDataFrame([(1,), (2,), (3,)], schema=["x"])
    with pytest.raises(RuntimeError, match=r"no persist level"):
        assert_persisted(df, name="never_persisted_df")


def test_unsupported_type_raises_typeerror():
    with pytest.raises(TypeError, match="unsupported type"):
        assert_persisted([1, 2, 3], name="not_a_spark_thing")
