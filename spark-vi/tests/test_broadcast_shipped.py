"""`spark_vi.core.broadcast.ship_array` / `load_shipped`: a large array reaches the
executors as a memory-mapped file, bit-identical to the driver's."""
import numpy as np

from spark_vi.core.broadcast import load_shipped, ship_array


def test_shipped_array_is_bit_identical_and_read_only_on_executors(spark):
    sc = spark.sparkContext
    rng = np.random.default_rng(0)
    arr = rng.random((7, 11))
    token = ship_array(sc, arr, "probe")

    def _check(it):
        a = load_shipped(token)
        yield (a.shape, a.dtype.str, a.flags.writeable, a.tobytes())

    got = sc.parallelize(range(4), 4).mapPartitions(_check).collect()
    assert len(got) == 4
    for shape, dt, writeable, raw in got:
        assert shape == arr.shape and dt == arr.dtype.str and not writeable
        assert raw == arr.tobytes()


def test_each_call_ships_its_own_file(spark):
    sc = spark.sparkContext
    t1 = ship_array(sc, np.zeros(3), "x")
    t2 = ship_array(sc, np.ones(3), "x")
    assert t1 != t2
    vals = sc.parallelize([0], 1).map(
        lambda _: (float(load_shipped(t1).sum()), float(load_shipped(t2).sum()))
    ).collect()
    assert vals == [(0.0, 3.0)]
