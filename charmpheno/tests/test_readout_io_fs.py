"""spark_vi.io.fs: local paths behave as before; any other scheme goes through the
SparkContext's Hadoop FileSystem (exercised here by forcing the Hadoop branch on a
file:// URI, which Hadoop's LocalFileSystem serves)."""
import os

import numpy as np
import pytest

from spark_vi.io import fs


@pytest.fixture()
def hadoop_branch(monkeypatch):
    """Route every URI through the Hadoop backend."""
    monkeypatch.setattr(fs, "is_local", lambda uri: False)


def test_heads_file_through_the_hadoop_branch(spark, tmp_path, hadoop_branch):
    from charmpheno.readout.heads import read_heads, write_heads
    C, K = 3, 4
    V = np.arange(C * K, dtype=float).reshape(C, K)
    uri = "file://" + str(tmp_path / "run")
    assert write_heads(uri, "arm", V, np.zeros(C), np.zeros(C), np.zeros(C, bool), C, K, 0)
    got = read_heads(uri, "arm", C=C, K=K, theta_topm=0)
    np.testing.assert_array_equal(got[0], V)
