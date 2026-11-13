"""spark_vi.io.fs: local paths behave as before; any other scheme goes through the
SparkContext's Hadoop FileSystem (exercised here by forcing the Hadoop branch on a
file:// URI, which Hadoop's LocalFileSystem serves)."""
import os

import numpy as np
import pytest

from spark_vi.io import fs


def test_scheme_dispatch_and_join():
    assert fs.is_local("/tmp/x") and fs.is_local("rel/x") and fs.is_local("file:///tmp/x")
    assert not fs.is_local("gs://b/x") and not fs.is_local("hdfs://nn/x")
    assert fs.local_path("file:///tmp/x") == "/tmp/x"
    assert fs.join("gs://b/run", "params", "a.npy") == "gs://b/run/params/a.npy"
    # Hadoop's qualified local form (one slash) is local too
    assert fs.is_local("file:/tmp/x") and fs.local_path("file:/tmp/x") == "/tmp/x"


def test_single_slash_hadoop_uris_are_remote():
    """Code review B5: `hdfs:/x` and `gs:/b/x` (Hadoop's single-slash forms) carry a
    scheme; a relative path with a colon, or a bare name, does not."""
    for uri in ("hdfs:/x", "gs:/b/x", "s3a://b/k", "HDFS://nn/x", "viewfs:/a"):
        assert not fs.is_local(uri), uri
    for uri in ("/tmp/x", "rel/x", "a:b", "run:1/x", "x", "./c:/d"):
        assert fs.is_local(uri), uri
        assert fs.local_path(uri) == uri


def test_file_uris_keep_hash_and_query_characters():
    """Code review B5: `#` / `?` are path characters here, not a fragment / query."""
    assert fs.local_path("file:///tmp/a#1/b?c") == "/tmp/a#1/b?c"
    assert fs.local_path("file:/tmp/run#2") == "/tmp/run#2"
    assert fs.local_path("file://localhost/tmp/x") == "/tmp/x"
    assert fs.local_path("FILE:///tmp/x") == "/tmp/x"
    with pytest.raises(ValueError):
        fs.local_path("file://otherhost/tmp/x")


def test_local_roundtrip_through_a_file_uri_with_a_hash(tmp_path):
    uri = "file://" + str(tmp_path / "run#3" / "f.bin")
    fs.write_bytes(uri, b"abc")
    assert (tmp_path / "run#3" / "f.bin").read_bytes() == b"abc"
    assert fs.read_bytes(uri) == b"abc"


def test_qualify_without_spark_leaves_the_uri():
    from pyspark import SparkContext
    if SparkContext._active_spark_context is not None:
        pytest.skip("a SparkContext is active")
    assert fs.qualify("/tmp/x") == "/tmp/x"
    assert fs.qualify("gs://b/x") == "gs://b/x"


def test_qualify_resolves_against_the_default_fs(spark, tmp_path):
    """A path with no scheme resolves as Spark resolves it (fs.defaultFS, file: on
    local Spark), and the result is still handled by the local branch."""
    q = fs.qualify(str(tmp_path / "a"))
    assert q == f"file:{tmp_path}/a" and fs.is_local(q)
    assert fs.qualify("gs://b/x") == "gs://b/x"
    assert fs.qualify("file:///tmp/x") == "file:///tmp/x"


def test_remove_tree(tmp_path):
    (tmp_path / "d" / "e").mkdir(parents=True)
    (tmp_path / "d" / "e" / "f").write_text("x")
    fs.remove_tree("file://" + str(tmp_path / "d"))
    assert not (tmp_path / "d").exists()
    fs.remove_tree(str(tmp_path / "d"))          # absent: a no-op


def test_local_roundtrip_is_atomic_and_leaves_no_tmp(tmp_path):
    p = str(tmp_path / "d" / "f.bin")
    fs.write_bytes(p, b"abc")
    assert fs.read_bytes(p) == b"abc" and fs.exists(p)
    assert os.listdir(tmp_path / "d") == ["f.bin"]
    fs.write_text("file://" + str(tmp_path / "t.txt"), "hi")
    assert fs.read_text(str(tmp_path / "t.txt")) == "hi"
    fs.remove(p)
    assert not fs.exists(p)
    fs.remove(p)                                  # absent: a no-op


@pytest.fixture()
def hadoop_branch(monkeypatch):
    """Route every URI through the Hadoop backend."""
    monkeypatch.setattr(fs, "is_local", lambda uri: False)


def test_hadoop_branch_remove_tree(spark, tmp_path, hadoop_branch):
    (tmp_path / "d" / "e").mkdir(parents=True)
    (tmp_path / "d" / "e" / "f").write_text("x")
    fs.remove_tree("file://" + str(tmp_path / "d"))
    assert not (tmp_path / "d").exists()


def test_hadoop_branch_roundtrip(spark, tmp_path, hadoop_branch):
    uri = "file://" + str(tmp_path / "h" / "f.bin")
    data = bytes(range(256)) * 1000
    fs.write_bytes(uri, data)
    assert fs.exists(uri) and fs.read_bytes(uri) == data
    assert "f.bin" in fs.listdir("file://" + str(tmp_path / "h"))
    fs.remove(uri)
    assert not fs.exists(uri)


def test_save_and_load_result_through_the_hadoop_branch(spark, tmp_path, hadoop_branch):
    from spark_vi.core.result import VIResult
    from spark_vi.io.export import load_result, save_result
    r = VIResult(global_params={"lambda": {0: np.ones((2, 3)), 1: np.zeros((2, 2))},
                                "alpha": np.array([0.5, 0.5])},
                 elbo_trace=[-2.0, -1.0], n_iterations=2, converged=False,
                 diagnostic_traces={"a": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]})
    uri = "file://" + str(tmp_path / "res")
    save_result(r, uri)
    back = load_result(uri)
    np.testing.assert_array_equal(back.global_params["lambda"][0], np.ones((2, 3)))
    np.testing.assert_array_equal(back.diagnostic_traces["a"][1], [3.0, 4.0])
    assert back.n_iterations == 2


def test_heads_file_through_the_hadoop_branch(spark, tmp_path, hadoop_branch):
    from spark_vi.readout.heads import read_heads, write_heads
    C, K = 3, 4
    V = np.arange(C * K, dtype=float).reshape(C, K)
    uri = "file://" + str(tmp_path / "run")
    assert write_heads(uri, "arm", V, np.zeros(C), np.zeros(C), np.zeros(C, bool), C, K, 0)
    got = read_heads(uri, "arm", C=C, K=K, theta_topm=0)
    np.testing.assert_array_equal(got[0], V)
