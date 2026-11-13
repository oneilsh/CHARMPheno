"""Filesystem access by URI: plain Python for local paths, Hadoop for everything else.

Every file spark-vi writes or reads (result checkpoints, readout checkpoints and
heads, model saves) goes through here, so a path may be a local path, a `file://`
URI, or any Hadoop-filesystem URI the running SparkContext can reach (`hdfs://`,
`gs://`, `s3a://`, ...), as MLlib's own writers allow.

Dispatch by scheme:

* **no scheme, or `file://`** -> plain Python I/O (`open`, `os.replace`). This keeps
  local behaviour exactly as it was, including on a FUSE-mounted object store (a
  gcsfuse run dir is a local path), and avoids Hadoop's checksumming
  `LocalFileSystem`, which would leave a `.<name>.crc` beside every file.
* **any other scheme** -> the active SparkContext's Hadoop `FileSystem` through py4j
  (no new dependency). Writes go through a byte buffer in chunks; reads load the whole
  file into memory (np.load needs a seekable file; every object written here is
  driver-sized already).

A URI has a scheme when it starts `<scheme>:/` (RFC 3986 scheme characters, then a
colon and a slash), so Hadoop's single-slash forms (`hdfs:/x`, `gs:/b/x`) are remote
like `hdfs://nn/x`; anything else, including a relative path with a colon in it, is a
plain local path. `file:` URIs are stripped by hand, not parsed, so a `#` or `?` in
the path survives.

Atomicity: `write_bytes(..., atomic=True)` writes a sibling temp then renames on a
local path. A non-local write overwrites in place with no rename. On an object store
(`gs://`, `s3a://`) an object becomes visible only when its upload completes, so a
single object is never torn (a "rename" there is a copy). On HDFS that does NOT hold:
`create(overwrite=True)` truncates first, so a crash mid-write leaves a torn file.
Multi-file writers therefore never overwrite their data files in place: they write a
fresh set and then the small file that names it (a manifest) last
(`spark_vi.io.export.save_result`).

Never wrap a URI in `pathlib.Path`: it collapses `gs://b/x` to `gs:/b/x`. Use `join`.
"""
from __future__ import annotations

import io
import os
import re

__all__ = ["is_local", "local_path", "qualify", "join", "exists", "makedirs",
           "write_bytes", "read_bytes", "write_text", "read_text", "remove",
           "remove_tree", "listdir"]

_CHUNK = 16 * 1024 * 1024


# `<scheme>:/...`: RFC 3986 scheme characters, a colon, then a slash. Requiring the
# slash keeps a relative local path with a colon in it (`a:b`) local, and takes both
# Hadoop forms of a URI: `hdfs://nn/x` and the single-slash `hdfs:/x` / `file:/x`.
_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):/")


def _scheme(uri) -> str:
    m = _SCHEME_RE.match(str(uri))
    return m.group(1).lower() if m else ""


def is_local(uri) -> bool:
    """True for a plain path or a `file://` URI (handled with plain Python I/O)."""
    return _scheme(uri) in ("", "file")


def local_path(uri) -> str:
    """The filesystem path of a local URI (`file:///x` and `file:/x` -> `/x`).

    Stripped by hand, not with `urlparse`, which would cut the path at a `#` or `?`
    (fragment / query) and so read or write a different file. `file://host/x` (an
    authority other than empty or `localhost`) is not a local path and is refused."""
    s = str(uri)
    if _scheme(s) != "file":
        return s
    rest = s[len("file:"):]
    if rest.startswith("//"):
        auth, sep, path = rest[2:].partition("/")
        if auth not in ("", "localhost"):
            raise ValueError(f"{uri}: a file: URI with a host is not a local path")
        return "/" + path
    return rest


def qualify(uri) -> str:
    """`uri` as Hadoop resolves it, for writers that mix Spark / Hadoop I/O with ours.

    A path with no scheme means different things to the two: to the functions here
    it is local, to Spark (``saveAsTextFile``, ``DataFrameWriter``, MLlib's
    ``DefaultParamsWriter``) it resolves against ``fs.defaultFS``, which is HDFS on a
    cluster. A writer that does both (MLlib persistence: Params metadata through
    Spark, data files through here) qualifies the path once with this, MLlib's rule,
    and hands the result to both: ``/x`` -> ``file:/x`` locally, ``hdfs://nn/x`` on
    a cluster. A URI with a scheme, or no active SparkContext, comes back as is."""
    s = str(uri)
    if _scheme(s):
        return s
    from pyspark import SparkContext
    sc = SparkContext._active_spark_context
    if sc is None:
        return s
    path = sc._jvm.org.apache.hadoop.fs.Path(s)
    fsys = path.getFileSystem(sc._jsc.hadoopConfiguration())
    return str(fsys.makeQualified(path).toString())


def join(base, *parts) -> str:
    """`base/part/...` without collapsing a URI's `//`."""
    s = str(base).rstrip("/")
    for p in parts:
        s = f"{s}/{str(p).strip('/')}"
    return s


# --------------------------------------------------------------------------- #
# Hadoop backend                                                               #
# --------------------------------------------------------------------------- #
def _hadoop(uri):
    """`(FileSystem, Path)` for `uri` from the active SparkContext."""
    from pyspark import SparkContext
    sc = SparkContext._active_spark_context
    if sc is None:
        raise RuntimeError(f"{uri}: a non-local URI needs an active SparkContext")
    jvm = sc._jvm
    path = jvm.org.apache.hadoop.fs.Path(str(uri))
    return path.getFileSystem(sc._jsc.hadoopConfiguration()), path


def _h_write(uri, data: bytes):
    fs, path = _hadoop(uri)
    out = fs.create(path, True)
    try:
        mv = memoryview(data)
        for i in range(0, len(mv), _CHUNK):
            out.write(bytearray(mv[i:i + _CHUNK]))
    finally:
        out.close()


def _h_read(uri) -> bytes:
    from pyspark import SparkContext
    fs, path = _hadoop(uri)
    stream = fs.open(path)
    try:
        jvm = SparkContext._active_spark_context._jvm
        return bytes(jvm.org.apache.commons.io.IOUtils.toByteArray(stream))
    finally:
        stream.close()


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def exists(uri) -> bool:
    if is_local(uri):
        return os.path.exists(local_path(uri))
    fs, path = _hadoop(uri)
    return bool(fs.exists(path))


def makedirs(uri) -> None:
    if is_local(uri):
        os.makedirs(local_path(uri), exist_ok=True)
    else:
        fs, path = _hadoop(uri)
        fs.mkdirs(path)


def write_bytes(uri, data: bytes, *, atomic: bool = True) -> None:
    """Write `data` to `uri`, creating parent directories. On a local path with
    `atomic`, through a sibling `.tmp` and `os.replace`, so readers never see a
    partial file."""
    if is_local(uri):
        p = local_path(uri)
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not atomic:
            with open(p, "wb") as f:
                f.write(data)
            return
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
        return
    _h_write(uri, data)


def read_bytes(uri) -> bytes:
    if is_local(uri):
        with open(local_path(uri), "rb") as f:
            return f.read()
    return _h_read(uri)


def write_text(uri, text: str, *, atomic: bool = True) -> None:
    write_bytes(uri, text.encode("utf-8"), atomic=atomic)


def read_text(uri) -> str:
    return read_bytes(uri).decode("utf-8")


def remove(uri) -> None:
    """Delete a file if present (best-effort no-op when absent)."""
    if is_local(uri):
        try:
            os.remove(local_path(uri))
        except FileNotFoundError:
            pass
        return
    fs, path = _hadoop(uri)
    if fs.exists(path):
        fs.delete(path, False)


def remove_tree(uri) -> None:
    """Delete a directory and everything under it, if present (no-op when absent)."""
    if is_local(uri):
        import shutil
        shutil.rmtree(local_path(uri), ignore_errors=True)
        return
    fs, path = _hadoop(uri)
    if fs.exists(path):
        fs.delete(path, True)


def listdir(uri) -> list:
    """Names (not paths) directly under `uri`; [] when it does not exist."""
    if is_local(uri):
        p = local_path(uri)
        return sorted(os.listdir(p)) if os.path.isdir(p) else []
    fs, path = _hadoop(uri)
    if not fs.exists(path):
        return []
    return sorted(str(st.getPath().getName()) for st in fs.listStatus(path))


def npz_bytes(**arrays) -> bytes:
    """`np.savez` into memory (the caller writes the bytes where it wants)."""
    import numpy as np
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()
