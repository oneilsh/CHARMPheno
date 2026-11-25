"""Broadcast cleanup shared by every Spark seam in spark_vi (ADR 0047), and shipping
large read-only arrays to executors as memory-mapped files.

One definition of `destroy_broadcast`, imported by the runner, the gate seeds, the
scalable spectral init, the coherence eval, the STM shim and the predictive-gain
eval. `ship_array` / `load_shipped` replace a broadcast for an array too large to hold
once per Python worker (the DAG-gated model's scoring table).
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def destroy_broadcast(bcast) -> None:
    """Fully release a broadcast: executor blocks, the driver block-manager copy,
    AND the driver-local pickled temp file.

    ``unpersist`` frees only the executor copies; the other two survive until
    ``destroy`` and accumulate at one pickle per call — at K~3,800 x V~11,600 a
    λ-sized (~355 MB) pickle per fit iteration, enough to fill the master's disk
    over a long fit. Cleanup must never outrank the work it follows, so failures
    degrade to unpersist and a log line rather than raising."""
    try:
        bcast.destroy()
    except Exception as exc:
        try:
            bcast.unpersist(blocking=False)
        except Exception:
            pass
        log.warning("broadcast destroy fell back to unpersist: %s", exc)


# --- Large read-only arrays shipped as memory-mapped files ----------------------------

_SHIPPED_DIR = None
_MAPPED: dict = {}


def ship_array(sc, arr, stem: str) -> str:
    """Ship a large read-only array to the executors as an ``.npy`` file; return the
    token a task passes to ``load_shipped``.

    Why not ``sc.broadcast``: PySpark unpickles a broadcast separately in EVERY Python
    worker process, so an executor with c cores holds c copies (a DAG-gated
    E[log β] is ~400 MB at K ≈ 3,250 over ~15,500 terms, so ~1.6 GB on a 4-core
    executor), and a pickled broadcast is one object, capped near 2 GB, which a
    larger user DAG can reach. A file registered with ``sc.addFile`` is copied once
    per executor; ``load_shipped`` memory-maps it read-only, so every worker on the
    executor reads the same OS page-cache pages: one copy per executor, nothing
    pickled, values bit-identical (same dtype, same layout).

    The driver's file lives in a process-lifetime temp directory removed at exit;
    the executors' copies live in Spark's per-application work directory and go
    with the application. Each call ships a new file (uuid-named: ``addFile``
    refuses a second, different file under one name), so a session that scores many
    times keeps one file per call on executor disk until it ends."""
    global _SHIPPED_DIR
    import atexit
    import shutil
    import tempfile
    import uuid
    from pathlib import Path

    import numpy as np

    if _SHIPPED_DIR is None:
        _SHIPPED_DIR = tempfile.mkdtemp(prefix="spark_vi_shipped_")
        atexit.register(shutil.rmtree, _SHIPPED_DIR, True)
    token = f"{stem}-{uuid.uuid4().hex}.npy"
    path = Path(_SHIPPED_DIR) / token
    np.save(path, np.ascontiguousarray(arr), allow_pickle=False)
    sc.addFile(str(path))
    return token


def load_shipped(token: str):
    """The array ``ship_array`` shipped under ``token``, memory-mapped read-only (see
    there for why). Cached per worker process, so a task pays the open once."""
    arr = _MAPPED.get(token)
    if arr is None:
        import numpy as np
        from pyspark import SparkFiles
        arr = np.load(SparkFiles.get(token), mmap_mode="r", allow_pickle=False)
        _MAPPED[token] = arr
    return arr
