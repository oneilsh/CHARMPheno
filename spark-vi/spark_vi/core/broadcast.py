"""Broadcast cleanup shared by every Spark seam in spark_vi (ADR 0047).

One definition of `destroy_broadcast`, imported by the runner, the gate seeds, the
scalable spectral init, the coherence eval, the STM shim and the predictive-gain
eval.
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
