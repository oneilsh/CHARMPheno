"""Broadcast cleanup shared by every Spark seam in spark_vi (ADR 0047).

One definition of `destroy_broadcast`, imported by the runner, the gate seeds, the
scalable spectral init, the coherence eval, the STM shim and the predictive-gain
eval (six copies before ADR 0050).
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def destroy_broadcast(bcast) -> None:
    """Fully release a broadcast: executor blocks, the driver block-manager copy,
    AND the driver-local pickled temp file.

    ``unpersist`` frees only the executor copies; the other two survive until
    ``destroy`` and accumulate at one pickle per call — at whole-Mondo scale a
    λ-sized (~355 MB) pickle per fit iteration filled the master's disk over a
    100-iteration fit and killed the run's post-fit readout with ENOSPC (exp 0104,
    2026-08-30). Cleanup must never outrank the work it follows, so failures
    degrade to unpersist and a log line rather than raising."""
    try:
        bcast.destroy()
    except Exception as exc:
        try:
            bcast.unpersist(blocking=False)
        except Exception:
            pass
        log.warning("broadcast destroy fell back to unpersist: %s", exc)
