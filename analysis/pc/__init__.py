"""Readout metric primitives that outlived the in-memory PC reference.

The faithful in-memory Prediction-Constrained topic model (ADR 0038) that used to
live here was removed with the supervised head (ADR 0050); it stays recoverable
from git (commit fff7bf4067). Two modules remain, both live:

  * :mod:`analysis.pc.evaluate` — per-label metric primitives (`_score_label`,
    `_macro`, the per-label logistic baselines, `_bundle_masked`) the readout
    drivers import;
  * :mod:`analysis.pc.batched_lr` — a re-export of
    `spark_vi.models.topic.batched_lr`, the batched multi-head L-BFGS solver of
    the post-hoc readout (where it lives is ADR 0051's business).
"""
