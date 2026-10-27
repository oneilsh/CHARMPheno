"""Readout metric primitives that outlived the in-memory PC reference.

The faithful in-memory Prediction-Constrained topic model (ADR 0038) that used to
live here was removed with the supervised head (ADR 0050); it stays recoverable
from git (commit fff7bf4067). One module remains:

  * :mod:`analysis.pc.evaluate` — per-label metric primitives (`_score_label`,
    `_macro`, the per-label logistic baselines, `_bundle_masked`) the readout
    drivers import. They move to `spark_vi.eval.metrics` with ADR 0051 step 4.

The batched multi-head L-BFGS solver is `spark_vi.readout.solver` (ADR 0051).
"""
