"""Per-node readout heads over ANY frozen feature column (ADR 0051).

Given a vector column per row (topic proportions θ, raw-code presence, any
representation), a multi-label target `label (C,)` with an observation mask
`labelMask (C,)`, the readout fits C independent L2 logistic heads in one batched
L-BFGS (`heads.fit_heads`, `solver.solve_batched_lr`) and scores rows with them;
`stacked` forms the closure product over a label DAG. Two backends share every
numerical kernel (`kernels`): `spark.SparkReadoutData` (a pass is a treeAggregate)
and `local.LocalReadoutData` (a pass is a numpy loop over rows held in memory).

Depends on `spark_vi.core` only; never imports `spark_vi.models` (the features are
an input, not a model). Metrics over the scored probabilities live with the
evaluation code, not here.
"""
from spark_vi.readout.heads import fit_heads, read_heads, write_heads
from spark_vi.readout.local import LocalReadoutData
from spark_vi.readout.spark import SparkReadoutData, collect_lean_proba
from spark_vi.readout.stacked import closure_matrix, stacked_proba

__all__ = ["LocalReadoutData", "SparkReadoutData", "closure_matrix",
           "collect_lean_proba", "fit_heads", "read_heads", "stacked_proba",
           "write_heads"]
