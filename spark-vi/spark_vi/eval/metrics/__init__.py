"""Exact, model-agnostic ranking metrics (ADR 0051 §3): AUC, AP, precision@recall,
recall@FDR, top-k screening, ROC/PR curves and binned calibration.

`core` defines every metric on a TALLY of one scored column (distinct scores with
positive / negative counts); tallies merge, so the `spark` backend computes the same
numbers as the in-memory `local` one without collecting cells to the driver. `report`
turns per-column metrics into the readout's records (the skip rules) and macros.

Depends on NumPy only (no scikit-learn, no pyspark import at module load).
"""
from spark_vi.eval.metrics.core import (calibration_bins, curve, ece_from_bins, merge,
                                        node_metrics, tally)
from spark_vi.eval.metrics.report import (SKIP_CODES, SKIP_CONSTANT, SKIP_DEGENERATE,
                                          SKIP_SMALL, macro, pr_macro, pr_record,
                                          score_record)

__all__ = ["tally", "merge", "node_metrics", "curve", "calibration_bins",
           "ece_from_bins", "score_record", "macro", "pr_record", "pr_macro",
           "SKIP_CODES", "SKIP_DEGENERATE", "SKIP_SMALL", "SKIP_CONSTANT"]
