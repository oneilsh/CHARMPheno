"""Evaluation utilities for spark_vi models.

Subpackages are scoped by model class:
  spark_vi.eval.topic   — held-out coherence and related metrics for topic
                          models (OnlineLDA, OnlineHDP).
  spark_vi.eval.metrics — exact, model-agnostic ranking metrics (AUC, AP,
                          precision@recall, recall@FDR, top-k, ROC/PR curves,
                          calibration) from mergeable score tallies, on Spark
                          without a driver collect (ADR 0051 step 4).
"""
