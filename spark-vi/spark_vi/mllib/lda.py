"""MLlib Estimator/Transformer shim for VanillaLDA.

Wraps spark_vi.models.lda.VanillaLDA + spark_vi.core.runner.VIRunner so the
model behaves like a pyspark.ml.clustering.LDA-shaped Estimator/Model pair.
The shim is a translation layer; all SVI logic lives in VanillaLDA. See
docs/superpowers/specs/2026-05-04-mllib-shim-design.md and ADR 0009.
"""
from __future__ import annotations

from pyspark.ml.base import Estimator, Model


class VanillaLDAEstimator(Estimator):
    """Stub — params and _fit added in subsequent tasks."""


class VanillaLDAModel(Model):
    """Stub — state and methods added in subsequent tasks."""
