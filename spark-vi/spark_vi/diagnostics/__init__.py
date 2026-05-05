"""Diagnostic helpers for distributed VI.

Currently exposes assert_persisted; future home for similar pre-fit checks
(broadcast leak detection, partition-skew warnings, etc.).
"""
from spark_vi.diagnostics.persist import assert_persisted

__all__ = ["assert_persisted"]
