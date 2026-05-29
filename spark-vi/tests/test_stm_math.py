"""Tests for STM per-doc inference math: STMDocument, gradient, Hessian, MAP."""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.types import STMDocument


class TestSTMDocument:
    def test_constructs_with_indices_counts_length_x(self):
        doc = STMDocument(
            indices=np.array([0, 3, 5], dtype=np.int32),
            counts=np.array([2.0, 1.0, 3.0], dtype=np.float64),
            length=6,
            x=np.array([1.0, 0.5, -1.2], dtype=np.float64),
        )
        assert doc.length == 6
        assert doc.x.shape == (3,)
        assert doc.x.dtype == np.float64

    def test_is_frozen(self):
        doc = STMDocument(
            indices=np.array([0], dtype=np.int32),
            counts=np.array([1.0]),
            length=1,
            x=np.array([0.0]),
        )
        with pytest.raises((AttributeError, TypeError)):
            doc.length = 99
