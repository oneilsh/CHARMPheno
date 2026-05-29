"""Tests for the STM dashboard bundle adapter (adapt_stm).

Surfaces Γ̂ + covariate names in the bundle so the dashboard can
display per-topic per-covariate effects. β / α-equivalent surfacing
re-uses the existing LDA adapter; only the Γ̂ piece is new.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class TestAdaptSTM:
    def test_writes_gamma_json_with_covariate_labels(self, tmp_path: Path):
        from charmpheno.export.dashboard import adapt_stm

        K, P = 4, 3
        Gamma = np.array([
            [+1.2, -0.3, 0.0, +0.5],
            [-0.4, +0.8, -1.1, +0.2],
            [+0.0, +0.1, +0.7, -0.3],
        ])
        covariate_names = ["Intercept", "sex[T.M]", "age"]
        out_dir = tmp_path / "bundle"
        out_dir.mkdir()
        adapt_stm(
            out_dir=out_dir,
            Gamma=Gamma,
            covariate_names=covariate_names,
            K=K, P=P,
        )

        gamma_json = json.loads((out_dir / "covariate_effects.json").read_text())
        # Schema: list of {"covariate": str, "per_topic": [float, ...]} entries.
        assert len(gamma_json) == P
        assert all("covariate" in row and "per_topic" in row for row in gamma_json)
        assert [row["covariate"] for row in gamma_json] == covariate_names
        assert all(len(row["per_topic"]) == K for row in gamma_json)
        # Check first row values match Gamma row 0.
        np.testing.assert_allclose(gamma_json[0]["per_topic"], Gamma[0])

    def test_rejects_size_mismatch(self, tmp_path: Path):
        from charmpheno.export.dashboard import adapt_stm
        with pytest.raises(ValueError, match="covariate_names|shape"):
            adapt_stm(
                out_dir=tmp_path,
                Gamma=np.zeros((3, 4)),
                covariate_names=["a", "b"],  # wrong length: 2 vs P=3
                K=4, P=3,
            )
