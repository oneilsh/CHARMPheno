from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from charmpheno.export.dashboard import write_model_and_vocab_bundles


def test_top_n_trim_reorders_and_renormalizes(tmp_path: Path):
    K, V = 2, 5
    # β: row 0 mostly on col 4; row 1 mostly on col 1. Other cols low.
    lambda_ = np.array([
        [1, 1, 1, 1, 100],
        [1, 100, 1, 1, 1],
    ], dtype=float)
    alpha = np.array([0.1, 0.1])
    marginals = [0.10, 0.30, 0.01, 0.01, 0.58]   # col 4, then col 1 are top-2
    vocab_ids = [101, 202, 303, 404, 505]
    descriptions = {101: "A", 202: "B", 505: "E"}
    domains = {101: "condition", 202: "drug", 505: "procedure"}

    write_model_and_vocab_bundles(
        out_dir=tmp_path,
        lambda_=lambda_, alpha=alpha,
        vocab_ids=vocab_ids, descriptions=descriptions, domains=domains,
        code_marginals=marginals, top_n=2,
    )

    model = json.loads((tmp_path / "model.json").read_text())
    vocab = json.loads((tmp_path / "vocab.json").read_text())

    # vocab kept top-2 by marginal: col 4 (505), then col 1 (202)
    assert [c["code"] for c in vocab["codes"]] == ["505", "202"]
    assert [c["description"] for c in vocab["codes"]] == ["E", "B"]
    assert [c["corpus_freq"] for c in vocab["codes"]] == pytest.approx([0.58, 0.30])
    # ids are 0..top_N-1
    assert [c["id"] for c in vocab["codes"]] == [0, 1]

    # model V matches trimmed vocab; β rows renormalize
    assert model["K"] == 2
    assert model["V"] == 2
    beta = np.array(model["beta"])
    np.testing.assert_allclose(beta.sum(axis=1), np.ones(2), atol=1e-6)
    # row 0 was concentrated on the kept col 4 → still concentrated there (column 0 of trimmed)
    assert beta[0, 0] > 0.9
    # row 1 was concentrated on col 1 → trimmed-column 1
    assert beta[1, 1] > 0.9


def test_returns_v_displayed(tmp_path: Path):
    K, V = 2, 4
    lambda_ = np.ones((K, V))
    alpha = np.array([0.1, 0.1])
    marginals = [0.4, 0.3, 0.2, 0.1]
    vocab_ids = [10, 20, 30, 40]
    v_disp = write_model_and_vocab_bundles(
        out_dir=tmp_path,
        lambda_=lambda_, alpha=alpha,
        vocab_ids=vocab_ids, descriptions={}, domains={},
        code_marginals=marginals, top_n=10,  # > V, should cap to V
    )
    assert v_disp == V
