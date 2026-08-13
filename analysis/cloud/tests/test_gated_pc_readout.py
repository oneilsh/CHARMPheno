"""Tests for the gated_pc post-hoc readout: pure helpers (parser, key recompute,
model reconstruction). The end-to-end reload+transform is the cluster smoke."""

import numpy as np
import pytest


def _manifest(**over):
    m = {
        "model_class": "gated_pc", "disease": "rare6", "min_n": 20,
        "strip_mode": "both", "label_mask_mode": "full", "window_mode": "lookback",
        "lookback_days": 365, "label_window_days": 365, "n_bg": 40, "tpn": 5,
        "K": 170, "C": 27, "weight_y": 50.0, "max_iter": 200, "min_label_count": 20,
        "corpus_manifest": {
            "cdr": "p.d", "source_table": "condition_era", "person_mod": 1,
            "vocab_size": 5000, "min_df": 20, "min_patient_count": 20,
            "prior_obs_days": 0, "window_days": 365, "holdout_frac": 0.2,
            "doc_min_length": 10, "emit_labels": True},
    }
    m.update(over)
    return m


def test_build_parser_surface():
    from gated_pc_readout import build_parser
    a = build_parser().parse_args(["--run-dir", "/runs/0076-x", "--cache-uri",
                                   "hdfs:///c", "--doc-min-length", "10"])
    assert a.run_dir == "/runs/0076-x" and a.cache_uri == "hdfs:///c"
    assert a.doc_min_length == 10
    assert a.recall_targets == "0.5,0.8,0.9" and a.fdr_targets == "0.1,0.25,0.5"


def test_bundle_key_from_manifest_matches_direct_key():
    from gated_pc_readout import bundle_key_from_manifest
    from _case_finding_cache import compute_bundle_cache_key
    m = _manifest()
    key = bundle_key_from_manifest(m)
    # Must equal the key the driver would have used at fit time (emit_labels=True).
    direct = compute_bundle_cache_key(
        source_table="condition_era", person_mod=1, vocab_size=5000, min_df=20,
        min_patient_count=20, doc_min_length=10, prior_obs_days=0, window_days=365,
        disease="rare6", min_n=20, holdout_frac=0.2, n_bg=40, tpn=5, cdr="p.d",
        strip_mode="both", window_mode="lookback", lookback_days=365,
        label_window_days=365, emit_labels=True, label_mask_mode="full")
    assert key == direct


def test_bundle_key_doc_min_length_override_for_old_manifest():
    from gated_pc_readout import bundle_key_from_manifest
    m = _manifest()
    del m["corpus_manifest"]["doc_min_length"]        # older manifest omits it
    with pytest.raises(KeyError, match="doc_min_length"):
        bundle_key_from_manifest(m)
    # override supplies it -> matches the full-manifest key.
    assert bundle_key_from_manifest(m, doc_min_length=10) == \
        bundle_key_from_manifest(_manifest())


def test_reconstruct_model_from_npz(tmp_path):
    from gated_pc_readout import reconstruct_model
    K, V, C = 170, 5000, 27
    np.savez(tmp_path / "gated_pc_result.npz",
             **{"lambda": np.abs(np.random.default_rng(0).normal(size=(K, V))) + 1e-3,
                "alpha": np.full(K, 1.0 / K), "w_CK": np.zeros((C, K))})
    model = reconstruct_model(tmp_path, _manifest())
    assert model.vocabSize() == V
    assert model.headWeights().shape == (C, K)
    assert float(model.getOrDefault("weightY")) == 50.0   # >0 -> transform emits proba
    assert int(model.getOrDefault("numLabels")) == C
