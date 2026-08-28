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


def test_resolve_run_dir_disambiguates_collision(tmp_path):
    from gated_pc_readout import resolve_run_dir
    runs = tmp_path / "runs"
    ours = runs / "0076-gated-pc-rare6-smooshed"
    other = runs / "0076-multidomain-rare-priority-cond-drug-obs"
    ours.mkdir(parents=True)
    other.mkdir(parents=True)
    (ours / "gated_pc_result.npz").write_bytes(b"x")   # only ours has the artifact
    # glob matches BOTH; resolve keeps the gated_pc one.
    assert resolve_run_dir(str(runs / "0076-*")) == ours
    # exact dir works too.
    assert resolve_run_dir(str(ours)) == ours


def test_resolve_run_dir_errors_when_none_finished(tmp_path):
    import pytest
    from gated_pc_readout import resolve_run_dir
    runs = tmp_path / "runs"
    (runs / "0076-gated-pc-rare6-smooshed").mkdir(parents=True)   # no npz yet
    with pytest.raises(SystemExit, match="no run dir with gated_pc_result.npz"):
        resolve_run_dir(str(runs / "0076-*"))


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


class TestResolveReadoutMaxIter:
    """The solver budget a recovery re-readout runs under.

    A re-readout has to REPRODUCE the run it is rescuing, and the iteration cap is
    the one input the tool used to have no way to learn: a lost 60-iter CHARM_DEV
    smoke and a lost 200-iter record run look identical from the run dir, so the
    tool defaulted to 200 and the operator had to remember the difference. The fit
    manifest now records `readout_max_iter` (post-dev-capping, i.e. what the lost
    readout would actually have used), and this is the precedence that consumes it."""

    def test_cli_wins_over_manifest(self):
        from gated_pc_readout import resolve_readout_max_iter
        assert resolve_readout_max_iter(30, _manifest(readout_max_iter=60)) == \
            (30, "CLI")

    def test_manifest_wins_when_no_cli_value(self):
        from gated_pc_readout import resolve_readout_max_iter
        assert resolve_readout_max_iter(None, _manifest(readout_max_iter=60)) == \
            (60, "manifest")

    def test_legacy_default_when_manifest_predates_the_field(self):
        from gated_pc_readout import (_LEGACY_READOUT_MAX_ITER,
                                      resolve_readout_max_iter)
        # `_manifest()` has no readout_max_iter — exactly the exp 0104 fit whose
        # recovery still needs an explicit --readout-max-iter 60.
        assert resolve_readout_max_iter(None, _manifest()) == \
            (_LEGACY_READOUT_MAX_ITER, "legacy default")
        assert _LEGACY_READOUT_MAX_ITER == 200

    def test_cli_wins_even_against_a_legacy_manifest(self):
        from gated_pc_readout import resolve_readout_max_iter
        assert resolve_readout_max_iter(60, _manifest()) == (60, "CLI")

    def test_zero_or_null_recorded_value_falls_back(self):
        from gated_pc_readout import resolve_readout_max_iter
        # A 0/None cap is not a budget; treat it as "not recorded" rather than
        # letting the solver take zero passes on a rescue run.
        assert resolve_readout_max_iter(None, _manifest(readout_max_iter=0))[1] == \
            "legacy default"
        assert resolve_readout_max_iter(None,
                                        _manifest(readout_max_iter=None))[1] == \
            "legacy default"

    def test_parser_default_is_none_so_the_manifest_can_speak(self):
        from gated_pc_readout import build_parser
        a = build_parser().parse_args(["--run-dir", "/runs/0104-x"])
        assert a.readout_max_iter is None
        b = build_parser().parse_args(["--run-dir", "/runs/0104-x",
                                       "--readout-max-iter", "60"])
        assert b.readout_max_iter == 60


class TestMondoSpecMismatch:
    """The saved fit's int2cid is the witness for its label space: a manifest that
    predates corpus_manifest recording dag_source resolves to the snomed default,
    and for a Mondo fit that keys (and on the guaranteed MISS, rebuilds) the wrong
    corpus — exp 0104's fresh-cluster recovery. The guard fires before the key."""

    def test_mondo_ids_with_snomed_spec_is_a_mismatch(self):
        from gated_pc_readout import mondo_spec_mismatch
        m = _manifest(int2cid=["MONDO:0005267", "MONDO:0002869"])
        assert mondo_spec_mismatch({"dag_source": "snomed"}, m)

    def test_mondo_ids_with_mondo_spec_is_fine(self):
        from gated_pc_readout import mondo_spec_mismatch
        m = _manifest(int2cid=["MONDO:0005267"])
        assert not mondo_spec_mismatch({"dag_source": "mondo"}, m)

    def test_snomed_ids_never_mismatch(self):
        from gated_pc_readout import mondo_spec_mismatch
        # A genuine SNOMED run: numeric concept ids, snomed default is correct.
        m = _manifest(int2cid=["44054006", "73211009"])
        assert not mondo_spec_mismatch({"dag_source": "snomed"}, m)

    def test_dict_shaped_int2cid_and_missing_field(self):
        from gated_pc_readout import mondo_spec_mismatch
        # int2cid serialized as {engine-int: cid} instead of a list, and a
        # manifest with no int2cid at all (nothing to witness -> no mismatch).
        m = _manifest(int2cid={"0": "MONDO:0005267"})
        assert mondo_spec_mismatch({"dag_source": "snomed"}, m)
        assert not mondo_spec_mismatch({"dag_source": "snomed"}, _manifest())
