"""Profile-eta prior wired into the Gated-PC path — WP-3 wiring tests.

The estimator side of the 2026-09-06 profile-eta-prior plan, mirroring
test_pc_spectral_init.py's split of labor: the BOOST machinery itself is tested
in test_gated_eta_boost.py (engine) and tests/scripts/test_profile_eta.py (the
driver-owned builder); here we pin only what pc.py adds —

  * the four provenance Params (profileEta / profileEtaStrength /
    profileEtaTopics / profileEtaMinCoverage) exist, default off, and
    round-trip (the run manifest / model params carry the knobs even though
    the boost itself arrives pre-built via setEtaBoost — plan D4);
  * the D5 guard: profileEta (or the mechanism Param etaBoost) is incompatible
    with resumeFrom/warmStartFrom, in exactly the spectral-init guard's shape
    (a fit-long prior vs a checkpoint fit without it).
"""
import json

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")


# --------------------------------------------------------------------------- #
# Spark-free: param plumbing                                                   #
# --------------------------------------------------------------------------- #
def test_pc_estimator_accepts_profile_eta_params_and_defaults_off():
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    est = OnlinePCLDAEstimator(numLabels=3, weightY=0.0)
    # defaults: no profile prior, plan's pre-registered knob defaults
    assert est.getOrDefault("profileEta") == ""
    assert est.getOrDefault("profileEtaStrength") == 1.0
    assert est.getOrDefault("profileEtaTopics") == 1
    assert est.getOrDefault("profileEtaMinCoverage") == 0.0
    est2 = OnlinePCLDAEstimator(
        numLabels=3, weightY=0.0,
        profileEta="data/ontology/profile_eta.tsv", profileEtaStrength=3.0,
        profileEtaTopics=2, profileEtaMinCoverage=0.25)
    assert est2.getOrDefault("profileEta") == "data/ontology/profile_eta.tsv"
    assert est2.getOrDefault("profileEtaStrength") == 3.0
    assert est2.getOrDefault("profileEtaTopics") == 2
    assert est2.getOrDefault("profileEtaMinCoverage") == 0.25


# --------------------------------------------------------------------------- #
# Spark: the D5 resume/warm-start guard                                        #
# --------------------------------------------------------------------------- #
def _two_node_df(spark, V=6):
    from pyspark.ml.linalg import SparseVector
    rows = []
    for _ in range(8):
        rows.append((SparseVector(V, [0, 1], [3.0, 2.0]), [0.0, 0.0, 0.0], [1]))
        rows.append((SparseVector(V, [2, 3], [3.0, 2.0]), [0.0, 0.0, 0.0], [2]))
    return spark.createDataFrame(rows, ["features", "label", "frontier"])


def _pc_est(**kw):
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    est = OnlinePCLDAEstimator(
        featuresCol="features", frontierCol="frontier", labelCol="label",
        numLabels=3, weightY=0.0, maxIter=1, seed=0, subsamplingRate=1.0, **kw)
    return est.setGateParent({1: 0, 2: 0})._set(gateNBg=2, gateTpn=1)


def _checkpoint_dir(tmp_path):
    """A directory that LOOKS like a save dir (manifest.json present), so the
    path-existence checks pass and the D5 guard is what actually fires."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "manifest.json").write_text(json.dumps({}))
    return str(ckpt)


def test_profile_eta_incompatible_with_warm_start(spark, tmp_path):
    df = _two_node_df(spark)
    est = _pc_est(profileEta="profile_eta.tsv",
                  warmStartFrom=_checkpoint_dir(tmp_path))
    with pytest.raises(ValueError, match="incompatible with resumeFrom/warmStartFrom"):
        est.fit(df)


def test_profile_eta_incompatible_with_resume(spark, tmp_path):
    df = _two_node_df(spark)
    est = _pc_est(profileEta="profile_eta.tsv",
                  resumeFrom=_checkpoint_dir(tmp_path))
    with pytest.raises(ValueError, match="incompatible with resumeFrom/warmStartFrom"):
        est.fit(df)


def test_eta_boost_mechanism_param_also_guarded(spark, tmp_path):
    """The guard keys on the MECHANISM Param too: a boost handed straight to
    setEtaBoost (no profileEta provenance) must not slip past D5 either."""
    df = _two_node_df(spark)
    est = _pc_est(warmStartFrom=_checkpoint_dir(tmp_path))
    est.setEtaBoost({2: ([0, 1], [0.5, 0.25])})
    with pytest.raises(ValueError, match="incompatible with resumeFrom/warmStartFrom"):
        est.fit(df)


def test_profile_eta_params_alone_do_not_change_the_fit(spark):
    """The four Params are provenance: without an etaBoost the fit is
    byte-identical to a plain run (the boost is what changes the trajectory,
    and it arrives only via setEtaBoost)."""
    df = _two_node_df(spark)
    lam_plain = _pc_est().fit(df)._result.global_params["lambda"]
    lam_prof = _pc_est(profileEta="profile_eta.tsv", profileEtaStrength=3.0,
                       profileEtaTopics=2, profileEtaMinCoverage=0.25
                       ).fit(df)._result.global_params["lambda"]
    np.testing.assert_array_equal(lam_plain, lam_prof)


def test_eta_boost_changes_the_fit_through_the_estimator(spark):
    """End-to-end through the shim: a driver-built boost handed to setEtaBoost
    reaches the gated engine and moves the fitted lambda (the WP-3 seam works),
    while the un-boosted twin on the same seed does not move."""
    df = _two_node_df(spark)
    lam_plain = _pc_est().fit(df)._result.global_params["lambda"]
    est = _pc_est(profileEta="profile_eta.tsv")
    est.setEtaBoost({2: ([4, 5], [1.0, 0.5])})           # node 1's topic
    lam_boost = est.fit(df)._result.global_params["lambda"]
    assert lam_plain.shape == lam_boost.shape == (4, 6)
    assert not np.allclose(lam_plain, lam_boost)
