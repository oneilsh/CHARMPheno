"""VI-backend wiring tests for the PC antidepressant driver.

Covers the pieces the ``--backend vi`` path adds, on the local Spark fixture
(conftest ``spark``, local[2]) — no BigQuery:

  * :func:`attach_multitask_label_columns` — the Spark ``ArrayType`` label/mask
    columns must equal the numpy :func:`assemble_multitask_labels`, cell-for-cell,
    for the SAME drug-column order (the faithfulness invariant between backends).
  * :func:`person_hash_split` — seeded, DataFrame-level, deterministic, disjoint,
    and partition-independent.
  * ``--backend`` + SVI-knob argparse plumbing (Spark-free).
  * ``_log_convergence`` — the fit-health passthrough for both backends.
  * A tiny end-to-end ``PCEstimator.fit(...).transform(...)`` smoke on a synthetic
    labeled BOW DataFrame: proves the driver's VI wiring (ArrayType label columns,
    numLabels=C, probabilityCol, collect back to numpy) is type-correct. The real
    BigQuery run is the user's (``make exp ID=71``).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_CLOUD = str(Path(__file__).resolve().parents[1])
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)

import pc_antidepressant_cloud as drv  # noqa: E402


def _bow_df(spark, feats_by_person, V):
    """A one-row-per-person BOW DataFrame (person_id: long, features: VectorUDT)."""
    from pyspark.ml.linalg import SparseVector
    from pyspark.sql import Row

    rows = [
        Row(person_id=pid, features=SparseVector(V, mapping))
        for pid, mapping in feats_by_person.items()
    ]
    return spark.createDataFrame(rows)


# --------------------------------------------------------------------------- #
# attach_multitask_label_columns == assemble_multitask_labels (cell-for-cell) #
# --------------------------------------------------------------------------- #
def test_attach_label_columns_matches_numpy_assembler(spark):
    V = 5
    feats = {1: {0: 2.0}, 2: {1: 1.0}, 3: {2: 3.0, 0: 1.0}, 4: {3: 1.0}}
    bow_df = _bow_df(spark, feats, V)
    drug_order = ["sertraline", "bupropion"]
    outcome = {
        1: ("sertraline", True),    # col 0, positive
        2: ("bupropion", False),    # col 1, negative (mask 1, y 0)
        3: ("sertraline", False),   # col 0, negative
        # person 4 absent from outcome -> all-unobserved row
    }

    out = drv.attach_multitask_label_columns(bow_df, outcome, drug_order, spark)
    got = {
        r["person_id"]: ([float(v) for v in r["y"]], [float(v) for v in r["label_mask"]])
        for r in out.select("person_id", "y", "label_mask").collect()
    }
    person_order = sorted(got)
    y_np, mask_np = drv.assemble_multitask_labels(outcome, person_order, drug_order)
    for i, pid in enumerate(person_order):
        y_got, mask_got = got[pid]
        np.testing.assert_array_equal(y_got, y_np[i], err_msg=f"y mismatch pid={pid}")
        np.testing.assert_array_equal(mask_got, mask_np[i], err_msg=f"mask pid={pid}")
    # Spot-check the semantics the numpy assembler enforces.
    np.testing.assert_array_equal(got[1], ([1.0, 0.0], [1.0, 0.0]))   # worked
    np.testing.assert_array_equal(got[2], ([0.0, 0.0], [0.0, 1.0]))   # not worked
    np.testing.assert_array_equal(got[4], ([0.0, 0.0], [0.0, 0.0]))   # unlabeled


def test_attach_label_columns_are_array_not_vector_type(spark):
    # The PCEstimator shim reads the label with isinstance(raw, (list,tuple,ndarray))
    # and wraps a miss (e.g. a DenseVector) to a wrong (1,C) shape — so the columns
    # MUST be ArrayType, which deserializes to a Python list.
    from pyspark.sql.types import ArrayType, DoubleType

    bow_df = _bow_df(spark, {1: {0: 1.0}}, 3)
    out = drv.attach_multitask_label_columns(
        bow_df, {1: ("a", True)}, ["a", "b", "c"], spark
    )
    assert out.schema["y"].dataType == ArrayType(DoubleType())
    assert out.schema["label_mask"].dataType == ArrayType(DoubleType())
    r = out.select("y", "label_mask").head()
    assert isinstance(r["y"], list) and isinstance(r["label_mask"], list)


# --------------------------------------------------------------------------- #
# attach_fullyobserved_label_columns == assemble_fullyobserved_labels          #
# (cell-for-cell; the fully-observed stable-treatment path, all-ones mask)     #
# --------------------------------------------------------------------------- #
def test_attach_fullyobserved_columns_match_numpy_assembler(spark):
    V = 5
    feats = {1: {0: 2.0}, 2: {1: 1.0}, 3: {2: 3.0, 0: 1.0}, 4: {3: 1.0}}
    bow_df = _bow_df(spark, feats, V)
    drug_order = ["fluoxetine", "sertraline", "bupropion"]
    label_by_person = {
        1: ["fluoxetine"],                 # single -> col 0
        2: ["fluoxetine", "sertraline"],   # combination -> cols 0 and 1
        3: ["bupropion"],                  # col 2
        # person 4 absent -> all-zero, all-UNOBSERVED row
    }

    out = drv.attach_fullyobserved_label_columns(
        bow_df, label_by_person, drug_order, spark,
    )
    got = {
        r["person_id"]: ([float(v) for v in r["y"]], [float(v) for v in r["label_mask"]])
        for r in out.select("person_id", "y", "label_mask").collect()
    }
    person_order = sorted(got)
    y_np, mask_np = drv.assemble_fullyobserved_labels(
        label_by_person, person_order, drug_order,
    )
    for i, pid in enumerate(person_order):
        y_got, mask_got = got[pid]
        np.testing.assert_array_equal(y_got, y_np[i], err_msg=f"y mismatch pid={pid}")
        np.testing.assert_array_equal(mask_got, mask_np[i], err_msg=f"mask pid={pid}")
    # Spot-check the fully-observed semantics: present -> all-ones mask.
    np.testing.assert_array_equal(got[1], ([1.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    np.testing.assert_array_equal(got[2], ([1.0, 1.0, 0.0], [1.0, 1.0, 1.0]))  # combo
    np.testing.assert_array_equal(got[4], ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))  # absent


def test_attach_fullyobserved_columns_are_array_not_vector_type(spark):
    from pyspark.sql.types import ArrayType, DoubleType

    bow_df = _bow_df(spark, {1: {0: 1.0}}, 3)
    out = drv.attach_fullyobserved_label_columns(
        bow_df, {1: ["a"]}, ["a", "b", "c"], spark,
    )
    assert out.schema["y"].dataType == ArrayType(DoubleType())
    assert out.schema["label_mask"].dataType == ArrayType(DoubleType())
    r = out.select("y", "label_mask").head()
    assert isinstance(r["y"], list) and isinstance(r["label_mask"], list)


# --------------------------------------------------------------------------- #
# person_hash_split                                                            #
# --------------------------------------------------------------------------- #
def test_person_hash_split_deterministic_disjoint_and_covers(spark):
    df = spark.range(0, 200).withColumnRenamed("id", "person_id")
    tr1, te1 = drv.person_hash_split(df, test_frac=0.25, seed=7)
    tr2, te2 = drv.person_hash_split(df, test_frac=0.25, seed=7)
    tr_ids = {r["person_id"] for r in tr1.collect()}
    te_ids = {r["person_id"] for r in te1.collect()}
    # Deterministic across calls.
    assert tr_ids == {r["person_id"] for r in tr2.collect()}
    assert te_ids == {r["person_id"] for r in te2.collect()}
    # Disjoint + covering.
    assert tr_ids & te_ids == set()
    assert tr_ids | te_ids == set(range(200))
    # Roughly the requested fraction (hash buckets are near-uniform; a loose
    # band — this is a sanity check on the split, not an exactness claim).
    assert 0.10 < len(te_ids) / 200 < 0.40


def test_person_hash_split_independent_of_partitioning(spark):
    df = spark.range(0, 120).withColumnRenamed("id", "person_id")
    _, te_a = drv.person_hash_split(df, test_frac=0.3, seed=1)
    _, te_b = drv.person_hash_split(df.repartition(7), test_frac=0.3, seed=1)
    assert {r["person_id"] for r in te_a.collect()} == \
        {r["person_id"] for r in te_b.collect()}


def test_person_hash_split_seed_changes_membership(spark):
    df = spark.range(0, 200).withColumnRenamed("id", "person_id")
    _, te1 = drv.person_hash_split(df, test_frac=0.25, seed=1)
    _, te2 = drv.person_hash_split(df, test_frac=0.25, seed=2)
    assert {r["person_id"] for r in te1.collect()} != \
        {r["person_id"] for r in te2.collect()}


# --------------------------------------------------------------------------- #
# --backend + SVI-knob argparse plumbing (Spark-free)                          #
# --------------------------------------------------------------------------- #
def test_backend_defaults_to_inmem():
    ns = drv._build_parser().parse_args(["--cdr", "p.d", "--billing", "b"])
    assert ns.backend == "inmem"
    # SVI knobs carry their defaults even on the inmem path.
    assert ns.subsampling_rate == 0.05 and ns.tau0 == 1024.0 and ns.kappa == 0.51


def test_backend_vi_and_svi_knobs_parse():
    ns = drv._build_parser().parse_args([
        "--cdr", "p.d", "--billing", "b", "--backend", "vi",
        "--subsampling-rate", "0.1", "--tau0", "64", "--kappa", "0.6",
    ])
    assert ns.backend == "vi"
    assert ns.subsampling_rate == 0.1 and ns.tau0 == 64.0 and ns.kappa == 0.6


def test_backend_rejects_unknown_value():
    with pytest.raises(SystemExit):
        drv._build_parser().parse_args([
            "--cdr", "p.d", "--billing", "b", "--backend", "bogus",
        ])


# --------------------------------------------------------------------------- #
# _log_convergence passthrough (both backends)                                 #
# --------------------------------------------------------------------------- #
def test_log_convergence_inmem_line(capsys):
    drv._log_convergence({"pc_convergence": {
        "n_iter": 3, "success": False, "init_obj": 1.0, "final_obj": 0.4,
        "w_CK_absmax": 0.0,
    }})
    out = capsys.readouterr().out
    assert "PC fit:" in out and "nit=3" in out and "success=False" in out
    assert "|w_CK|max=0" in out and "head UNTRAINED" in out


def test_log_convergence_vi_line(capsys):
    drv._log_convergence({"vi_convergence": {
        "n_iter": 50, "final_elbo": -1234.5, "converged": True, "w_CK_absmax": 2.3,
    }})
    out = capsys.readouterr().out
    assert "VI-PC fit:" in out and "n_iter=50" in out and "converged=True" in out
    assert "final_elbo=-1234.5" in out and "|w_CK|max=2.3" in out


# --------------------------------------------------------------------------- #
# End-to-end smoke: PCEstimator.fit -> transform on a synthetic labeled BOW    #
# --------------------------------------------------------------------------- #
def test_vi_fit_transform_smoke(spark):
    pytest.importorskip("spark_vi")
    pytest.importorskip("autograd")
    from spark_vi.mllib.topic.pc import PCEstimator

    V, C = 6, 2
    drug_order = ["a", "b"]
    # 12 persons; alternate index drug so both heads see observed cells, and give
    # each drug both outcomes so no head is degenerate.
    feats, outcome = {}, {}
    rng = np.random.default_rng(0)
    for pid in range(12):
        idx = {int(j): float(rng.integers(1, 4)) for j in rng.choice(V, size=3, replace=False)}
        feats[pid] = idx
        drug = drug_order[pid % 2]
        outcome[pid] = (drug, bool(pid % 4 < 2))

    bow_df = _bow_df(spark, feats, V)
    labeled = drv.attach_multitask_label_columns(bow_df, outcome, drug_order, spark)
    # Tiny fit: full-batch SVI, few iters — a wiring smoke, not a convergence test.
    est = PCEstimator(
        featuresCol="features", labelCol="y", labelMaskCol="label_mask",
        numLabels=C, weightY=1.0, k=3, docConcentration=[1.1],
        subsamplingRate=1.0, learningOffset=16.0, learningDecay=0.6,
        maxIter=2, seed=0, probabilityCol="probability",
    )
    model = est.fit(labeled)
    scored = model.transform(labeled)
    assert "probability" in scored.columns
    X, y_DC, mask_DC, proba_DC, order = drv.collect_labeled_bow(
        scored, V, C, prob_col="probability",
    )
    assert X.shape == (12, V)
    assert y_DC.shape == (12, C) and mask_DC.shape == (12, C)
    assert proba_DC.shape == (12, C)
    assert np.all(np.isfinite(proba_DC)) and np.all((proba_DC >= 0) & (proba_DC <= 1))
    # The trained head is (C, K) and readable for the convergence signal.
    assert model.headWeights().shape == (C, 3)


# --------------------------------------------------------------------------- #
# Distributed-SVI two-stage baseline (option B): PCEstimator(weightY=0) topics #
# -> per-label LR, collected via _collect_topics_labels (no dense BOW).        #
# --------------------------------------------------------------------------- #
def _args_ns(**over):
    from types import SimpleNamespace
    base = dict(
        K=3, alpha=1.1, subsampling_rate=1.0, tau0=16.0, kappa=0.6,
        max_iter=2, seed=0, min_label_count=0, baseline_max_iter=-1,
        skip_two_stage=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_collect_topics_labels_shapes(spark):
    pytest.importorskip("spark_vi")
    pytest.importorskip("autograd")
    from spark_vi.mllib.topic.pc import PCEstimator

    V, C = 6, 2
    drug_order = ["a", "b"]
    feats, outcome = {}, {}
    rng = np.random.default_rng(0)
    for pid in range(12):
        feats[pid] = {int(j): float(rng.integers(1, 4))
                      for j in rng.choice(V, size=3, replace=False)}
        outcome[pid] = (drug_order[pid % 2], bool(pid % 4 < 2))
    labeled = drv.attach_multitask_label_columns(
        _bow_df(spark, feats, V), outcome, drug_order, spark)
    est = PCEstimator(
        featuresCol="features", labelCol="y", labelMaskCol="label_mask",
        numLabels=C, weightY=0.0, k=3, docConcentration=[1.1],
        subsamplingRate=1.0, learningOffset=16.0, learningDecay=0.6,
        maxIter=2, seed=0, topicDistributionCol="topicDistribution")
    scored = est.fit(labeled).transform(labeled)
    assert "topicDistribution" in scored.columns
    Pi, y_DC, mask_DC, order = drv._collect_topics_labels(scored, C)
    assert Pi.shape == (12, 3)                       # (D, K), NOT (D, V)
    assert y_DC.shape == (12, C) and mask_DC.shape == (12, C)
    assert len(order) == 12


def test_vi_two_stage_bundle_smoke(spark):
    pytest.importorskip("spark_vi")
    pytest.importorskip("autograd")

    V, C = 6, 2
    drug_order = ["a", "b"]
    feats, subset = {}, {}
    rng = np.random.default_rng(1)
    for pid in range(24):
        feats[pid] = {int(j): float(rng.integers(1, 4))
                      for j in rng.choice(V, size=3, replace=False)}
        # fully-observed: each person on one drug's stable regimen (both classes
        # present per column so no head is degenerate at this tiny size).
        subset[pid] = [drug_order[pid % 2]] if pid % 3 else []
    labeled = drv.attach_fullyobserved_label_columns(
        _bow_df(spark, feats, V), subset, drug_order, spark)
    train_df, test_df = drv.person_hash_split(labeled, 0.5, 0)
    bundle = drv._vi_two_stage_bundle(train_df, test_df, C, _args_ns())
    assert set(bundle.keys()) == {"per_label", "macro"}
    assert set(bundle["per_label"].keys()) == {0, 1}
    assert "n_labels_scored" in bundle["macro"]


def test_vi_two_stage_bundle_baseline_max_iter_caps_iters(spark):
    # baseline_max_iter overrides max_iter for the unsupervised fit (cheaper).
    pytest.importorskip("spark_vi")
    pytest.importorskip("autograd")
    V, C = 6, 2
    drug_order = ["a", "b"]
    feats, subset = {}, {}
    rng = np.random.default_rng(2)
    for pid in range(20):
        feats[pid] = {int(j): float(rng.integers(1, 4))
                      for j in rng.choice(V, size=3, replace=False)}
        subset[pid] = [drug_order[pid % 2]] if pid % 3 else []
    labeled = drv.attach_fullyobserved_label_columns(
        _bow_df(spark, feats, V), subset, drug_order, spark)
    train_df, test_df = drv.person_hash_split(labeled, 0.5, 0)
    # max_iter huge but baseline_max_iter tiny -> must still return promptly.
    bundle = drv._vi_two_stage_bundle(
        train_df, test_df, C, _args_ns(max_iter=999, baseline_max_iter=1))
    assert set(bundle.keys()) == {"per_label", "macro"}
