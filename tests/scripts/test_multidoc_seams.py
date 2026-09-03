"""WP-A2/A3/A4 (exp 0111, spec R5.5/R5.6/R5.7) — the three multi-doc seams WP-A1
(spec R5.4, the int64 doc-key synthesis) left open. WP-A1's own commit named them
explicitly out of its scope: "driver-path calibration split, detection dedup,
sample_frac doc-keying." This file pins the fix for each.

  * **A2** (`_doc_key_sample`, R5.5 seam 6) — the A/B gate's `sample_frac` used to
    be `DataFrame.sample()`, a per-ROW-INDEX Bernoulli draw: a function of the
    query PLAN (partitioning), not of document identity. The alignment dict it
    feeds (`readout_ab_report`'s `pos = {int(k): i for i, k in
    enumerate(ids_dist)}`) was ALREADY re-keyed onto the doc key by A1 — verified
    here as a no-op check, not re-fixed. What A1 left is pinned by
    `test_doc_key_sample_is_order_and_partition_independent`: hashing the doc key
    makes the kept set a pure function of WHICH DOCUMENTS exist plus the seed,
    reproducible under a repartition the way `.sample()` never promised to be.

  * **A3** (`_person_keyed_cal_split`, R5.6) — the driver-path calibration/fit
    split used to be `rng.random(n_rows) < 0.25`, a per-ROW draw. Under multi-doc
    a chronic person's several documents could land on BOTH sides: some fit the
    isotonic calibrator, one held out to grade it — the calibrator graded partly
    on the same person's own correlated covariates it was fit on, an in-sample
    leak dressed as an out-of-sample ECE improvement (the exp 0079 run-2
    failure). `test_person_keyed_cal_split_no_person_straddles` reproduces the old
    straddle by construction, then proves the person-keyed split never does it.

  * **A4** (`detection_readout(doc_keys=...)`, R5.7) — the case-vs-background
    detection pool used to be per-DOCUMENT, so a chronic 3-doc person voted up to
    3x toward the same pooled AUC point. `test_detection_readout_...` fixtures a
    case where episode-weighted and person-deduped detection AUC DISAGREE by
    construction (one weak episode drags a doc-level row below a background
    person's score; the person's own PEAK episode does not), and a companion
    proves the dedup is a byte-identical no-op on a single-doc corpus.

A2's oracle needs Spark (`@pytest.mark.slow`, the `spark` fixture from
`tests/scripts/conftest.py`); A3 and A4 are pure numpy and need neither.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUD = REPO_ROOT / "analysis" / "cloud"
for _p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
                str(REPO_ROOT / "spark-vi"), os.environ.get("PYTHONPATH", ""))
    if p)

import distributed_readout as dr  # noqa: E402
import gated_pc_cloud as g  # noqa: E402

RADIX = dr.DOC_KEY_RADIX


# --------------------------------------------------------------------------- #
# A2 — doc-key-grained sample_frac (R5.5, seam 6). Local-Spark (@slow).       #
# --------------------------------------------------------------------------- #
def _scored_df(spark, rows, *, with_episode):
    """rows: list of (person_id, episode_no, theta, label, mask). Same shape as
    test_doc_key.py's helper — a document row, keyed on (person_id, episode_no)
    when `with_episode`."""
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql.types import (ArrayType, DoubleType, LongType, StructField,
                                   StructType)
    fields = [StructField("person_id", LongType(), False)]
    if with_episode:
        fields.append(StructField("episode_no", LongType(), False))
    fields += [StructField("topicDistribution", VectorUDT(), False),
               StructField("label", ArrayType(DoubleType()), False),
               StructField("labelMask", ArrayType(DoubleType()), False)]
    data = []
    for person, ep, theta, label, mask in rows:
        rec = [int(person)]
        if with_episode:
            rec.append(int(ep))
        rec += [Vectors.dense(theta), [float(v) for v in label],
                [float(v) for v in mask]]
        data.append(tuple(rec))
    return spark.createDataFrame(data, StructType(fields))


@pytest.mark.slow
def test_doc_key_sample_is_order_and_partition_independent(spark):
    """A2 oracle: `_doc_key_sample` (what `readout_ab_report`'s `sample_frac<1`
    branch now calls) must select the SAME set of documents no matter how the
    frame is partitioned — `DataFrame.sample()`'s per-partition Bernoulli draw
    does not have this property (it is a function of row POSITION within a
    partition, i.e. of the query PLAN), so two physically different
    arrangements of the IDENTICAL rows could silently disagree at the same
    seed, breaking the gate's own promise ("both paths restricted to the SAME
    sample"). Two documents per person, the same 120 rows repartitioned two
    different ways: the kept doc-key set must be identical both times, and a
    real (non-trivial) subset."""
    rng = np.random.default_rng(0)
    rows = []
    for person in range(60):
        for ep in (0, 1):                       # every person has TWO documents
            theta = rng.random(4).tolist()
            rows.append((person, ep, theta, [1.0, 0.0], [1.0, 1.0]))

    df_1 = _scored_df(spark, rows, with_episode=True).repartition(1)
    df_11 = _scored_df(spark, rows, with_episode=True).repartition(11)

    def _kept_keys(df):
        sampled = g._doc_key_sample(df, 0.4, seed=13)
        return {int(r["doc_key"]) for r in
                sampled.select(g._doc_key_column(sampled)).collect()}

    kept_1 = _kept_keys(df_1)
    kept_11 = _kept_keys(df_11)
    assert kept_1 == kept_11, ("doc-key sampling must not depend on partitioning "
                               "— this is exactly what `.sample()` could not "
                               "promise")
    assert 0 < len(kept_1) < len(rows)          # a real subset, not everything/nothing

    # Whole documents, never a fragment or a duplicate: every kept key is one of
    # the 120 real doc keys, and the set has no repeats.
    all_keys = {p * RADIX + e for p in range(60) for e in (0, 1)}
    assert kept_1 <= all_keys


@pytest.mark.slow
def test_ab_alignment_dict_is_already_doc_keyed_by_a1(spark, capsys):
    """A1 verification (not a re-fix): `readout_ab_report`'s row-wise alignment
    dict must be keyed on the int64 doc key, not `person_id` — a two-docs-per-
    person fixture is the fixture that catches a person-keyed dict silently
    overwriting one document's row with another's (the seam-6 bug A1's commit
    message names). This asserts the gate runs clean on such a fixture and
    reports a per-row agreement line, which a person-keyed dict could not do
    (it would either raise on the ensuing shape mismatch or silently compare
    the wrong rows) — A2's job was `sample_frac` alone."""
    C, K = 3, 4
    rng = np.random.default_rng(5)
    rows = []
    for person in range(40):
        n_docs = 2 if person < 20 else 1        # half the corpus is multi-doc
        for ep in range(n_docs):
            theta = rng.dirichlet(np.full(K, 0.5)).tolist()
            label = (rng.random(C) < 0.5).astype(float).tolist()
            rows.append((person, ep, theta, label, [1.0] * C))
    train_df = _scored_df(spark, rows, with_episode=True).cache()
    test_df = _scored_df(spark, rows, with_episode=True).cache()

    out = g.readout_ab_report(
        train_df, test_df, C, K, recall_targets=[0.5], fdr_targets=[0.25],
        min_count=0, label="multidoc-ab", seed=1)
    txt = capsys.readouterr().out
    assert "A/B readout equality gate" in txt
    assert "max |Δp|" in txt              # the doc-key-aligned row comparison ran
    assert out["distributed"]["ranking"]["n_labels_scored"] == \
        out["driver"]["ranking"]["n_labels_scored"]
    train_df.unpersist(); test_df.unpersist()


# --------------------------------------------------------------------------- #
# A3 — person-keyed calibration split, driver path (R5.6). Pure numpy.        #
# --------------------------------------------------------------------------- #
def test_person_keyed_cal_split_no_person_straddles():
    """A3 oracle: the exp 0079 run-2 failure, pinned. 30 persons shaped like an
    episode corpus (cap 3): 10 single-doc, 10 two-doc, 10 three-doc.

    First, reproduce the OLD bug by construction: a per-ROW draw at this seed
    demonstrably splits at least one multi-doc person's rows across both sides
    — the fixture is chosen to exercise the failure, not just assert an
    abstract probability of it.

    Then the fix: `_person_keyed_cal_split`, tried across 20 seeds, never lets
    a person straddle cal/fit — and still produces a non-degenerate split (both
    sides non-empty) so the check is not vacuously true of an empty slice."""
    doc_keys = []
    for pid in range(10):
        doc_keys.append(pid * RADIX)
    for pid in range(10, 20):
        doc_keys += [pid * RADIX + 0, pid * RADIX + 1]
    for pid in range(20, 30):
        doc_keys += [pid * RADIX + e for e in range(3)]
    doc_keys = np.asarray(doc_keys, dtype=np.int64)
    persons = dr.person_of(doc_keys)

    # THE BUG THIS FIXES: a per-row draw straddles a multi-doc person.
    old_row_cal = np.random.default_rng(0).random(len(doc_keys)) < 0.25
    straddled = any(old_row_cal[persons == pid].any()
                    and not old_row_cal[persons == pid].all()
                    for pid in set(persons.tolist()))
    assert straddled, ("fixture must reproduce the row-level straddle A3 "
                       "fixes — if this stops firing, pick a new seed/shape")

    # THE FIX, across many seeds.
    saw_nonempty_split = False
    for seed in range(20):
        cal_sel, fit_sel = g._person_keyed_cal_split(doc_keys, seed)
        assert np.array_equal(~cal_sel, fit_sel)
        assert cal_sel.shape == doc_keys.shape
        for pid in set(persons.tolist()):
            rows = cal_sel[persons == pid]
            assert rows.all() or not rows.any(), \
                f"person {pid} straddles cal/fit at seed={seed}"
        if cal_sel.any() and fit_sel.any():
            saw_nonempty_split = True
    assert saw_nonempty_split, "the split must be non-degenerate for SOME seed"


def test_person_keyed_cal_split_is_row_order_independent():
    """`np.unique` sorts by VALUE, so which persons land in `cal` is a function
    of person identity + seed alone — shuffling `doc_keys`'s row order (as a
    different upstream collect order would) must not change which PERSONS are
    selected, only where their rows sit in the returned boolean arrays."""
    doc_keys = np.array([pid * RADIX + e for pid in range(20) for e in range(2)],
                        dtype=np.int64)
    rng = np.random.default_rng(3)
    perm = rng.permutation(len(doc_keys))
    shuffled = doc_keys[perm]

    cal_a, _ = g._person_keyed_cal_split(doc_keys, seed=7)
    cal_b, _ = g._person_keyed_cal_split(shuffled, seed=7)

    persons_cal_a = set(dr.person_of(doc_keys[cal_a]).tolist())
    persons_cal_b = set(dr.person_of(shuffled[cal_b]).tolist())
    assert persons_cal_a == persons_cal_b


# --------------------------------------------------------------------------- #
# A4 — person-level detection dedup (R5.7). Pure numpy.                       #
# --------------------------------------------------------------------------- #
def test_detection_readout_person_dedup_disagrees_with_episode_weighting():
    """A4 oracle: episode-weighted (`doc_keys=None`) and person-deduped
    (`doc_keys=...`) detection MUST disagree here, by construction. Person 1000
    is a real case with a clear PEAK episode (score 0.9) but two weaker
    episodes (0.2, 0.3/0.4) that drag two of their three document-level rows
    below a background person's single score (0.5).

      * document-level (no dedup): 5 rows, 3 "positive". The Mann-Whitney count
        by hand: {0.9,0.3,0.4} positives vs {0.5,0.15} negatives — 4 of 6
        pairs are correctly ordered -> AUC = 4/6.
      * person-level (deduped, max-pooled): 3 persons, 1 case (score = max of
        their three episodes = 0.9) vs 2 controls (0.5, 0.15) -> the case
        outranks both -> AUC = 1.0, perfect separation.

    Both numbers are asserted exactly (not just "different") — proof the dedup
    fires AND that it fires at the value the stated MAX-pooling semantic
    predicts (see `detection_readout`'s docstring for which semantic and why)."""
    proba = np.array([
        [0.5, 0.9, 0.1],     # person 1000, episode 0 — the peak episode
        [0.5, 0.2, 0.3],     # person 1000, episode 1 — weak
        [0.5, 0.1, 0.4],     # person 1000, episode 2 — weak
        [0.5, 0.5, 0.5],     # person 2000 — background, single doc
        [0.5, 0.15, 0.15],   # person 3000 — background, single doc
    ])
    y = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=float)
    doc_keys = [1000 * RADIX + 0, 1000 * RADIX + 1, 1000 * RADIX + 2,
                2000 * RADIX, 3000 * RADIX]

    per_doc = g.detection_readout(proba, y, [0.5])
    per_person = g.detection_readout(proba, y, [0.5], doc_keys=doc_keys)

    assert per_doc["grain"] == "document" and per_person["grain"] == "person"
    assert per_doc["n_units"] == 5 and per_person["n_units"] == 3
    assert per_doc["auc"] == pytest.approx(4 / 6)
    assert per_person["auc"] == pytest.approx(1.0)
    assert per_doc["auc"] != pytest.approx(per_person["auc"])   # the dedup FIRES

    # Independent hand-check of the pools themselves.
    assert per_doc["prevalence"] == pytest.approx(3 / 5)         # 3 of 5 DOCS
    assert per_person["prevalence"] == pytest.approx(1 / 3)      # 1 of 3 PERSONS


def test_detection_readout_single_doc_noop():
    """On a single-doc-per-person corpus `person_of(doc_key)` is a bijection
    onto the rows, so passing `doc_keys` must reproduce the `doc_keys=None`
    result exactly — every recorded pre-multi-doc run (0104/0109/0110) is
    compared on the `doc_keys=None` numbers, and this is the guarantee they
    stay byte-identical once callers start passing doc keys everywhere."""
    rng = np.random.default_rng(0)
    C, D = 5, 40
    proba = rng.random((D, C))
    y = (rng.random((D, C)) < 0.3).astype(float)
    y[:, 0] = (rng.random(D) < 0.5).astype(float)         # root: the foreground flag
    doc_keys = [pid * RADIX for pid in range(D)]

    plain = g.detection_readout(proba, y, [0.5, 0.9])
    keyed = g.detection_readout(proba, y, [0.5, 0.9], doc_keys=doc_keys)

    assert plain["grain"] == "document" and keyed["grain"] == "person"
    assert plain["n_units"] == keyed["n_units"] == D
    assert keyed["auc"] == pytest.approx(plain["auc"])
    assert keyed["ap"] == pytest.approx(plain["ap"])
    assert keyed["prevalence"] == pytest.approx(plain["prevalence"])
    assert keyed["par"] == pytest.approx(plain["par"])


def test_readout_from_proba_forwards_doc_keys_to_detection_only():
    """`readout_from_proba`'s `doc_keys` reaches `detection`'s pool and nothing
    else — the ranking/per-node axes are unaffected by the SAME multi-doc
    fixture that flips the detection AUC above, because R5.7 is scoped to the
    pooled detection signal (spec R5.7), not the per-node axes."""
    C = 3
    proba = np.array([
        [0.5, 0.9, 0.1], [0.5, 0.2, 0.3], [0.5, 0.1, 0.4],
        [0.5, 0.5, 0.5], [0.5, 0.15, 0.15],
    ])
    y = np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    m = np.ones_like(y)
    doc_keys = [1000 * RADIX + 0, 1000 * RADIX + 1, 1000 * RADIX + 2,
                2000 * RADIX, 3000 * RADIX]

    plain = g.readout_from_proba(proba, y, m, C, recall_targets=[0.5],
                                 fdr_targets=[0.25])
    keyed = g.readout_from_proba(proba, y, m, C, recall_targets=[0.5],
                                 fdr_targets=[0.25], doc_keys=doc_keys)

    assert plain["ranking"] == keyed["ranking"]
    assert plain["per_node"] == keyed["per_node"]
    assert plain["detection"]["auc"] != pytest.approx(keyed["detection"]["auc"])
