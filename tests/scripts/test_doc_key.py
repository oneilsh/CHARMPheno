"""The int64 doc-key seam — exp 0111 WP-A1 (spec R5.4).

Until 0111 a document WAS a person: `doc_id == person_id`, an int64, and the whole
readout/eval stack leaned on that. 0111 gives a person up to a few episode
documents (a STRING `doc_id`), so the stack synthesizes one int64 DOC KEY per
document — `doc_key = person_id * RADIX + episode_no` — that (a) never collides,
(b) yields the person back by `// RADIX`, and (c) leaves every single-doc corpus
on its current ids scaled by RADIX (order-preserving, so nothing about the
byte-identical single-doc path moves).

What is pinned here, and why each failure would otherwise be silent:

  1. **The synthesis + its inverse** (pure): the round-trip, the order-preserving
     single-doc mapping, and the two overflow guards (`person_id < 2**57`,
     `0 <= episode_no < RADIX`) — the low bits carrying past the radix is exactly
     the cross-person collision the seam exists to prevent.
  2. **The lean kernel path, two docs per person** (`@slow`, local Spark): the
     collect keys rows on the doc key, so a person's two documents survive as two
     distinct rows (the person-keyed dict silently overwrote one — the seam-6
     bug), and a per-person join lands each document on its own person via
     `person_of`.
  3. **The single-doc path is unchanged** (`@slow`): the doc keys are exactly the
     person ids times RADIX, and the collected proba/labels are what the kernel
     always produced — the seam is a rename of the id, not a change of behavior.
  4. **The corpus-level uniqueness tripwire and `score_cells_df`'s doc-key
     column** — the two smaller guards the seam ships.

Groups 2-4's Spark cases are `@slow` + the `spark` fixture from conftest; group 1
and the tripwire are pure.
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
# 1. The synthesis and its inverse (pure — no SparkSession).                  #
# --------------------------------------------------------------------------- #
def test_scalar_round_trip():
    for person, ep in [(0, 0), (1, 0), (12345, 3), (2 ** 31, RADIX - 1)]:
        key = dr.synthesize_doc_key(person, ep)
        assert key == person * RADIX + ep
        assert dr.person_of(key) == person
        assert dr.episode_of(key) == ep


def test_single_doc_maps_to_person_times_radix():
    # episode_no defaults to 0, so a pre-0111 int64 person id becomes id*RADIX.
    for person in (7, 41, 999_999):
        assert dr.synthesize_doc_key(person) == person * RADIX
        assert dr.person_of(person * RADIX) == person
        assert dr.episode_of(person * RADIX) == 0


def test_synthesis_is_order_preserving_and_collision_free():
    # Sorting by doc_key sorts by (person_id, episode_no): person blocks are
    # disjoint and the bounded episode_no occupies the low bits.
    persons = np.array([5, 3, 9, 3, 5], dtype=np.int64)
    episodes = np.array([0, 1, 0, 0, 2], dtype=np.int64)
    keys = dr.synthesize_doc_key(persons, episodes)
    assert len(set(keys.tolist())) == len(keys)          # collision-free
    by_key = np.argsort(keys)
    by_pair = np.lexsort((episodes, persons))
    assert np.array_equal(by_key, by_pair)               # order-preserving


def test_array_inverse_round_trips():
    persons = np.array([1, 2, 1_000_000_000], dtype=np.int64)
    episodes = np.array([0, 2, 4], dtype=np.int64)
    keys = dr.synthesize_doc_key(persons, episodes)
    assert np.array_equal(dr.person_of(keys), persons)
    assert np.array_equal(dr.episode_of(keys), episodes)


def test_max_key_stays_in_int64():
    # The overflow guard's boundary: the largest permitted key is exactly the
    # largest int64 — `(2**57 - 1) * 64 + 63 == 2**63 - 1` — so it fits and its
    # inverse still round-trips, and the guard's job is to keep anything larger
    # out.
    person = dr.DOC_KEY_MAX_PERSON_ID - 1
    key = dr.synthesize_doc_key(person, RADIX - 1)
    assert key == 2 ** 63 - 1 == int(np.iinfo(np.int64).max)
    assert dr.person_of(key) == person and dr.episode_of(key) == RADIX - 1


@pytest.mark.parametrize("person,ep", [
    (-1, 0),                          # negative person
    (dr.DOC_KEY_MAX_PERSON_ID, 0),    # person at the overflow boundary
    (dr.DOC_KEY_MAX_PERSON_ID + 5, 0),
    (1, -1),                          # negative episode
    (1, RADIX),                       # episode at the radix (would carry)
    (1, RADIX + 10),                  # WP-D1's raw ordinal leaking in
])
def test_overflow_guards_raise(person, ep):
    with pytest.raises(ValueError):
        dr.synthesize_doc_key(person, ep)


# --------------------------------------------------------------------------- #
# 4a. The corpus-level uniqueness tripwire (pure).                            #
# --------------------------------------------------------------------------- #
def test_uniqueness_tripwire_raises_on_a_collision():
    with pytest.raises(ValueError, match="doc-key collision"):
        g._assert_unique_doc_keys([1, 2, 2, 3], where="test")


def test_uniqueness_tripwire_passes_distinct_keys():
    g._assert_unique_doc_keys([10, 20, 30], where="test")   # no raise


# --------------------------------------------------------------------------- #
# Local-Spark fixtures (AGENTS.md: local Spark => @slow).                      #
# --------------------------------------------------------------------------- #
C, K = 3, 4


def _fit_params(seed=0):
    """A fixed (V, b_raw) so the collected proba is a deterministic reference."""
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(C, K))
    b_raw = rng.normal(size=C)
    return V, b_raw


def _expected_proba(theta, V, b_raw):
    z = np.clip(V @ np.asarray(theta, float) + b_raw, -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _scored_df(spark, rows, *, with_episode):
    """rows: list of (person_id, episode_no, theta, label, mask)."""
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
    return spark.createDataFrame(data, StructType(fields)).repartition(2)


@pytest.mark.slow
class TestLeanKernelDocKeyPath:
    def test_two_docs_per_person_survive_as_distinct_keys(self, spark):
        # Two persons, two documents each (episode_no 0 and 1). Under the OLD
        # person-keyed alignment these four rows would fold to two; keyed on the
        # doc key they must stay four distinct, person-derivable rows.
        rng = np.random.default_rng(1)
        rows = []
        for person in (100, 250):
            for ep in (0, 1):
                theta = rng.random(K)
                label = (rng.random(C) < 0.5).astype(float)
                mask = np.ones(C)
                rows.append((person, ep, theta, label, mask))
        df = _scored_df(spark, rows, with_episode=True)
        V, b_raw = _fit_params()
        proba, y, mask, doc_keys, elig = g._collect_lean_proba(df, C, V, b_raw)

        # Four distinct doc keys, each = person*RADIX + episode_no.
        assert len(doc_keys) == 4
        assert len(set(doc_keys)) == 4
        expected_keys = {p * RADIX + e for p in (100, 250) for e in (0, 1)}
        assert set(doc_keys) == expected_keys
        # person-grain recovery: the two keys per person share a person_of value.
        persons = dr.person_of(np.asarray(doc_keys))
        assert sorted(persons.tolist()) == [100, 100, 250, 250]
        assert elig is None

        # The proba is the real kernel output, aligned back BY DOC KEY — proof the
        # per-document identity is what indexes the eval arrays.
        by_key = {p * RADIX + e: theta
                  for p, e, theta, _, _ in rows}
        for i, key in enumerate(doc_keys):
            assert np.allclose(proba[i], _expected_proba(by_key[key], V, b_raw),
                               atol=1e-6)

    def test_person_grain_join_lands_each_document_on_its_person(self, spark):
        # A per-PERSON covariate (ADR-0025 grain) joined to a per-DOCUMENT eval
        # array through `person_of` at the seam — the grain crossing the plan
        # names. Every one of a person's documents must pick up that person's
        # covariate, and only that person's.
        rows = [(100, 0, [0.1, 0.2, 0.3, 0.4], [1, 0, 0], [1, 1, 1]),
                (100, 1, [0.4, 0.3, 0.2, 0.1], [0, 1, 0], [1, 1, 1]),
                (250, 0, [0.2, 0.2, 0.2, 0.4], [0, 0, 1], [1, 1, 1])]
        df = _scored_df(spark, rows, with_episode=True)
        V, b_raw = _fit_params()
        _, _, _, doc_keys, _ = g._collect_lean_proba(df, C, V, b_raw)

        covariate = {100: "cohortA", 250: "cohortB"}      # per-person sidecar
        landed = {key: covariate[int(dr.person_of(key))] for key in doc_keys}
        assert landed == {100 * RADIX + 0: "cohortA",
                          100 * RADIX + 1: "cohortA",
                          250 * RADIX + 0: "cohortB"}

    def test_single_doc_path_is_unchanged_relative_to_the_id_times_radix_map(
            self, spark):
        # No episode_no column (every corpus that exists today): one document per
        # person, doc_key = person_id * RADIX, and the collected arrays are what
        # the kernel always produced.
        rng = np.random.default_rng(2)
        rows = []
        for person in (5, 3, 9):
            theta = rng.random(K)
            label = (rng.random(C) < 0.5).astype(float)
            rows.append((person, None, theta, label, np.ones(C)))
        df = _scored_df(spark, rows, with_episode=False)
        V, b_raw = _fit_params()
        proba, y, mask, doc_keys, _ = g._collect_lean_proba(df, C, V, b_raw)

        assert set(doc_keys) == {5 * RADIX, 3 * RADIX, 9 * RADIX}
        assert np.array_equal(dr.person_of(np.asarray(sorted(doc_keys))),
                              np.array([3, 5, 9]))
        by_key = {p * RADIX: theta for p, _, theta, _, _ in rows}
        for i, key in enumerate(doc_keys):
            assert np.allclose(proba[i], _expected_proba(by_key[key], V, b_raw),
                               atol=1e-6)

    def test_collect_theta_labels_returns_doc_keys(self, spark):
        # The DRIVER-path collect (the A/B gate's driver side) must key on the
        # same doc key, so the two sides align on document identity.
        rows = [(100, 0, [0.1, 0.2, 0.3, 0.4], [1, 0, 0], [1, 1, 1]),
                (100, 1, [0.4, 0.3, 0.2, 0.1], [0, 1, 0], [1, 1, 1]),
                (250, 0, [0.2, 0.2, 0.2, 0.4], [0, 0, 1], [1, 1, 1])]
        df = _scored_df(spark, rows, with_episode=True)
        _, _, _, doc_key_order = g._collect_theta_labels(df, C)
        assert set(doc_key_order) == {100 * RADIX, 100 * RADIX + 1, 250 * RADIX}

    def test_collect_raises_on_a_doc_key_collision(self, spark):
        # Two documents that synthesize the SAME key (same person, same episode_no)
        # trip the corpus-level uniqueness assertion rather than silently
        # overwriting one another downstream.
        rows = [(100, 0, [0.1, 0.2, 0.3, 0.4], [1, 0, 0], [1, 1, 1]),
                (100, 0, [0.4, 0.3, 0.2, 0.1], [0, 1, 0], [1, 1, 1])]
        df = _scored_df(spark, rows, with_episode=True)
        V, b_raw = _fit_params()
        with pytest.raises(ValueError, match="doc-key collision"):
            g._collect_lean_proba(df, C, V, b_raw)

    def test_overflow_guard_fires_in_the_spark_column(self, spark):
        # An out-of-range person id trips the folded raise_error during the scan,
        # before the int64 multiply could wrap it into another person's block.
        rows = [(dr.DOC_KEY_MAX_PERSON_ID, 0, [0.1, 0.2, 0.3, 0.4],
                 [1, 0, 0], [1, 1, 1])]
        df = _scored_df(spark, rows, with_episode=True)
        V, b_raw = _fit_params()
        with pytest.raises(Exception, match="person_id out of"):
            g._collect_lean_proba(df, C, V, b_raw)


@pytest.mark.slow
class TestScoreCellsDocKey:
    def test_score_cells_df_carries_doc_key_and_metrics_are_unchanged(self, spark):
        # `score_cells_df(id_col=...)` prepends the doc key to every exploded cell
        # so a person-grain distributed eval can group on `person_of`, while
        # `per_node_metric_rows` (which selects node/y/p by name) scores
        # identically with or without it.
        rows = [(100, 0, [0.1, 0.2, 0.3, 0.4], [1, 0, 0], [1, 1, 1]),
                (100, 1, [0.4, 0.3, 0.2, 0.1], [0, 1, 0], [1, 1, 1]),
                (250, 0, [0.2, 0.2, 0.2, 0.4], [0, 0, 1], [1, 1, 1])]
        scored = _scored_df(spark, rows, with_episode=True)
        scored = scored.withColumn("doc_key", g._doc_key_column(scored))
        V, b_raw = _fit_params()

        plain = dr.score_cells_df(scored, V, b_raw, C)
        keyed = dr.score_cells_df(scored, V, b_raw, C, id_col="doc_key")
        assert "doc_key" in keyed.columns and "doc_key" not in plain.columns

        # Every keyed cell's doc_key is a real, person-derivable document key.
        keyed_rows = keyed.collect()
        seen = {int(r["doc_key"]) for r in keyed_rows}
        assert seen == {100 * RADIX, 100 * RADIX + 1, 250 * RADIX}

        # Per-node metrics agree whether or not the cells carry the doc key.
        m_plain = dr.per_node_metric_rows(plain, C)
        m_keyed = dr.per_node_metric_rows(keyed, C)
        for c in range(C):
            assert m_plain[c]["n_pos"] == m_keyed[c]["n_pos"]
            assert m_plain[c]["n_neg"] == m_keyed[c]["n_neg"]

    def test_score_cells_df_refuses_doc_key_with_topm(self, spark):
        rows = [(100, 0, [0.1, 0.2, 0.3, 0.4], [1, 0, 0], [1, 1, 1])]
        scored = _scored_df(spark, rows, with_episode=True)
        scored = scored.withColumn("doc_key", g._doc_key_column(scored))
        V, b_raw = _fit_params()
        with pytest.raises(ValueError, match="dense path"):
            dr.score_cells_df(scored, V, b_raw, C, id_col="doc_key", topm=2)
