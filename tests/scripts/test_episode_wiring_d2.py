"""Exp 0111 WP-D2 — wiring the episode / matched-random index into the fit driver.

WP-D1 built the index providers (`episode_index.py`) and `EpisodeDocSpec`; WP-C
opened multi_domain's `index_df=` / `doc_spec=` seams. WP-D2 is the driver glue
that consumes them: it turns an `--index-arm` into an external-index corpus spec,
folds the sampling design into the bundle cache key WITHOUT moving any existing
key, threads the driver-built index frame into the Mondo assemble closure, and
bakes the BOUNDED within-corpus `episode_no` the readout doc key needs onto the
bundle.

What is pinned here, and why each failure would otherwise be silent:

  1. **Cache-key folding** (pure): an episode/random spec keys DIFFERENTLY from a
     population/disease/mondo spec; the two arms differ only by `arm`; and a spec
     with NO episode sampling keys BYTE-IDENTICALLY to the pinned Mondo key — the
     compatibility guarantee (a key move silently orphans a ~20-min-to-rebuild
     bundle).
  2. **Spec construction** (pure): `--index-arm` sets `index_mode=external`,
     `doc_spec=episode`, and the sampling dict; a non-Mondo dag_source is refused.
  3. **The bounded doc index** (`@slow`, local Spark): `_attach_bounded_doc_index`
     writes a DENSE per-person `row_number()-1` in `[0, cap)` — the doc-key radix
     bound — from the doc_id alone, uniquely, so `_doc_key_column`'s `episode_no<64`
     guard never trips and `_assert_unique_doc_keys` passes; WP-D1's UNBOUNDED
     ordinal rides a separate `episode_ordinal` column for R7.5.
  4. **The external seam end to end** (`@slow`, local Spark): the Mondo assemble
     closure builds the index (sidecar monkeypatched), passes `index_mode=external`
     + a `source_cohort`-carrying `index_df` + an `EpisodeDocSpec` to the
     assembler, and the returned bundle carries a valid `episode_no`.

Groups 1-2 are pure; 3-4 use local Spark (`@slow` + the `spark` fixture).
"""
import os
import sys
import types
from datetime import date

import pytest

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
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
import gated_pc_cloud as gpc  # noqa: E402

RADIX = dr.DOC_KEY_RADIX


# --------------------------------------------------------------------------- #
# A fake argparse.Namespace carrying only what multidomain_corpus_spec reads.  #
# --------------------------------------------------------------------------- #
def _args(**over):
    base = dict(
        dag_source="mondo_native", disease="rare6", cdr="p.d", billing="proj",
        source_table="condition_era", person_mod=10, vocab_size=5000, min_df=20,
        min_patient_count=20, doc_min_length=0, min_n=50, holdout_frac=0.2,
        n_bg=2, tpn=1, strip_mode="test_only", lookback_days=365,
        label_window_days=365, label_mask_mode="full", window_mode="lookback",
        prior_obs_days=365, window_days=365, mondo_version="2026-06-02",
        mondo_branch="", min_positives=100, mondo_cache_dir="data/mondo",
        dag_collapse=False, preindex_closure=False,
        # exp 0111 WP-D2 knobs (defaults mirror parse_args):
        index_arm="", episode_gap_days=90, episode_cap=3, episode_salt="0111",
        episode_prior_obs_days=365, episode_window_days=365,
        episode_sidecar_uri="")
    base.update(over)
    return types.SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# 1. Cache-key folding (pure — no SparkSession).                              #
# --------------------------------------------------------------------------- #
def test_no_episode_sampling_keys_byte_identically_to_the_pinned_mondo_key():
    # The compatibility guarantee: a plain population-index Mondo spec (no
    # --index-arm) must key exactly where it did before WP-D2 existed. This
    # reuses the tripwire's own pinned value so the two suites cannot drift.
    from _case_finding_cache import compute_bundle_cache_key
    md_base = dict(
        source_table="condition_era", person_mod=10, vocab_size=5000, min_df=20,
        min_patient_count=20, doc_min_length=0, prior_obs_days=365,
        window_days=365, disease="diabetes", min_n=50, holdout_frac=0.2,
        split_salt=20260716, n_bg=2, tpn=1, cdr="p.d", multidomain=True,
        extra_domains=("drug",), index_mode="population", mondo=True,
        mondo_version="2026-06-02", mondo_branch="", min_positives=100)
    assert compute_bundle_cache_key(**md_base) == "0ba6393f0d92af07"
    # Passing an explicit empty/none episode_sampling must ALSO not move it —
    # the fold fires only on a truthy dict.
    assert compute_bundle_cache_key(**md_base, episode_sampling=None) \
        == "0ba6393f0d92af07"
    assert compute_bundle_cache_key(**md_base, episode_sampling={}) \
        == "0ba6393f0d92af07"


def test_episode_spec_keys_differently_from_population_and_disease():
    pop = gpc.multidomain_corpus_spec(_args(), extra_domains=())
    epi = gpc.multidomain_corpus_spec(_args(index_arm="episode"),
                                      extra_domains=())
    kpop = gpc.multidomain_cache_key(pop)
    kepi = gpc.multidomain_cache_key(epi)
    assert kpop != kepi
    # index_mode moved to external, and the doc unit moved to episode — both are
    # real corpus-identity changes.
    assert pop["index_mode"] == "population" and epi["index_mode"] == "external"
    assert epi["doc_spec"] == "episode"


def test_the_two_arms_get_distinct_keys_differing_only_by_arm():
    epi = gpc.multidomain_corpus_spec(_args(index_arm="episode"),
                                      extra_domains=())
    rnd = gpc.multidomain_corpus_spec(_args(index_arm="random"),
                                      extra_domains=())
    assert gpc.multidomain_cache_key(epi) != gpc.multidomain_cache_key(rnd)
    # Everything else about the two sampling dicts is identical — only `arm`
    # differs, which is exactly what splits the two bundles.
    e, r = dict(epi["episode_sampling"]), dict(rnd["episode_sampling"])
    assert e.pop("arm") == "episode" and r.pop("arm") == "random"
    assert e == r


def test_sampling_knobs_move_the_key():
    base = _args(index_arm="episode")
    k = lambda a: gpc.multidomain_cache_key(
        gpc.multidomain_corpus_spec(a, extra_domains=()))
    k0 = k(base)
    assert k(_args(index_arm="episode", episode_gap_days=60)) != k0
    assert k(_args(index_arm="episode", episode_cap=5)) != k0
    assert k(_args(index_arm="episode", episode_salt="other")) != k0
    assert k(_args(index_arm="episode", episode_prior_obs_days=90)) != k0
    assert k(_args(index_arm="episode", episode_window_days=180)) != k0


def test_sidecar_uri_is_not_part_of_the_key():
    # The sidecar has its own independent key; where it lives must not split the
    # bundle cache.
    a = gpc.multidomain_corpus_spec(_args(index_arm="episode"), extra_domains=())
    b = gpc.multidomain_corpus_spec(
        _args(index_arm="episode", episode_sidecar_uri="gs://bucket/sc"),
        extra_domains=())
    assert gpc.multidomain_cache_key(a) == gpc.multidomain_cache_key(b)


# --------------------------------------------------------------------------- #
# 2. Spec construction (pure).                                                #
# --------------------------------------------------------------------------- #
def test_index_arm_populates_the_sampling_block():
    spec = gpc.multidomain_corpus_spec(_args(index_arm="episode"),
                                       extra_domains=())
    samp = spec["episode_sampling"]
    assert samp == {"arm": "episode", "gap_days": 90, "cap": 3, "salt": "0111",
                    "prior_obs_days": 365, "window_days": 365}
    assert spec["index_mode"] == "external"
    assert spec["doc_spec"] == "episode"


def test_no_index_arm_leaves_the_spec_free_of_episode_fields():
    spec = gpc.multidomain_corpus_spec(_args(), extra_domains=())
    assert "episode_sampling" not in spec
    assert "episode_sidecar_uri" not in spec
    assert spec["index_mode"] == "population"
    assert spec["doc_spec"] == gpc.doc_spec_identity()


def test_index_arm_on_a_non_mondo_source_is_refused():
    with pytest.raises(ValueError, match="Mondo path"):
        gpc.multidomain_corpus_spec(
            _args(dag_source="snomed", index_arm="episode"), extra_domains=())


def test_re_readout_recovers_the_episode_key_from_the_manifest():
    # The driver writes the corpus spec into corpus_manifest verbatim; a re-readout
    # rebuilds the spec from it and MUST recompute the fit's own key, or it MISSES
    # the cache and rebuilds the wrong corpus. Pins that episode_sampling survives
    # the round-trip so the key does.
    import gated_pc_readout as gpr
    spec = gpc.multidomain_corpus_spec(
        _args(dag_source="mondo_native", index_arm="episode",
              episode_sidecar_uri="gs://b/sc"), extra_domains=())
    manifest = {"corpus_manifest": dict(spec)}
    recovered = gpr.corpus_spec_from_manifest(manifest)
    assert recovered["episode_sampling"] == spec["episode_sampling"]
    assert (gpc.multidomain_cache_key(recovered)
            == gpc.multidomain_cache_key(spec))


def test_re_readout_of_a_population_corpus_is_unaffected():
    # A plain Mondo (no-arm) manifest recovers a spec with no episode_sampling and
    # keys exactly where the fit did — the compatibility half of the round-trip.
    import gated_pc_readout as gpr
    spec = gpc.multidomain_corpus_spec(_args(dag_source="mondo_native"),
                                       extra_domains=())
    recovered = gpr.corpus_spec_from_manifest({"corpus_manifest": dict(spec)})
    assert not recovered.get("episode_sampling")
    assert (gpc.multidomain_cache_key(recovered)
            == gpc.multidomain_cache_key(spec))


# --------------------------------------------------------------------------- #
# Local-Spark fixtures (AGENTS.md: local Spark => @slow).                      #
# --------------------------------------------------------------------------- #
def _episode_bundle_frames(spark, docs):
    """A bundle-like train/test pair with EpisodeDocSpec doc_ids.

    `docs`: list of (person_id, index_date_str, split) — one document each. The
    doc_id is "episode:{person}:{index_date}", exactly what EpisodeDocSpec emits
    and `_attach_bounded_doc_index` parses."""
    from pyspark.sql.types import (LongType, StringType, StructField,
                                   StructType)
    fields = StructType([
        StructField("doc_id", StringType(), False),
        StructField("person_id", LongType(), False)])
    tr = [(f"episode:{p}:{d}", int(p)) for p, d, s in docs if s == "train"]
    te = [(f"episode:{p}:{d}", int(p)) for p, d, s in docs if s == "test"]
    return (spark.createDataFrame(tr, fields).repartition(2),
            spark.createDataFrame(te, fields).repartition(2))


class _FakeBundle:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df


@pytest.mark.slow
class TestAttachBoundedDocIndex:
    def test_dense_zero_based_index_per_person_within_cap(self, spark):
        # Person 100 has three episodes (2010, 2012, 2015); person 250 has one.
        # The bounded index must be 0,1,2 for 100 (date order) and 0 for 250 —
        # dense, 0-based, and strictly < the cap.
        docs = [(100, "2015-03-01", "train"), (100, "2010-01-01", "train"),
                (100, "2012-07-15", "train"), (250, "2011-05-05", "train")]
        train_df, test_df = _episode_bundle_frames(spark, docs)
        b = gpc._attach_bounded_doc_index(_FakeBundle(train_df, test_df))
        got = {(r["person_id"], r["doc_id"]): r["episode_no"]
               for r in b.train_df.collect()}
        assert got[(100, "episode:100:2010-01-01")] == 0
        assert got[(100, "episode:100:2012-07-15")] == 1
        assert got[(100, "episode:100:2015-03-01")] == 2
        assert got[(250, "episode:250:2011-05-05")] == 0
        assert all(0 <= v < 3 for v in got.values())     # within cap

    def test_doc_keys_are_unique_and_pass_the_tripwire_and_guard(self, spark):
        # The full doc-key path: build the bounded index, synthesize the doc key
        # via _doc_key_column (which reads episode_no and enforces the [0,64)
        # guard), and run the corpus-level uniqueness tripwire — the two guards
        # WP-A1 ships that D2 must satisfy.
        docs = [(100, "2010-01-01", "train"), (100, "2012-07-15", "train"),
                (100, "2015-03-01", "train"), (250, "2011-05-05", "test"),
                (250, "2013-02-02", "test")]
        train_df, test_df = _episode_bundle_frames(spark, docs)
        b = gpc._attach_bounded_doc_index(_FakeBundle(train_df, test_df))
        for df, where in ((b.train_df, "train"), (b.test_df, "test")):
            keyed = df.withColumn("doc_key", gpc._doc_key_column(df))
            keys = [int(r["doc_key"]) for r in keyed.collect()]  # guard runs here
            gpc._assert_unique_doc_keys(keys, where=where)       # no raise
            # doc_key = person*RADIX + episode_no, person recoverable.
            for r in keyed.collect():
                assert dr.person_of(int(r["doc_key"])) == r["person_id"]
                assert 0 <= dr.episode_of(int(r["doc_key"])) < 3

    def test_unbounded_ordinal_rides_a_separate_column(self, spark):
        # WP-D1's UNBOUNDED chronological ordinal (a chronic patient's 70th
        # episode) is carried as `episode_ordinal`, joined from the index frame,
        # and never leaks into the bounded `episode_no` the doc key uses.
        docs = [(100, "2010-01-01", "train"), (100, "2012-07-15", "train")]
        train_df, test_df = _episode_bundle_frames(spark, docs)
        ordinal_df = spark.createDataFrame(
            [(100, date(2010, 1, 1), 4), (100, date(2012, 7, 15), 70)],
            "person_id long, index_date date, episode_ordinal long")
        b = gpc._attach_bounded_doc_index(_FakeBundle(train_df, test_df),
                                          ordinal_df=ordinal_df)
        rows = {r["doc_id"]: (r["episode_no"], r["episode_ordinal"])
                for r in b.train_df.collect()}
        assert rows["episode:100:2010-01-01"] == (0, 4)
        assert rows["episode:100:2012-07-15"] == (1, 70)   # 70 stays OUT of the key


@pytest.mark.slow
class TestExternalSeamEndToEnd:
    def test_mondo_closure_threads_the_external_index_and_bakes_episode_no(
            self, spark, monkeypatch):
        # Drive the anchor-Mondo assemble closure with injected build + assemble
        # hooks and a monkeypatched sidecar, and assert the seam hands the
        # assembler an EpisodeDocSpec + a source_cohort-carrying external index,
        # and that the returned bundle carries a valid bounded episode_no.
        from pyspark.sql import functions as F

        from charmpheno.omop.doc_spec import EpisodeDocSpec

        # (a) fake first-attestation sidecar: two persons, several attested dates
        #     >90d apart so each becomes its own episode.
        first = spark.createDataFrame(
            [(100, 10, date(2010, 1, 1)), (100, 11, date(2010, 8, 1)),
             (100, 12, date(2011, 4, 1)), (250, 10, date(2012, 1, 1))],
            "person_id long, node_cid long, first_attested_date date")
        obs = spark.createDataFrame(
            [(100, date(2005, 1, 1), date(2020, 1, 1)),
             (250, date(2005, 1, 1), date(2020, 1, 1))],
            "person_id long, observation_period_start_date date, "
            "observation_period_end_date date")
        monkeypatch.setattr(gpc, "_load_or_build_first_attestation",
                            lambda *a, **k: first)
        import conversion_sidecar as cs
        monkeypatch.setattr(cs, "load_observation_period",
                            lambda *a, **k: obs)

        # (b) fake DAG build: a real climb_sdf (normalize_code_map selects its two
        #     columns) + a terminals list; the DAG object is opaque (the real
        #     assembler is replaced).
        climb = spark.createDataFrame(
            [(10, 111), (11, 111), (12, 111)],
            "descendant_concept_id long, ancestor_concept_id long")

        def _fake_build(*_a, **_k):
            return object(), climb, [111], {}, {"n_classes": 0}

        # (c) fake assembler: capture kwargs, and SIMULATE the assembler by
        #     deriving EpisodeDocSpec doc_ids from the external index it received
        #     (person-keyed split is irrelevant here — one side suffices).
        captured = {}

        def _fake_assemble(spark_, *, before_dag, attested_provider, index_df,
                           doc_spec, **assembly_params):
            captured["index_df"] = index_df
            captured["doc_spec"] = doc_spec
            captured["index_mode"] = assembly_params.get("index_mode")
            ev = index_df.select(
                "person_id",
                F.col("source_cohort").cast("string").alias("source_cohort"),
                "index_date")
            docs = doc_spec.derive_docs(ev).select("doc_id", "person_id")
            return _FakeBundle(docs, docs.limit(0))

        spec = gpc.multidomain_corpus_spec(
            _args(dag_source="mondo", index_arm="episode",
                  episode_sidecar_uri="gs://x/sc"),
            extra_domains=())
        assembly, key_extra = gpc._multidomain_params(spec)
        fn = gpc.mondo_assemble_fn(spec, _build_inputs=_fake_build,
                                   _assemble=_fake_assemble,
                                   cache_uri="gs://x/cache")
        bundle = fn(spark, **assembly)

        # The seam selected the external path.
        assert captured["index_mode"] == "external"
        assert isinstance(captured["doc_spec"], EpisodeDocSpec)
        # The index frame carries the columns the external seam requires.
        assert set(("person_id", "index_date", "source_cohort")).issubset(
            set(captured["index_df"].columns))
        sc = {r["source_cohort"] for r in captured["index_df"].collect()}
        assert sc == {"episode"}
        # Person 100 got up to cap=3 episode documents; person 250 got one.
        counts = {r["person_id"]: r["n"] for r in
                  captured["index_df"].groupBy("person_id").count()
                  .withColumnRenamed("count", "n").collect()}
        assert counts[100] == 3 and counts[250] == 1

        # The returned bundle carries a valid bounded episode_no + doc key.
        keyed = bundle.train_df.withColumn(
            "doc_key", gpc._doc_key_column(bundle.train_df))
        keys = [int(r["doc_key"]) for r in keyed.collect()]
        gpc._assert_unique_doc_keys(keys, where="e2e")
        for r in keyed.collect():
            assert 0 <= dr.episode_of(int(r["doc_key"])) < 3
            assert dr.person_of(int(r["doc_key"])) == r["person_id"]
