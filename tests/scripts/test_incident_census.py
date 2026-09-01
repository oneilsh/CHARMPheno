"""The incident-eligibility census (E-census) — the GO/NO-GO gate's counting kernel.

The whole diagnostic is one `treeAggregate` plus a pure classifier, so the whole
diagnostic is testable off-Spark, and it must be: this is the number that decides
whether E2/E3/E4 get built at all, and the spec's own validation item is "counts
reproduce on a fixture with hand-computed eligibility".

Four groups:

  1. **The counts.** A tiny hand-computable `(label, labelMask, preindexClosure)`
     triple where every one of `(n_incident_eligible, n_incident_pos,
     n_incident_neg)` can be read off by eye — including the two asymmetries that
     are easy to get wrong: eligibility is symmetric across BOTH classes (D2), and
     a negative requires the MASK while a positive does not.

  2. **The C2.1 population (RC.3).** The constructed case E2/R2.1's guard exists
     for: a node whose TRAIN cell was degenerate (all-positive, so its head is a
     constant column) and which ACQUIRES negatives once prior carriers leave. Such
     a node stops being skipped, scores `roc_auc_score(y, const)` = exactly 0.5,
     and enters the macro.

  3. **The closure discipline (RC.4 / ADR 0047 addendum).** The reduction identity
     is a `None` sentinel — nothing array-shaped rides the task closure — so the
     combiner has to handle `None` on either side and an empty partition has to
     produce no partial at all.

  4. **The two refusals and one end-to-end run.** A cache MISS is refused (a
     diagnostic never pays a rebuild) and a bundle without E1's witness is refused
     BY NAME rather than dying on a missing column — plus one local-Spark run of
     the whole tool over a real cached bundle, which is what covers the pieces the
     pure kernel cannot: the `array<int>` decode, the Row subscript, and that the
     column survives `save`/`try_load`.

Groups 1-3 are pure (no Spark); group 4's end-to-end case is `@slow`.
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

import diag_incident_census as dic  # noqa: E402


class _Row(dict):
    """A row is subscript-by-name; the kernel also tolerates Spark ML vectors via
    `toArray`, which plain lists do not have — that branch is exercised on the
    cluster, the list branch here."""


def _row(label, mask, preindex):
    return _Row(label=list(label), labelMask=list(mask),
                preindexClosure=list(preindex))


def _tally(rows, C):
    parts = dic.census_partial(iter(rows), C)
    out = None
    for p in parts:
        out = dic.census_combine(out, p)
    return out


# --------------------------------------------------------------------------- #
# 1. The counts, hand-computed.                                                #
# --------------------------------------------------------------------------- #
def test_counts_reproduce_a_hand_computed_fixture():
    """C=3 and four documents, every cell readable by eye.

        doc  label      mask       R_d      -> node 0        node 1        node 2
        A    [1,1,0]    [1,1,1]    []          elig,pos      elig,pos      elig,neg
        B    [1,0,0]    [1,1,1]    [0]         INELIGIBLE    elig,neg      elig,neg
        C    [0,1,0]    [1,1,0]    [2]         elig,neg      elig,pos      INELIGIBLE
        D    [1,1,1]    [1,0,1]    [1]         elig,pos      INELIGIBLE    elig,pos

    node 0: eligible A,C,D = 3; positive A,D = 2; negative C (observed) = 1
    node 1: eligible A,B,C = 3; positive A,C = 2; negative B (observed) = 1
    node 2: eligible A,B,D = 3; positive D = 1; negative A (observed) = 1,
            B is observed and negative too => 2. C is ineligible AND unobserved.
    """
    rows = [
        _row([1, 1, 0], [1, 1, 1], []),
        _row([1, 0, 0], [1, 1, 1], [0]),
        _row([0, 1, 0], [1, 1, 0], [2]),
        _row([1, 1, 1], [1, 0, 1], [1]),
    ]
    n_obs, n_pos, n_elig, n_ipos, n_ineg = _tally(rows, 3)

    assert list(n_elig) == [3, 3, 3]
    assert list(n_ipos) == [2, 2, 1]
    assert list(n_ineg) == [1, 1, 2]
    # ...and the prevalent side is the readout's own (n_obs, n_pos), unchanged.
    assert list(n_obs) == [4, 3, 3]
    assert list(n_pos) == [3, 2, 1]


def test_a_prior_carrier_leaves_BOTH_classes():
    """D2 is symmetric. The same doc, with and without node 0 in `R_d`: as a
    carrier it contributes to neither the positives nor the negatives, and not to
    the eligible denominator either. Dropping it from the positives only would be
    a different — and wrong — estimator."""
    free = _tally([_row([1, 0], [1, 1], [])], 2)
    carrier = _tally([_row([1, 0], [1, 1], [0])], 2)
    assert (free[2][0], free[3][0], free[4][0]) == (1, 1, 0)
    assert (carrier[2][0], carrier[3][0], carrier[4][0]) == (0, 0, 0)
    # node 1 (not carried) is untouched in both.
    assert free[2][1] == carrier[2][1] == 1


def test_a_negative_needs_the_MASK_but_a_positive_does_not():
    """Under `label_mask_mode="closure"` a node is observed only on rows inside
    its parent's closure, so an UNMASKED zero means "not asked", not "asked and
    answered no". A masked-out positive is still an attested fact about the label
    window, so it still counts."""
    unobserved_zero = _tally([_row([0, 0], [0, 0], [])], 2)
    assert list(unobserved_zero[4]) == [0, 0]          # no negatives
    assert list(unobserved_zero[2]) == [1, 1]          # but still eligible

    masked_positive = _tally([_row([1, 0], [0, 0], [])], 2)
    assert masked_positive[3][0] == 1                  # positive counted
    assert masked_positive[4][0] == 0
    # so pos + neg <= eligible, and the gap is unobserved-negative mass
    for t in (unobserved_zero, masked_positive):
        assert np.all(t[3] + t[4] <= t[2])


def test_preindex_ids_outside_the_label_space_are_ignored():
    """A witness column from a wider vintage must not index out of bounds."""
    t = _tally([_row([1, 0], [1, 1], [0, 5, -1])], 2)
    assert list(t[2]) == [0, 1]


# --------------------------------------------------------------------------- #
# 2. The classifier: the gate count, the C2.1 population, the fate breakdown.  #
# --------------------------------------------------------------------------- #
def _vecs(**kw):
    C = len(next(iter(kw.values())))
    return [np.asarray(kw.get(k, [0] * C), float) for k in
            ("n_obs", "n_pos", "n_elig", "n_ipos", "n_ineg")]


def test_the_gate_counts_nodes_clearing_min_count_on_BOTH_classes():
    """RC.2, and the number the whole program turns on. One node of each shape."""
    v = _vecs(n_obs=[100, 100, 100, 100, 100],
              n_pos=[50, 50, 50, 50, 50],
              n_elig=[80, 80, 80, 5, 0],
              n_ipos=[40, 40, 2, 2, 0],
              n_ineg=[40, 2, 40, 2, 0])
    buckets, summary = classify(v, min_count=20)
    assert buckets["powered_both"] == [0]
    assert buckets["powered_pos_only"] == [1]
    assert buckets["powered_neg_only"] == [2]
    assert buckets["starved"] == [3]
    assert buckets["no_eligible"] == [4]
    assert summary["n_nodes_clearing_both"] == 1
    # every node is in exactly one powering bucket
    assert sum(len(buckets[k]) for k in
               ("powered_both", "powered_pos_only", "powered_neg_only",
                "starved", "no_eligible")) == 5


def classify(v, **kw):
    return dic.classify_census(*v, **kw)


def test_the_c21_population_is_the_degenerate_head_that_acquires_negatives():
    """RC.3 / the R1 risk. Node 0's TRAIN cell is all-positive (n_pos == n_obs),
    so its head is a CONSTANT column and the prevalent mask skips it as a
    degenerate test column. Under the incident mask the prior carriers that made
    it all-positive leave, it acquires negatives, becomes non-degenerate, and
    scores exactly 0.5 inside the macro — unless E2/R2.1's guard catches it. This
    is the count that sizes that guard's job BEFORE the guard is written.

    Node 1 is degenerate and acquires nothing (still all-positive). Node 2 is
    degenerate and loses its cell entirely. Node 3 is not degenerate at all and
    is never in this population however many negatives it has."""
    v = _vecs(n_obs=[100, 100, 100, 100],
              n_pos=[100, 100, 100, 50],
              n_elig=[40, 40, 0, 80],
              n_ipos=[30, 40, 0, 40],
              n_ineg=[10, 0, 0, 40])
    buckets, summary = classify(v, min_count=20)
    assert buckets["c21_population"] == [0]
    assert summary["n_c21_population"] == 1
    assert summary["n_train_degenerate"] == 3

    # The fate breakdown PARTITIONS the degenerate set — no node is counted twice
    # and none is lost.
    assert buckets["fate_acquired_negatives"] == [0]
    assert buckets["fate_still_all_positive"] == [1]
    assert buckets["fate_no_longer_eligible"] == [2]
    assert (summary["fate_acquired_negatives"]
            + summary["fate_still_all_positive"]
            + summary["fate_no_longer_eligible"]) == summary["n_train_degenerate"]


def test_degeneracy_uses_the_readouts_exact_train_side_rule():
    """`(n_obs<=0)|(n_pos<=0)|(n_pos>=n_obs)` — `diag_sibling_support.py:78`,
    reproduced rather than re-derived so the two diagnostics cannot disagree about
    which nodes are degenerate."""
    v = _vecs(n_obs=[0, 10, 10, 10], n_pos=[0, 0, 10, 5],
              n_elig=[1, 1, 1, 1], n_ipos=[0, 0, 0, 0], n_ineg=[0, 0, 0, 0])
    _, summary = classify(v, min_count=20)
    assert summary["n_train_degenerate"] == 3          # all but the last


def test_the_go_line_reports_the_number_and_the_bar_without_deciding():
    """The tool prints; the human decides and records the call in the 0110 run
    log. So `go` is a plain comparison against a NAMED bar, and both appear in the
    banner — a reader can disagree with the bar without re-running anything."""
    below = _vecs(n_obs=[10] * 3, n_pos=[5] * 3, n_elig=[10] * 3,
                  n_ipos=[30] * 3, n_ineg=[30] * 3)
    _, s = classify(below, min_count=20)
    assert s["n_nodes_clearing_both"] == 3
    assert s["go_threshold"] == dic.GO_NODE_THRESHOLD
    assert s["go"] is False
    banner = dic.format_census_report(s)
    assert "NO-GO" in banner and str(dic.GO_NODE_THRESHOLD) in banner
    assert "20/20" in banner

    big = {**s, "n_nodes_clearing_both": dic.GO_NODE_THRESHOLD + 1,
           "go": True}
    assert "=> GO" in dic.format_census_report(big)


def test_the_banner_reports_only_counts_of_nodes():
    """EGRESS: cells under 20 are not disclosable, so the printed banner carries
    counts of NODES and label-space totals only — never a per-node cell. The
    per-node table goes to the workspace-internal JSON sidecar."""
    v = _vecs(n_obs=[100, 100], n_pos=[100, 50], n_elig=[40, 80],
              n_ipos=[30, 40], n_ineg=[3, 40])
    _, s = classify(v, min_count=20)
    banner = dic.format_census_report(s)
    assert "EGRESS" in banner
    # the small cell (n_ineg=3) must not appear as such anywhere in the banner
    assert "n_incident_neg" not in banner


# --------------------------------------------------------------------------- #
# 3. Closure discipline (RC.4 / ADR 0047 addendum).                            #
# --------------------------------------------------------------------------- #
def test_an_empty_partition_emits_no_partial_at_all():
    """The identity is a `None` SENTINEL, not a tuple of dense `(C,)` zeros: at
    C=3,820 five float64 partials pickle to ~153 KB, which is under the 1 MB
    auto-broadcast threshold — but the sentinel is doctrine, it costs nothing, and
    this diagnostic is the template the next one copies. An empty partition must
    therefore allocate nothing."""
    assert dic.census_partial(iter([]), 3820) == []


def test_the_combiner_handles_the_None_identity_on_either_side():
    part = dic.census_partial(iter([_row([1, 0], [1, 1], [])]), 2)[0]
    assert dic.census_combine(None, part) is part
    assert dic.census_combine(part, None) is part
    assert dic.census_combine(None, None) is None
    both = dic.census_combine(part, part)
    assert list(both[2]) == [2, 2]


def test_partials_are_allocated_inside_the_kernel_not_captured():
    """The arrays exist only after the kernel sees a row, so nothing array-shaped
    can be captured by the closure that ships to the executors."""
    src = Path(dic.__file__).read_text()
    # The identity handed to treeAggregate is the sentinel, never an array.
    assert "treeAggregate(None," in src
    # Exactly two allocations in the whole module: the partition kernel's
    # executor-side partials, and the driver's empty-corpus fallback. A third
    # would mean something array-shaped was built where a closure could capture
    # it.
    assert src.count("np.zeros") == 2


def test_two_partitions_sum_to_the_single_partition_answer():
    """The reduction is a plain elementwise sum, so partitioning must not matter —
    the property the treeAggregate depends on."""
    rows = [_row([1, 1, 0], [1, 1, 1], []),
            _row([1, 0, 0], [1, 1, 1], [0]),
            _row([0, 1, 0], [1, 1, 0], [2]),
            _row([1, 1, 1], [1, 0, 1], [1])]
    whole = _tally(rows, 3)
    split = dic.census_combine(_tally(rows[:1], 3), _tally(rows[1:], 3))
    for a, b in zip(whole, split):
        assert list(a) == list(b)


@pytest.mark.parametrize("bucket", ["powered_both", "c21_population",
                                    "no_eligible"])
def test_bucket_ids_are_sorted_engine_ids(bucket):
    v = _vecs(n_obs=[100] * 4, n_pos=[100, 50, 50, 100],
              n_elig=[40, 80, 0, 40], n_ipos=[30, 40, 0, 30],
              n_ineg=[30, 40, 0, 0])
    buckets, _ = classify(v, min_count=20)
    assert buckets[bucket] == sorted(buckets[bucket])
    assert all(isinstance(c, int) for c in buckets[bucket])


# --------------------------------------------------------------------------- #
# 4. The two refusals: no cache HIT, and no E1 witness.                        #
# --------------------------------------------------------------------------- #
_MANIFEST = {
    "model_class": "gated_pc", "C": 3, "K": 5, "min_label_count": 20,
    "dag_source": "mondo_native", "extra_domains": ["drug"], "disease": "rare6",
    "min_n": 0, "strip_mode": "test_only", "label_mask_mode": "closure",
    "window_mode": "lookback", "lookback_days": 1825, "label_window_days": 365,
    "n_bg": 8, "tpn": 1,
    "corpus_manifest": {
        "dag_source": "mondo_native", "disease": "rare6", "cdr": "p.d",
        "billing": "bp", "source_table": "condition_era",
        "extra_domains": ["drug"], "index_mode": "population", "person_mod": 1,
        "vocab_size": 5000, "min_df": 20, "min_patient_count": 20,
        "doc_min_length": 10, "min_n": 0, "holdout_frac": 0.2, "n_bg": 8,
        "tpn": 1, "strip_mode": "test_only", "lookback_days": 1825,
        "label_window_days": 365, "label_mask_mode": "closure",
        "emit_labels": True, "window_mode": "lookback", "prior_obs_days": 0,
        "window_days": 0, "mondo_version": "2026-06-02", "mondo_branch": "",
        "min_positives": 100, "mondo_cache_dir": "data/mondo",
        "dag_collapse": False, "preindex_closure": True, "cache_uri": None,
    },
}


class _NoWitnessBundle:
    train_df = None
    test_df = None
    parent_int = {}
    int2cid = {}
    cid2int = {}


def _census_run_dir(tmp_path):
    import json as _json
    run = tmp_path / "0110-native-mondo-label-space"
    run.mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(_json.dumps(_MANIFEST))
    # `resolve_run_dir` identifies a finished gated_pc run by this artifact.
    np.savez(run / "gated_pc_result.npz", alpha=np.ones(5))
    return run


def _patch_session(monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def _session(**_kw):
        yield None

    monkeypatch.setattr(dic, "make_spark_session", _session)


def test_a_cache_MISS_is_refused_rather_than_rebuilt(monkeypatch, tmp_path,
                                                     capsys):
    """A diagnostic never pays a rebuild (`diag_sibling_support`'s rule): a MISS
    means run the readout first, not spend ~5 min of BigQuery here."""
    import _case_finding_cache as ccache

    _patch_session(monkeypatch)
    monkeypatch.setattr(ccache, "try_load", lambda *a, **k: None)
    rc = dic.main(["--run-dir", str(_census_run_dir(tmp_path)),
                   "--cache-uri", "file:///nowhere"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "cache MISS" in out and "never rebuilds" in out


def test_a_bundle_without_the_E1_witness_is_refused_by_name(monkeypatch,
                                                            tmp_path, capsys):
    """R1.4's acceptance: the census asks a bundle for `preindexClosure`, and a
    bundle built without `--preindex-closure` must fail with a message naming the
    key, the cache uri and the fix — NOT with a Spark AnalysisException from a
    missing column, which names none of them."""
    import _case_finding_cache as ccache

    _patch_session(monkeypatch)
    monkeypatch.setattr(ccache, "try_load", lambda *a, **k: _NoWitnessBundle())
    rc = dic.main(["--run-dir", str(_census_run_dir(tmp_path)),
                   "--cache-uri", "hdfs:///c"])
    assert rc == 3
    out = capsys.readouterr().out
    assert "NO pre-index closure witness" in out
    assert "hdfs:///c" in out and "--preindex-closure" in out


@pytest.mark.slow
def test_end_to_end_over_a_real_cached_bundle(spark, tmp_path, capsys,
                                              monkeypatch):
    """The treeAggregate path against real Spark rows and a real cached bundle:
    the key recompute, the HIT, the witness check, the counts, the banner and the
    JSON sidecar. Catches everything the pure kernel cannot — the array<int>
    decode, the Row subscript, and that the column survives save/try_load."""
    import json as _json
    from contextlib import contextmanager

    import _case_finding_cache as ccache
    import gated_pc_readout as gpr
    import preindex_closure as pic
    from charmpheno.omop.multi_domain import MultiDomainBundle
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql.types import (ArrayType, DoubleType, IntegerType, LongType,
                                   StringType, StructField, StructType)

    schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("person_id", LongType(), False),
        StructField("features_0", VectorUDT(), False),
        StructField("label", ArrayType(DoubleType()), False),
        StructField("labelMask", ArrayType(DoubleType()), False),
        StructField(pic.PREINDEX_CLOSURE_COL, ArrayType(IntegerType()), False),
    ])
    # The same four documents as the hand-computed fixture above, so the numbers
    # this run prints are the ones already checked by eye.
    fixture = [([1., 1., 0.], [1., 1., 1.], []),
               ([1., 0., 0.], [1., 1., 1.], [0]),
               ([0., 1., 0.], [1., 1., 0.], [2]),
               ([1., 1., 1.], [1., 0., 1.], [1])]
    rows = [(f"population:{i}", i, Vectors.dense([1.0, 0.0]), y, m, r)
            for i, (y, m, r) in enumerate(fixture)]
    df = spark.createDataFrame(rows, schema)
    bundle = MultiDomainBundle(
        train_df=df, test_df=df, parent_int={0: [], 1: [0], 2: [0]},
        int2cid={0: -1, 1: 1001, 2: 1002}, cid2int={-1: 0, 1001: 1, 1002: 2},
        vocab_maps=[{101: 0, 102: 1}],
        name_by_id={-1: "root", 1001: "a", 1002: "b"}, ledger={"K_nodes": 2})
    bundle.preindex_closure = pic.preindex_witness()

    uri = f"file://{tmp_path}/census-cache"
    ccache.save(spark, bundle, uri, gpr.bundle_key_from_manifest(_MANIFEST))

    @contextmanager
    def _session(**_kw):
        yield spark

    monkeypatch.setattr(dic, "make_spark_session", _session)
    run = _census_run_dir(tmp_path)
    rc = dic.main(["--run-dir", str(run), "--cache-uri", uri, "--min-count", "1"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "incident-eligibility census over C=3" in out
    assert "GATE:" in out and "EGRESS:" in out

    payload = _json.loads((run / "incident_census.json").read_text())
    assert payload["per_node"]["n_incident_eligible"] == [3.0, 3.0, 3.0]
    assert payload["per_node"]["n_incident_pos"] == [2.0, 2.0, 1.0]
    assert payload["per_node"]["n_incident_neg"] == [1.0, 1.0, 2.0]
    assert payload["summary"]["n_nodes_clearing_both"] == 3
    assert payload["summary"]["preindex_witness"] == pic.preindex_witness()
    assert "NOT disclosable" in payload["egress_note"]
