"""The INCIDENT-ELIGIBILITY CENSUS (E-census) — the program's GO/NO-GO gate.

WHY THIS RUNS BEFORE ANYTHING ELSE IS BUILT
-------------------------------------------
Every per-node AUC/AP in exps 0104/0109 is a BLEND of two different questions
that share one metric (spec §1):

  * **tracking** — the patient already carried the condition before the index, and
    it is coded again in the label window; the lookback is full of its workup;
  * **onset prediction** — the condition is absent from the pre-index record and
    appears in the label window.

Root prevalence at 0109 is **0.9609**, so the blend is not a rounding effect. The
incident arm removes the tracking rows: for node c, a document that already
carried `closure(c)` before the index is excluded from **BOTH** classes (spec D2 —
excluding it from the positives only would be a different and wrong estimator).

That subtraction may leave nothing behind. If most patients are prior carriers of
most of the upper DAG, the incident cells starve and **the incident macro is not a
deliverable**. This diagnostic answers that question for a few minutes of cluster
time and no fit at all — the same move `diag-sibling-support` made for degeneracy,
and for the same reason: corpus properties are checkable before any model is fit.

    GO / NO-GO (spec §E-census): GO iff a few hundred nodes clear min_count on
    BOTH incident classes. On NO-GO, E2/E3/E4 are NOT BUILT. Either way the
    numbers are recorded as a finding — the distribution of incident-eligible
    support across the label space is the most decision-relevant number in the
    program and the cheapest. THE THRESHOLD IS PRINTED, THE HUMAN DECIDES.

WHAT IT COUNTS (RC.1)
---------------------
One `treeAggregate` over the TRAIN split's `(label, labelMask, preindexClosure)`,
giving per node c:

  * `n_incident_eligible` — docs with `c ∉ R_d` (D2);
  * `n_incident_pos`      — eligible ∧ `label[c] == 1` (D3);
  * `n_incident_neg`      — eligible ∧ observed under the mask ∧ `label[c] == 0`
                            (D4). The mask is what makes a negative a NEGATIVE
                            rather than an unobserved cell: under
                            `label_mask_mode="closure"` a node is observed only on
                            rows inside its parent's closure, so an unmasked zero
                            is "not asked", not "asked and answered no".

and the prevalent-side `(n_obs, n_pos)` the readout's own degeneracy rule reads,
so the two populations can be compared in one pass.

WHAT IT REPORTS (RC.2 / RC.3)
-----------------------------
  * per-bucket totals and the count of nodes clearing `min_count` on **both**
    incident classes — the GO/NO-GO number;
  * the **C2.1 population** (RC.3): nodes whose TRAIN cell was DEGENERATE
    (`(n_obs<=0)|(n_pos<=0)|(n_pos>=n_obs)`, `diag_sibling_support.py:78`) and
    which ACQUIRE incident negatives. Those carry a constant prediction column
    (`gated_pc_cloud.py:856-858`); under the prevalent mask they are all-positive
    and `_score_label` skips them, but under the incident mask the prior carriers
    that made them all-positive leave both classes, so they become non-degenerate,
    score `roc_auc_score(y, const)` = **exactly 0.5**, `skipped: None`, and enter
    the macro. This is the size of the population E2/R2.1's guard has to catch,
    measured BEFORE that guard is written rather than hoped about. Post-0110 it
    should be small (the native build makes the subsumed-sibling trap structurally
    impossible); the audit's "up to 619" is a 0104/0109-vintage figure and is NOT
    a post-0110 expectation.
  * the constant-head FATE breakdown: still-all-positive / acquired-negatives /
    no-longer-eligible-at-all.

EGRESS (spec §8, `evaluate.py:76-78`)
-------------------------------------
Counts under 20 are NOT DISCLOSABLE. The per-node table is written to the run dir
(workspace-internal); only the counts-OF-NODES summary in the banner is
disclosable. Nothing under 20 leaves the workspace.

CLOSURE DISCIPLINE (RC.4 / ADR 0047 addendum)
---------------------------------------------
The reduction identity is a `None` SENTINEL with identity handling in the
combiner; partials are allocated EXECUTOR-side by the partition kernel; the driver
substitutes zeros only in the empty-corpus case. At C≈3,820 five float64 `(C,)`
partials pickle to ~153 KB — under the 1 MB auto-broadcast threshold, so a dense
zero would not itself be a violation here. The sentinel is used ANYWAY: it is
doctrine, it costs nothing, and this diagnostic is the template the next one
copies. Verify by telemetry — the `disk_telemetry:` `pyspark-*` dir must be flat
in passes, not linear.

Bundle located exactly like `gated_pc_readout` and `diag_sibling_support`:
recompute the cache key from the run's manifest and REQUIRE a HIT. A diagnostic
never pays a rebuild — run the readout first if the cache is cold. And it REFUSES
LOUDLY a bundle without E1's witness, rather than dying on a missing column.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from _driver_common import _phase, configure_logging, make_spark_session
from gated_pc_readout import bundle_key_from_manifest, resolve_run_dir
from preindex_closure import (PREINDEX_CLOSURE_COL, bundle_preindex_witness,
                              require_preindex_closure)

# The publishing floor AND the powering floor happen to share a value here; they
# are different things and are kept structurally separate everywhere else (0110
# plan §: never conflate `min_positives`, an internal dial, with the egress
# floor). 20 is the All-of-Us disclosure floor (`evaluate.py:76-78`).
DEFAULT_MIN_COUNT = 20

# The spec's GO bar: "a few hundred nodes clear min_count on BOTH classes". Not a
# threshold the tool enforces — it PRINTS the number and the bar, and the human
# makes the call and records it in the 0110 experiment doc's run log.
GO_NODE_THRESHOLD = 300


# --------------------------------------------------------------------------- #
# The pure kernel: one document's contribution, and the whole-corpus tally.    #
# --------------------------------------------------------------------------- #
def census_partial(rows, C, *, label_col="label", mask_col="labelMask",
                   preindex_col=PREINDEX_CLOSURE_COL):
    """Fold an iterable of rows into five `(C,)` float64 vectors, or None.

    Returns `None` for an empty partition — the reduction's identity is a
    sentinel, and the arrays are allocated HERE, executor-side, so nothing
    array-shaped ever rides the task closure (ADR 0047 addendum).

    The five, per node c:
      n_obs     prevalent observed cells (the readout's own denominator);
      n_pos     prevalent positives;
      n_elig    incident-eligible docs — `c ∉ R_d`, D2, independent of the mask
                because eligibility is about the PRE-index record, not about
                whether this run chose to observe the cell;
      n_ipos    eligible ∧ label==1 (D3);
      n_ineg    eligible ∧ observed ∧ label==0 (D4).

    `n_ipos` is deliberately NOT masked while `n_ineg` is: a positive is attested
    (the doc gained `closure(c)`, which is a fact about the label window), whereas
    a negative only exists where the cell was OBSERVED — an unmasked zero under
    `label_mask_mode="closure"` means "not asked", not "asked and answered no".
    So `n_ipos + n_ineg <= n_elig`, and the gap is the unobserved-negative mass.
    """
    C = int(C)
    out = None
    for r in rows:
        y = np.asarray(getattr(r[label_col], "toArray",
                               lambda: r[label_col])(), float)
        m = np.asarray(getattr(r[mask_col], "toArray",
                               lambda: r[mask_col])(), float)
        if out is None:
            out = [np.zeros(C) for _ in range(5)]
        n_obs, n_pos, n_elig, n_ipos, n_ineg = out
        n_obs += m
        n_pos += y * m
        # D2: eligible everywhere EXCEPT the pre-index closure. Built per row and
        # thrown away per row; the only (C,) arrays that survive the loop are the
        # five accumulators.
        elig = np.ones(C)
        for c in (r[preindex_col] or ()):
            c = int(c)
            if 0 <= c < C:
                elig[c] = 0.0
        n_elig += elig
        n_ipos += elig * y
        n_ineg += elig * m * (1.0 - y)
    return [out] if out is not None else []


def census_combine(a, b):
    """The `None`-identity combiner: whichever side is missing contributes
    nothing, and two present sides add elementwise."""
    if a is None:
        return b
    if b is None:
        return a
    return [x + y for x, y in zip(a, b)]


def classify_census(n_obs, n_pos, n_elig, n_ipos, n_ineg, *,
                    min_count=DEFAULT_MIN_COUNT):
    """Bucket every node from the five count vectors. Pure; no Spark.

    Returns `(buckets, summary)`.

    Buckets (engine ids, sorted):
      powered_both        — `n_ipos >= min_count` AND `n_ineg >= min_count`: the
                            nodes the incident macro could actually be computed
                            on. **This count IS the gate.**
      powered_pos_only / powered_neg_only — one class clears, the other does not;
                            named separately because WHICH side starves says which
                            mechanism is biting (all-carrier vs never-observed).
      starved             — neither class clears.
      no_eligible         — `n_elig <= 0`: every document is a prior carrier, so
                            the node has no incident cell at all.

      c21_population      — RC.3: the node's TRAIN cell was degenerate
                            (`(n_obs<=0)|(n_pos<=0)|(n_pos>=n_obs)`) AND it
                            acquires incident negatives (`n_ineg > 0`). These are
                            exactly the constant-prediction columns that stop
                            being skipped under the incident mask and would score
                            a hard 0.5 inside the macro without E2/R2.1's guard.

    Constant-head FATE (a partition of the degenerate set, so the three sum to it):
      fate_still_all_positive — degenerate and still no incident negatives, but
                            some eligible rows remain;
      fate_acquired_negatives — == `c21_population`;
      fate_no_longer_eligible — degenerate and `n_elig <= 0`: the node did not
                            become scoreable, it disappeared.
    """
    n_obs, n_pos, n_elig, n_ipos, n_ineg = (
        np.asarray(v, float) for v in (n_obs, n_pos, n_elig, n_ipos, n_ineg))
    C = int(n_obs.shape[0])

    # The readout's exact TRAIN-side degeneracy rule (diag_sibling_support.py:78),
    # reproduced rather than re-derived so the two diagnostics cannot disagree.
    degenerate = (n_obs <= 0) | (n_pos <= 0) | (n_pos >= n_obs)

    pos_ok = n_ipos >= min_count
    neg_ok = n_ineg >= min_count
    none_elig = n_elig <= 0

    buckets = {
        "powered_both": [], "powered_pos_only": [], "powered_neg_only": [],
        "starved": [], "no_eligible": [], "c21_population": [],
        "fate_still_all_positive": [], "fate_acquired_negatives": [],
        "fate_no_longer_eligible": [],
    }
    for c in range(C):
        if none_elig[c]:
            buckets["no_eligible"].append(c)
        elif pos_ok[c] and neg_ok[c]:
            buckets["powered_both"].append(c)
        elif pos_ok[c]:
            buckets["powered_pos_only"].append(c)
        elif neg_ok[c]:
            buckets["powered_neg_only"].append(c)
        else:
            buckets["starved"].append(c)

        if not degenerate[c]:
            continue
        if none_elig[c]:
            buckets["fate_no_longer_eligible"].append(c)
        elif n_ineg[c] > 0:
            buckets["c21_population"].append(c)
            buckets["fate_acquired_negatives"].append(c)
        else:
            buckets["fate_still_all_positive"].append(c)

    summary = {
        "C": C,
        "min_count": int(min_count),
        "go_threshold": int(GO_NODE_THRESHOLD),
        "n_nodes_clearing_both": len(buckets["powered_both"]),
        "n_nodes_clearing_pos_only": len(buckets["powered_pos_only"]),
        "n_nodes_clearing_neg_only": len(buckets["powered_neg_only"]),
        "n_nodes_starved": len(buckets["starved"]),
        "n_nodes_no_eligible": len(buckets["no_eligible"]),
        "n_train_degenerate": int(degenerate.sum()),
        "n_c21_population": len(buckets["c21_population"]),
        "fate_still_all_positive": len(buckets["fate_still_all_positive"]),
        "fate_acquired_negatives": len(buckets["fate_acquired_negatives"]),
        "fate_no_longer_eligible": len(buckets["fate_no_longer_eligible"]),
        # Totals over the label space, not per-node cells: no small cell here.
        "total_incident_eligible": float(n_elig.sum()),
        "total_incident_pos": float(n_ipos.sum()),
        "total_incident_neg": float(n_ineg.sum()),
        "total_prevalent_obs": float(n_obs.sum()),
        "total_prevalent_pos": float(n_pos.sum()),
        "go": bool(len(buckets["powered_both"]) >= GO_NODE_THRESHOLD),
    }
    return buckets, summary


def format_census_report(summary) -> str:
    """The banner, in `diag_sibling_support` / `format_collapse_report` style.

    Counts of NODES only — every number here is a count over the label space, not
    a per-node cell count, so nothing under the disclosure floor appears."""
    go = "GO" if summary["go"] else "NO-GO"
    mc = summary["min_count"]
    return "\n".join([
        f"[census] incident-eligibility census over C={summary['C']} nodes "
        f"(TRAIN split; eligibility = c NOT IN R_d, spec D2)",
        f"[census]   nodes clearing min_count={mc} on BOTH incident classes: "
        f"{summary['n_nodes_clearing_both']}",
        f"[census]   ...positives only: {summary['n_nodes_clearing_pos_only']}; "
        f"negatives only: {summary['n_nodes_clearing_neg_only']}; "
        f"neither: {summary['n_nodes_starved']}; "
        f"no eligible doc at all: {summary['n_nodes_no_eligible']}",
        f"[census]   train-degenerate nodes: {summary['n_train_degenerate']}; "
        f"of those, C2.1 POPULATION (acquire incident negatives, would score a "
        f"hard 0.5 in the macro without the R2.1 constant-column guard): "
        f"{summary['n_c21_population']}",
        f"[census]   constant-head fate: still-all-positive "
        f"{summary['fate_still_all_positive']} / acquired-negatives "
        f"{summary['fate_acquired_negatives']} / no-longer-eligible "
        f"{summary['fate_no_longer_eligible']}",
        f"[census] GATE: {summary['n_nodes_clearing_both']} nodes clear "
        f"{mc}/{mc} vs a bar of ~{summary['go_threshold']} => {go}",
        "[census] The bar is the spec's 'a few hundred'; this line REPORTS, it "
        "does not decide. Record the call (and these counts) in the 0110 "
        "experiment doc's run log. On NO-GO, E2/E3/E4 are not built and that "
        "entry is the program's terminal record for them.",
        "[census] EGRESS: the per-node table is workspace-internal (cells < 20 "
        "are not disclosable); only these counts-of-nodes are.",
    ])


# --------------------------------------------------------------------------- #
# The driver.                                                                  #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cache-uri", default=None)
    p.add_argument("--label-col", default="label")
    p.add_argument("--mask-col", default="labelMask")
    p.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                   help="per-class floor for 'this node is scoreable in the "
                        "incident arm'. ALSO the All-of-Us disclosure floor: the "
                        "per-node table stays in the workspace regardless.")
    p.add_argument("--out", default=None,
                   help="JSON sidecar path (default: <run-dir>/incident_census.json).")
    args = p.parse_args(argv)
    configure_logging()

    run_dir = resolve_run_dir(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    C = int(manifest["C"])
    cm = manifest.get("corpus_manifest") or {}
    cache_uri = args.cache_uri or cm.get("cache_uri")
    key = bundle_key_from_manifest(manifest)

    with make_spark_session(app_name="diag-incident-census") as spark:
        from _case_finding_cache import try_load

        with _phase("load cached bundle"):
            bundle = try_load(spark, cache_uri, key)
            if bundle is None:
                print(f"[census] ERROR: cache MISS at {cache_uri}/{key} — this "
                      "diagnostic never rebuilds; run gated_pc_readout first so "
                      "the bundle is cached.", flush=True)
                return 2
            # R1.4: refuse a bundle without the witness BY NAME, before any Spark
            # column reference, so the failure says "rebuild with
            # --preindex-closure" instead of raising an AnalysisException.
            try:
                preindex_col = require_preindex_closure(
                    bundle, key=key, cache_uri=cache_uri)
            except ValueError as exc:
                print(f"[census] ERROR: {exc}", flush=True)
                return 3

        with _phase("incident census (one treeAggregate over TRAIN)"):
            cols = (args.label_col, args.mask_col, preindex_col)

            def _local(rows, _C=C, _cols=cols):
                # The partial arrays are allocated HERE, on the executor, and the
                # identity below is a None sentinel: nothing array-shaped rides
                # the task closure (ADR 0047 addendum / RC.4).
                return census_partial(rows, _C, label_col=_cols[0],
                                      mask_col=_cols[1], preindex_col=_cols[2])

            totals = (bundle.train_df.select(*cols).rdd
                      .mapPartitions(_local)
                      .treeAggregate(None, census_combine, census_combine,
                                     depth=2))
        if totals is None:                       # empty corpus: the only case the
            totals = [np.zeros(C) for _ in range(5)]   # driver allocates zeros
        n_obs, n_pos, n_elig, n_ipos, n_ineg = totals

        buckets, summary = classify_census(n_obs, n_pos, n_elig, n_ipos, n_ineg,
                                           min_count=args.min_count)
        summary["bundle_key"] = key
        summary["preindex_witness"] = bundle_preindex_witness(bundle)
        print(format_census_report(summary), flush=True)

        def _name(c):
            cid = bundle.int2cid.get(c, c)
            return f"{cid} {bundle.name_by_id.get(cid, '?')}"

        # A handful of eyeball-checkable examples per interesting bucket. These
        # print NODE IDENTITIES, never their cell counts, so the banner stays
        # disclosable; the counts live in the workspace-internal sidecar.
        for k in ("powered_both", "c21_population", "no_eligible"):
            for c in buckets[k][:5]:
                print(f"[census]   {k}: {_name(c)}", flush=True)

        out_path = args.out or str(run_dir / "incident_census.json")
        payload = {
            "summary": summary,
            "buckets": {k: [int(c) for c in v] for k, v in buckets.items()},
            # WORKSPACE-INTERNAL (R3.5 / egress): per-node cells, many under 20.
            "per_node": {
                "int2cid": {str(i): int(c) for i, c in bundle.int2cid.items()},
                "n_obs": [float(x) for x in n_obs],
                "n_pos": [float(x) for x in n_pos],
                "n_incident_eligible": [float(x) for x in n_elig],
                "n_incident_pos": [float(x) for x in n_ipos],
                "n_incident_neg": [float(x) for x in n_ineg],
            },
            "egress_note": (
                "per_node cells are NOT disclosable (counts < 20 leave the "
                "workspace never); only `summary`'s counts-of-nodes are."),
        }
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[census] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
