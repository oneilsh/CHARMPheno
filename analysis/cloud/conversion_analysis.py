"""FUTURE-CONVERSION analysis (spec E4) — PU channel 1, measured as a LOWER BOUND.

THE QUESTION
------------
Of the documents scored INCIDENT NEGATIVE for node c — eligible (c ∉ R_d, spec D2)
and not gaining `closure(c)` in the label window — how many are diagnosed with c
LATER, beyond the label horizon?

Every such document is a negative the corpus asserts and the record eventually
contradicts. Their fraction is **PU channel 1**: label noise from a diagnosis that
had not happened yet at labelling time.

    EVERY NUMBER HERE IS A LOWER BOUND, and the framing is mandatory (R4.6).
    Channel 1 is the only channel that can be measured from this record at all.
    Channel 2 (never diagnosed anywhere) and Channel 3 (diagnosed in care this CDR
    does not see) are UNMEASURED, and no bound on care fragmentation is claimed or
    implied. A conversion rate of X% means "at least X% of these negatives are
    wrong", never "X% of them are wrong".

    EXPECT THE BOUND TO BE LOOSE (R4.7). At 0109's root prevalence 0.9609 most
    nodes' "did not gain c this year" is a weak negative and conversion will be
    high across the board. That is a FINDING, recorded as such — not a failure, and
    not something to tune toward a tighter number.

WHAT IT REPORTS (R4.5)
----------------------
Per node and pooled, at 1y / 2y / 3y past the end of the label window:

    conversion(c, h) = |{d : incident-negative for c, first attestation of
                          closure(c) lands in (label_end, label_end + h]}|
                     / |{d : incident-negative for c, OBSERVED through
                          label_end + h}|

and the same rates split by MODEL-SCORE DECILE within each node.

RIGHT-CENSORING IS THE DENOMINATOR (R4.4). A person whose observation period ends
before `label_end + h` cannot be seen converting at h, so counting them as a
non-converter turns a coverage artifact into a "finding". The denominator is
therefore gated on `observation_period_end_date` at EACH horizon — the follow-up
clause of `_window_observed_cohort` (`cohorts.py:693-729`), re-derived driver-side
in `conversion_sidecar.observation_gate_frame` rather than imported, because
`cohorts.py` is folded into every bundle cache key through `cohort_defs_version()`
and editing it would orphan every cached bundle in the repo. The denominators
therefore SHRINK MONOTONICALLY with the horizon, which is the check that the gate
is actually applied.

THE DECILE TABLE IS CASE-FINDING VALIDATION (R4.8). If the top score decile
converts at a materially higher rate than the bottom, the model is finding FUTURE
CASES among its own "negatives" — which is the case-finding claim, measured on the
one population where it can be measured. A flat decile profile says the opposite,
just as loudly.

WHAT IT NEEDS, AND REFUSES WITHOUT (diag discipline)
-----------------------------------------------------
  * a cached bundle — HIT REQUIRED (exit 2). A diagnostic never pays a rebuild.
  * E1's pre-index closure WITNESS (exit 3) — without eligibility there is no
    incident negative to ask the question about, and the failure must name the fix
    rather than dying on a missing Spark column.
  * the E4 sidecar's witness + a sidecar HIT (exit 4) — `make
    build-conversion-sidecar ID=N` builds it. Nothing else in the repo carries a
    post-label-window date.
  * the run's saved fit, to transform the test split (exit 5).
  * for the score deciles (`--deciles on`), the run's PERSISTED readout heads
    (`readout_heads_gated_pc.npz`, written by every readout fit under current
    code). Scoring is one distributed mapPartitions over those saved params — it
    NEVER re-fits (the re-fit was the disk cascade that killed the cluster on
    0110). If the npz is absent (a fit predating that code) `--deciles on` does
    NOT re-fit: it prints the one-time fix (`make gated-pc-readout ID=<n>`) and
    produces the OVERALL table only. `--deciles off` (the default) skips scoring
    entirely.

EGRESS
------
The per-node table is WORKSPACE-INTERNAL (`conversion_analysis.json` in the run
dir): per-node numerators and denominators are cell counts and many are under 20.
The printed banner carries only pooled rates and counts OF NODES.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from _driver_common import _phase, configure_logging, make_spark_session
from conversion_sidecar import (FIRST_DATE_COL, INDEX_COL, NODE_COL, OBS_END_COL,
                                PERSON_COL, try_load_index_horizon,
                                try_load_sidecar)
from gated_pc_readout import bundle_key_from_manifest, resolve_run_dir
from preindex_closure import require_preindex_closure

# The horizons, in days past the END of the label window. Three years is where the
# All-of-Us EHR record thins out enough that the censoring gate is doing most of
# the work; past that the denominator is small and the rate is noise.
DEFAULT_HORIZON_DAYS = (365, 730, 1095)

# The disclosure floor (`evaluate.py:76-78`). Applied to the PRINTED table and to
# every pooled figure; the per-node file stays in the workspace.
EGRESS_MIN_COUNT = 20

# The framing string. Carried in the JSON and printed on every table, because a
# conversion number quoted without it becomes "the contamination rate" within one
# copy-paste.
LOWER_BOUND_NOTE = (
    "LOWER BOUND on PU channel 1 (label noise from a not-yet-made diagnosis). "
    "Channels 2 (never diagnosed) and 3 (diagnosed in care this CDR cannot see) "
    "are UNMEASURED; no care-fragmentation bound is claimed. Read every rate as "
    "'at least this many of these negatives are wrong'.")


# --------------------------------------------------------------------------- #
# Pure core: the closure fold, the horizon arithmetic, the decile split.       #
# --------------------------------------------------------------------------- #
def conversion_counts(first_date, index_date, obs_end, label_window_days,
                      horizons=DEFAULT_HORIZON_DAYS):
    """`(converted, observed)` boolean arrays per horizon for one node's negatives.

    All three inputs are day offsets or absolute days as float arrays of the same
    length (one entry per candidate document); `first_date` may be `nan`, meaning
    "never attested anywhere in the record", which is a non-conversion at every
    horizon rather than a missing value.

    Per horizon `h`, with `end = index_date + label_window_days`:

      * OBSERVED  — `obs_end >= end + h`. This is the follow-up clause of
        `_window_observed_cohort`, at the horizon rather than at the label window.
        A person who drops out of the record cannot be seen converting, so they
        leave the DENOMINATOR; counting them as a non-converter is exactly the
        censoring artifact R4.4 names.
      * CONVERTED — observed AND `end < first_date <= end + h`. Strictly AFTER the
        label window: an attestation inside the window is not a conversion, it is a
        positive the label already carries (and such a document is not an incident
        negative in the first place).

    Both are monotone in `h` by construction — `observed` shrinks, `converted`
    grows within the shrinking set — which is the property the fixtures assert.
    """
    first = np.asarray(first_date, dtype=float)
    end = np.asarray(index_date, dtype=float) + float(label_window_days)
    oe = np.asarray(obs_end, dtype=float)
    out = {}
    for h in horizons:
        h = float(h)
        observed = oe >= (end + h)
        converted = observed & np.isfinite(first) & (first > end) & (first <= end + h)
        out[int(h)] = (converted, observed)
    return out


def decile_of(scores):
    """0-9 decile index per row, by rank — ties broken by position, empty-safe.

    Rank-based rather than value-based: a node's probabilities can be piled on a
    handful of values (a shallow head, a constant fallback), and equal-width value
    bins would then put 90% of the cohort in one decile and call the result a
    gradient. Decile 9 is the TOP (highest score), which is the direction R4.8's
    claim is stated in."""
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(s, kind="stable")
    rank = np.empty(n, dtype=float)
    rank[order] = np.arange(n, dtype=float)
    return np.minimum((rank * 10.0 / n).astype(int), 9)


def _rate(num, den):
    return (float(num) / float(den)) if den else None


def node_conversion_table(converted_observed, scores=None,
                          min_count=EGRESS_MIN_COUNT):
    """One node's row: overall and per-decile `(n_converted, n_observed, rate)`.

    `min_count` gates only the DISCLOSABLE flag on each cell, not the arithmetic —
    the caller decides what leaves the workspace. Per-decile cells are ten times
    smaller than the overall one and will mostly be under the floor; that is
    expected and is why the decile table is a workspace-internal artifact whose
    pooled summary is what gets published."""
    out = {"horizons": {}}
    for h, (converted, observed) in sorted(converted_observed.items()):
        n_obs = int(observed.sum())
        n_conv = int(converted.sum())
        row = {"n_observed": n_obs, "n_converted": n_conv,
               "rate": _rate(n_conv, n_obs),
               "disclosable": bool(n_obs >= min_count)}
        if scores is not None and n_obs:
            dec = decile_of(np.asarray(scores, dtype=float)[observed])
            c_obs = np.asarray(converted, dtype=bool)[observed]
            row["deciles"] = [
                {"decile": int(d),
                 "n_observed": int((dec == d).sum()),
                 "n_converted": int(c_obs[dec == d].sum()),
                 "rate": _rate(int(c_obs[dec == d].sum()), int((dec == d).sum())),
                 "disclosable": bool(int((dec == d).sum()) >= min_count)}
                for d in range(10)]
        out["horizons"][int(h)] = row
    return out


def pool_tables(per_node, min_count=EGRESS_MIN_COUNT):
    """Pooled rates over nodes, and the decile profile that validates case-finding.

    Pooling is over CELLS, not a mean of per-node rates: the question "how many of
    these negatives are wrong" is a count question, and a macro over nodes would
    weight a 25-document node like a 25,000-document one. Only nodes whose cell
    clears `min_count` at that horizon contribute, so nothing under the floor is
    inside a published number either."""
    horizons = sorted({h for t in per_node.values() for h in t["horizons"]})
    pooled = {}
    for h in horizons:
        rows = [t["horizons"][h] for t in per_node.values()
                if h in t["horizons"] and t["horizons"][h]["disclosable"]]
        n_obs = sum(r["n_observed"] for r in rows)
        n_conv = sum(r["n_converted"] for r in rows)
        dec = [{"decile": d, "n_observed": 0, "n_converted": 0} for d in range(10)]
        for r in rows:
            for cell in r.get("deciles", ()):
                dec[cell["decile"]]["n_observed"] += cell["n_observed"]
                dec[cell["decile"]]["n_converted"] += cell["n_converted"]
        for cell in dec:
            cell["rate"] = _rate(cell["n_converted"], cell["n_observed"])
        pooled[int(h)] = {
            "n_nodes": len(rows), "n_observed": n_obs, "n_converted": n_conv,
            "rate": _rate(n_conv, n_obs), "deciles": dec,
            "top_minus_bottom": (
                None if (dec[9]["rate"] is None or dec[0]["rate"] is None)
                else dec[9]["rate"] - dec[0]["rate"]),
        }
    return pooled


def format_conversion_report(pooled, meta) -> str:
    """The banner: pooled rates and counts of NODES only — nothing per-cell."""
    lines = [
        f"[conversion] future-conversion of INCIDENT NEGATIVES "
        f"({meta.get('arm', 'gated_pc')}; label window "
        f"{meta.get('label_window_days')}d)",
        f"[conversion] {LOWER_BOUND_NOTE}",
        "[conversion] claim type: PU CONTAMINATION FLOOR — not a prospective "
        "incidence estimate, not a calibration statement",
        "[conversion]   horizon   nodes   negatives(obs)   converted   rate",
    ]
    for h, p in sorted(pooled.items()):
        rate = "n/a" if p["rate"] is None else f"{p['rate']:.4f}"
        lines.append(f"[conversion]   {h:>5}d   {p['n_nodes']:>5}   "
                     f"{p['n_observed']:>13}   {p['n_converted']:>9}   {rate}")
    lines.append("[conversion]   (denominators MUST shrink with the horizon — that "
                 "is the right-censoring gate on observation_period_end_date "
                 "working, spec R4.4)")
    lines.append("[conversion] score-decile profile = CASE-FINDING VALIDATION "
                 "(R4.8): top decile converting materially above the bottom means "
                 "the model is finding future cases among its own negatives")
    for h, p in sorted(pooled.items()):
        cells = "  ".join(
            ("  n/a" if c["rate"] is None else f"{c['rate']:.3f}")
            for c in p["deciles"])
        d = ("n/a" if p["top_minus_bottom"] is None
             else f"{p['top_minus_bottom']:+.4f}")
        lines.append(f"[conversion]   {h:>5}d  d0..d9: {cells}   (top-bottom={d})")
    lines.append("[conversion] EGRESS: the per-node table is workspace-internal "
                 "(cells < 20 are not disclosable); only these pooled figures and "
                 "counts-of-nodes are.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The driver.                                                                  #
# --------------------------------------------------------------------------- #
def _to_ordinals(series):
    """A pandas date column -> float day ordinals, `nan` where NULL.

    Every date in this analysis is compared as a NUMBER of days (index + window +
    horizon), so the conversion happens once, at the boundary, rather than in the
    inner loop. Proleptic-Gregorian ordinals are exact in float64 and exact in
    float32 up to 2^24 days, comfortably past year 9999."""
    out = np.full(len(series), np.nan, dtype=float)
    for i, v in enumerate(series.tolist()):
        if v is None or v != v:
            continue
        out[i] = float(v.toordinal())
    return out


def _collect_labels_and_eligibility(df, C, preindex_col, id_col="person_id",
                                    label_col="label", mask_col="labelMask"):
    """`(y u8, mask u8, person_order, elig u8)` — the `--deciles off` collect.

    The incident mask is a pure function of the CORPUS (spec R2.3), so when no
    scores are wanted there is nothing to fit and nothing to broadcast: four
    columns come back, densified the same way `_densify_lean_blocks` densifies
    them, and the batched readout solve — the expensive half of this tool — never
    runs."""
    rows = df.select(id_col, label_col, mask_col, preindex_col).collect()
    D = len(rows)
    y = np.zeros((D, int(C)), dtype=np.uint8)
    mask = np.zeros((D, int(C)), dtype=np.uint8)
    elig = np.ones((D, int(C)), dtype=np.uint8)
    persons = []
    for i, r in enumerate(rows):
        persons.append(r[id_col])
        y[i] = np.asarray(getattr(r[label_col], "toArray",
                                  lambda: r[label_col])(), float) != 0
        mask[i] = np.asarray(getattr(r[mask_col], "toArray",
                                     lambda: r[mask_col])(), float) != 0
        for c in (r[preindex_col] or ()):
            c = int(c)
            if 0 <= c < int(C):
                elig[i, c] = 0
    return y, mask, persons, elig


def first_attestation_matrix(sidecar_person, sidecar_node, sidecar_day, persons,
                             cid2int, lay, C):
    """`(D, C)` float32 of "first attestation of closure(c)", `inf` where never.

    The three `sidecar_*` sequences are COLUMNS, not rows — `person_id`,
    `node_cid`, and the date as a day ordinal — because the sidecar is millions of
    rows at `person_mod: 1` and the driver reads it through Arrow (`toPandas`) as
    three numpy arrays rather than as Spark `Row` objects, which would cost ~100
    bytes of Python overhead each.

    THE GRAIN CHANGE HAPPENS HERE AND IS NAMED HERE (R4.3): the sidecar is per
    `(person_id, node_cid)`; `persons` is the DOCUMENT order of the collected test
    split. Under today's `PatientCohortDocSpec` + `index_mode=population` there is
    exactly one document per person, so the map is 1:1 on `person_id`; under exp
    0111's episode doc unit it would not be, and this function is where that has to
    change (a person with two documents needs two index dates, which the per-person
    horizon frame does not carry) rather than fanning out silently. A duplicate
    `person_id` in `persons` therefore RAISES here.

    The fold goes UP, not down: for each attested frontier node `f` with day `t`,
    every ancestor `a ∈ lay.closure(f)` gets `min(a, t)`. That is the same direction
    `frontier_to_label` writes the label in, it visits only the nodes the person
    actually carries, and it costs `Σ_f |closure(f)|` rather than the
    `Σ_c |subtree(c)|` a downward gather would pay per document.

    Dense because the consumer is a per-node column scan: float32 at
    (D_te≈80k, C≈2,714) is ~870 MB, the same order as the readout's own probability
    matrix and on the same driver budget. `inf` (not `nan`) is the identity for
    `min`, and `conversion_counts` reads a non-finite value as "never attested"."""
    row_of = {}
    for i, pid in enumerate(persons):
        pid = int(pid)
        if pid in row_of:
            raise ValueError(
                f"person_id {pid} appears twice in the scored split. The E4 "
                "sidecar's index/horizon frame is PER PERSON (R4.3), so a "
                "multi-document person has no single index date here — this is "
                "the doc-unit change exp 0111 makes, and this analysis needs a "
                "per-(person, index) horizon frame before it can run on one.")
        row_of[pid] = i
    out = np.full((len(persons), int(C)), np.inf, dtype=np.float32)
    # `lay.closure` is called once per DISTINCT node, not once per row: the
    # sidecar has one row per (person, node) and the same node recurs across
    # hundreds of thousands of people.
    closure_of = {}
    for pid, node_cid, day in zip(sidecar_person, sidecar_node, sidecar_day):
        i = row_of.get(int(pid))
        if i is None:                          # not in this split
            continue
        node = cid2int.get(int(node_cid))
        if node is None:                       # a code-map node the DAG pruned
            continue
        if day is None or day != day:          # NULL / NaT
            continue
        anc = closure_of.get(node)
        if anc is None:
            anc = closure_of[node] = [int(a) for a in lay.closure(int(node))
                                      if 0 <= int(a) < int(C)]
        t = np.float32(day)
        row = out[i]
        for a in anc:
            if t < row[a]:
                row[a] = t
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cache-uri", default=None)
    p.add_argument("--sidecar-uri", default=None)
    p.add_argument("--horizon-days", default=",".join(
        str(h) for h in DEFAULT_HORIZON_DAYS))
    p.add_argument("--deciles", choices=("on", "off"), default="off",
                   help="'on' SCORES the test split from the run's SAVED readout "
                        "heads (readout_heads_gated_pc.npz — one distributed "
                        "mapPartitions, NO re-fit) to get per-document scores for "
                        "the decile split; it requires that npz, which every "
                        "readout fit under current code persists (re-run `make "
                        "gated-pc-readout ID=<n>` once for a pre-existing fit), "
                        "and falls back to the overall table (never a re-fit) if "
                        "it is absent. 'off' (default) reports the overall table "
                        "only. NB: 'on' used to RE-FIT all heads from scratch — "
                        "the disk-killer that took the cluster down on 0110.")
    p.add_argument("--min-count", type=int, default=EGRESS_MIN_COUNT)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    configure_logging()

    run_dir = resolve_run_dir(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    C = int(manifest["C"])
    cm = manifest.get("corpus_manifest") or {}
    cache_uri = args.cache_uri or cm.get("cache_uri")
    key = bundle_key_from_manifest(manifest)
    horizons = tuple(int(h) for h in str(args.horizon_days).split(",") if h)
    label_window_days = int(cm.get("label_window_days") or 365)

    wit_path = run_dir / "conversion_sidecar.json"
    if not wit_path.exists():
        print(f"[conversion] ERROR: no {wit_path.name} in the run dir — this run "
              "has no E4 sidecar witness, and nothing else in the repo carries a "
              "post-label-window date. Build it first:  make "
              "build-conversion-sidecar ID=<n>", flush=True)
        return 4
    witness = json.loads(wit_path.read_text())
    sidecar_uri = args.sidecar_uri or witness.get("sidecar_uri")

    with make_spark_session(app_name="conversion-analysis") as spark:
        from _case_finding_cache import try_load

        with _phase("load cached bundle"):
            bundle = try_load(spark, cache_uri, key)
            if bundle is None:
                print(f"[conversion] ERROR: cache MISS at {cache_uri}/{key} — this "
                      "analysis never rebuilds; run gated_pc_readout first so the "
                      "bundle is cached.", flush=True)
                return 2
            try:
                preindex_col = require_preindex_closure(
                    bundle, key=key, cache_uri=cache_uri)
            except ValueError as exc:
                print(f"[conversion] ERROR: {exc}", flush=True)
                return 3

        with _phase("load the E4 sidecar"):
            first_df = try_load_sidecar(spark, sidecar_uri, witness["key"])
            horizon_df = try_load_index_horizon(spark, sidecar_uri, witness["key"])
            if first_df is None or horizon_df is None:
                print(f"[conversion] ERROR: sidecar MISS at {sidecar_uri}/"
                      f"{witness['key']} (the witness in {wit_path.name} names it). "
                      "Rebuild it:  make build-conversion-sidecar ID=<n>",
                      flush=True)
                return 4

        deciles_scored = False
        with _phase("score the test split (for the incident mask and the deciles)"):
            from gated_pc_cloud import (_collect_lean_proba, _read_readout_heads,
                                        incident_eval_mask)
            from gated_pc_readout import reconstruct_model
            try:
                model = reconstruct_model(run_dir, manifest)
            except (FileNotFoundError, KeyError) as exc:
                print(f"[conversion] ERROR: no scoreable saved fit in {run_dir} "
                      f"({exc}). The incident mask needs the corpus only, but the "
                      "score deciles need the fit; re-run with --deciles off if "
                      "the fit is gone.", flush=True)
                return 5
            test_scored = model.transform(bundle.test_df).cache()
            proba = None
            if args.deciles == "on":
                # SCORE from the SAVED readout heads — never re-fit. The re-fit
                # (a full batched readout solve, run purely to get per-doc scores)
                # is what filled the worker local disks and cascaded the cluster
                # dead on 0110. `_read_readout_heads` loads the params
                # `distributed_score_arm` now persists at the end of every readout
                # fit; if they are ABSENT (an old fit predating that code) we do
                # NOT fall back to a re-fit — we print the fix and produce the
                # OVERALL table only, which is the primary deliverable.
                heads = _read_readout_heads(
                    run_dir, "gated_pc", C=C,
                    K=(int(manifest.get("K") or 0) or None),
                    theta_topm=int(manifest.get("readout_theta_topm") or 0))
                if heads is None:
                    print(f"[conversion] no persisted readout heads in {run_dir} "
                          "(readout_heads_gated_pc.npz absent) — the score deciles "
                          "need them, and this analysis NEVER re-fits (the re-fit "
                          "is the disk-killer that took the cluster down on 0110). "
                          "Re-run `make gated-pc-readout ID=<n>` ONCE under current "
                          "code to persist them, or pass --deciles off. Producing "
                          "the OVERALL conversion table only.", flush=True)
                    y_te, m_te, persons, elig = _collect_labels_and_eligibility(
                        test_scored, C, preindex_col)
                else:
                    V, b_raw, const, degenerate, _hC, _hK, _hm = heads
                    # DISTRIBUTED scoring — the SAME lean kernel
                    # `distributed_score_arm` uses internally, NOT the fitting
                    # path: one mapPartitions over the test split, no moments pass,
                    # no batched solve, no train split. `elig_col` rides the same
                    # collect so the incident mask still gets E1's eligibility.
                    proba, y_te, m_te, persons, elig = _collect_lean_proba(
                        test_scored, C, V, b_raw, degenerate=degenerate,
                        const=const, theta_topm=_hm, elig_col=preindex_col)
                    deciles_scored = True
            else:
                # No scores wanted, so no solve: collect only what the INCIDENT
                # MASK needs. Deliberately not `_collect_lean_proba` — that packs a
                # per-doc (C,) probability, and handing it raw θ (K,) would build a
                # (D, K) array into a (D, C) destination.
                y_te, m_te, persons, elig = _collect_labels_and_eligibility(
                    test_scored, C, preindex_col)
            test_scored.unpersist()

        m_incident = incident_eval_mask(y_te, m_te, elig)
        if m_incident is None:
            print("[conversion] ERROR: the scored split came back with no "
                  "eligibility matrix even though the bundle carries E1's "
                  f"witness (column {preindex_col!r}). Nothing here can be "
                  "computed without it — do not read a conversion rate off the "
                  "prevalent mask.", flush=True)
            return 3
        # An INCIDENT NEGATIVE (D4): scored under the incident mask, label 0.
        neg = (m_incident.astype(bool)) & (np.asarray(y_te) == 0)

        with _phase("fold first-attestation to closure and count conversions"):
            from spark_vi.models.topic.dag_placement import DagLayout
            lay = DagLayout(bundle.parent_int,
                            n_bg=int(manifest.get("n_bg") or 0),
                            tpn=int(manifest.get("tpn") or 1))
            # Restrict BOTH sidecar frames to this split's persons IN SPARK before
            # anything reaches the driver: the first-attestation frame is one row
            # per (person, node) over the whole sampled population, which is
            # millions of rows at `person_mod: 1`. The join is against a
            # driver-built id frame (one long column, D_te rows), not a broadcast
            # of anything array-shaped.
            ids_df = spark.createDataFrame(
                [(int(p),) for p in persons], f"{PERSON_COL} long")
            first_pd = (first_df.join(ids_df, on=PERSON_COL, how="inner")
                        .toPandas())
            first_mat = first_attestation_matrix(
                first_pd[PERSON_COL].to_numpy(),
                first_pd[NODE_COL].to_numpy(),
                _to_ordinals(first_pd[FIRST_DATE_COL]),
                persons, bundle.cid2int, lay, C)
            hz_pd = (horizon_df.join(ids_df, on=PERSON_COL, how="inner")
                     .toPandas())
            hz_index = dict(zip(hz_pd[PERSON_COL].to_numpy().tolist(),
                                _to_ordinals(hz_pd[INDEX_COL]).tolist()))
            hz_obs = dict(zip(hz_pd[PERSON_COL].to_numpy().tolist(),
                              _to_ordinals(hz_pd[OBS_END_COL]).tolist()))
            # A person with no index row cannot be placed on the timeline at all
            # (nan propagates through every comparison as False); one with no
            # observation end is CENSORED EVERYWHERE (-inf fails the gate at every
            # horizon), which is the correct reading of "never observed to have
            # been observed" — excluded from the denominator, never a non-converter.
            index_day = np.array([hz_index.get(int(p), np.nan) for p in persons],
                                 dtype=float)
            obs_day = np.array([hz_obs.get(int(p), -np.inf) for p in persons],
                               dtype=float)
            obs_day[~np.isfinite(obs_day)] = -np.inf

            per_node = {}
            for c in range(C):
                rows_c = np.flatnonzero(neg[:, c])
                if rows_c.size == 0:
                    continue
                co = conversion_counts(
                    first_mat[rows_c, c], index_day[rows_c], obs_day[rows_c],
                    label_window_days, horizons=horizons)
                # Index the COLUMN, then widen: `np.asarray(proba, dtype=float)`
                # would copy the whole (D, C) float32 matrix to float64 once per
                # node, which at C≈2,700 is the difference between a diagnostic
                # and an OOM.
                scores = (None if proba is None
                          else proba[rows_c, c].astype(float))
                per_node[c] = node_conversion_table(co, scores,
                                                    min_count=args.min_count)

        pooled = pool_tables(per_node, min_count=args.min_count)
        meta = {
            "arm": "gated_pc (pc_topics_lr)",
            "claim": "PU channel 1 contamination FLOOR (discrimination-adjacent; "
                     "NOT a prospective incidence claim)",
            "lower_bound_note": LOWER_BOUND_NOTE,
            "tags": {"arm": "incident", "node_set": "incident-negative cohort",
                     "cell_type": "marginal", "claim_type": "PU lower bound"},
            "label_window_days": label_window_days,
            "horizon_days": list(horizons),
            "min_count": int(args.min_count),
            "deciles": args.deciles,
            "deciles_scored": bool(deciles_scored),
            "bundle_key": key,
            "sidecar_key": witness.get("key"),
            "preindex_col": preindex_col,
            "n_nodes_with_incident_negatives": len(per_node),
            "censoring": ("denominators are gated on observation_period_end_date "
                          "at each horizon (R4.4); a person who leaves the record "
                          "before label_end + h is EXCLUDED, never counted as a "
                          "non-converter"),
            "expect_loose": ("R4.7: at 0109's root prevalence 0.9609 most nodes' "
                             "'did not gain c this year' is a weak negative and "
                             "conversion is expected to be high across the board. "
                             "That is a finding, not a failure."),
        }
        print(format_conversion_report(pooled, meta), flush=True)
        out_path = args.out or str(run_dir / "conversion_analysis.json")
        with open(out_path, "w") as fh:
            json.dump({"meta": meta, "pooled": pooled,
                       "per_node": {str(c): t for c, t in per_node.items()},
                       "int2cid": {str(i): int(cid)
                                   for i, cid in bundle.int2cid.items()},
                       "egress_note": (
                           "per_node cells are NOT disclosable (counts < 20 leave "
                           "the workspace never); only `pooled` and the "
                           "counts-of-nodes in `meta` are.")}, fh, indent=2)
        print(f"[conversion] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
