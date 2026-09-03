"""EPISODE PROBES for exp 0111 (plan WP8a) — measure BEFORE any fit exists.

Two numbers the 0111 plan cannot be written without, both catalogued as
pre-measurements in the incident-program spec (§E5) and plan (WP8a):

  R5.9  THE EPISODE MULTIPLIER. Distinct gap-and-islands FIRST-ATTESTATION
        episodes per person. No repo data exists on episodes-per-person; the
        multiplier is a guess until measured, and R5.8's ×3 distributed-eval
        wiring threshold cannot be evaluated without it.
  R5.10 THE PRIOR-OBS-GATE KILL RATE. `_LOOKBACK_PRIOR_OBS_DAYS = 365` is
        hardcoded and un-overridable; `index >= op_start + 365` drops a person's
        EARLIEST (most unambiguously incident) episodes by construction, and
        `index + W <= op_end` drops the last of every record. What fraction dies,
        and is the death anti-correlated with incidence (first episodes dying
        more than later ones)?

GATE OCCUPANCY (`--gate-occupancy on`, spec R5.14)
--------------------------------------------------
A THIRD measurement, additive to the two above and off the SAME sidecar with no
fit. exp 0111's two arms — EPISODE-anchored vs uniform-RANDOM index — share the
365-day label, the two observation gates, and the per-person cap; they differ
ONLY in where the index sits. Each document's GATE is the label nodes whose
first attestation falls in the half-open forward window `[index, index+90d)`.
Episode indices sit one day before a presentation, so their gate is non-empty
essentially by construction (~100%); a uniform-random index rarely lands in the
90-day run-up to a new diagnosis, so the random arm may be mostly EMPTY-gated —
carrying no incident signal. This probe MEASURES that empty-gate rate per arm
(pooled non-empty fraction + gate-size distribution) BEFORE a fit is spent, so
the random arm's viability is known, not guessed. Egress-safe: pooled figures
and counts of documents only, banner + `gate_occupancy_<tag>.json`.

WHAT AN EPISODE IS, HERE
------------------------
A gap-and-islands cluster of a person's FIRST-ATTESTATION dates — the dates on
which some label node appears in their record for the first time ever. Two new
diagnoses ≤ gap days apart are one episode; a new diagnosis more than gap days
after the previous new one starts the next. The candidate index for an episode
is `episode_start − 1 day`: the label window is half-open `[index, index+W)`
(conversion_analysis pins this), so the episode's own first codes land INSIDE
the label window and OUTSIDE the lookback — the model stands just before the
presentation it is asked to predict.

First attestations, not raw condition rows, on purpose: a chronic patient
refilling one diagnosis for a decade is ONE episode under this clustering, not
sixty. The clustering counts moments where something NEW enters the record —
exactly the moments 0111 wants to anchor documents at.

WHY IT RUNS OFF THE E4 SIDECAR (no condition_era rescan)
--------------------------------------------------------
`(person_id, node_cid, first_attested_date)` over the whole record is PRECISELY
the E4 first-attestation sidecar (`conversion_sidecar.py`), already built and
cached per corpus identity. This probe is a read of that parquet plus one small
`observation_period` table read — no full-history scan, no code-map rebuild, no
fit, no cache-key impact. It therefore REQUIRES the sidecar witness in the run
dir, same refusal discipline as `conversion_analysis` (exit 4 with the fix).

THE GATES ARE THE ASSEMBLER'S OWN
---------------------------------
Survival is decided by CALLING `_window_observed_cohort` (`cohorts.py:693-729`)
— the very function the assembler's index builders gate with — three ways:

    both gates      prior_obs_days=365, window_days=W   (what 0111 would keep)
    prior-only      prior_obs_days=365, window_days=0   (isolates the lookback)
    follow-up-only  prior_obs_days=0,   window_days=W   (isolates the tail)

Calling it (free) rather than re-deriving it keeps the probe's kill rates the
assembler's kill rates, not a lookalike's. Editing `cohorts.py` is what moves
`cohort_defs_version()`; calling it moves nothing.

GRAIN AND EGRESS
----------------
Everything computed here is per-(person, episode) internally and POOLED before
it leaves Spark: counts of episodes, means, quantiles, band histograms over
millions of persons. The banner and `episode_probe_gap<g>.json` carry pooled
figures only — no per-node cells, nothing under the All of Us egress floor.

ADR 0047: Spark-native windows/joins/groupBys only; no UDF, no `sc.broadcast`,
nothing array-shaped in a task closure.
"""
from __future__ import annotations

# Bands for the per-person episode-count histogram. Pooled over millions of
# persons; the top band is open so no knob ever needs retuning.
COUNT_BANDS = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, None))

# The clustering itself, and its column names, live in episode_index.py now:
# episode_index_frame (WP-D's gated + capped provider) needs the SAME
# build_episodes this probe already tests, and a library the fit driver
# depends on may not import a driver-CLI tool script (lib<-tool, never the
# reverse — see episode_index.py's module docstring). Re-exported under their
# original names so this module's own call sites (`build_episodes(...)`,
# `PERSON_COL`, ...) and `tests/scripts/test_episode_probe.py`'s
# `ep.build_episodes` / `ep.PERSON_COL` usage are both unchanged.
from episode_index import (  # noqa: E402
    DATE_COL,
    EPISODE_COL,
    INDEX_COL,
    NNODES_COL,
    PERSON_COL,
    START_COL,
    build_episodes,
    episode_index_frame,
    random_index_frame,
)

# The frontier gate window: a document's GATE is the set of label nodes whose
# FIRST attestation falls in the half-open forward window [index, index+90d) —
# the "frontier" exp 0111 grants each document topic-block access to. 90d is the
# spec's fixed gate; the constant names it so the probe's default and the
# report banner cannot drift apart.
GATE_DAYS = 90


def node_yield(assignments, gated_episodes, *, bars=(20, 100)):
    """How many label nodes gain >= bar GATED first-attestation episodes — the
    direct answer to whether episode anchoring un-starves the incident-thin
    nodes (0110 dropped 923 as <20 incident positives).

    A LOWER BOUND, stated as such: `node_cid` is the frontier node a code
    resolves to, and the incident label folds first attestation over
    `closure(c)` — the fold only ADDS episodes to ancestors, never removes.
    Output is counts OF NODES (pooled), never per-node counts: egress-safe.
    """
    from pyspark.sql import functions as F

    per_node = (assignments
                .join(gated_episodes.select(PERSON_COL, EPISODE_COL),
                      on=[PERSON_COL, EPISODE_COL], how="inner")
                .groupBy("node_cid")
                .agg(F.count("*").alias("n_ep")))
    agg = [F.count("*").alias("nodes_any")]
    for bar in bars:
        agg.append(F.sum((F.col("n_ep") >= int(bar)).cast("long"))
                   .alias(f"nodes_ge_{bar}"))
    row = per_node.agg(*agg).collect()[0]
    out = {"nodes_with_any_gated_episode": int(row["nodes_any"])}
    for bar in bars:
        out[f"nodes_ge_{bar}"] = int(row[f"nodes_ge_{bar}"])
    return out


def gate_occupancy(index_frame, first_attestation, *, gate_days=GATE_DAYS,
                   quantiles=(0.5, 0.9, 0.99)):
    """Pooled 90-day-gate occupancy for ONE index arm.

    For each document (a `(person_id, index_date)` in `index_frame`), the GATE is
    the set of label nodes whose first attestation lands in the half-open window
    `[index_date, index_date + gate_days)` — half-open matching the label window
    `conversion_analysis` pins (a positive is a first attestation in
    `[index, index+W)`, a conversion is STRICTLY after; a code at `index+gate_days`
    is OUTSIDE the gate). `countDistinct` over the in-window rows is the gate
    size; a document with no in-window first attestation has size 0 — an EMPTY
    gate, the thing the random arm is feared to produce.

    FRONTIER LOWER BOUND, stated as such (same framing as `node_yield` and
    `episode_index`): `node_cid` is the frontier node a code resolves to, and the
    real gate rolls each frontier node UP the Mondo DAG, adding its ancestors.
    That fold only ADDS nodes, so the NON-EMPTY FRACTION here is EXACT (a gate
    with >=1 frontier node stays non-empty after ancestors are added, and one
    with zero frontier nodes has zero ancestors to add), while the gate SIZE
    (distinct-node count) is a LOWER BOUND on the rolled-up gate.

    Returns a pooled dict — doc count, non-empty count and fraction, and gate-size
    mean / p50 / p90 / p99. Counts of documents and pooled moments only, never a
    per-node or per-document cell: egress-safe by construction. `None` fields on
    an empty arm (no documents) so a degenerate input reports rather than raises.
    """
    from pyspark.sql import functions as F

    idx = index_frame.select(PERSON_COL, INDEX_COL).distinct()
    in_win = ((F.col(DATE_COL) >= F.col(INDEX_COL))
              & (F.col(DATE_COL) < F.date_add(F.col(INDEX_COL), int(gate_days))))
    # Left join so a document whose person has NO in-window first attestation
    # still surfaces as one row with gate_size 0 (countDistinct ignores the
    # null the `when` yields off-window) rather than vanishing from the count.
    per_doc = (idx.join(
                   first_attestation.select(PERSON_COL, DATE_COL, "node_cid"),
                   on=PERSON_COL, how="left")
               .groupBy(PERSON_COL, INDEX_COL)
               .agg(F.countDistinct(F.when(in_win, F.col("node_cid")))
                    .alias("gate_size")))
    per_doc = per_doc.cache()
    n_docs = per_doc.count()
    if n_docs == 0:
        per_doc.unpersist()
        return {"n_docs": 0, "n_nonempty": 0, "nonempty_fraction": None,
                "gate_size_mean": None, "gate_size_p50": None,
                "gate_size_p90": None, "gate_size_p99": None}
    row = per_doc.agg(
        F.sum((F.col("gate_size") >= 1).cast("long")).alias("nonempty"),
        F.avg("gate_size").alias("mean")).collect()[0]
    qs = per_doc.approxQuantile("gate_size", list(quantiles), 0.001)
    per_doc.unpersist()
    n_nonempty = int(row["nonempty"])
    return {"n_docs": int(n_docs), "n_nonempty": n_nonempty,
            "nonempty_fraction": n_nonempty / n_docs,
            "gate_size_mean": float(row["mean"]),
            "gate_size_p50": qs[0], "gate_size_p90": qs[1],
            "gate_size_p99": qs[2]}


def run_gate_occupancy(first_attestation, observation_period, *, gap_days=90,
                       gate_days=GATE_DAYS, cap=3, salt, prior_obs_days=365,
                       window_days=365):
    """Both arms' gate occupancy, side by side, off the cached sidecar. NO FIT.

    exp 0111's two arms share the 365-day label and the SAME two observation
    gates and per-person cap; they differ ONLY in index location. Before a fit is
    spent we measure how occupied each arm's 90-day forward gate is:

      * EPISODE arm — `episode_index_frame`: index = episode_start - 1, so the
        episode's own first codes fall at `index+1`, INSIDE the gate. Non-empty
        essentially by construction — this arm is the ~100% floor.
      * RANDOM arm — `random_index_frame`: a uniform-random valid index. A
        population-random index rarely sits in the 90-day run-up to a new
        diagnosis, so this arm's gate may be mostly EMPTY — the quantity that
        decides whether the random arm carries any incident signal at all.

    The random arm is drawn on the EPISODE arm's surviving persons (passed as
    `persons=`), so the two arms compare on an identical population and the only
    difference left is where the index sits.
    """
    ep_idx = episode_index_frame(
        first_attestation, observation_period, gap_days=gap_days, cap=cap,
        salt=salt, prior_obs_days=prior_obs_days, window_days=window_days).cache()
    ep_persons = ep_idx.select(PERSON_COL).distinct()
    rnd_idx = random_index_frame(
        observation_period, cap=cap, salt=salt, prior_obs_days=prior_obs_days,
        window_days=window_days, persons=ep_persons).cache()
    episode_arm = gate_occupancy(ep_idx, first_attestation, gate_days=gate_days)
    random_arm = gate_occupancy(rnd_idx, first_attestation, gate_days=gate_days)
    ep_idx.unpersist()
    rnd_idx.unpersist()
    return {"gap_days": int(gap_days), "gate_days": int(gate_days),
            "cap": int(cap), "salt": str(salt),
            "prior_obs_days": int(prior_obs_days), "window_days": int(window_days),
            "episode_arm": episode_arm, "random_arm": random_arm}


def format_gate_occupancy_report(res) -> str:
    """The gate-occupancy banner. Pooled figures only — egress-safe."""
    ep, rnd = res["episode_arm"], res["random_arm"]

    def _pct(x):
        return f"{100 * x:.1f}%" if x is not None else "n/a"

    def _f(x):
        return f"{x:.2f}" if x is not None else "n/a"

    lines = [
        f"[gate-occ] gap={res['gap_days']}d  gate=[index, index+{res['gate_days']}d)"
        f"  cap={res['cap']}  prior_obs={res['prior_obs_days']}d  "
        f"W={res['window_days']}d  salt={res['salt']}",
        "[gate-occ] FRONTIER measure: non-empty fraction is EXACT (the Mondo-DAG "
        "roll-up only ADDS ancestor nodes); gate SIZE is a lower bound on the "
        "rolled-up gate",
    ]
    for name, arm in (("EPISODE", ep), ("RANDOM ", rnd)):
        lines.append(
            f"[gate-occ] {name} arm: {arm['n_docs']} docs, non-empty "
            f"{_pct(arm['nonempty_fraction'])} (n={arm['n_nonempty']}); gate size "
            f"mean={_f(arm['gate_size_mean'])}, p50={arm['gate_size_p50']}, "
            f"p90={arm['gate_size_p90']}, p99={arm['gate_size_p99']}")
    lines.append(
        f"[gate-occ] side by side — EPISODE non-empty {_pct(ep['nonempty_fraction'])}"
        f" vs RANDOM {_pct(rnd['nonempty_fraction'])}: the empty-gate cost of a "
        "uniform-random index, measured with no fit spent")
    return "\n".join(lines)


def gate_episodes(episodes, observation_period, *, prior_obs_days, window_days):
    """The episodes surviving `_window_observed_cohort` at the given gates.

    The assembler's own gate function, called on the episode `(person_id,
    index_date)` frame. Distinct index dates per person are guaranteed (distinct
    attestation dates → distinct episode starts → distinct indexes), so joining
    the survivors back on `(person, index)` flags episodes one-to-one.
    """
    from charmpheno.omop.cohorts import _window_observed_cohort
    surviving = _window_observed_cohort(
        episodes.select(PERSON_COL, INDEX_COL), observation_period,
        prior_obs_days=int(prior_obs_days), window_days=int(window_days))
    return episodes.join(surviving, on=[PERSON_COL, INDEX_COL], how="inner")


def _per_person_stats(episodes, *, caps):
    """Pooled multiplier stats for one episode frame (raw or gated).

    Returns a plain dict: totals, per-person mean/quantiles, the band histogram,
    capped totals (`sum(min(n, cap))` — what a per-person doc cap would keep),
    and mean/median new-nodes-per-episode.
    """
    from pyspark.sql import functions as F

    per_person = (episodes.groupBy(PERSON_COL)
                  .agg(F.count("*").alias("n_ep")))
    per_person = per_person.cache()
    n_persons = per_person.count()
    if n_persons == 0:
        per_person.unpersist()
        return {"n_persons": 0, "n_episodes": 0}
    agg = {"n_episodes": F.sum("n_ep"), "max_ep": F.max("n_ep")}
    for i, (lo, hi) in enumerate(COUNT_BANDS):
        cond = (F.col("n_ep") >= lo) if hi is None else \
            ((F.col("n_ep") >= lo) & (F.col("n_ep") <= hi))
        agg[f"band_{i}"] = F.sum(cond.cast("long"))
    for cap in caps:
        agg[f"cap_{cap}"] = F.sum(F.least(F.col("n_ep"), F.lit(int(cap))))
    row = per_person.agg(*(v.alias(k) for k, v in agg.items())).collect()[0]
    q50, q90, q99 = per_person.approxQuantile("n_ep", [0.5, 0.9, 0.99], 0.001)
    per_person.unpersist()

    nn = episodes.agg(F.avg(NNODES_COL).alias("mean")).collect()[0]["mean"]
    nn_q = episodes.approxQuantile(NNODES_COL, [0.5, 0.9], 0.001)
    n_episodes = int(row["n_episodes"])
    out = {
        "n_persons": int(n_persons),
        "n_episodes": n_episodes,
        "episodes_per_person_mean": n_episodes / n_persons,
        "episodes_per_person_p50": q50, "episodes_per_person_p90": q90,
        "episodes_per_person_p99": q99, "episodes_per_person_max": int(row["max_ep"]),
        "person_count_bands": {
            (f"{lo}" if hi == lo else f"{lo}-{hi}" if hi else f"{lo}+"):
                int(row[f"band_{i}"])
            for i, (lo, hi) in enumerate(COUNT_BANDS)},
        "capped_totals": {str(c): int(row[f"cap_{c}"]) for c in caps},
        "new_nodes_per_episode_mean": float(nn),
        "new_nodes_per_episode_p50": nn_q[0],
        "new_nodes_per_episode_p90": nn_q[1],
    }
    return out


def _first_vs_later_kill(episodes, gated_both):
    """R5.10's anti-correlation check: do FIRST episodes die more than later ones?

    `episode_no == 1` is a person's earliest new-diagnosis cluster — the one the
    prior-obs gate kills by construction when it sits at record start. Survival
    is membership in the both-gates frame.
    """
    from pyspark.sql import functions as F

    surv = gated_both.select(PERSON_COL, EPISODE_COL).withColumn(
        "_alive", F.lit(1))
    flagged = (episodes.select(PERSON_COL, EPISODE_COL)
               .join(surv, on=[PERSON_COL, EPISODE_COL], how="left")
               .withColumn("_first", (F.col(EPISODE_COL) == 1).cast("int")))
    row = (flagged.groupBy("_first")
           .agg(F.count("*").alias("n"),
                F.sum(F.coalesce(F.col("_alive"), F.lit(0))).alias("alive"))
           .collect())
    out = {}
    for r in row:
        key = "first_episodes" if r["_first"] == 1 else "later_episodes"
        n, alive = int(r["n"]), int(r["alive"])
        out[key] = {"n": n, "surviving": alive,
                    "kill_rate": (n - alive) / n if n else None}
    return out


def run_probe(first_attestation, observation_period, *, gap_days, window_days,
              prior_obs_days=365, caps=(3, 5)):
    """One gap value, end to end. Returns the pooled results dict."""
    episodes, assignments = build_episodes(
        first_attestation, gap_days=gap_days, return_assignments=True)
    episodes = episodes.cache()
    raw = _per_person_stats(episodes, caps=caps)

    both = gate_episodes(episodes, observation_period,
                         prior_obs_days=prior_obs_days,
                         window_days=window_days).cache()
    gated = _per_person_stats(both, caps=caps)
    prior_only = gate_episodes(episodes, observation_period,
                               prior_obs_days=prior_obs_days, window_days=0)
    followup_only = gate_episodes(episodes, observation_period,
                                  prior_obs_days=0, window_days=window_days)
    n_raw = raw.get("n_episodes", 0)
    decomposition = {
        "raw": n_raw,
        "surviving_both": gated.get("n_episodes", 0),
        "surviving_prior_only": prior_only.count(),
        "surviving_followup_only": followup_only.count(),
    }
    if n_raw:
        decomposition["kill_rate_both"] = 1 - decomposition["surviving_both"] / n_raw
        decomposition["kill_rate_prior_only"] = (
            1 - decomposition["surviving_prior_only"] / n_raw)
        decomposition["kill_rate_followup_only"] = (
            1 - decomposition["surviving_followup_only"] / n_raw)
    first_later = _first_vs_later_kill(episodes, both)
    yield_counts = node_yield(assignments, both)
    episodes.unpersist()
    both.unpersist()
    return {"gap_days": int(gap_days), "window_days": int(window_days),
            "prior_obs_days": int(prior_obs_days),
            "raw": raw, "gated_both": gated,
            "gate_decomposition": decomposition,
            "first_vs_later_kill": first_later,
            "node_yield": yield_counts}


def format_probe_report(res) -> str:
    """The banner. Pooled figures only — egress-safe by construction."""
    g = res["gap_days"]
    raw, gated = res["raw"], res["gated_both"]
    dec, fl = res["gate_decomposition"], res["first_vs_later_kill"]

    def _mult(d):
        m = d.get("episodes_per_person_mean")
        return f"{m:.3f}" if m is not None else "n/a"

    def _pct(x):
        return f"{100 * x:.1f}%" if x is not None else "n/a"

    lines = [
        f"[probe] gap={g}d  W={res['window_days']}d  prior_obs={res['prior_obs_days']}d",
        f"[probe] R5.9 multiplier — RAW: {raw.get('n_episodes', 0)} episodes / "
        f"{raw.get('n_persons', 0)} persons = {_mult(raw)} per person "
        f"(p50={raw.get('episodes_per_person_p50')}, p90={raw.get('episodes_per_person_p90')}, "
        f"p99={raw.get('episodes_per_person_p99')}, max={raw.get('episodes_per_person_max')})",
        f"[probe]      GATED (both gates): {gated.get('n_episodes', 0)} episodes / "
        f"{gated.get('n_persons', 0)} persons with >=1 = {_mult(gated)} per such person",
        f"[probe]      gated capped totals: " + ", ".join(
            f"cap {c}: {n}" for c, n in sorted(
                (gated.get('capped_totals') or {}).items(), key=lambda kv: int(kv[0]))),
        f"[probe]      new nodes per episode (gated): "
        f"mean={gated.get('new_nodes_per_episode_mean', float('nan')):.2f}, "
        f"p50={gated.get('new_nodes_per_episode_p50')}, p90={gated.get('new_nodes_per_episode_p90')}",
        f"[probe] R5.10 kill — both: {_pct(dec.get('kill_rate_both'))} of {dec['raw']}; "
        f"prior-obs gate alone: {_pct(dec.get('kill_rate_prior_only'))}; "
        f"follow-up gate alone: {_pct(dec.get('kill_rate_followup_only'))}",
    ]
    fe, le = fl.get("first_episodes") or {}, fl.get("later_episodes") or {}
    lines.append(
        f"[probe]      first episodes killed {_pct(fe.get('kill_rate'))} "
        f"(n={fe.get('n')}) vs later episodes {_pct(le.get('kill_rate'))} "
        f"(n={le.get('n')}) — the R5.10 anti-correlation, measured")
    ny = res.get("node_yield") or {}
    if ny:
        lines.append(
            f"[probe] node yield (frontier LOWER BOUND; closure fold only adds): "
            f"{ny.get('nodes_ge_20', 0)} nodes with >=20 gated episodes, "
            f"{ny.get('nodes_ge_100', 0)} with >=100, "
            f"{ny.get('nodes_with_any_gated_episode', 0)} with any — compare "
            "0110's 923 nodes dropped at <20 incident positives")
    lines.append(
        "[probe] R5.8 wiring rule: the GATED per-person multiplier (or its capped "
        "variant, whichever 0111 adopts) >= 3 makes the distributed eval path "
        "MANDATORY before any episode fit")
    return "\n".join(lines)


def main(argv=None) -> int:
    """Cluster-only (one small BigQuery table read + a cached sidecar parquet)."""
    import argparse
    import json as _json

    from _driver_common import _phase, configure_logging, make_spark_session
    from conversion_sidecar import (load_observation_period, try_load_sidecar)
    from gated_pc_readout import resolve_run_dir

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gap-days", default="60,90",
                   help="Comma-separated gap values; one probe pass per value.")
    p.add_argument("--caps", default="3,5",
                   help="Per-person doc caps to size (sum of min(n, cap)).")
    p.add_argument("--prior-obs-days", default="365",
                   help="Comma-separated prior-observation-gate values; one "
                        "probe pass per (gap, prior_obs) pair. Default 365 is "
                        "the assembler's hardcoded `_LOOKBACK_PRIOR_OBS_DAYS` — "
                        "the 0111 PRIMARY arm. Pass e.g. `365,90,0` for the "
                        "R5.10 relaxed-gate SENSITIVITY (spec §4 / WP-H): what "
                        "the incidence definition costs. This probe RELAXES the "
                        "gate only in its own survival call (the gate function "
                        "takes the value as an argument); it does NOT edit the "
                        "hardcoded assembler constant, so no fit and no corpus "
                        "changes — a measurement, not a knob turned in anger.")
    p.add_argument("--gate-occupancy", choices=("on", "off"), default="off",
                   help="Additive: also measure how occupied each arm's 90-day "
                        "GATE ([index, index+gate-days)) is, EPISODE vs "
                        "uniform-RANDOM index, per (gap, prior_obs) pass. The "
                        "random arm rarely sits before a presentation, so its "
                        "gate may be mostly EMPTY — this quantifies that off the "
                        "cached sidecar with NO fit, before a fit is spent. "
                        "Writes `gate_occupancy_<tag>.json` next to the probe "
                        "json.")
    p.add_argument("--gate-days", default=str(GATE_DAYS),
                   help="Forward gate window in days (default 90): a document's "
                        "gate is the first-attestation nodes in the half-open "
                        "[index, index+gate-days).")
    p.add_argument("--gate-cap", default="3",
                   help="Per-person doc cap for BOTH gate-occupancy arms (default "
                        "3) — the arms share the cap so they differ only in index "
                        "location.")
    p.add_argument("--gate-salt", default="0111",
                   help="Salt for the deterministic episode cap sample and random "
                        "index draw. Same salt => identical draw on any rerun; a "
                        "different salt reshuffles both. Never `F.rand()`.")
    args = p.parse_args(argv)
    configure_logging()

    run_dir = resolve_run_dir(args.run_dir)
    manifest = _json.loads((run_dir / "manifest.json").read_text())
    cm = manifest.get("corpus_manifest") or {}
    witness_path = run_dir / "conversion_sidecar.json"
    if not witness_path.exists():
        print("[probe] ERROR: no conversion_sidecar.json in the run dir. The "
              "probe reads the E4 first-attestation sidecar — run "
              "`make build-conversion-sidecar ID=N` first.", flush=True)
        return 4
    witness = _json.loads(witness_path.read_text())
    gaps = [int(x) for x in str(args.gap_days).split(",") if x.strip()]
    caps = tuple(int(x) for x in str(args.caps).split(",") if x.strip())
    prior_obs = [int(x) for x in str(args.prior_obs_days).split(",") if x.strip()]
    window_days = int(cm.get("label_window_days") or 365)
    gate_days = int(args.gate_days)
    gate_cap = int(args.gate_cap)

    with make_spark_session(app_name="diag-episode-probe") as spark:
        first = try_load_sidecar(spark, witness["sidecar_uri"], witness["key"])
        if first is None:
            print(f"[probe] ERROR: sidecar MISS at {witness['first_attestation']} "
                  "— the witness names a parquet that is not there. Rebuild with "
                  "`make build-conversion-sidecar ID=N`.", flush=True)
            return 4
        obs = load_observation_period(spark, cdr=cm.get("cdr"),
                                      billing=cm.get("billing"))
        first = first.cache()
        for gap in gaps:
            for pod in prior_obs:
                # The filename carries prior_obs ONLY when it is relaxed off the
                # 365 primary, so the original `episode_probe_gap<g>.json` names
                # (the recorded primary run) are byte-stable and a sensitivity
                # sweep never overwrites them.
                tag = f"gap{gap}" if pod == 365 else f"gap{gap}_prior{pod}"
                with _phase(f"episode probe gap={gap}d prior_obs={pod}d"):
                    res = run_probe(first, obs, gap_days=gap,
                                    window_days=window_days, caps=caps,
                                    prior_obs_days=pod)
                    print(format_probe_report(res), flush=True)
                    out = run_dir / f"episode_probe_{tag}.json"
                    out.write_text(_json.dumps(res, indent=2))
                    print(f"[probe] wrote {out}", flush=True)
                if args.gate_occupancy == "on":
                    with _phase(f"gate occupancy gap={gap}d prior_obs={pod}d"):
                        gocc = run_gate_occupancy(
                            first, obs, gap_days=gap, gate_days=gate_days,
                            cap=gate_cap, salt=args.gate_salt,
                            prior_obs_days=pod, window_days=window_days)
                        print(format_gate_occupancy_report(gocc), flush=True)
                        gout = run_dir / f"gate_occupancy_{tag}.json"
                        gout.write_text(_json.dumps(gocc, indent=2))
                        print(f"[gate-occ] wrote {gout}", flush=True)
        first.unpersist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
