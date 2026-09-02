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

PERSON_COL = "person_id"
DATE_COL = "first_attested_date"
INDEX_COL = "index_date"
EPISODE_COL = "episode_no"       # 1-based per person, in date order
START_COL = "episode_start"
NNODES_COL = "n_new_nodes"


def build_episodes(first_attestation, *, gap_days, return_assignments=False):
    """`(person_id, episode_no, episode_start, index_date, n_new_nodes)`.

    Gap-and-islands over each person's DISTINCT first-attestation dates — the
    `_stable_drug_intervals` idiom (`cohorts.py:2237-2252`): lag, break flag,
    running sum. A break is `datediff > gap_days`, so two dates exactly
    `gap_days` apart are the SAME episode (the boundary is inclusive; the gap
    must be exceeded to split).

    The island id is then rejoined to the per-(person, node, date) rows to count
    how many distinct nodes each episode newly attests — the payoff density: an
    episode doc contributes that many incident positives.

    `return_assignments=True` additionally returns that per-(person, episode,
    node) join, so the node-yield count reuses one clustering instead of
    rebuilding it.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    dates = first_attestation.select(PERSON_COL, DATE_COL).distinct()
    w = Window.partitionBy(PERSON_COL).orderBy(DATE_COL)
    islands = (dates
               .withColumn("_prev", F.lag(DATE_COL).over(w))
               .withColumn("_break",
                           (F.col("_prev").isNull()
                            | (F.datediff(F.col(DATE_COL), F.col("_prev"))
                               > int(gap_days))).cast("int"))
               .withColumn(EPISODE_COL, F.sum("_break").over(
                   w.rowsBetween(Window.unboundedPreceding, 0)))
               .select(PERSON_COL, DATE_COL, EPISODE_COL))
    per_node = first_attestation.join(islands, on=[PERSON_COL, DATE_COL],
                                      how="inner")
    episodes = (per_node
                .groupBy(PERSON_COL, EPISODE_COL)
                .agg(F.min(DATE_COL).alias(START_COL),
                     F.countDistinct("node_cid").alias(NNODES_COL))
                .withColumn(INDEX_COL, F.date_sub(F.col(START_COL), 1)))
    if return_assignments:
        assignments = per_node.select(PERSON_COL, EPISODE_COL, "node_cid")
        return episodes, assignments
    return episodes


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
    window_days = int(cm.get("label_window_days") or 365)

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
            with _phase(f"episode probe gap={gap}d"):
                res = run_probe(first, obs, gap_days=gap,
                                window_days=window_days, caps=caps)
                print(format_probe_report(res), flush=True)
                out = run_dir / f"episode_probe_gap{gap}.json"
                out.write_text(_json.dumps(res, indent=2))
                print(f"[probe] wrote {out}", flush=True)
        first.unpersist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
