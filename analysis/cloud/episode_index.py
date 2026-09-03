"""Episode index provider — exp 0111 WP-D1 (spec D8-D11, plan WP-D).

WHAT THIS MODULE OWNS
----------------------
The driver-owned machinery WP-C's injection seams (`index_df=`, `doc_spec=` on
`assemble_multidomain_case_finding_corpus`) will receive once they land:

  * `build_episodes` — the gap-and-islands clustering of a person's
    FIRST-ATTESTATION dates into episodes (D8), MOVED here from
    `diag_episode_probe.py` (the probe imports it back — see below).
  * `episode_index_frame` — the GATED, CAPPED `(person_id, index_date,
    episode_no)` frame (D9-D11): both observation gates, then a deterministic
    salted-hash sample keeping at most `cap` surviving episodes per person.
  * `random_index_frame` — the RANDOM arm's sibling provider (spec R5.14): the
    SAME gates and cap as `episode_index_frame`, but the index is drawn UNIFORMLY
    over the person's fully-observed calendar interval instead of anchored on a
    presentation. The two arms differ in index location and nothing else — what
    the gate-occupancy probe measures the cost of before a fit is spent.
  * `min_doc_length_drop_rate_by_ordinal` — the R7.5 monitoring table: does
    `min_doc_length` drop episode docs non-uniformly toward the incident
    (first-episode) end.

`EpisodeDocSpec` is NOT here. See "PLACEMENT DECISION" below.

WHY build_episodes MOVED (lib <- tool, never tool <- lib)
----------------------------------------------------------
`diag_episode_probe.py` is a cluster driver script (a `main()`, an argparse CLI,
a Makefile target). `episode_index_frame` — the thing WP-D2 wires into the fit
driver's assemble closure — needs the SAME clustering `diag_episode_probe`
already built and tested (`tests/scripts/test_episode_probe.py`'s gap-boundary
and index-convention tests). Importing a tool script's function from a library
module the fit driver depends on would make the fit driver's import graph
reach through a probe CLI; moving the clustering here and having the probe
import it back keeps the dependency arrow libraries -> nothing, tools ->
libraries, the direction every other module in this program follows
(`preindex_closure.py`, `conversion_sidecar.py`). `diag_episode_probe.py`
re-exports `build_episodes` and the column constants under their old names so
`tests/scripts/test_episode_probe.py` (`import diag_episode_probe as ep`,
`ep.build_episodes`, `ep.PERSON_COL`, ...) is unchanged and still green.

THE GATES ARE THE ASSEMBLER'S OWN (same discipline as the probe)
------------------------------------------------------------------
`episode_index_frame` judges survival by CALLING
`charmpheno.omop.cohorts._window_observed_cohort` — exactly what
`diag_episode_probe.gate_episodes` already does, on the same three-way split
(both gates / prior-only / follow-up-only is the probe's concern, not this
provider's; the provider only ever needs "both"). The one-liner is duplicated
here rather than imported from the probe (same lib<-tool reason as above,
and the same "duplicated rather than imported" call `conversion_sidecar.py`
makes for its own `_module_source_hash`): three lines, no logic invented,
nothing to drift. Calling `cohorts.py` is free — nothing here imports its
private helpers for anything OTHER than delegation, and NOTHING in this file
edits `cohorts.py`. Editing it is what moves `cohort_defs_version()` and every
cache key in the repo; calling it moves nothing (WP-C, not WP-D, is where that
edit is scoped, and only there).

THE CAP: DETERMINISTIC, SALTED, NEVER F.rand() (D11)
-------------------------------------------------------
`cohorts._random_event_windows` picks one anchor per person by
`min hash(person_id, event_date, salt)` — explicitly resume-stable, explicitly
not `F.rand()` (`cohorts.py:1083-1145`). `preindex_closure.feature_window_index_
table` calls that same picker rather than reimplementing it, for the same
reason: a pure function of its inputs reproduces identically on any rerun.
`episode_index_frame`'s cap needs the same property but a DIFFERENT shape — up
to `cap` keeps per person, not one — so it cannot simply call the population
picker; it applies the identical IDIOM (`F.sha2` over a salted, delimited
concatenation of the identifying columns, ascending rank, keep rank <= cap)
that `hdp_bigquery_cloud.py`'s deterministic person-hash split and the
holdout-split design (`docs/superpowers/specs/2026-05-11-topic-coherence-
evaluation-design.md`) already use elsewhere in this repo for the same
resume-stability property. `F.hash` (Murmur3, 32-bit, used by
`_random_event_windows`) would work as well for uniformity, but `F.sha2` is
the idiom named in the 0111 plan (WP-D) and the wider hash space makes an
accidental cross-person tie between two candidate episodes vanishingly
unlikely; either is deterministic, and this module uses the one named.

EPISODE_NO IN THE OUTPUT IS THE ORIGINAL ORDINAL, NEVER A RE-RANK
---------------------------------------------------------------------
`episode_no` on a surviving, capped row is the SAME value `build_episodes`
assigned before any gate or sample ran: a person's chronologically fourth
episode that happens to survive the gate and win the cap sample still reports
`episode_no=4`, not `episode_no=1` for "first kept episode." R7.5 (the ordinal
drop-rate table) and the plan's first-vs-later kill decomposition both depend
on `episode_no` meaning "this many new-diagnosis clusters into this person's
record" — re-ranking after sampling would silently erase the exact bias (the
66.2% first-episode kill) the whole experiment exists to report honestly.

PLACEMENT DECISION: EpisodeDocSpec lives in charmpheno/charmpheno/omop/doc_spec.py
-------------------------------------------------------------------------------------
ADR 0018 sanctions `doc_spec.py` as the extension point for new doc units, but
the sanction is conditional here: if the MODULE'S SOURCE folded into any
bundle cache key, adding a class there would silently move every key that
folds it (the same trap `preindex_closure.py`'s docstring names for
`case_finding_assembly`/`multi_domain`). It was checked, not assumed:

  * `grep -rn "_module_source_hash" analysis/cloud charmpheno` finds it called
    on `condition_dag`, `case_finding_assembly`, `multi_domain`, `mondo_dag`,
    `mondo_collapse`, `mondo_native_dag`, `mondo_usage_core`,
    `preindex_closure`, and (self-referentially) `conversion_sidecar` — never
    on `doc_spec`.
  * `_case_finding_cache.compute_bundle_cache_key`'s `doc_spec` payload field
    is a STRING read off `doc_spec_identity()`
    (`gated_pc_cloud.py:2307-2326`), which returns `PatientCohortDocSpec().name`
    — a class-level string constant, not a source hash. Adding `EpisodeDocSpec`
    to `doc_spec.py` cannot change `PatientCohortDocSpec().name` and therefore
    cannot move `doc_spec_identity()`'s return value for any caller that has
    not switched to the new class. `DEFAULT_DOC_SPEC` in `_case_finding_cache.py`
    stays `"patient_cohort"`; every existing key is untouched.
  * `doc_spec.py` is absent from the WP standing rule's never-edit list
    (`cohorts.py`, `multi_domain.py`, `case_finding_assembly.py`, the Mondo/
    condition-DAG/preindex-closure modules) — it was never source-hashed into
    anything, so it was never meant to be untouchable.

So the guarded branch of the instruction does not fire: `EpisodeDocSpec` is
added to `doc_spec.py` proper, mirroring `PatientCohortDocSpec`'s interface
exactly as ADR 0018 intends. `episode_index.py` stays free of doc-shaping
logic — it hands WP-D2's injection seam an index FRAME; what a document's
`doc_id` looks like is `doc_spec.py`'s question, not this one's.

R7.5 MONITORING
----------------
`min_doc_length_drop_rate_by_ordinal` takes a per-doc frame the CALLER builds
(person_id, episode_no, doc_length) — this module has no opinion on how a doc
length is counted (that is `to_bow_dataframe`'s CountVectorizer-derived
feature-vector length, per `DocSpec`'s own docstring) and does not compute it.
It only pools kept/dropped counts into the three ordinal bands the plan names
(1 / 2 / 3+) so the table can ride every 0111 smoke log next to the 66.2%
first-episode gate-kill it is expected to echo (audit seam 9).

ADR 0047: Spark-native window/join/groupBy only; no UDF, no `sc.broadcast`,
nothing array-shaped in a task closure.
"""
from __future__ import annotations

# Column names, MOVED from diag_episode_probe.py (which now imports them back
# under the same names — see the module docstring's lib<-tool section).
PERSON_COL = "person_id"
DATE_COL = "first_attested_date"
INDEX_COL = "index_date"
EPISODE_COL = "episode_no"       # 1-based per person, in date order
START_COL = "episode_start"
NNODES_COL = "n_new_nodes"

# The R7.5 ordinal bands: first episode on its own (the 66.2%-kill anchor),
# second episode on its own, third-and-beyond pooled — open-ended so a
# megapatient's 63rd episode needs no new band.
ORDINAL_BAND_1 = "1"
ORDINAL_BAND_2 = "2"
ORDINAL_BAND_3PLUS = "3+"
ORDINAL_BANDS = (ORDINAL_BAND_1, ORDINAL_BAND_2, ORDINAL_BAND_3PLUS)


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


def _gated_episodes(episodes, observation_period, *, prior_obs_days, window_days):
    """Episodes surviving `_window_observed_cohort` at the given gates.

    Duplicated from `diag_episode_probe.gate_episodes` rather than imported
    (lib<-tool direction — see the module docstring): both call the SAME
    imported `cohorts._window_observed_cohort`, so there is no forked gate
    logic to drift, only a forked three-line wrapper. Distinct index dates per
    person are guaranteed (distinct attestation dates -> distinct episode
    starts -> distinct indexes), so joining survivors back on
    `(person_id, index_date)` flags episodes one-to-one.
    """
    from charmpheno.omop.cohorts import _window_observed_cohort
    surviving = _window_observed_cohort(
        episodes.select(PERSON_COL, INDEX_COL), observation_period,
        prior_obs_days=int(prior_obs_days), window_days=int(window_days))
    return episodes.join(surviving, on=[PERSON_COL, INDEX_COL], how="inner")


def episode_index_frame(first_attestation, observation_period, *, gap_days=90,
                        cap=3, salt, prior_obs_days=365, window_days=365):
    """`(person_id, index_date, episode_no)` — the gated, capped episode index.

    The provider WP-D2 wires into the assembler's `index_df=` seam once WP-C
    lands. Three steps, each already spec'd (D8-D11):

      1. `build_episodes` — cluster first-attestation dates into episodes
         (D8), candidate index = episode_start - 1 day (D9).
      2. `_gated_episodes` — keep only episodes surviving BOTH of the
         assembler's own observation gates (D10): `index >= op_start +
         prior_obs_days` and `index + window_days <= op_end`.
      3. The per-person cap (D11): rank surviving episodes by ascending
         `F.sha2(concat(person_id, episode_start, salt))` and keep rank
         <= `cap`. Deterministic and resume-stable (never `F.rand()`);
         a different `salt` reshuffles the ranking (a different sample), the
         SAME `salt` on the SAME input reproduces the SAME sample byte for
         byte, on any resume or rerun.

    `episode_no` on every OUTPUT row is the ORIGINAL chronological ordinal
    `build_episodes` assigned, never a post-cap re-rank (see the module
    docstring) — R7.5 and the first-vs-later kill decomposition both need it
    to keep meaning "how many new-diagnosis clusters into this person's
    record", not "how many-th kept document is this."

    `salt` has no default deliberately: unlike `gap_days`/`cap`/the gate days
    (all fixed by the probe, `docs/reports/2026-09-02-0111-episode-probe-
    results.md`), the salt is an experiment-identity choice (front matter,
    same discipline `split_salt` already follows) that must be named at each
    call site rather than silently inherited from a module constant a caller
    might forget to override.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    episodes = build_episodes(first_attestation, gap_days=gap_days)
    gated = _gated_episodes(episodes, observation_period,
                            prior_obs_days=prior_obs_days,
                            window_days=window_days)
    ranked = gated.withColumn(
        "_h", F.sha2(F.concat_ws("|",
                                 F.col(PERSON_COL).cast("string"),
                                 F.col(START_COL).cast("string"),
                                 F.lit(str(salt))), 256)
    ).withColumn(
        "_rank", F.row_number().over(
            Window.partitionBy(PERSON_COL)
            .orderBy(F.col("_h").asc(), F.col(EPISODE_COL).asc()))
    )
    kept = ranked.where(F.col("_rank") <= int(cap))
    return kept.select(PERSON_COL, INDEX_COL, EPISODE_COL)


def random_index_frame(observation_period, *, cap=3, salt, prior_obs_days=365,
                       window_days=365, persons=None):
    """`(person_id, index_date)` — the UNIFORM-RANDOM index arm, gate-matched.

    The sibling of `episode_index_frame` for exp 0111's RANDOM arm (spec R5.14).
    Both arms share the SAME two observation gates (`prior_obs_days`,
    `window_days`) and the SAME per-person cap; they differ in the ONE thing the
    experiment is designed to isolate — where the index sits. `episode_index_
    frame` anchors on presentations (episode_start - 1); this one draws days
    UNIFORMLY over the person's fully-observed calendar interval, with no regard
    to when the person's coding actually happens.

    WHY UNIFORM CALENDAR DAYS, NOT EVENT-SHIFTED DATES
    ---------------------------------------------------
    The candidate set is `cap` uniform day-offset draws inside each observation
    period's valid interval `[op_start + prior_obs_days, op_end - window_days]`
    (the exact span `_window_observed_cohort` admits), NOT the person's first-
    attestation dates shifted. Anchoring on attestation dates would put every
    random index a fixed offset before a real presentation — which is precisely
    what `episode_index_frame` already does, and would erase the contrast the
    probe exists to measure: that a population-random index rarely lands in the
    90-day run-up to a new diagnosis, so its forward gate is usually EMPTY.
    `cohorts._random_event_windows` deliberately anchors on events because an
    empty document is useless for a FIT; here we WANT the empty-gate rate a truly
    uniform index incurs, so we draw uniform on purpose and MEASURE it.

    WHY A HASH-TO-OFFSET DRAW, NOT RANK-OVER-EVERY-DAY
    ---------------------------------------------------
    A strictly uniform draw-without-replacement would rank every calendar day in
    the valid interval by `hash(person, day, salt)` and keep the `cap` lowest —
    the `_random_event_windows` min-hash idiom (`cohorts.py:1129-1143`), extended
    from keep-1 to keep-cap. But enumerating every day explodes a person's multi-
    year valid interval into thousands of rows before ranking (~10^9 rows over
    the cohort). Drawing `offset = pmod(hash(person, op_start, draw_no, salt),
    n_valid_days)` is the SAME uniform distribution at `cap` rows per period
    (modulo bias ~ n_valid / 2^32 ≈ 10^-6, negligible), deterministic and
    resume-stable for the identical reason (a pure function of the salted
    identifying columns, never `F.rand()`). `draw_no` in the hash separates the
    `cap` draws; a rare collision of two draws onto the same day simply yields
    fewer than `cap` distinct indices — "up to `cap`", exactly as the population
    picker yields "up to 1".

    VALIDITY IS THE ASSEMBLER'S OWN GATE, BELT-AND-SUSPENDERS
    ---------------------------------------------------------
    Draws are constructed to land in a valid interval, but they are then passed
    through `_window_observed_cohort` (the SAME function the episode arm and the
    assembler gate with) so validity is the assembler's own semantics, not this
    module's interval arithmetic. A candidate that any off-by-one would have
    slipped past is dropped by the gate function itself — the probe's random
    indices are, by construction AND by re-check, all fully observed.

    `persons`, when given (a frame carrying `person_id`), restricts the draw to
    that population — exp 0111 passes the EPISODE arm's surviving persons so the
    two arms compare on an identical person set and the only difference left is
    index location. `salt` has no default for the same experiment-identity reason
    `episode_index_frame`'s does not.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    op = observation_period.select(
        PERSON_COL, "observation_period_start_date",
        "observation_period_end_date")
    if persons is not None:
        op = op.join(persons.select(PERSON_COL).distinct(), on=PERSON_COL,
                     how="inner")

    # The valid interval is EXACTLY what `_window_observed_cohort` admits:
    # index >= op_start + prior_obs_days  AND  index + window_days <= op_end,
    # i.e. index in the inclusive day range [valid_start, valid_end].
    valid = (op
             .withColumn("_valid_start",
                         F.date_add("observation_period_start_date",
                                    int(prior_obs_days)))
             .withColumn("_valid_end",
                         F.date_add("observation_period_end_date",
                                    -int(window_days)))
             .withColumn("_n_valid",
                         F.datediff(F.col("_valid_end"), F.col("_valid_start"))
                         + F.lit(1))
             .where(F.col("_n_valid") >= 1))

    # `cap` deterministic uniform draws per (person, period). crossJoin against a
    # tiny range keeps this array-free (ADR 0047): no Python list rides a
    # closure, the draw index is a plain column.
    draws = valid.sparkSession.range(int(cap)).withColumnRenamed("id", "_draw")
    drawn = (valid.crossJoin(draws)
             .withColumn("_off",
                         F.pmod(F.hash(F.col(PERSON_COL),
                                       F.col("observation_period_start_date"),
                                       F.col("_draw"), F.lit(str(salt))),
                                F.col("_n_valid")))
             .withColumn(INDEX_COL, F.date_add(F.col("_valid_start"),
                                               F.col("_off")))
             .select(PERSON_COL, INDEX_COL)
             .distinct())

    from charmpheno.omop.cohorts import _window_observed_cohort
    valid_draws = _window_observed_cohort(
        drawn, observation_period, prior_obs_days=int(prior_obs_days),
        window_days=int(window_days))

    # A person with several observation periods can produce more than `cap`
    # distinct valid draws; rank by the same salted-hash idiom and keep `cap`,
    # so the per-person doc budget matches the episode arm exactly.
    ranked = valid_draws.withColumn(
        "_h", F.sha2(F.concat_ws("|",
                                 F.col(PERSON_COL).cast("string"),
                                 F.col(INDEX_COL).cast("string"),
                                 F.lit(str(salt))), 256)
    ).withColumn(
        "_rank", F.row_number().over(
            Window.partitionBy(PERSON_COL)
            .orderBy(F.col("_h").asc(), F.col(INDEX_COL).asc()))
    )
    return ranked.where(F.col("_rank") <= int(cap)).select(PERSON_COL, INDEX_COL)


def min_doc_length_drop_rate_by_ordinal(doc_lengths, *, min_doc_length,
                                        length_col="doc_length"):
    """R7.5: pooled kept/dropped doc counts per episode-ordinal band.

    `doc_lengths`: one row per assembled episode document, carrying
    `person_id`, `episode_no` (the ORIGINAL ordinal — see
    `episode_index_frame`) and `length_col` (the feature-vector length
    `to_bow_dataframe` would filter `min_doc_length` on). This function does
    not assemble that frame or count features itself; it only pools an
    already-counted one, so it has no opinion on vocabulary or CountVectorizer
    — the caller's `to_bow_dataframe` pass owns that.

    Returns a dict `{"1": {...}, "2": {...}, "3+": {...}}`, each entry
    `{"n", "kept", "dropped", "drop_rate"}` — pooled counts, egress-safe by
    construction (no per-node, no per-person cells). `drop_rate` is `None` for
    an empty band rather than a division error, so a fixture (or a corpus)
    with no episodes in a band reports cleanly instead of raising.
    """
    from pyspark.sql import functions as F

    banded = (doc_lengths
              .withColumn("_band",
                          F.when(F.col(EPISODE_COL) <= 1, F.lit(ORDINAL_BAND_1))
                          .when(F.col(EPISODE_COL) == 2, F.lit(ORDINAL_BAND_2))
                          .otherwise(F.lit(ORDINAL_BAND_3PLUS)))
              .withColumn("_kept",
                          (F.col(length_col) >= int(min_doc_length)).cast("long")))
    rows = (banded.groupBy("_band")
            .agg(F.count("*").alias("n"), F.sum("_kept").alias("kept"))
            .collect())
    out = {band: {"n": 0, "kept": 0, "dropped": 0, "drop_rate": None}
           for band in ORDINAL_BANDS}
    for r in rows:
        n, kept = int(r["n"]), int(r["kept"])
        dropped = n - kept
        out[r["_band"]] = {"n": n, "kept": kept, "dropped": dropped,
                           "drop_rate": (dropped / n) if n else None}
    return out
