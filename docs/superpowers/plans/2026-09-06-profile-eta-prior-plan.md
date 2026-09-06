# Plan: HPO-profile word-side eta prior for Gated-PC (exp 0116)

**Date:** 2026-09-06 · **Status:** approved, ready for agent execution
**Feeds from:** insight 0083 (spectral init costs case-finding; live candidate =
random init + knowledge prior), the 2026-09-06 HPOA scouting report (stage 1/2/2b:
GO — rolled-up profiles reach 132/299 label nodes at median 0.87 positive-doc
coverage), insight 0082 (the alignment residual this attacks).

## Goal

Give each credited label node a PERSISTENT, soft, knowledge-aligned word-side prior:
node-specific Dirichlet eta boosted on the OMOP condition tokens its rolled-up HPO
phenotype profile maps to. Then run the record A/B: **0116 = 0113's exact config
(random init) + profile-eta**, one effective knob, judged by `gated-pc-readout`
against 0113's recorded macro AUC **0.7813** (per insight 0083, evidence/sharpness
diagnostics alone are NOT acceptance — the readout is).

Why eta and not the init: an init can wash out or lock a wrong basin (0083); the
prior re-enters every lambda update (lambda = eta + counts), so a topic at its
floor still has sharp E[log beta] on profile tokens — anti-starvation AND
alignment — while real counts trivially override it. The gate already supplies
patient-side guidance; this is the missing word side.

## Normative design decisions (pre-registered; defaults chosen, knobs exposed)

- **D1 — one aligned topic per node block.** With `tpn=5`, ONLY the block's FIRST
  topic receives the boost; the other four stay flat/free. Rationale: forcing all
  five toward one profile recreates within-block redundancy and kills discovery
  capacity; one aligned + four free lets the block hold both the phenotype and
  unannotated structure. Knob `--profile-eta-topics N` (default 1) for the A/B
  later if wanted.
- **D2 — weights.** Per (node k, HP term t): `w = freq_k(t) * idf(t)`,
  `idf(t) = log(N / df(t))` with N = credited label nodes and df counted over the
  ROLLED-UP label-node profiles (the inflated df is the point — roll-up and IDF
  are a package). Unknown freq -> 0.5. A term's weight lands on each of its
  evidence concept ids; a concept claimed by several terms takes the MAX. NOT
  rows: multiplicative downweight `eta * 0.5`, floored at `0.1 * eta_base`
  (garnish; never zero — Dirichlet params must stay positive).
- **D3 — scale.** Per boosted topic, the boost vector is normalized to sum to
  `profile_eta_strength * eta_base * V_condition` added pseudo-mass
  (default strength **1.0**: doubles the topic's prior mass, concentrated on its
  ~40-300 profile tokens — a sharp floor tilt, still dominated by real counts).
- **D4 — artifact flow (keeps BQ out of the fit).** `hpoa_stage2_probe.py` gains
  `--emit-eta PATH`: writes (mondo_id, concept_id, weight, neg, coverage) with
  freq x IDF already folded — CONCEPT ids, not vocab indices, so the file is
  bundle-agnostic and patient-free EXCEPT the coverage column (a train-derived
  fraction) — the file stays workspace-internal in `data/` (gitignored), never
  committed. The FIT driver reads it, maps concept_id -> condition-domain vocab
  index via the bundle meta it already holds, and builds the sparse per-topic
  boost. Optional `--profile-eta-min-coverage` (default 0 = off) drops nodes in
  the annotation-population-mismatch tail (varicose 0.09, mitral valve 0.23).
- **D5 — scope and guards.** Condition domain (domain 0) only — HPO maps no drugs
  or measurements. Nodes without a credited profile keep flat eta (167/299; they
  are the INTERNAL CONTROL in the readout slice). Incompatible with resume /
  warm-start (guard exactly like spectral's). No bundle/corpus cache-key change:
  this is a fit parameter recorded in the run manifest, and NO source-hashed
  module is touched.

## Work packages (each: implement + unit tests + the validation gate below)

- **WP-0 (scout, first):** read `spark_vi` GatedOnlineLDA + VIRunner and list
  EVERY site where scalar `eta` enters (lambda init, lambda update, ELBO terms,
  anything executor-side under ADR-0027 lazy blocks). Write the micro-design as
  a comment block in the WP-1 commit. ADR 0047 applies: the sparse boost reaches
  executors via explicit broadcast, never a task closure.
- **WP-1 (spark-vi core):** optional sparse per-topic eta boost
  ({topic_index: (vocab_idx_array, weight_array)}, domain 0) honored at every
  site WP-0 found. Tests: boost tilts the zero-count floor's E[log beta]; absent
  boost is byte-identical to today; update = eta_vec + counts; non-boosted
  domains/topics untouched.
- **WP-2 (probe `--emit-eta`):** D2/D4 semantics; IDF over credited label nodes;
  coverage column from the existing support pass. Tests: IDF math on a fixture,
  max-over-terms per concept, NOT floor, unknown-freq default.
- **WP-3 (wiring):** estimator params in `pc.py`
  (`profileEta`/`profileEtaStrength`/`profileEtaTopics`/`profileEtaMinCoverage`)
  building the topic-indexed boost from the eta file + bundle meta via the SAME
  node_order/block layout `inspect_topics.topic_labels` documents; argparse in
  `gated_pc_cloud.py`; emission in `run_experiment.py` ONLY when set (existing
  runs' args stay byte-identical). Tests mirror the spectral-wiring suite.
- **WP-4 (experiment 0116):** doc = 0113 front matter VERBATIM (init: random,
  diag_only: true, max_iter 50) + the profile-eta block; run fit, then
  `gated-pc-readout ID=116`, then `inspect_topics --readout-auc` and `--digest`.
  Results + insight write-up per the acceptance below.

## Validation gate (every WP)

`PYTHONPATH=spark-vi poetry run pytest <new/touched suites>
tests/scripts/test_case_finding_cache_mondo.py -p no:randomly -q` all green — the
60-test tripwire stays byte-identical (source-hashed modules are read-only:
charmpheno/omop/{cohorts,multi_domain,case_finding_assembly}.py,
analysis/cloud/{mondo_*,condition_dag,preindex_closure}.py). `make test` in
spark-vi for WP-1. Egress floor 20 on anything printed. No model names in any
committed artifact. Per-step timings ("X.Xs"), never cumulative "elapsed".

## Pre-registered acceptance for 0116 (decided BEFORE the run)

1. **Primary:** readout macro AUC over the shared scored nodes >= 0.7813 - noise
   (non-inferiority); a WIN is credited-node improvement: per-node AUC on the 132
   credited nodes UP vs 0113 while the 167 uncredited nodes (internal control)
   move ~0. Read via `--readout-auc` both runs.
2. **Secondary (legibility):** starvation fraction vs 0113's 72% on credited deep
   nodes (the tilted-floor hypothesis predicts credited nodes feed under random
   init); digest `--grep 'cardiomyopath'` topics anchor on phenotype, not
   pregnancy/demographics.
3. **Failure reads:** credited nodes DOWN -> prior misweighted (suspect IDF/scale
   before abandoning); everything flat -> strength too low (try 3.0); uncredited
   nodes moved -> a wiring bug (the boost leaked past its blocks), stop and fix.

## Cluster commands (every cluster command carries the git preamble)

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
make -C analysis/cloud hpoa-stage2-probe ID=114 GPR_ARGS="--emit-eta ~/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv"   # after WP-2
make exp ID=116                                                                                                                        # after WP-3/4
make -C analysis/cloud gated-pc-readout ID=116 GPR_ARGS="--readout-mode distributed"
```

HDFS caches die with the cluster (AGENTS.md); the bundle rebuild via
`gated-pc-readout` reproduces recorded numbers exactly (verified 2026-09-06).

## Out of scope (parked, named)

All-topics-boost A/B (D1 knob), coverage-gating ON (D4 knob), the DAG-aware
increment-over-ancestor IDF variant, profile-overlap as an alignment METRIC,
sparse spectral group accumulators, PC revival. Spectral init itself stays a
demoted legibility/diagnostic tool (insight 0083).
