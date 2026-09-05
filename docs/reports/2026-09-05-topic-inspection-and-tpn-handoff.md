# Handoff — topic-inspection tooling + the depth-starvation investigation + exp 0113 (tpn test) — 2026-09-05

Written before a context compaction. Branch **`claude/gated-conditional-voi`**, tree
clean, everything below committed and pushed (**HEAD `1466f29`**). Read `AGENTS.md` first
(cache-key landmine, egress floor, cluster-command preamble). Supersedes the
2026-09-05 WP-E handoff for the topic-inspection line of work; the 0111 build itself is
done.

Session: `https://claude.ai/code/session_01CWHmrp8SXQU127fTYksP1c`.

---

## TL;DR — where we are

The 0111 episode fit finished; inspecting its **topics** (not just its AUC) opened a
long, productive investigation into **why the deep node topics starve**, built a
full **off-cluster topic-inspection toolkit**, produced four insights (0079–0081 +
corrections), and landed on a sharp, testable hypothesis — **topic budget (`tpn`)** —
now under test as **exp 0113** (cardiovascular branch at `tpn=5`, fit-only, JUST FIT,
awaiting inspection). **0112 (the matched-random control) was killed** by a cluster
idle-out and is **deprioritized** — comparing arms on a model we now believe is
`tpn`-starved would measure the starvation twice.

**Immediate next action:** inspect exp 0113's fit (below) — does the deep CV evidence
floor lift under `tpn=5`? That decides the whole modeling direction.

---

## The tooling built this session (all committed, all off-cluster / off-YARN)

`analysis/cloud/inspect_topics.py` — **standalone, pure numpy+stdlib**, reads a saved fit
(`gated_pc_result.npz` λ + `manifest.json`, optionally the readout heads and the bundle
meta) and renders a topics view. **No Spark, no bundle reload** — runs on the master
while a fit is going. Sections/flags:
- per-node **sharpness** (evidence = λ pseudo-count mass; eff.support = exp(entropy);
  frac = support/V — flat→1.0), a **sharpness distribution** + evidence quantiles header.
- **readout loadings** from the REAL decoder — reads `readout_heads_<label>.npz` (V) or
  falls back to `readout_ckpt` (W_std) then the untrained co-fit `w_CK` (all labelled);
  prefers the **standardized** W_std for loadings because raw-θ V explodes (÷σ) for
  low-variance topics.
- `--bundle-meta <meta.json>` — off-YARN `hdfs dfs -cat <cache>/<key>/meta/part-*`
  (an HDFS *client* read, **zero YARN containers**, safe mid-fit) → topic **words**
  (vocab_maps) + node **depth** (parent_int) + a **sharpness-by-depth** rollup. Guards
  against a wrong bundle key (checks vocab sizes + int2cid vs the run manifest).
- `--tour N` — tree tour: the N best-fed nodes at EACH depth, indented by level, all
  three domains, STARVED/DEGENERATE flags.
- `--redundancy N` — per-parent cosine among a parent's FED children; flags UNIFORM
  collapse (capacity starvation) vs partial (fine); correlates with fan-out.
- `--grep REGEX` — look up specific nodes by name (evidence/depth/words) — the
  evidence-vs-depth probe.
- `--sort {sharpness,evidence,alpha,depth}`, `--top-topics`, `--top-words`, best-fed
  section, background-topic words.

`analysis/cloud/resolve_vocab_names.py` — names the topic WORDS off-YARN: reads the bundle
meta's vocab_maps, DECODES measurement synthetic tokens (`concept_id*100 + state` →
"Creatinine [high]", via `charmpheno.omop.measurement_tokens`), runs a **targeted
`bq query`** on the CDR concept table (cdr+billing from the run manifest — a few-thousand-id
IN-list, not the 8M-row table, no Spark), writes `concept_names.csv`. `--dry-run` /
`--names-json` for testing without BigQuery.

`analysis/cloud/Makefile` — **`make inspect-topics ID=N`** (sources `.workspace_env` then
plain `python`, no Spark, run-dir glob-resolved). Knobs: `INSPECT_KEY=<bundle key>`
(auto `hdfs dfs -cat` the meta), `RESOLVE_NAMES=1` (auto-run the bq name resolve),
`INSPECT_NAMES=<csv>` (reuse a resolved csv), `INSPECT_ARGS="--tour 2 --redundancy 30 --grep '...'"`.

Tests: `tests/scripts/test_inspect_topics.py` (13), `test_resolve_vocab_names.py` (6) —
all green. Both scripts are driver-owned; no source-hashed module touched; the 60-test
tripwire is unaffected.

**How to run the inspector (off-cluster, safe any time):**
```bash
# headline read needs NO bundle key (node names come from the run's manifest.json):
make -C analysis/cloud inspect-topics ID=113 INSPECT_ARGS="--grep 'cardiomyopathy|arrhythmia|...'"
# words + depth + tour: find the key, then:
hdfs dfs -ls -t hdfs:///user/dataproc/charm/case_finding_cache | head    # newest non-sidecar dir = the run's bundle
make -C analysis/cloud inspect-topics ID=113 INSPECT_KEY=<key> RESOLVE_NAMES=1 INSPECT_ARGS="--tour 2 --redundancy 30"
```

---

## Cluster run state

- **0111 (episode, whole-Mondo, tpn=1) — FIT DONE, inspected.** macro AUC 0.7203 over
  2241 nodes, cond_AUC 0.63–0.75 by depth, ECE 0.0195. Its calibration-fit was **killed**
  (it only refines conditional ECE; recoverable any time via `make gated-pc-readout ID=111`,
  no re-fit). Bundle key `d1e8e3282108efd5`; names csv cached at `/tmp/concept_names_111.csv`.
- **0112 (matched random control) — KILLED** (exit 130 / SIGINT, overnight Dataproc
  **cluster idle-out**, confirming it's cluster-idle not UI-idle). **Deprioritized:** the
  episode-vs-random comparison is uninformative on a `tpn`-starved model; revisit only
  after the model is fixed. Its doc (`docs/experiments/0112-*.md`) is unchanged and can be
  re-run later (`rm -rf <RUNS_DIR>/0112-… && make exp ID=112`).
- **0113 (CV branch, tpn=5, fit-only) — JUST FIT (2026-09-05 ~13:23), AWAITING INSPECTION.**
  K=1498 (8 bg + **298 CV nodes × 5**), 50 iters, 1504 s, `diag_only` saved the fit-only λ
  to `<RUNS_DIR>/0113-mondo-cardiovascular-tpn5`. The "299/299 heads DEAD" banner is
  EXPECTED and irrelevant (it's the co-fit head probe; `weight_y=0` = head untrained by
  design; we want the topics). Fit-log tease: per-topic `Σλ_k min` = 26–30 (m0/m1), 8 (m2)
  — above the bare prior floor (~3.3/1.2) where 0111's starved topics sat at 1.00 — but this
  could be ADR-0027 lazy-update leaving unused `tpn=5` blocks at random-init mass, NOT real
  feeding. **The `frac`/`eff.support` columns from `inspect_topics` are the real read.**

`<RUNS_DIR>` = `/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs`.

---

## The scientific arc (insights 0079–0081), and what's SETTLED vs OPEN

Inspecting 0111's topics showed **most deep (depth ≥5) node topics are flat/starved**
(evidence ≈ the Dirichlet prior floor). The investigation — with several of my over-reads
corrected by Shawn, each walked back in the insight record — converged on:

**SETTLED (ruled out with data, not intuition):**
- **NOT prevalence / doc-count.** Well-populated deep nodes split cleanly: ischemic stroke
  (n=2618), hemorrhoid (n=2138), varicella (n=2222) are FED; breast-carcinoma subtypes and
  strep syndromes (n~1000+) are STARVED. So it tracks **code-separability after the leakage
  strip** (`strip_mode=both` drops every DAG-node code from the BOW; a node learns only from
  its distinctive UNSTRIPPED drug/lab/non-Mondo signal). — **insight 0079**.
- **NOT sibling redundancy / capacity-at-fed-nodes.** Measured: 0/194 parents show uniform
  child collapse; corr(fan-out, median child-cosine) = −0.09. Where topics are fed they
  differentiate well. — **insight 0080**.
- **NOT `n_bg`.** 0062 found doubling background NULL for detection; and background
  under-capacity spills into the SHALLOW node blocks (gate-reachable), never the deep ones
  — "deep-unused + shallow-catch-all" is the signature. The 8 background topics ARE the rich
  multi-domain interpretable structure (cardiometabolic / acute / PCP / endocrine-onc /
  renal-hepatic; measurement value-states discriminate here). — **insight 0081**.
- **PC / `weight_y>0` is not a quick lever** — parked (not dead): engine pathologies fixed
  (0072/0073, ADR 0044/0045), but the co-fit head (0.567) is worse than the gate (0.739) so
  shaping hurts (exp 0102); revival needs a full-K head beating the O(C·K²) collect wall
  (matrix-free L-BFGS / MI-selected support, scaffolded, unbuilt). — **insight 0080**.

**THE LEADING OPEN HYPOTHESIS — topic budget (`tpn`):** checking the experiment log showed
signal-existence is already settled (0035: a rare EDS foreground recovered POTS/MCAS/
vascular-EDS subphenotypes; 0019: K=80→~40 phenotypes; rare6 coherent node topics) — so it
is **NOT an information ceiling**. But **every deep-recovery success used a generous budget
(rare6/diabetes `tpn=5`, EDS a 20-topic block)** while whole-Mondo forced **`tpn=1`** for
compute. So "recovers vs starves" is **confounded with topic budget**: a one-topic node
can't out-compete its sharp ancestor stack for its residual; five give it room. **Deflation
IS the flat-start trap** — which also **restores spectral init** as a candidate (0063's
"init null" was a shallow 170-block DAG). — recorded in **insight 0079** (mechanism
UNRESOLVED; `tpn` is the leading structural lever).

**Secondary candidates still live:** spectral init (dead code on this branch — a broken
`spectral_init_scalable` import, §5.6 of the whole-Mondo handoff); **closure-only strip**
(mask only a node's own+ancestor codes per-node in the gated E-step, keeping comorbidity
codes — a real engine change); softer deflation gate. `tpn` is tested first because the
history most directly implicates it.

---

## exp 0113 — the acceptance criterion (what to read next)

`docs/experiments/0113-mondo-cardiovascular-tpn5.md`. Inspect the saved fit:

**Does the depth-≥5 CV node evidence floor LIFT under `tpn=5` vs 0111's `tpn=1`?**
- Read the **sharpness-distribution header** (% starved — compare to 0111's **73%**) and
  `--grep` the CV subtypes (cardiomyopathy / arrhythmia / valve / MI / heart-failure) for
  their `evidence`/`frac`; compare to the SAME nodes in 0111 (`--grep` on 0111, they were
  ev≈5 / frac≈1.0).
- **LIFT** (deep CV subtypes go sharp/fed) → **topic budget is the lever** → run the 0071
  **cascade at `tpn≥5` per branch**; the whole-Mondo monolith under-fed depth purely for
  compute.
- **NO LIFT** → budget isn't binding → go to **closure-only strip** / spectral init /
  softer gate next.

Caveat baked into the doc: 0113 is STANDARD-indexed, 0111 is EPISODE — a second-order
confound for a topic-*feeding* question. A **`tpn=1` companion on this same branch/index**
(a second fit-only 50-iter run, K≈280, trivial) is the airtight A/B — recommended follow-on,
easy to add as 0114 or a one-line `tpn` override.

---

## NEXT actions, in order

1. **Inspect 0113** (headline needs no key): `make -C analysis/cloud inspect-topics ID=113
   INSPECT_ARGS="--grep 'cardiomyopathy|arrhythmia|atrial fibrillation|heart valve|mitral|aortic|myocardial infarction|heart failure'"`
   → does the CV deep floor lift? This is THE decision.
2. Per the result: either (a) spec the **cascade at `tpn=5`** (the real architectural
   direction, 0071), or (b) prototype the **closure-only strip** (per-node E-step feature
   masking) / repair spectral init, or (c) add the **`tpn=1` CV companion** to isolate `tpn`
   cleanly first.
3. **Backlog, deferred:** 0112 random control (only meaningful once the model is fixed);
   evaluation should separate **learnable** from **code-definitional** nodes (scoring
   histology subtypes is scoring the impossible); the **min-support knob** Shawn wanted
   bumped — `min_positives` is ALREADY 100 in 0110/0111/0113; confirm whether he meant
   `min_label_count`/`min_patient_count`/`min_df` (all 20) before changing (cache-key input,
   future docs only).

---

## Config / mechanics notes (so they're not re-derived)

- **`diag_only: true`** in front matter → `--diag-only` → fit + save λ + head-probe, THEN
  return before any readout/θ-collect. The fit-only npz has λ (topics) but the co-fit
  `w_CK` is untrained at `weight_y=0` — irrelevant for topic inspection.
- **`mondo_branch: MONDO:xxxx`** + `dag_source: mondo_native` restricts the kept label DAG
  to that subtree (`mondo_native_dag.build` `branch_root`); the CORPUS stays all-patients.
- **`tpn`** is a front-matter knob (K = `n_bg + kept_nodes*tpn`, emergent, printed at build).
  `run_experiment.build_gated_pc_args` forwards `tpn` / `mondo_branch` / `diag_only`.
- **inspect_topics needs no bundle key for evidence/sharpness** (node names from the run
  manifest); the key only adds words + depth (from the off-YARN meta cat).
- **Bundle key**: `hdfs dfs -ls -t <cache_uri> | head` — newest non-`conversion_sidecar` dir.
  `cache_uri = hdfs:///user/dataproc/charm/case_finding_cache` (cluster-ephemeral).
- **Cluster idles out** (Dataproc, not just UI) — long fits die with exit 130; size runs to
  finish, or babysit. `spark.executor.instances: 20`.

---

## Standing constraints (unchanged, do not break)

Work ONLY on `claude/gated-conditional-voi`. Never commit patient-level data; egress floor
(any aggregate cell <20 not disclosable; reports/banners carry pooled figures + counts-of-
nodes only — the inspector reports only MODEL PARAMETERS + node names, egress-safe). No model
identifiers in committed content. Source-hashed modules never edited except in a deliberate
announced cache-drop; the 60-test tripwire stays byte-identical otherwise (nothing this
session touched them). Every cluster command carries the `cd ~/repos/CHARMPheno && git fetch
origin claude/gated-conditional-voi && git checkout … && git pull --ff-only` preamble. Don't
merge to main without explicit permission. Verify every subagent's work independently.
Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Claude-Session:
https://claude.ai/code/session_01CWHmrp8SXQU127fTYksP1c`.
