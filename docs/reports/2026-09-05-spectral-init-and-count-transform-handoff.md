# Handoff — spectral init resolves deep-node starvation; the residual is genuine anchor misalignment (2026-09-05, session 2)

Continues the 2026-09-05 topic-inspection/tpn handoff. Branch
**`claude/gated-conditional-voi`**, HEAD **`1387cca`** (tree clean, all pushed).
Model note: session ran mostly on Opus; a few turns on Fable. Ignore for the science.

## TL;DR — the arc this session closed

The session-long question was WHY deep per-node topics starve at whole-Mondo depth
(insight 0079). It is now **resolved**, and a new, sharper question has replaced it.

1. **0113 (CV branch, tpn=5, random init)** — falsified topic BUDGET as the lever: 72%
   of node topics still starved; depth-4 evidence floor at ~61 (the Dirichlet prior).
2. **0114 (same, `init: spectral`)** — the decisive A/B. **72% → 1% starved; depth-4
   floor 62 → 1000.** Only the seed differs, so: **the flat-start / deflation trap was
   the binding constraint** (insight 0079 RESOLVED → insight **0082**). Starvation was
   never fundamental; a sharp anchor-word seed fixes it.
3. **The 0114 residual:** most fed deep topics are coherent (AF, systolic/diastolic HF,
   valve), but a minority anchor on the WRONG signal — intrinsic/dilated **cardiomyopathy**
   topics dominated by **pregnancy** codes. Shawn's standing fear: spectral's criterion
   (max co-occurrence separability) is not aligned with node meaning.
4. **0115 (0114 + `count_transform: binary`)** — tested whether that was a token-MASS /
   burst artifact (conditions/drugs are raw per-visit counts; only measurement was binary,
   0077). **FALSIFIED:** per-doc presence held the floor (1% starved, redundancy cleaner
   at 0.17) but left the pregnancy anchoring essentially unchanged. So the misalignment is
   **genuine objective misalignment** (separability ≠ phenotype): at presence level the
   demographic/etiologic stratum (peripartum, genetic CM) IS the most-separable direction
   for these heterogeneous nodes. Shawn's fear confirmed as the deep kind, not the shallow.

## THE OPEN DECISION POINT (start here)

The residual is a topic-legibility blemish. **It only matters if it costs the downstream
goal (case-finding).** Three paths, cheapest first — **recommended: (1).**

1. **`gated-pc-readout` on 0114 (or 0115)** — NO re-fit; reuses saved λ, fits the L-BFGS
   readout heads, writes `readout_heads_gated_pc.npz` + `results_readout.json`. Tells us
   whether the demographic anchors cost macro/cond AUC or the localized head just doesn't
   read them. This GATES whether alignment is worth chasing. Command:
   `make -C analysis/cloud gated-pc-readout ID=114 GPR_ARGS="--readout-mode distributed"`
2. **Nuisance deflation** — deflate each node's anchor against a corpus-wide demographic
   basis (age/sex/pregnancy axes). Cheaper than PC, principled, UNBUILT.
3. **Supervision / PC revival** — the real alignment fix, and plausibly insight 0066's
   payoff regime (low-separability cardiac signal buried under a more-separable
   demographic one), UNLIKE the AoU antidepressant task where PC was marginal. BUT PC is
   parked for real reasons (see below).

## Why PC is parked (recalled this session, grounded in the record)

- **Unified co-fit head (the same-model-post-fit Shawn wants) carries a small AUC tax**
  vs a fresh two-stage LR readout: 0.706 vs 0.724 macro (insight 0069), and on gated-CV
  the shaping actively hurt (0102: co-fit 0.567 « gate 0.739). It IS well-calibrated
  (ECE better than two-stage), just less discriminative, worst at small n.
- **Compute wall on all-Mondo:** dense full-K co-fit head is O(C·K²) Hessian-collect, not
  shuffleable (0101 superseded). The localized head dodges it (O(|support|³)) but shaping
  still underperformed.
- **0066 wrinkle:** PC shaping bought ~0.005 AUC on AoU antidepressant because "the signal
  is already in the unsupervised topics"; PC only pays in the hidden-low-mass regime. The
  alignment residual here is plausibly that regime — the one place PC's premise holds.

## Tooling built this session (all tested, pushed)

- **Spectral init wired into Gated-PC** (`spark-vi/spark_vi/mllib/topic/pc.py`): the shipped
  block-aligned anchor-word seed (`gated_init.py`), never before connected to the PC path.
  Params: `init/spectralMethod/spectralMaxVocab/spectralD/spectralMinDocFreq/anchorScope/
  spectralTopoOrder`. Builds `data_summary` from `pc_rdd`, passes to `VIRunner.fit`.
- **Scalable-seed performance rework** (`gated_init.py` / `spectral_init_scalable.py`):
  - Fixed a PRE-EXISTING crash: `precompute_projection_rows` was a dangling import that
    broke the ENTIRE scalable path (9 red tests) — restored.
  - BATCHED by depth level (`_NodeGroups`), B auto-sized to `spark.driver.maxResultSize`
    (`_safe_batch_cap`, models the ~sqrt(P) treeReduce fan-out — the first attempt OOM'd
    at 4.1 GiB), `CHARM_SPECTRAL_BATCH=<n>` overrides.
  - `_scalable_projection_dim`: dropped the dense K-floor on `d` (scalable places only
    ~tpn+|seed_rows| anchors PER NODE, never K) → d ~1000 not 1498, ~2× smaller sketches.
  - `pooled=False` on batch passes: only docs training a batch node are projected.
  - **Known limitation:** per-batch time did NOT fall with depth as hoped — the dense
    (V,d) group-sketch treeReduce+collect is the depth-independent floor. Real fix =
    SPARSE group accumulators (deep sketches touch ~1-3k words, stored dense). UNBUILT;
    the next big perf lever if whole-Mondo spectral is needed. Seed ~1.5h on the lean
    cluster (6 workers) at B=7, d=768.
- **`count_transform` knob** (`pc.py:_transform_counts`, `--count-transform` none|binary|
  log1p): per-doc presence / log-damp, applied in-memory before seed AND fit; the cached
  bundle stays raw (NO cache-key change). Insight 0077's measurement fix extended to all
  domains.
- **`inspect_topics.py --digest`** (earlier this session): compact single-block topics
  view (header + depth rollup + fed exemplars + `--grep` + redundancy), condition/drug-led
  word lines. `make inspect-topics ID=N INSPECT_META=... INSPECT_NAMES=... INSPECT_ARGS="--digest ..."`.

## Docs written this session

Experiments **0113** (done, budget falsified), **0114** (done, spectral MET), **0115**
(done, burst-bias falsified). Insights **0082** (init lifts floor + alignment residual),
**0079** status → RESOLVED. Reports: this file + the prior tpn handoff.

## Reproduction / read commands (off-cluster, reuse cached meta at /tmp/inspect_meta_113.json)

```bash
make -C analysis/cloud inspect-topics ID=114 \
    INSPECT_META=/tmp/inspect_meta_113.json INSPECT_NAMES=/tmp/concept_names_113.csv \
    INSPECT_ARGS="--digest --redundancy 30 --grep 'cardiomyopathy|atrial fibrillation|heart valve|myocardial infarction|heart failure|pregnan|gestation'"
```
Cluster commands need the preamble:
`cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only`

## Standing constraints (unchanged)

Work ONLY on `claude/gated-conditional-voi`. Egress floor: no cell < 20; reports carry
pooled figures + counts-of-nodes only (the digest is model-params + node-names, safe).
Source-hashed modules (cohorts/case_finding_assembly/multi_domain/mondo_*) untouched all
session; the 60-test tripwire stays byte-identical. NO PC runs without explicit go. Every
cluster command carries the git preamble. Commit trailer: Co-Authored-By: Claude Opus 4.8
+ Claude-Session. Never commit patient-level data.

## Config quick-ref for the next experiment

0114/0115 front matter is the current template: CV branch `mondo_branch: MONDO:0004995`,
`tpn: 5`, `init: spectral`, `spectral_method: scalable`, `spectral_d: 768`, `diag_only:
true`, `max_iter: 50`, `min_positives: 100`, 20 executors. Add `count_transform: binary`
for presence (0115). A full-run (readout, not fit-only) drops `diag_only`. The seed is
batch-size-invariant; d is a speed/quality knob (512 → faster, mild quality).
