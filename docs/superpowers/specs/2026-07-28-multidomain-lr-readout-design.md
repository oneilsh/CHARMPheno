# Multidomain LR / per-domain placement-lift readout — Design

**Date:** 2026-07-28
**Status:** Approved (user, 2026-07-28). Ready for a plan. **Depends on SP3a–SP3c.**
**Arc:** the multi-domain gated LDA arc (`docs/superpowers/specs/2026-07-24-multidomain-gated-lda-arc-design.md`); this is the quantitative case-finding read that SP4's ω work will build on.

## Goal

Give the multi-domain gated fits (exp 0071/0072 and future) a **post-hoc
likelihood-ratio placement readout** that (a) scores held-out case-finding by the
parameter-free alpha→∞ LR lift (the same "fork-settler" the single-domain
`lr_readout.py` provides), and (b) **decomposes the placement lift by domain**
(condition vs drug vs observation) — answering "which domain drives finding which
rare disease," and "did the LR lens recover disease signal that the
observation-dominated θ-mass buries." The single-domain LR readout is
dag_placement-only and cannot read a multi-domain artifact (it loads
`dag_placement_result.npz` + a single-`source_table` manifest); this builds the
multi-domain analog.

## Layer note

The **scorer** lives in `spark_vi/` and stays integer-id and domain-neutral (a
dict of per-domain BOW matrices + a dict-λ + the shared `DagLayout`). The
**readout driver** lives in `analysis/cloud/` and is the clinical-semantics layer:
it maps rare6 anchor concept-ids → engine nodes → per-disease rows, and labels the
domains condition/drug/observation.

## Decision: persist the test set in the run directory (choice C)

**User decision (2026-07-28).** The readout obtains the held-out test docs by
**loading them from the run directory**, where the fit persists them — NOT by
re-assembling from BigQuery, and NOT via the dag_placement content-hashed bundle
cache. Rationale (established in brainstorming): the dag_placement readout reloads
a cached parquet bundle keyed by a content-hash of the assembly source, which
carries a "no re-fit protection" fragility (any assembly edit → cache miss →
manual `--bundle-path`); re-assembling from BQ is heavier still (a live cluster
round-trip per readout + deterministic split/vocab reproduction). Persisting the
test set in the run dir makes the artifact **self-contained**: the readout loads
the run dir alone — no BQ, no cache key, no content-hash fragility.

**Consequence: exp 0071/0072 must be re-fit once** so their artifacts contain the
persisted test set. Thereafter the readout is pure post-hoc, and every future
multi-domain fit is LR-ready.

## Decision: reuse the single-domain LR math, summed over domains

The multi-domain placement score is the **per-domain sum** of the existing,
validated single-domain scorer. Because each domain's λ_m shares the same K
topics and the same `DagLayout` (only the vocabulary differs), the node-placement
LR score is additive across domains:

```
lr_score(u | doc) = Σ_m  lr_placement_scores(bow_m, lam_dict[m], lay, alpha, background_m)[u]
```

`lr_placement_scores(bow, lam, lay, *, alpha, background=None, ...)` already
exists in `spark_vi.models.topic.dag_placement`; `background=None` derives the
per-vocab base rate from the BOW matrix itself (`_lr_base_rate`). The per-domain
decomposition is that same sum **restricted to a domain subset** — cond-only,
drug-only, obs-only, or leave-one-out — so the decomposition falls out of the
generalization for free, with no new scoring math.

## Components

### 1. Fit-time persistence (`analysis/cloud/multidomain_cloud.py`)

After the fit, write to the run dir (the one re-fit):

- **`test_docs/` (parquet):** `bundle.test_df` projected to `person_id`,
  `features_0 … features_{N-1}`, `frontier`. The held-out per-domain BOWs + the
  set-valued frontier labels — the input to LR scoring.
- **`test_affinities/` (parquet):** `model.transform(bundle.test_df)` projected to
  `person_id`, the node-affinity vector, `frontier`. The model's **native
  θ-mass placement** — the readout's baseline, so no CAVI re-run is needed
  post-hoc. This adds one `transform` call at fit time.
- **`parent_int` in `manifest.json`:** the DAG parent map (engine-id → parent
  engine-id), so the readout can reconstruct `lay = DagLayout(parent_int, n_bg,
  tpn)` (n_bg/tpn already recorded). Currently the manifest saves `int2cid` /
  `name_by_id` but not `parent_int`.

No other fit behavior changes. Persistence is guarded so a fit still succeeds if
the test split is empty (write nothing, log it).

### 2. Multi-domain LR scorer (`spark_vi/spark_vi/models/topic/dag_placement.py`)

Pure-numpy additive wrappers over the single-domain functions (no Spark):

- `lr_placement_scores_multidomain(bows, lam_dict, lay, *, alpha, domains=None,
  backgrounds=None) -> np.ndarray [n_docs × n_nodes]`: sums
  `lr_placement_scores(bows[m], lam_dict[m], lay, alpha=alpha,
  background=(backgrounds or {}).get(m))` over `m in (domains or bows.keys())`.
  `domains` selects the subset for the decomposition.
- `lr_auc_sweep_multidomain(bows, lam_dict, lay, is_fg, *, alpha_grid,
  domains=None, backgrounds=None) -> {alpha: auc}`: max-over-nodes score vs
  `is_fg`, mirroring the single-domain `lr_auc_sweep`.

These are the only new library functions; both are unit-tested.

### 3. Readout driver (`analysis/cloud/multidomain_lr_readout.py`, spark-submit)

Post-hoc, no re-fit. Loads the run dir: dict-λ via `spark_vi.io.export.load_result`,
`manifest.json` (→ `lay`, `int2cid`, `name_by_id`, per-domain vocab sizes,
rare6 anchors via `disease`), `test_docs` and `test_affinities` parquet. Builds
per-domain sparse BOW matrices `bows = {m: [n_docs × V_m]}` (one collect, mirroring
`build_test_bow`), the boolean `is_fg` (frontier ∩ `lay.nodes`, so LR and θ-mass
score the SAME positive set), and per-disease positive sets. Emits:

- **Headline — per-rare-disease × domain-subset LR-AUC table.** For each rare6
  anchor node d, and each domain subset in {all, cond-only, drug-only, obs-only,
  drop-cond, drop-drug, drop-obs}: `AUC( max_{u ∈ subtree(d)} scores_subset[:, u],
  has_disease_d )`, where **`subtree(d)`** = d and its DESCENDANTS ∩ `lay.nodes`
  (a patient coded at a subtype must still count for the anchor; note this is the
  descendant subtree, NOT the ancestral gating "closure"), `has_disease_d` = the
  doc's frontier intersects `subtree(d)`, and `scores_subset =
  lr_placement_scores_multidomain(..., domains=subset)`. Both `subtree(d)` and the
  frontier test come from `parent_int` (invert to a children map). This is the
  "which domain finds which disease" matrix, framed as anchor-level detection.
- **θ-mass baseline.** Overall and per-disease AUC from the saved
  `test_affinities` (the model's native node scores) beside the all-domain
  LR-AUC — the "did LR recover signal the observation-dominated θ-mass buries"
  comparison (exp 0071/0072: observation is ~58% of θ).
- **Overall detection sweep.** The single-domain-style `alpha_grid` sweep of
  max-over-nodes LR-AUC vs `is_fg`, all-domains, for continuity with the
  dag_placement readout's output shape.
- **Optional per-case decompose viewer** (behind a flag): reuse `lr_decompose`
  per domain to show, for a sampled case, which domain's which tokens drive its
  placement (each row tagged with its domain).

### 4. Makefile (`analysis/cloud/Makefile`)

A new `multidomain-lr-readout ID=N` target (separate from the dag_placement
`lr-readout`, which is unchanged), dispatching `multidomain_lr_readout.py` via
spark-submit against `$(RUNS_DIR)/NNNN-slug/`. Post-hoc; no re-fit, no `--cache-uri`.

## Validation / acceptance

Per the arc's realism discipline, acceptance asserts structural + tie-out
correctness, not a single AUC number.

1. **Scorer unit tests (pure numpy, no Spark, no CDR):**
   - **Additivity:** `lr_placement_scores_multidomain(all domains)` equals the
     elementwise sum of the per-domain single-domain `lr_placement_scores`.
   - **Single-domain tie-out:** with one domain, the multidomain scorer reproduces
     `lr_placement_scores` exactly (regression guard — the generalization must not
     change the N=1 answer).
   - **Subset correctness:** `domains=[k]` equals that domain's single-domain
     score; leave-one-out equals all minus the dropped domain.
2. **Persistence unit test:** the driver's test-set projection writes the expected
   columns (`person_id`, `features_0…`, `frontier`) and the affinity projection
   the expected shape — Spark-free where possible, or a tiny local-Spark fixture.
3. **Cluster smoke (user-run) on the re-fit 0071/0072:** the readout loads the run
   dir with no BQ; the per-disease × domain-subset AUC table computes with all
   six rare6 rows present; the θ-mass baseline column is present; the additive
   decomposition is internally consistent (all-domain ≈ observed sum).

## Out of scope

- **The ω sweep and ω-weighted scoring — SP4.** These fits are ω=1; the readout
  reports the RAW per-domain lift (ω tempers θ, not the alpha→∞ lift limit). An
  ω-weighted readout is a later concern.
- **A quality/deployment threshold.** Like the single-domain LR readout, this is a
  triage RANKER read (AUC + per-domain lift), not a deployable classifier;
  low-prevalence precision is not a gate here.
- **Backfilling old artifacts.** Only re-fit artifacts (with `test_docs`) are
  readable; there is no path to score a pre-persistence fit (accepted in choice C).
- **`explain_away` routing-weighted scoring** as the default — the plain LR is the
  headline; the decompose viewer may expose it as an option, mirroring the
  single-domain readout, but it is not required.

## Risks

- **The re-fit is mandatory.** 0071/0072 must be re-run once; there is no readout
  for the current artifacts (they lack `test_docs`). Accepted (choice C tradeoff).
- **Test-set size.** rare6 at `person_mod=1` has a large background arm; the
  `test_docs` collect + BOW build is the same held-out scale the fit already
  materializes, so it is bounded, but the parquet is non-trivial — persist only
  the columns the readout needs.
- **Per-disease positive sets are small.** rare6 anchors have ~`min_n`-scale test
  positives, so per-disease AUCs are noisy; report the positive count beside each
  AUC so a small-n row is not over-read (mirrors the single-domain readout's
  discipline about junk-topic / small-n artifacts).
