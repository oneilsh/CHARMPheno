---
id: 90
slug: mondo-cardiovascular-templatebranch
status: pending
model_class: gated_pc
cohort: population_mondo_cardiovascular
cohort_def: population_mondo_cardiovascular
disease: rare_priority
# STEP A of the whole-Mondo fit (plan 2026-08-17): the FIRST fit whose label DAG
# is the Mondo powered hierarchy (exp 0088) instead of the SNOMED concept_ancestor
# anchor forest. Restricted to ONE body-system branch — cardiovascular disorder
# (MONDO:0004995) — so it validates the NEW machinery (Mondo engine DAG, SNOMED-
# climb per-patient frontier, population index, localized head) at real-but-bounded
# scale (K~few-hundred, the range we run comfortably) BEFORE the whole K~3,800 run.
dag_source: mondo
mondo_branch: MONDO:0004995
min_positives: 100
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
# Population index (all-comers, random event-anchored window) + SNOMED-climb
# attestation: a patient's cardiovascular Mondo frontier = their condition codes
# rolled up to mapped cardiovascular anchors. closure mask = conditional/sharpening
# readout (compare to 0089's within-branch conditional AUC). localized head (0089).
extra_domains: measurement,drug
label_mask_mode: closure
localize_head: true
# --- everything else identical to 0089/0085 ---
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback
lookback_days: 1825
label_window_days: 365
strip_mode: both
n_bg: 8
tpn: 1
optimize_doc_concentration: true
weight_y: 50.0
head_optimizer: newton
head_lr: 0.3
head_newton_ridge: 0.05
head_l2: 0.01
grad_cavi_iters: 30
topic_trust: 0.05
weight_y_warmup_iters: 25
max_iter: 100
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
skip_unsup_gated: false
# Re-enabled the weight_y=0 twin for the SHAPING ablation (does supervision improve
# the representation, or does the gate structure alone carry it?), now that the
# readout collect is bounded (run-1 died there). readout_sample_frac subsamples the
# driver-side theta/proba arrays; 0.3 of the whole-pop cardiovascular test is ample
# for the per-node LRs at min_label_count=20, and it de-risks the Step-B readout wall.
readout_sample_frac: 0.3
with_dag_head: false
baseline_max_iter: 100
min_label_count: 20
eval_every: 0
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# 0090 — Mondo cardiovascular template branch (Step A of the whole-Mondo fit)

The first fit off the **Mondo** backbone. Everything about the model is 0089
(multi-domain gated-PC, closure mask, localized head, ridge-only); the ONE change
is the label DAG's *source*: instead of the disease's SNOMED `concept_ancestor`
anchor forest, the DAG is the **Mondo powered hierarchy** (exp 0088) restricted to
the cardiovascular-disorder branch (`MONDO:0004995`), and patients are placed on it
by **SNOMED-climb** (their condition codes rolled up to mapped Mondo anchors) over a
**population index** (all-comers, no single disease anchoring the window).

This is the go/no-go that the *new plumbing* works end to end at real scale before
the whole K≈3,800 run (Step B): the Mondo engine DAG (`mondo_dag.build_mondo_engine_dag`),
the per-patient frontier (`make_mondo_attested_provider`), the population index
(`case_finding_population_index_table`), and the `before_dag`/`attested_provider`
seams into the multi-domain assembler — all reusing 0089's split/prune/ledger/
frontier/BOW/strip/labels + localized head verbatim.

## What to read

- **`[mondo]` line** — powered terminals + class nodes in the cardiovascular branch
  (the realized DAG size); **`[cost]`** — K, fan-out, localized-vs-dense head
  matrix memory/compute (watch high-fan-out parents; this is the whole-Mondo
  risk surfaced at bounded scale).
- **ledger** — coarsening rate + test coverage: does the climb place a sensible
  fraction of patients on cardiovascular nodes (sanity vs exp 0087's population
  reach)?
- **conditional readout (per-node reliability)** — cond AUC by depth + mean/max
  ECE, the SAME readout as 0089. Within-branch discrimination is where 0089's
  localized head matched dense EXACTLY (depths 1/2), so the Mondo tree (which has
  real depth: cardiovascular → subtypes) should localize *at least* as well as the
  flat 41-anchor worst case.
- **|w_CK|max** — bounded (ridge unchanged); 0089 saw a transient excursion to
  ~500, watch it (bump `head_l2` if it diverges).

## Why cardiovascular

Big, adult-powered, clinically legible, and a genuinely DEEP subtree (cardiovascular
disorder → heart disease / vascular disease → specific entities), so it exercises
the within-branch conditional structure the whole-Mondo fit relies on — unlike the
flat root→41 layout of 0089. `min_positives=100` (exp 0088's floor) keeps K bounded;
raise it to shrink K, lower it to densify the tree.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=90
```

Paste the `[mondo]` + `[cost]` lines, the ledger, and the two conditional-readout
blocks (incl. per-node reliability) so we can put Mondo-DAG placement side by side
with 0089's SNOMED-anchor placement before scaling to the whole tree.

## Run log

### Run 1 — OOM at the head-stat collect; Step A caught the scaling gap it exists for

Build + assemble were clean and the profile was exactly the bounded-scale target:
`powered terminals=273, class nodes=163`, K=444, ledger `dropped=0` /
`coarsening=0.0` / `test_fg_docs=45992` (healthy cardiovascular coverage via the
climb), and a DEEP tree (`kept_by_depth` 0→10, mass at 5–6 — the within-branch
regime where localization matched dense exactly in 0089). Fan-out maxed at 14,
support/node at 24.

**Then the fit died** collecting the head sufficient-statistic:
`Total size of serialized results of 6 tasks (4.3 GiB) is bigger than
spark.driver.maxResultSize (4.0 GiB)`. Root cause: the localized head localized the
Newton SOLVE (indexing sub-blocks) but the Fisher was still EMITTED dense — each
partition built and shipped the full `(C, K, K)` Hessian (the cost profile's
`dense C*K^2 = 657.3MB`; ×6 tasks ≈ 4.3 GB). At 0089's K=101 that was 8 MB
(invisible); at K=444 it OOMs; at Step B's K≈3,800 it is the 850 GB wall the whole
localization was meant to dodge. **Localization was realized in compute, not in
collection.**

**Fix:** emit the Fisher as a compact padded `(C, S, S)` per-node block stack
(S = max support = 24) instead of dense `(C, K, K)` — exact (each block is the dense
Fisher's support sub-block, term-for-term; the padded tail is never read), ~2 MB
instead of 690 MB/partition. The cost profile now reports the true collected size
(`collected C*S^2`).

### Run 2 — fit COMPLETES; readout LR conditional beats 0089; two follow-on findings

The emission fix worked: the **gated_pc fit ran to completion** (K=444, 5223 s), and
`|w_CK|max=272.5` settled smoothly (no blowup — the ridge held at this scale).

**Deliverable (readout LR, `pc_topics_lr`) — excellent, and better than 0089:**
- macro AUC 0.7374 over 241 nodes; conditional `P(child|parent)` cond_AUC **0.73–0.78
  across depths 1–8** (0089's flat 41-anchor best was ~0.70), pooled ECE 0.0027,
  held-out isotonic raw 0.0030 → **calibrated 0.0015** (VOI-ready). Per-node
  reliability mean ECE 0.047 (max 0.42 on a tiny degenerate retinal-artery node). This
  is the deep-tree-localizes-well prediction confirmed: within-branch conditional
  discrimination is strong at every depth.

**Finding 1 — the co-fit (unified) head is near-chance conditionally (cond_AUC ~0.52,
ECE 0.12), a real regression vs 0089's ~0.69.** Not a bug (the emission fix is exact;
the solve inputs are byte-identical): it's the LOCALIZATION. A node's co-fit head reads
only its local support (block + ancestors + siblings, ≤24 topics); in the DEEP Mondo
tree the child-vs-sibling discriminative signal often lives in OTHER (distant) topics,
which the head can't see but the readout LR (all K features) can — hence readout 0.74 vs
head 0.52 on the SAME topics. So the localization that lets the unified head SCALE also
guts its conditional quality at depth. **Practical conclusion: at whole-Mondo scale the
two-stage readout LR is the deliverable; the unified co-fit head is not viable once
localized (and it can't be un-localized — that's the 850 GB wall).** This resolves the
unified-vs-two-stage tension (insight 0071) in favor of two-stage AT SCALE.

**Finding 2 — the run then died `exit 143` (SIGTERM) in the UNSUP twin arm**, right
after its 100th iter, before printing its readout — i.e. in the twin's full θ-collect +
C-logistic readout, on top of the gated arm's retained arrays, past the 8 GB driver
heap. The gated arm (the deliverable) had already fully completed. Fix for Step A: drop
the weight_y=0 twin (`skip_unsup_gated: true`) — it's an A/B we don't need for mechanics
validation and it's the sole thing that OOM'd; the rerun completes clean and saves the
model. **Next scaling wall for Step B (flagged, not yet fixed):** the readout itself
collects the full `N×K` θ matrix to the driver to fit the per-node logistics — at
K≈3,800 over the whole population that is multi-GB per arm and will not fit an 8 GB
driver. Step B needs a subsampled or distributed readout before the whole-tree run.

### Run 3 (planned) — diagnose WHY localization killed the co-fit head

Two diagnostics added to the driver, both on the gated arm's collected θ:
- **Oracle localized readout** — the best-possible per-node logistic fit on EXACTLY
  the co-fit head's support (`allowed_with_siblings`), same hypothesis class as the
  head but fit optimally. The A-vs-B decider, printed as
  `[driver] A-vs-B cond_AUC: full-K readout=… oracle-localized=… co-fit head=…`:
  - oracle ≈ full-K ⇒ the signal IS in the support; the co-fit head is merely
    UNDER-FIT (recoverable — tune the head fit; the unified model lives);
  - oracle ≈ co-fit head ⇒ the signal is OUTSIDE the local support (localization
    fundamentally lossy — widen support or concede two-stage).
- **Shaping ablation** — the weight_y=0 twin readout restored (memory-safe via
  `readout_sample_frac`): does supervision improve the representation, or does the
  gate structure carry it? (0089 was ~neutral.)

### Run 3 — the A-vs-B verdict: the co-fit head is UNDER-FIT, not locality-limited

The memory fix held (readout did not OOM; the `exit 143` was a *worker* node
preempted mid-fit, which Spark recovered from). Mean conditional AUC across depths:

| head | mean cond_AUC | mean ECE |
|---|---|---|
| full-K readout LR | **0.739** | 0.061 |
| oracle-localized (support-only LR) | **0.676** | 0.031 |
| co-fit head (as trained) | **~0.52** | ~0.16 |

Decomposing the co-fit head's deficit:
- **co-fit 0.52 → oracle 0.676 (+0.156)** — the big piece — is **pure under-fitting**:
  a logistic on the co-fit head's *exact support topics*, fit optimally, reaches
  0.676. The signal IS in the support; the co-fit Newton head isn't extracting it.
- **oracle 0.676 → full-K 0.739 (+0.063)** — the small piece — is the only genuine
  locality tax (signal outside the support).

So localization costs ~0.06; the other ~0.16 is the head under-fitting what's in
front of it. **Run-2's "localization is fundamentally lossy" conclusion was wrong on
the mechanism — the unified head is recoverable.** Corroborating: the oracle-localized
is *better calibrated* than full-K (ECE 0.031 vs 0.061) and clinically sane
per-parent, so the local support carries most of the real signal; and `|w_CK|=273`
is the tell — the head over-committed to the marginal-prevalence direction and
saturated, losing the within-sibling contrast.

**Shaping ablation (the other half): supervision is NEUTRAL on the representation.**
`gated_pc` (supervised) vs `unsup_gated` (weight_y=0) readout, same gated topics:
- pc_topics_lr AUC 0.7320 vs 0.7289 (Δ+0.003); AP 0.516 vs 0.511 (Δ+0.006);
  cond AUC 0.7320 vs 0.7289 (Δ+0.003); multiclass top-1 0.827 vs 0.829 (Δ−0.002).

So PC supervision does **not** shape the topics — the **DAG gate** (structural per-node
topic blocks) produces the whole conditional representation, with or without the head.
Confirms the marginal-PC-benefit pattern on information-limited EHR (insights
0064/0066/0089), now on the Mondo backbone. The per-node domain λ-mass is likewise
near-identical supervised vs unsup (specialization is a gated-multi-domain property,
insight 0078, not a PC effect).

**Combined verdict.** The co-fit head earns nothing here: it neither predicts
(under-fit, 0.52 vs 0.666 achievable on its own support) nor improves the
representation (neutral). The good conditional numbers (readout 0.73) come from the
**unsupervised gated topics**. This reframes the "unified model": keep the single
calibrated artifact, but as a **POST-FIT head on the gated topics**, not a co-fit head
— one saved model (gated topics + head), calibrated conditional output, single
inference path, WITHOUT paying for co-training that doesn't pay off. `weight_y`
becomes optional. (PC may still help on richer data; on this data it's neutral.)

### Run 4 (planned) — HEAD-FORMULATION LADDER: isolate WHICH difference costs the head

The oracle diagnostic was CONFOUNDED — my sklearn oracle uses `fit_intercept=True` +
`StandardScaler`, while the co-fit head has neither, plus a different (relative) ridge
and is under-converged. So "oracle 0.666 vs co-fit 0.52" bundles four differences.
The ladder converges a localized head on the frozen gated θ, stepping the co-fit
head's EXACT formulation toward the sklearn oracle one factor at a time, printing mean
cond_AUC for each (`[driver] HEAD-FORMULATION LADDER`):

1. co-fit head (as trained) — ~0.52
2. engine Newton [relative ridge, no intercept], **CONVERGED** — isolates convergence
3. + FIXED ridge (`--head-fixed-ridge`) — isolates the ridge that VANISHES at
   separation (the `|w|=273` blowup: relative ridge ∝ trace(H) → 0 as p→0/1)
4. + unpenalized INTERCEPT — isolates the missing bias (θ sums to 1, so a rare node's
   marginal is otherwise fit through ridge-penalized topic weights, saturating)
5. sklearn [no-intercept, standardized] — isolates the intercept in the well-reg regime
6. sklearn [intercept, NOT standardized] — isolates feature standardization
7. sklearn oracle [intercept, standardized] — 0.666
8. full-K readout — 0.732

Each step's delta names its factor's cost. The winning factor(s) become the engine
head fix (e.g. add an unpenalized per-node intercept and/or fixed L2), which — if the
head's saturation was also corrupting its topic-shaping gradient — could flip the PC
shaping result from neutral to positive. `|w|max` for the engine variants is printed
too (the fixed ridge should tame the 273 blowup). Hypothesis (mechanism for the
small-model→large-model regression): the relative ridge + missing intercept are cheap
at 41 shallow anchors but bite at 436 deep/rare nodes.

### Run 4 RESULT — under-CONVERGENCE dominates (+0.087), then INTERCEPT (+0.060)

| step (one factor added per row) | cond_AUC | Δ | \|w\|max |
|---|---|---|---|
| co-fit head (as trained) | 0.523 | — | 273 |
| **+ CONVERGE** (rel-ridge, no-icpt) | 0.610 | **+0.087** | 71 |
| + fixed ridge | 0.602 | −0.008 | 19 |
| **+ INTERCEPT** | 0.662 | **+0.060** | 12 |
| sklearn [no-icpt, standardized] | 0.677 | +0.015 (standardize) | — |
| sklearn oracle [icpt, standardized] | 0.680 | — | — |
| full-K readout | 0.737 | +0.057 (locality) | — |

**Verdict (I was wrong that convergence wouldn't matter):** the co-fit head is badly
UNDER-CONVERGED — one damped (`head_lr=0.3`) Newton step/iter against a moving θ never
settles, sitting at `|w|=273`; a converged localized head reaches 0.610 at `|w|=71`.
Convergence is the biggest lever (+0.087), the unpenalized INTERCEPT second (+0.060);
the ridge TYPE barely matters (fixed alone −0.008, though it tames `|w|`). Engine-
fixable ceiling (converge + intercept, localized) ≈ 0.66–0.68 = the oracle; the last
0.057 to full-K is locality (widen support, optional).

**The deeper finding — supervision is a NO-OP on the topics.** `gated_pc` vs
`unsup_gated` this run: AUC 0.7365 vs 0.7365 (Δ−0.0000), **bit-identical ELBO
(−25944399.7250 both arms)**, identical per-node AUC and domain λ-mass. weight_y=50
produced the EXACT same topic model as weight_y=0. Mechanism (unified with the head):
the under-converged, intercept-less head saturates (`|w|=273`) → its `∂loss/∂θ`
vanishes → the topic correction `ρ·wy·∂loss/∂θ ≈ 0` (capped at `topic_trust·λ_unsup`
but starved of gradient) → topics don't move → PC neutral BY CONSTRUCTION. **So fixing
the head gates whether PC works at all**, not just the head's own AUC. Next: exp 0091
(`head_lr=1.0`, full Newton) tests convergence + the new `corr_relΔλ` diagnostic (does
the un-saturated head finally move the topics?); then the engine intercept.
