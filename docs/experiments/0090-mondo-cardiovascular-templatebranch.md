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
skip_unsup_gated: true
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
