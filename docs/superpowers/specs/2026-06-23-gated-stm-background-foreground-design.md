# Gated STM: Background / Foreground Topic Blocks — Design

**Date:** 2026-06-23
**Status:** Approved design, pending implementation plan
**Branch context:** extends the unmerged `stm` branch (canonical prevalence-only STM)

## Goal

Add an opt-in "topic-block gating" mode to the canonical `OnlineSTM` so that a
**rare group of documents can be given its own foreground topics — with their
own vocabulary content — while a shared background block borrows the statistical
strength of the full (much larger) corpus.** This serves the rare-disease /
rare-cohort modeling problem: use the large cohort to model the common
structure, while isolating phenotype profiles specific to the minority group.

The canonical no-gating path must remain mathematically identical to today's STM.

## Motivation

Prevalence-only STM gives a rare group **prevalence fidelity** (its documents
get a distinct topic-prevalence profile via Γ) but **not content fidelity** (no
group-specific vocabulary clusters), because the topic-word distribution β is
shared and majority-led. This is documented empirically in
[docs/insights/0026](../../insights/0026-stm-prevalence-gives-prevalence-not-content-fidelity.md):
the prevalence covariate "resizes bubbles, doesn't move them," so if pooled β
never surfaces a minority phenotype, the covariate has nothing minority-specific
to up-weight.

The desired behavior is a **background / foreground decomposition**: shared
background topics estimated with full-cohort power, plus a small set of
group-specific foreground topics estimated on the residual. The hierarchical
(tree-HDP) route to this is sketched in
[docs/architecture/TOPIC_STATE_MODELING.md](../../architecture/TOPIC_STATE_MODELING.md#L1004-L1069),
but (a) HDP under-delivered in practice on this data (insights
[0001](../../insights/0001-hdp-gamma-collapse-at-low-gamma0.md),
[0002](../../insights/0002-hdp-catchall-hoarding-at-last-stick.md),
[0017](../../insights/0017-hdp-gamma-sensitivity-is-prior-dominance.md): prior-dominated,
catch-all hoarding, γ-collapse), and (b) standard HDP shares topic *atoms*
globally, so it has the same "resize, don't move" ceiling as prevalence STM.

STM hosts the decomposition more cleanly: **the prevalence model is the gating
mechanism, and shared-β SVI gives the minority its own content as a side effect
of gating.** Fixed K is an advantage here — insight
[0019](../../insights/0019-lda-large-k-with-full-convergence-gracefully-unused-slots.md)
shows STM/LDA at large K with full convergence leaves excess capacity gracefully
unused, so an oversized foreground block recovers most of the "let the data
decide how many" benefit of nonparametrics without HDP's failure modes.

## Core idea

Partition the K topics into a **background block** (every document may express
it) and one **foreground block per group** (only documents in that group may
express it). Gating is enforced by **hard masking** in per-document inference:
a document's allowed-topic set is `background ∪ (foreground blocks of its
groups)`; its η is optimized only over the allowed set, and θ is exactly 0 on
disallowed topics.

Content fidelity then falls out of ordinary SVI: a majority document puts θ ≡ 0
on every foreground topic, so it contributes **zero sufficient statistics** to
those foreground β rows. The foreground β is therefore estimated entirely from
group documents — it learns the group's distinctive vocabulary, uncontaminated
by majority token mass. Within a group document, softmax competition routes
common tokens to the strong shared background topics and leaves only the
residual for foreground. Background borrows full-cohort strength; foreground
explains the group's residual.

The teaching one-liner: **gating changes which documents train which topics,
nothing else.** β rows, Γ columns, and Σ entries for a foreground block simply
see a sub-corpus; the estimators are the same estimators on less data.

## Approach decision

Three approaches were considered; this design commits to **A**. The decision and
its alternatives are recorded in a companion ADR (next number in
[docs/decisions/](../../decisions/)).

- **A — Joint fit with hard masking (chosen).** Per-document allowed-topic set;
  block-aware M-step. Cleanest isolation (foreground content provably from group
  docs only), single joint model (background still adapts to the whole corpus),
  interpretable. Cost: per-doc inference and M-step become block-aware.
- **B — Joint fit with a soft prevalence prior (documented alternative).** No
  mask; an informative Gaussian prior on Γ drives majority foreground prevalence
  toward 0, so gating *emerges*. Smallest engine change, and its continuous /
  fully-joint nature may have its own benefits worth revisiting. Cost: soft
  isolation (mild foreground contamination), a prior-strength tuning knob.
- **C — Two-pass freeze-background.** Fit background on the full corpus, freeze
  β, then fit foreground on the group's residual. Simplest math; not a joint
  fit (background can't adapt), two stages/checkpoints.

## Terminology (collision discipline)

"Cohort" is already overloaded in the pipeline and must not gain a third sense:
- `corpus_manifest["cohort"]` — the corpus-level OMOP filter (e.g.
  `cancer_or_dementia`).
- `source_cohort` — the per-document cohort-of-origin within a combined corpus
  (e.g. `cancer` vs `dementia`).

The gating mechanism uses **"group"** (consistent with the tree-HDP design doc,
which already uses "group" for this concept) and **"background / foreground
block"** for the topic partition. The gating variable is the generic
`group_var`, which in the first experiment binds to the existing `source_cohort`
column — but the abstraction and all new identifiers stay "group." Foreground
topics are labeled in the dashboard by the group *value* (e.g. "dementia"),
exactly as that value already appears as a covariate level.

Note a separate pre-existing overload: the LLM labeler
([scripts/label_phenotypes.py](../../../scripts/label_phenotypes.py#L94)) uses
`background` as a *quality category* (a catch-all/baseline topic). That is a
quality judgment, distinct from the gating **background block** (a structural
partition). A topic in the gating background block can still be labeled
`phenotype`, `anchor`, or `dead`. This disambiguation is part of the deferred
labeler work (see Out of scope).

## Config & data model

**`TopicBlockPartition`** (new value object, engine-side in
`spark_vi.models.topic`): the single source of truth for the gating layout.
- `group_var`: the document column naming a doc's group (binds to
  `source_cohort` in the first experiment; already tagged on every document and
  recoverable from `doc_id`, so no new corpus plumbing to produce it).
- `background_k`: size of the background block.
- `foreground`: ordered mapping `group-label -> block size`, e.g.
  `{cancer: 10, dementia: 10}`.

It derives explicit, disjoint topic-index sets — background = `[0,
background_k)`, then each group's block laid out contiguously after it — and
asserts they cover exactly `[0, K)` (so `K == background_k + Σ foreground
sizes`; mismatch is a hard config error). Contiguous layout is a readability
convention; the engine only consumes the index sets.

**Document side.** [`STMDocument`](../../../spark-vi/spark_vi/models/topic/types.py#L49)
gains `groups: frozenset[str]` (default empty). Allowed-topic set =
`background ∪ ⋃_{g ∈ doc.groups} foreground[g]`. A document with no groups sees
background only; the set form means a patient in two rare groups opens both
blocks. (Single-string group column → singleton set; an array column → true
multi-group, supported but not exercised in the first experiment.)

**Activation / canonical preservation.** The partition lives on `OnlineSTM` as
an optional attribute (default `None`). When `None`: allowed set is all K for
every document, the mask is identity, per-group XᵀX collapses to the single
canonical XᵀX, every per-topic doc count equals D — the code path is
mathematically identical to today's STM. This is a hard requirement, pinned by a
byte-identical regression test on a fixed seed.

**Config flow.** Experiment YAML gains a `topic_blocks:` block (`group_var`,
`background_k`, `foreground:` map). The driver builds the partition, threads the
`group_var` column into the STM input DataFrame (reusing `source_cohort`), and
the shim attaches `groups` onto each `STMDocument`. The partition is persisted
into `corpus_manifest` as `topic_block_spec` so eval, resume, and the dashboard
can all reconstruct it.

**Guards.**
- `group_var` must not also appear as a term in the prevalence formula. Reason:
  a foreground block's Γ is solved over its group's documents only, where the
  group indicator is a constant column — collinear with the intercept,
  rank-deficient, only ridge-rescued, and uninterpretable. The gating already
  encodes group membership structurally. Validated at construction; hard error.
  (Group-shifted *background* prevalence via a separate covariate is a legitimate
  future option, noted in the ADR.)
- A foreground block with **zero** matching documents is a config error (raise,
  not silently ridge-rescue). A block with few documents emits a min-docs
  warning.

## Engine math

Two changes — a masked E-step and a block-aware M-step — both collapsing exactly
to today's code when the partition is `None`. (Notation: plain text; Greek as
Unicode. Per-document covariate vector x, prevalence regression Γ of shape
(P, K), diagonal residual covariance Σ of shape (K,).)

### Masked per-doc inference (E-step)

For document d with allowed set A_d ⊆ {0..K−1}, restrict every per-document
quantity to A_d before the L-BFGS in
[`_stm_doc_inference`](../../../spark-vi/spark_vi/models/topic/stm.py#L170):
- topic-word rows expElogβ[A_d] (shape |A_d| × n_unique),
- prior mean (Γᵀx)[A_d], prior variance Σ[A_d],
- η optimized as an |A_d|-vector; θ = softmax over A_d only, exactly 0 elsewhere.

The negative log joint, gradient, and Hessian are structurally unchanged — they
run on the sub-index. This is exact, not an approximation: a disallowed topic has
θ = 0 and contributes nothing to the data likelihood, so dropping it from the
optimization changes nothing. ν_d returns |A_d| × |A_d|.

### Block-aware M-step

Different topics now have different training sets, so the global updates respect
that (in [`local_update`](../../../spark-vi/spark_vi/models/topic/stm.py#L289) /
[`update_global`](../../../spark-vi/spark_vi/models/topic/stm.py#L371)):

- **λ / β (automatic).** Suff-stats `phi * counts` scatter into rows A_d of the
  K×V accumulator; disallowed rows receive 0. A foreground β row accumulates
  only from documents that allow it. No special-casing — the mask does it.

- **Γ (per-block normal equations).** Each topic's regression of η on x uses only
  the documents where that topic is allowed:
  - Background columns: Γ_bg = (XᵀX_all + ridge)⁻¹ XᵀMu_bg, where XᵀX_all =
    Σ_d x_d x_dᵀ over all documents.
  - Group g's foreground columns: Γ_fg(g) = (XᵀX_g + ridge)⁻¹ XᵀMu_fg(g), where
    XᵀX_g = Σ_{d : g ∈ groups_d} x_d x_dᵀ over group-g documents only.

  `local_update` accumulates XᵀX_all plus a fixed-shape (G, P, P) per-group XᵀX
  (a document adds x_d x_dᵀ to XᵀX_all and to XᵀX_g for each of its groups), and
  XᵀMu's columns accumulate over each topic's allowed documents. `update_global`
  solves background columns against XᵀX_all and each group's block against its
  XᵀX_g, then assembles the (P, K) Γ — **same shape as today**. Added memory is
  O((1+G)·P²), negligible (P is tens).

- **Σ (per-topic doc counts).** residual_diag accumulates (η̂ − Γx)² + diag(ν_d)
  only over allowed (topic, document) pairs, alongside a per-topic allowed-doc
  count n_k. Then σ²_k = residual_diag_k / n_k — background topics divide by D,
  a group-g foreground topic divides by D_g. (Today's single `n_docs` divisor is
  the n_k = D special case.)

### ELBO

doc_loglik and the η-KL are computed on the A_d sub-space (the Gaussian KL uses
Σ[A_d]); the global β-KL still sums over all K rows (every β_k is a valid
Dirichlet). The sum is a valid monotone objective for the gated model —
comparable across iterations within a fit, though not across different
partitions.

### Canonical collapse (the guarantee)

With partition `None`: A_d = {0..K−1} for all d, the mask is identity, no XᵀX_g
exists, XᵀX_all is the current XᵀX, every n_k = D. Every line reduces to the
current engine, asserted by a byte-identical regression test.

## Pipeline impact (layer by layer)

Shapes barely change: Γ stays **(P, K)** (foreground topics are columns
estimated on a subset, not a new tensor dimension), covariates stay per-`(person,
source_cohort)`, and persisted params keep identical shapes. The one genuinely
non-cosmetic change is the dashboard's prevalence math.

### 1. spark-vi engine + types (the substantive work)

- [`types.py`](../../../spark-vi/spark_vi/models/topic/types.py): `STMDocument`
  gains `groups: frozenset[str]` (default empty).
- New `TopicBlockPartition` value object (engine-side): resolves group +
  background_k + foreground map into disjoint index sets, validates coverage.
- [`stm.py`](../../../spark-vi/spark_vi/models/topic/stm.py): `OnlineSTM.__init__`
  gains `topic_blocks=None`. `local_update` does masked per-doc inference and
  accumulates the new stats (`XtX_all`, `(G, P, P)` per-group XᵀX, per-topic
  `n_docs_per_topic`). `update_global` solves Γ block-wise and Σ per-topic.
  Persisted global_params (`lambda`, `Gamma`, `Sigma`, `eta`) keep identical
  shapes — `VIResult` / `save_result` untouched.

### 2. mllib shim (StreamingSTM / STMModel)

- [`StreamingSTM`](../../../spark-vi/spark_vi/mllib/topic/stm.py#L58) gains
  `topic_blocks` (the partition) and `doc_group_col` (the column naming each
  doc's group). `fit` threads the group column through the RDD;
  `_vector_to_stm_document` attaches `groups` (single-string column → singleton
  set; array column → multi-group).
- [`STMModel.save/load`](../../../spark-vi/spark_vi/mllib/topic/stm.py#L287)
  roundtrip the partition. It is JSON-serializable (`group_var`, `background_k`,
  `foreground` map), so it rides in `metadata` next to the existing
  `model_spec.pkl` / `covariate_names.json` sidecars — no new sidecar file.

### 3. charmpheno driver + covariates + eval

- [`stm_bigquery_cloud.py`](../../../analysis/cloud/stm_bigquery_cloud.py):
  builds the partition from YAML, materializes `source_cohort` from `doc_id` as
  the `doc_group_col`, passes both to `StreamingSTM`, and writes
  `topic_block_spec` into `corpus_manifest`. `source_cohort` is a charmpheno
  domain label fed into the engine's domain-agnostic group slot — spark-vi and
  the shim never name a cohort.
- **`source_cohort`'s two roles are decoupled** (the original `composite` flag
  conflated them): `source_cohort_is_covariate` (`in cat_cols`) drives
  per-(person, cohort) covariate keying; `need_source_cohort` (covariate **or**
  gating) drives materialization from `doc_id`. Under gating `source_cohort` is
  the group label but NOT a covariate, so covariates key **per-person** — exact
  because age (`2025 - year_of_birth`) and sex are static per person. Gating
  requires the combined-cohort `patient_cohort` doc-spec (which encodes
  `source_cohort` in `doc_id`); a clear error otherwise. No `doc_block_id` key.
- Guards: `group_var` rejected if it also appears in the formula;
  `run_experiment.py`'s resume-compat check adds `topic_block_spec` (a changed
  partition is a different model, cannot resume).
- [`eval_coherence_cloud.py`](../../../analysis/cloud/eval_coherence_cloud.py):
  still scores all K λ-rows, reads `topic_block_spec` to label which topics are
  foreground, and scores each foreground block against its **group sub-corpus**
  (see Foreground-aware eval).

### 4. Dashboard (the part to get right)

- `build_dashboard_cloud.py` + `_stm_corpus_prevalence`: today it computes
  (1/D) Σ_d softmax(Γᵀx_d) over all K. Under gating that is wrong — a cancer
  document must have θ ≡ 0 on dementia foreground topics. The corpus-mean
  prevalence must use the **masked softmax per row** (allowed set = background ∪
  foreground[that row's `source_cohort`]), so the prevalence helper now needs
  each row's group label alongside its covariate vector. Real server-side work,
  not styling.
- `adapt_stm` / `covariate_schema.json`: emit a per-topic block label
  (`topic_id -> "background" | group-name`) so the client knows which topics
  gate. Γ and `CovariateEffects` keep their (P, K) shape — masking is applied at
  prevalence time, never to Γ.
- Front-end `covariate.ts::covariatePrevalence`: becomes masked — when the user
  selects `source_cohort = cancer`, dementia foreground topics are zeroed and the
  softmax is taken over the allowed set.
- Atlas UX (the correct interpretation): switching the group control makes the
  other group's foreground bubbles **vanish** (not resize) — a dementia-specific
  phenotype simply does not exist for a cancer patient — while background bubbles
  resize as before. "Bubbles resize; foreground bubbles appear/vanish with the
  group."

## Identifiability guardrails, diagnostics, eval

- **Primary failure mode — foreground stealing common content.** Hard masking
  guarantees a foreground topic only *trains* on its group's documents, not that
  it is *distinctive*. If `background_k` is too small, background cannot cover
  the common space and a group's ordinary comorbidity coding spills into its
  foreground block. Mitigations: (a) guidance to size `background_k` generously
  (background sees the whole corpus; extra slots go gracefully quiet per 0019);
  (b) a diagnostic flagging a foreground topic whose top terms are
  common-everywhere codes (the undersized-background signature).
- **Degenerate groups.** Tiny n_k makes σ²_k unstable (SIGMA_FLOOR catches it)
  and XᵀX_g ill-conditioned (ridge catches it, but the Γ block is meaningless);
  emit a per-group min-docs warning. A zero-document foreground block is a config
  error.
- **Diagnostics.** `iteration_summary` / `iteration_diagnostics` gain per-block
  readouts: foreground-doc count per group, Σλ mass per block, |Γ| per block — to
  watch whether foreground is populated or sitting as graceful-unused slots.
- **Foreground-aware eval.** A foreground topic for group g is scored on the
  group-g sub-corpus (the documents that could express it), not the full corpus —
  both because that is the reference those topics saw, and because scoring a rare
  phenotype against majority documents that can never contain it triggers the
  zero-pair penalty of insight
  [0007](../../insights/0007-npmi-zero-pair-floor-penalizes-rare-phenotypes.md).
  Background topics are scored on the full corpus. The report labels every topic
  by block.

## Testing strategy (TDD throughout)

Engine unit tests (synthetic, deterministic) carry the confidence:
1. **Canonical collapse** — partition `None` produces byte-identical global
   params to today's `OnlineSTM` on a fixed seed (the preservation pin).
2. **Mask correctness** — a document with a restricted allowed set gets θ exactly
   0 on disallowed topics; η optimized only over allowed coords.
3. **Zero foreground contribution** — in a corpus where majority documents are
   never foreground-allowed, foreground β rows accumulate suff-stats only from
   group documents (majority contribution exactly 0).
4. **Block-aware M-step** — Σ divides each topic by the right n_k (D vs D_g) and
   Γ blocks solve against the right document subsets, on a synthetic case with a
   known closed-form answer.
5. **Recovery test (the "it works" test)** — a synthetic generative corpus:
   shared background topics plus a group with a planted distinctive phenotype
   absent from the majority. Assert a foreground topic recovers that vocabulary
   and the majority contributes ≈ 0 to it.

Shim / driver / manifest: `STMDocument.groups` threading; partition save/load
roundtrip; `StreamingSTM(topic_blocks=…)` fits and returns a gated model;
`topic_block_spec` written to and reproduced from the manifest; the
group-var-in-formula guard raises; the resume guard rejects a changed partition.

Dashboard: server-side masked corpus prevalence zeroes out-of-group foreground
topics; `covariate_schema.json` carries per-topic block labels; the front-end
`covariatePrevalence` masks by selected group (a TS unit test).

Real-cohort validation (an experiment, not a unit test): `cancer_or_dementia`
with a background block plus cancer/dementia foreground blocks. Success
criterion: the dementia foreground surfaces a dementia-distinctive phenotype that
prevalence-only STM (insight 0026) could not.

## Out of scope (deferred, with sequencing rationale)

- **LLM topic-labeler guidance.** The rubric in
  [scripts/label_phenotypes.py](../../../scripts/label_phenotypes.py) will need
  updating for the gated model type — at minimum: (a) disambiguate the gating
  **background block** from the labeler's `background` quality category; (b) make
  the rubric aware that foreground topics are rare-by-construction and
  group-scoped, so a sparse-but-real foreground phenotype is not mislabeled
  `dead`. This is deferred **until after initial experiments**, deliberately:
  insight [0024](../../insights/0024-labeler-classifier-rules-have-regime-dependent-blind-spots.md)
  shows labeler rubrics have regime-dependent blind spots, so guidance written
  before we have seen how gated results actually read would be guessing at
  pitfalls we have not met. Run experiments, learn the interpretation traps, then
  update the rubric from evidence.
- **Content covariates / SAGE.** Out of scope. Foreground content fidelity here
  comes from gating (who may express a topic), not from covariate-dependent β.
  Content covariates remain the separately-tracked canonical STM gap.
- **Soft prevalence prior (approach B) and group-shifted background prevalence.**
  Documented in the companion ADR as alternatives that may be revisited.

## Companion ADR

A new ADR (next number in [docs/decisions/](../../decisions/)) records choosing
hard masking (A) over the soft prevalence prior (B) and two-pass (C), the
identifiability reasoning, and the deferred alternatives (B's continuous joint
fit; group-shifted background prevalence as a separate covariate).

## Success criteria

1. Canonical no-gating STM is byte-identical to today (regression pin).
2. With gating on, foreground β is provably estimated from group documents only.
3. On a synthetic recovery corpus, a planted minority phenotype is recovered as a
   foreground topic with ≈ 0 majority contribution.
4. On the real `cancer_or_dementia` corpus, the dementia foreground surfaces a
   dementia-distinctive phenotype absent from prevalence-only STM results.
5. The dashboard renders foreground bubbles that appear/vanish with the selected
   group, with correct masked prevalence on both server and client.
