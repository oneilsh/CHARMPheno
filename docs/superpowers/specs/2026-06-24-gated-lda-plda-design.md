# Gated LDA (Partially Labeled Dirichlet Allocation) — Implementation Sketch

**Date:** 2026-06-24
**Status:** Design sketch (proposed; not yet planned/built)
**Motivation:** insight [0028](../../insights/0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md)

## Goal

Add an opt-in background/foreground topic-block gating mode to `OnlineLDA`,
exactly mirroring the gating already shipped for `OnlineSTM`, so each document
may express a shared **background** block plus **only its own group's**
foreground block (hard topic-masking by a group variable). This is
**Partially Labeled Dirichlet Allocation** (Ramage, Manning & Dumais, KDD 2011).
Per 0028, the Dirichlet document-topic prior — not gating — is what recovers
rare minority phenotypes, so PLDA = gating + the LDA engine we already have.

**Gating is the general case** (same architecture decision as STM):
`background_k = K`, no foreground → byte-identical to plain LDA;
`background + per-group blocks` → PLDA; `0 background + per-group blocks` →
Labeled-LDA. The non-gated path must be pinned identical to current LDA.

## What we reuse unchanged

- **`TopicBlockPartition` + `allowed_indices`** (spark-vi/spark_vi/models/topic/partition.py) — the whole partition abstraction is model-agnostic.
- **Block-aware NPMI eval** — [eval_coherence_cloud.py](../../../analysis/cloud/eval_coherence_cloud.py) already reads `corpus['topic_block_spec']` and scores each block against its own reference for *any* `--model-class`. Gated LDA eval works with zero eval changes.
- **`gating.json` export + block-level k-anon** — [charmpheno/charmpheno/export/gating.py](../../../charmpheno/charmpheno/export/gating.py) takes only `partition` + `group_counts`; fully model-agnostic.
- **Driver `source_cohort` materialization** — the STM driver's pattern (`source_cohort` = `split(doc_id, ":")[0]` from the `patient_cohort` doc-spec) ports directly.
- **Per-iter top-terms fit logger** — already block-aware (we added `topic_labels` prefixing for STM; the LDA logger factory needs the same one-line addition).

## Changes, file by file

### 1. Data model — `BOWDocument` gets an optional group label
[types.py](../../../spark-vi/spark_vi/models/topic/types.py): add
`groups: frozenset[str] = frozenset()` to `BOWDocument` (mirror `STMDocument`),
and `from_spark_row(..., group_col=None)` to populate it. Default empty = the
non-gated path. `frozen=True, slots=True` allows a defaulted field.

### 2. Engine — `OnlineLDA` masking (the core change)
[lda.py](../../../spark-vi/spark_vi/models/topic/lda.py):
- Constructor: add `topic_blocks=None`; add `_effective_partition()` (returns
  an implicit all-background partition when None — copy STM's helper verbatim).
- `_cavi_doc_inference`: add an `allowed` parameter. Restrict the γ/φ CAVI loop
  to the allowed topic rows (`expElogbeta[allowed]`, `alpha[allowed]`,
  `gamma_init[allowed]`), and return a full-K `sstats_row` that is **zero on
  disallowed rows**. This is the exact analogue of `_stm_doc_inference(...,
  allowed=)`. A document's θ becomes a Dirichlet over its allowed sub-simplex;
  disallowed topics get no responsibility and contribute no β suff-stats.
- `local_update`: `allowed = part.allowed_indices(doc.groups)` per doc; thread
  it into `_cavi_doc_inference`. The per-doc content-hash seed is unchanged
  (masking doesn't affect determinism).

### 3. Engine subtlety — learned asymmetric-α under masking is deferred to v2
Originally scoped here as a simple per-topic-`D` change; implementing the plan
showed it is **deeper than that**. The closed form in `alpha_newton_step`
assumes one shared K-simplex (a single `D` and a single ψ(Σα)); under PLDA
masking each doc uses only its allowed set `A_d`, so the normalizer
ψ(Σ_{j∈A_d} α_j) **varies per document** and the single Sherman-Morrison step no
longer applies. **v1 therefore ships with fixed α** (`optimize_alpha=True` +
foreground blocks raises), which is correct and sufficient — α < 1 still supplies
the document-topic sparsity that recovers rare phenotypes (0028). The v2
per-group-simplex Newton derivation lives in the plan's Appendix
([2026-06-24-gated-lda-plda-model.md](../plans/2026-06-24-gated-lda-plda-model.md)).

### 4. MLlib shim — `OnlineLDAEstimator`
[mllib/topic/lda.py](../../../spark-vi/spark_vi/mllib/topic/lda.py): add
`topic_blocks` + `doc_group_col` params (mirror `StreamingSTM`); build gated
`BOWDocument`s from `doc_group_col`; pass `topic_blocks` to `OnlineLDA`. The
`setOnIteration` plumbing is untouched.

### 5. Cloud driver — `lda_bigquery_cloud.py`
Add `--background-k` / `--foreground` / `--group-var` (copy the STM driver's
argparse + `build_topic_block_partition`), materialize `source_cohort` from the
`patient_cohort` doc-spec (copy the STM driver's `need_source_cohort` block),
and write `topic_block_spec` into checkpoint metadata so eval + dashboard read
it. Block-aware fit logging: pass `topic_labels` into the existing LDA
`_make_topic_evolution_logger` (the STM one already does this).

### 6. Eval — nothing
Block-aware NPMI already dispatches on `topic_block_spec`, model-agnostic. Done.

### 7. Dashboard — one model-specific helper
`gating.json` + k-anon are free. The only LDA-specific piece is **masked
corpus-mean prevalence** (the appear/vanish-by-group bubble sizes): STM derives
it from Γ; LDA's analogue is the corpus-mean θ restricted to each group's
allowed topics — derivable from the fitted model (normalized per-group aggregate
γ, or λ-row-mass within block). Add `adapt_lda`-side gated prevalence mirroring
`adapt_stm`'s masked-prevalence helper.

### 8. Experiment runner — `build_lda_args`
Add the gating flags (mirror `build_stm_args`'s `gating` block). Defaults:
gating off; `optimize_doc_concentration` on.

### 9. Tests
- **Recovery (easy + hard):** port `test_gated_stm_recovers_planted_minority_phenotype`
  to LDA. Crucially, also add the *hard* regime the STM test never covered
  (weak, rare-within-group phenotype against a shared background) — gated LDA
  should pass where gated STM degraded. Our local repro harness is the seed.
- **Non-gated byte-identity:** `background_k=K`, no foreground ⇒ output identical
  to current `OnlineLDA` (the STM gating has the analogous all-background pin).
- **Masking integrity:** background topics may emit on all docs; a foreground
  topic's β stays ~0 on tokens that appear only in other groups' docs.

## Effort & sequencing

Small-to-moderate; almost entirely mirrors shipped STM gating. Suggested order:
1. Engine (`types.py` group field → `_cavi_doc_inference` masking → per-topic
   α aggregation) with unit tests (recovery easy+hard, non-gated identity).
2. Shim + driver + runner flag (reuse STM patterns) → first cloud run
   (experiment: gated LDA on `cancer_or_dementia`, 30 bg / 10 cancer / 10
   dementia, asymmetric α on — the direct rematch of 0004/0005).
3. Dashboard masked-prevalence helper for LDA.

## Open questions / next increments

- **Seeded β priors (MixEHR-Guided/-Nest nod):** generalize the scalar `eta`
  (β-Dirichlet prior) to a per-topic `(K, V)` pseudocount matrix and seed
  foreground topics toward known phenotype codes (PheCodes/CCS). Composes with
  PLDA; natural second increment. (0028 implications #1–2.)
- **Asymmetric-α benefit is corpus-dependent** (Wallach 2009 — stronger for
  short-text/small-vocab). OMOP condition_era docs are short bags of codes, so
  favorable; verify on the first gated-LDA run rather than assume.
- **Prevalence vs discovery tension:** gated LDA gives content discovery but
  drops STM's covariate-prevalence regression. If both are needed later, the
  routes are content-covariate SAGE or Dirichlet-multinomial regression — a
  separate, larger design.
