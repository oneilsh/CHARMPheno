# ADR 0017 — Topic coherence evaluation: NPMI on held-out

**Status:** Accepted
**Date:** 2026-05-11
**Supersedes:** none
**Superseded by:** none
**Companion spec:** docs/superpowers/specs/2026-05-11-topic-coherence-evaluation-design.md

## Context

The framework now has working OnlineLDA and OnlineHDP plus persistence on
both the framework and shim sides. What's missing is any *quantitative*
assessment of topic quality. ELBO does not compare across K or across model
classes; we need a held-out interpretability metric for K-selection,
T-selection, hyperparameter comparisons, and model-class comparisons.

The user's prior project used a modified-UCI coherence:
`sum log_2((1 + p(t_i, t_j)) / (p(t_i) * p(t_j)))` over top-N pairs, then
z-scored. The brainstorming session that preceded this ADR (and the
companion spec) settled the metric choice; this ADR records the decisions.

## Decisions

### Metric: NPMI (Roder et al. 2015)

```
NPMI(w_i, w_j) = log[ p(w_i, w_j) / (p(w_i) * p(w_j)) ]  /  -log p(w_i, w_j)
NPMI = -1 when p(w_i, w_j) = 0    (Roder convention)
NPMI =  1 when p(w_i, w_j) = 1    (perfect co-occurrence)
```

Per-topic score: mean NPMI over the unordered pairs of the top-N terms.

Rejected alternatives:
- *Modified UCI* (user's prior). Custom smoothing constant (`1 + ...`,
  `0.01 + ...` etc.) was tunable but un-principled; z-scoring was needed to
  interpret. NPMI is bounded in `[-1, 1]` without tuning and interpretable
  in absolute terms.
- *Gensim CoherenceModel adapter*. Stock `c_uci` uses sliding-window
  co-occurrence over tokenized text, which is not the right shape for
  patient bags; we'd be patching gensim rather than wrapping it. Gensim's
  release cadence has slowed and the CoherenceModel API has shifted
  between 3.x and 4.x; the maintenance cost of tracking that drift is
  larger than maintaining ~80 lines of NPMI in-tree.

### Co-occurrence shape: whole-document (patient bag)

OMOP records *are* temporally ordered, but choosing a sensible window for
clinical timelines (1-year rolling? episode-bounded?
observation-period-relative?) is its own research question. v1 treats each
patient bag as unordered. A temporal-window variant is plausible v2 work
if topic-quality assessment turns out to be sensitive to it.

### Layering: generic metric, domain split, driver-layer orchestration

- `spark_vi.eval.topic.coherence` is generic over the `(topic_term, holdout_bow)`
  contract. No patient or OMOP concepts.
- `charmpheno.omop.split_bow_by_person` is the deterministic, SHA-256-keyed
  split helper. Lives in charmpheno because it knows about the BOW shape
  and the person-keyed structure.
- `analysis/local/eval_coherence.py` orchestrates: load checkpoint → split
  BOW → score.

The split is **not** a Param on the estimator. MLlib's idiom is that
estimators don't split; users do. Wrapping the split into a Param would
couple every fit to an eval concern.

### Driver-side contract

The fit driver and the eval driver must agree on `(holdout_fraction, seed)`.
v1 documents this as a human contract in the driver headers and this ADR.
v2 may stamp the split provenance into `VIResult.metadata` so the eval
driver can verify before running.

### HDP topic selection

OnlineHDP carries (T, V) lambda where many topics in the tail of the
corpus stick have negligible usage. Scoring them produces near-random NPMI
and inflates the report. The default eval workflow passes a top-K-by-`E[beta]`
mask computed from the corpus-level (u, v); k defaults to 50.

### What's deferred

- Term relevance (Sievert-Shirley) and concept-named top-N tables — its
  own deliverable, same `E[beta]` plumbing.
- pyLDAvis adapter — requires materializing a full `doc_topic_dists`
  matrix; non-trivial plumbing of its own.
- Cloud driver — mirror of the local driver after we have a real model at
  AoU scale to evaluate.
- Gensim adapter and OCTIS integration — if and when the metric menu
  expands.
- Simulation / synthetic recovery testing — its own spec.
- Z-score normalization — NPMI is already bounded and self-interpretable.

## Consequences

- The eval module is purely additive; no compatibility surface.
- `charmpheno.omop.split` is new but additive.
- ADR 0016's `spark_vi.models.topic` namespace is now mirrored by
  `spark_vi.eval.topic`.

## Revisions

### 2026-05-12 — full-corpus reference + min-pair-count threshold

Empirical observation on the patient-year condition-era corpus (see
`docs/insights/0007-npmi-zero-pair-floor-penalizes-rare-phenotypes.md`):
the original holdout-only reference + Roder zero-pair floor combined
to bias the metric against real-but-rare phenotype topics. SLE +
antiphospholipid + chemo-pancytopenia scored −0.53 not because the
cluster was incoherent (it's a recognizable immunosuppression
phenotype) but because most top-N pairs had zero joint counts in the
20% holdout and were floored at −1 each. Same mechanism penalized
sarcoidosis-adjacent and Factor VIII deficiency topics.

Two changes adopted, both well-precedented in the topic-coherence
literature:

1. **Default reference corpus = full BOW (train ∪ holdout)** rather
   than holdout-only. Methodologically sound: NPMI is a coherence
   metric over a fixed (post-fit) topic-word distribution; there is
   no predictive-overfitting concern that requires hold-out. Using
   train ∪ holdout gives ~5× more documents per pair and dramatically
   reduces how many genuine pairs round to zero. Holdout-only remains
   available via `--npmi-reference holdout` for reproducing the prior
   metric.

2. **Min-pair-count threshold with coverage reporting** ("C_NPMI"
   handling per Aletras & Stevenson 2013; Röder et al. 2015). Pairs
   with joint count below `--npmi-min-pair-count` (default 3) are
   *skipped*, not floored at −1. Each topic now reports both its mean
   NPMI (over scored pairs only) and its coverage =
   scored_pairs / total_pairs. A rare-phenotype topic now reads as
   "NPMI=+0.4, cov=55%" instead of being dragged negative for
   sparsity. A topic where every top-N pair falls below threshold
   reports NPMI=NaN (unrated) and is excluded from summary
   statistics.

   The original Roder zero-pair convention (NPMI=−1 when p(w_i,w_j)=0)
   still lives in `_npmi_pair`, but is unreachable in the default
   path because pairs with count 0 fail the threshold first. Setting
   `--npmi-min-pair-count 1` recovers the historical behavior for
   missing pairs (skipped instead of floored — a deliberate
   asymmetry; the −1 floor was always a workaround for an
   undefined limit).

These changes are CLI-level (no checkpoint format change) and the
fit drivers are untouched. Old checkpoints evaluate fine; the
historical metric is reproducible with explicit flags.

References:
- Aletras, N. & Stevenson, M. (2013). "Evaluating Topic Coherence
  Using Distributional Semantics." IWCS.
- Röder, M., Both, A. & Hinneburg, A. (2015). "Exploring the Space
  of Topic Coherence Measures." WSDM.

## 2026-05-13 Revision — train/holdout split removed; eval is vocab-frozen

The held-out evaluation surface became methodologically vestigial after
the 2026-05-12 revision: with the full-corpus reference as the default
(no predictive-overfitting concern for NPMI), the train/holdout split
no longer earned its share of the driver complexity (CLI flags,
metadata stamping, contract checking). It is removed in full.

What this changes:

- Fit drivers no longer accept `--holdout-fraction` / `--holdout-seed`.
  The full corpus is the fit corpus. `metadata["split"]` is no longer
  stamped.
- Eval drivers no longer accept `--holdout-fraction` / `--seed` /
  `--npmi-reference`. The reference is always the full BOW the caller
  supplies.
- The `verify_split_contract` checker in `analysis/_eval_common.py`
  and the `split_bow_by_person` helper in
  `charmpheno/charmpheno/omop/split.py` are deleted. Old checkpoints
  with `metadata["split"]` still load — the field is simply ignored.

Companion change — vocab freezing at eval time:

- `to_bow_dataframe` gains an optional `vocab` parameter. When
  supplied, it skips the `CountVectorizer.fit` step and constructs a
  `CountVectorizerModel.from_vocabulary(...)` directly.
- Eval drivers read `metadata["vocab"]` from the checkpoint and pass
  it to `to_bow_dataframe(vocab=...)`. Tokens absent from the
  checkpoint's vocab are dropped from the eval reference; tokens in
  the vocab not present in the supplied parquet just contribute zero
  doc-frequency.
- Net effect: the eval is decoupled from the fit-time input. The eval
  signature is now `(checkpoint, any OMOP parquet) → CoherenceReport`.

If held-out evaluation matters in the future, the right place to
restore it is at data-prep time (partition the OMOP parquet into
train/holdout artifacts, fit consumes train, eval consumes whichever
artifact the user wants) rather than inside the fit/eval drivers. The
function itself was ~60 LOC; reimplementing if needed is cheap.
