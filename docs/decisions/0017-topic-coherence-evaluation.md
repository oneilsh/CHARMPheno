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
