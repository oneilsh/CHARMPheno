# Hybrid domain-reliability readout for multi-domain case-finding — design

**Date:** 2026-07-29
**Branch:** `multidomain-spectral-init`
**Status:** approved
**Motivates:** insights 0071–0074 and the domain-normalization result that no
single global rule is best across rare6 diseases

## Goal

Raise rare6 case-finding precision/recall using the three domains already fitted
(`condition`, `drug`, and `observation`) before adding measurement or other
domains.

The immediate experiment estimates a small, disease-specific supervised
combination of the existing per-domain α→∞ LR score matrices. It serves as a
**diagnostic ceiling**: how much ranking-head performance is available from
better domain combination alone, with the fitted topic model held fixed?

The longer-term target remains usable for MONDO-scale DAGs containing diseases
with few or zero confirmed examples. The same experiment therefore evaluates
predeclared model-derived reliability weights and establishes the seam for a
later shrinkage estimator:

```text
zero labels      -> model-derived domain weights
some labels      -> disease-specific weights shrunk toward that fallback
many labels      -> increasingly disease-specific weights
```

The current rare6 held-out split is explicitly a **development benchmark**:
method choices may use it. Final claims require validation on other rare
diseases or a fresh patient split.

## Why a hybrid

### Label-free weights alone

Weights derived from fitted λ and unlabeled documents extend naturally to new
DAG nodes and thousands of topics. They preserve a clean interpretation and do
not consume scarce labels. But model distinctiveness is not task utility:
administrative artifacts can be distinctive, and the existing explain-away
quantity measures token ownership rather than improvement in average precision.
No label-free statistic can guarantee that adding a domain will not hurt the
ranking head.

### Label-trained weights alone

A three-feature discriminative combiner directly optimizes the rare6 objective
and can learn that observation should help one disease and be ignored for
another. Multi-view topic representations followed by supervised prediction
have precedent in MixEHR, which evaluated a MixEHR-plus-logistic-regression
classifier with five-fold cross-validation (Li, Nair, Lu et al. 2020).

Unconstrained disease-specific supervision does not transfer to a MONDO node
with no positives, however, and estimates for diseases with roughly 79–80
positives can be unstable. It would also turn the readout into a collection of
independent disease classifiers.

### Decision

Use a small supervised combiner first to measure attainable headroom. In
parallel, evaluate three fixed model-derived candidates against that ceiling.
Do not build a hierarchical estimator until the experiment establishes both:

1. continuous disease-specific weighting materially improves PR; and
2. at least one model-derived reliability signal usefully approximates those
   weights, or the supervised gain is large enough to justify pooled supervised
   transfer instead.

## Inputs and fixed semantics

The first increment requires no topic-model refit. It reuses:

- per-domain fitted `params/lambda_<m>.npy` sidecars;
- persisted held-out per-domain BOW matrices;
- the DAG layout and parent mapping;
- held-out frontiers; and
- the validated tie-collapsing average-precision implementation.

For disease anchor `a`, a positive patient is defined exactly as in the current
readout: the patient's frontier intersects the descendant subtree of `a`.

Domain evidence is combined **at each ontology node before subtree
maximization**:

```text
score(d, a) =
    max over v in subtree(a) [
        sum over domains m [
            w(a,m) * S_m(d,v) / scale(a,m)
        ]
    ]
```

This prevents different domains from independently choosing incompatible
descendant nodes and then adding those separate maxima. It preserves the
current case-finding readout semantics.

All LR scores use the parameter-free α→∞ lift limit already established by the
case-finding arc.

## Supervised diagnostic ceiling

### Cross-validation

Use repeated stratified five-fold cross-validation separately for each of the
six disease anchors. With approximately 79 positives for the smallest diseases,
each outer test fold retains roughly 16 positives. Repeated partitions expose
whether a gain depends on a lucky allocation of those few cases.

Every reported supervised prediction must be out-of-fold. A patient's label may
influence neither the weights nor transformations used to score that patient.

If an artifact can contain multiple documents per person, folds must be grouped
by person. The current rare6 patient-document artifact may use row-level folds
only after asserting the one-row-per-person invariant.

### Fold-local transformations

The outer training fold alone determines:

- each domain's background code-frequency distribution;
- each disease/domain score scale; and
- the selected discrete rule or continuous weights.

The frozen quantities are then applied to the outer test fold. This removes two
transductive dependencies from the evaluation:

- `_lr_base_rate` otherwise derives the LR background from the full cohort being
  scored; and
- current `std` normalization estimates scale from the full scoring batch.

For disease `a` and domain `m`, estimate one positive scale from the training
patients' subtree-level domain scores. Apply that scalar to every node in the
subtree, preserving within-domain node and patient order. A non-finite or
non-positive scale falls back to `1.0`.

### Nonnegative weight search

Search a predeclared finite grid on the three-domain simplex:

```text
w_condition + w_drug + w_observation = 1
w_m >= 0
```

Nonnegativity lets a domain be ignored with weight zero but does not allow the
readout to reverse the meaning of its LR evidence. A finite grid is preferred
to a general classifier in this first increment because it is transparent,
deterministic, and sufficient to estimate a three-domain ceiling.

Select weights by inner stratified cross-validation using average precision
within the outer training data; never select them on the same outer-fold
patients used to report performance. Any choice among grid resolution, inner
fold count, outer repeat count, or tie-breaking policy must be fixed in the
implementation plan rather than adjusted after viewing results.

## Comparison ladder

Evaluate three increasingly flexible combination strategies on identical outer
folds.

### 1. Fixed policies

Report the existing fixed baselines, including:

- condition only;
- condition plus drug (`drop:observation`);
- all domains, unnormalized; and
- all domains under `std`, `length`, and `length+std`.

### 2. Discrete supervised selector

Inside each outer training fold, choose among the existing finite menu:

```text
domain subset: all / only:m / drop:m
normalization: none / std / length / length+std
```

This estimates the gain available merely from selecting an existing rule per
disease.

### 3. Continuous supervised ceiling

Select the nonnegative simplex weights inside the outer training data. Its
improvement over the discrete selector is the value attributable to graded
disease-specific domain reliability, rather than rediscovering an existing
hard drop or normalization rule.

## Model-derived fallback candidates

For node `u` and domain `m`, compute three quantities from fitted model state
without case labels.

### Distinctiveness

Sum and normalize the λ rows in node `u`'s topic block to obtain its fitted code
distribution in domain `m`. Compare it with the fitted domain background using
bounded Jensen–Shannon divergence. This asks whether the domain says anything
node-specific.

### Ownership

Use `_routing_rows` to measure whether codes characteristic of node `u` route
to its block rather than background or competing nodes:

```text
ownership(u,m) =
    expected r(u | code)
    for codes drawn from u's fitted domain distribution
```

This reuses the existing explain-away responsibility while interpreting it as
ownership, not as a proven reliability estimator.

### Viability

Downweight a node-domain block whose topics remain at or effectively near the
Dirichlet prior. Viability must operate at topic granularity so it does not
repeat the any-topic-alive blind spot corrected by `starved_topic_report`.

### Predeclared candidates

Evaluate only:

```text
A: distinctiveness
B: ownership
C: distinctiveness * ownership * viability
```

Normalize each candidate across domains to obtain weights summing to one. Do
not expand the formula menu until these establish whether fitted-model
reliability predicts task utility at all.

Evaluate candidates by:

- direct rare6 out-of-fold-compatible PR under the same fixed-background and
  fixed-scale semantics;
- agreement with the supervised domain ordering;
- patient-ranking agreement with the supervised ceiling; and
- consistency between the mini-batch and full-batch fitted artifacts.

## Metrics and interpretation

The primary metric is per-disease out-of-fold average precision. Also report:

- macro-average AP across the six diseases, giving each equal weight;
- prevalence and lift over prevalence;
- precision at 10%, 25%, 50%, and 80% recall;
- fold-to-fold and repeat-to-repeat variability;
- learned weight distributions by disease;
- top-ranked-patient overlap and rank stability; and
- model-derived versus supervised weight/ranking agreement.

Use the existing `_average_precision` implementation so tied scores share one
threshold and a constant scorer yields AP equal to prevalence.

Interpret the continuous ceiling's gain over the discrete selector as:

- **little headroom:** less than approximately 10% relative macro-AP lift with
  inconsistent disease-level movement;
- **useful headroom:** approximately 10–25% relative improvement stable across
  multiple diseases; or
- **strong headroom:** greater than approximately 25%, or a major stable
  improvement in precision at a clinically useful recall for the weakest rare
  diseases.

These bands guide interpretation; they are not significance thresholds.

## Decision after the experiment

| Supervised headroom | Model-derived agreement | Next step |
|---|---|---|
| strong/useful | useful | design shrinkage from supervised weights toward the model-derived fallback |
| strong/useful | poor | design pooled supervised transfer across diseases; do not claim λ alone identifies reliability |
| little | any | stop tuning domain combination and move to representation, observation curation, or new information |

The eventual shrinkage mechanism is deliberately not selected here. Candidate
forms include a prior-centered penalized combiner or a hierarchical domain
weight model, but choosing one before measuring the ceiling and fallback
agreement would be premature.

## Components

### `spark_vi` generic layer

Add or expose pure, domain-agnostic operations to:

- compute raw per-domain LR matrices with explicit background distributions;
- estimate and apply fixed domain scales;
- apply fixed domain weights and combine matrices before subtree maximization;
- compute distinctiveness, ownership, and viability; and
- preserve identity behavior for unit scales and equal weights.

No rare6 labels, disease names, cross-validation, or clinical concepts belong
in `spark_vi`.

### Analysis layer

Add a Spark-free multidomain weighting readout that:

- loads λ, BOW, DAG, and frontier artifacts;
- asserts patient/document fold invariants;
- constructs deterministic repeated stratified folds;
- recomputes fold-local backgrounds and scales;
- evaluates fixed, discrete, continuous, and model-derived strategies;
- emits out-of-fold predictions and diagnostics; and
- writes machine-readable JSON plus a concise Markdown result report.

Supervised search remains in `analysis` because it is task-specific evaluation,
not part of the topic-model engine.

## Experiment sequence

1. Develop and run unchanged on exp 0072, the mini-batch fit representing the
   production compute regime and currently stronger PR result.
2. Replicate unchanged on the corrected exp 0071 full-batch reference fit.
3. Compare PR headroom, domain-weight direction, model-derived reliability,
   and ranking-head patient overlap across the two fits.

No new fit is required initially. Refit only if:

- an existing artifact lacks information required for honest folds;
- a one-row-per-person invariant cannot be established; or
- a later design explicitly adds across-seed stability as a model-derived
  reliability signal.

New fits are operationally affordable, so lack of a reusable artifact is not a
reason to weaken the evaluation protocol.

## Testing

Behavioral tests must establish:

- a planted noise domain receives zero or near-zero supervised weight;
- a helpful second domain receives positive weight and improves held-out AP;
- two diseases with opposite optimal domain mixtures recover different weights;
- altering outer-test rows cannot change training-fold backgrounds or scales;
- domain combination occurs before subtree maximization;
- every reported supervised score is out-of-fold;
- tied-score AP delegates to the existing validated implementation;
- folds and grid ordering are deterministic;
- starved node-domain topics reduce model-derived viability;
- model-derived candidates use no case labels; and
- identity scales/equal weights reproduce the existing unweighted result up to
  the explicit constant implied by simplex normalization, which cannot change
  ranking metrics.

When implementing a regression test for a prevented leakage or ordering bug,
restore the bug and confirm that the test fails before trusting the gate.

## Out of scope

- fitting-time ω sweeps;
- measurement or value-aware laboratory representation;
- topic-model refits in the first increment;
- a production hierarchical estimator;
- MONDO ingestion or ontology mapping;
- arbitrary patient-level features beyond the three per-domain LR matrices; and
- changing the topic-model fitting objective.

## References

- Li, Y., Nair, P., Lu, X. H., et al. (2020). “Inferring multimodal latent
  topics from electronic health records.” *Nature Communications* 11:2536.
- Davis, J. & Goadrich, M. (2006). “The relationship between
  Precision-Recall and ROC curves.” ICML.
- `docs/insights/0071-multidomain-lr-readout-condition-carries-case-finding-observation-is-drag.md`
- `docs/insights/0072-pr-reveals-observation-drag-that-auc-hides-ppi-strip-insufficient.md`
- `docs/insights/0073-no-single-domain-normalization-rule-wins-length-trades-evidence-quantity-for-concentration.md`
- `docs/insights/0074-full-batch-vi-was-damped-fixing-it-did-not-explain-the-minibatch-gap.md`
- `docs/superpowers/specs/2026-07-29-domain-normalized-lr-combination-design.md`
