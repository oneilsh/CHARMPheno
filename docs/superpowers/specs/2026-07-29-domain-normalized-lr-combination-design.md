# Domain-normalized multi-domain LR combination — design

**Date:** 2026-07-29
**Branch:** `multidomain-spectral-init`
**Status:** approved
**Motivates:** insight 0072 (a low-signal domain costs most of the precision), plus the
owner's requirement: *"I don't like the idea of a domain dragging things down; ideally if
there's no signal it should at least not hurt, or at least not hurt badly"* — because as the
disease set grows, observation may carry real signal for some diseases and should be kept
in the model rather than curated out.

## Problem

The multi-domain LR placement score is an **unweighted, unnormalized sum** of per-domain
log-likelihood ratios:

```
score(doc, u) = Σ_m Σ_w count_m(doc, w) · log[ P(w | u, m) / bg_m(w) ]
```

Each domain's term scales with **its own token volume**. Observation carries many tokens per
patient, so its term is large in magnitude; when that term is mostly noise it adds variance
proportional to observation's volume on top of a condition term of fixed magnitude. This is
Naive-Bayes-across-domains, which assumes both conditional independence *and* equal
per-domain calibration. EHR domains satisfy neither.

Insight 0072 measured the cost: `drop:observation` beats `all` on PR-AUC for every disease,
by up to +263% relative (Ehlers-Danlos 0.038 → 0.138), while ROC-AUC moved only ~1–4%.
Average precision is a head-of-ranking metric, and the head is exactly where the added
variance does its damage.

**Two distinct pathologies produce this, and they need different fixes.**

**(i) Across-domain scale mismatch.** Observation's score term is systematically larger in
magnitude than condition's, so it dominates the sum regardless of content.

**(ii) Within-domain, across-doc volume heterogeneity.** A patient with 500 observation
tokens gets an extreme-magnitude observation term in either direction; a patient with 5 gets
a near-zero one. The top of the ranking therefore fills with *heavily-utilized* patients
rather than patients whose codes match the topic — the utilization confound already flagged
against learning ω from the fit. A single per-domain scalar does **not** fix this one.

## Approach

Add a per-domain **normalization** step to the combination rule, applied before summing.
Four rules, all label-free and all readout-side (no re-fit):

| rule | transform of domain m's score matrix `S_m` | fixes |
|---|---|---|
| `none` | `S_m` (today's behavior) | — |
| `std` | `S_m / σ_m`, `σ_m` = std of all entries of `S_m` | (i) |
| `length` | per-doc divide by domain m's token count (mean log-LR per token) | (ii) |
| `length+std` | length-normalize, then scale-equalize | (i) + (ii) |

**Caveat on `length` (not a clean win):** per-doc division also discards **evidence
quantity**, which at rare-disease base rates is real signal — removing it can cost more
than the confound it fixes. (The synthetic in the Acceptance-criterion section below
measured `length` costing a signal-carrying subset 0.716 → 0.337 PR-AUC, with the noise
domain not even present in that column.) There is also no denominator shrinkage: a doc
with a single token in a domain contributes that one code's full per-token log-LR as the
domain's entire term, so sparse-domain docs get high-variance extreme values that can
populate the head of the ranking after `std`. A `tok + k` or `sqrt(tok)` denominator would
temper this; out of scope here, noted only as a possible follow-up.

### Why scale-only, and why a single scalar

Centering is provably irrelevant here: adding a constant to a domain's whole score matrix
cancels in both operations the readout performs — ranking docs, and taking the max over a
subtree's columns. So the honest minimal transform for (i) is **scale, not z-score**.

Using one scalar per domain (rather than per-column) makes the transform **affine over the
whole matrix**, which preserves *every* within-domain ordering: doc ranking and
max-over-nodes alike. Consequences worth stating as contract:

- The `only:<m>` columns of the readout tables are **provably unchanged** by `std`. That is a
  free invariance test, and it keeps the per-domain decomposition comparable across rules.
- `σ_m` is computed from domain m alone and does **not** depend on which subset is being
  scored. A domain's contribution is therefore the same in `all` as in `drop:x`, so the
  decomposition table remains coherent. This is why per-domain matrices must be computed
  once and reused across subsets — a semantic requirement, not just an optimization.
- `σ_m` uses **no labels**, so nothing here is tuned or leaks: it stays in the same
  parameter-free spirit as the α→∞ lift limit.

### On "additivity"

`lr_placement_scores_multidomain` currently refuses `length_normalize` on the grounds that it
"breaks additivity." That reasoning was too strong. What additivity buys the readout is
*"a subset's score is the sum of its member domains' scores"*, and that survives any
per-domain transform. What a per-domain transform costs is the summed score being a joint
model's log-likelihood ratio — and the readout already gave that up at α→∞, where the score
is an uncalibrated lift limit rather than a posterior. The readout is a ranker, so
comparability across domains is worth more than joint-likelihood interpretation.

### Honest ceiling

Equal-variance weighting **bounds** a noise domain's damage; it cannot zero it. With D
equal-weight domains where one carries the signal, SNR dilutes roughly as 1/√D. Reaching
true "no harm" requires reliability weighting — per-domain (and probably per-node) weights
driven toward zero where a domain earns nothing, e.g. via the already-implemented
`explain_away_placement_scores`. This design is the prerequisite for that: weights over
incomparable scales are meaningless. The goal here is converting *catastrophic* into
*bounded and predictable*, and measuring how much of the observed drag that removes.

`_domain_scale`'s use of a single whole-matrix std has two further limits beyond the 1/√D
dilution bound above:

- **Conflates two spreads.** `σ_m` = std of the whole `[n_docs x n_nodes]` matrix mixes
  across-doc spread (the volume component that actually drives cross-domain dominance in
  the sum) with across-node spread (which only decides which node wins for a given doc, not
  the domain's overall magnitude). A domain with wide node-to-node but narrow doc-to-doc
  spread is over-shrunk relative to what dilution alone would call for.
- **Transductive, not deployable as-is.** `σ_m` is estimated from the batch being scored, so
  a single patient's score is not computable in isolation and is not stable as the scoring
  cohort changes. Fine for a batch research readout; if `std` is ever promoted to a deployed
  default, `σ_m` must be persisted from a reference cohort rather than recomputed per
  scoring batch.

## Components

### 1. Library — `spark-vi/spark_vi/models/topic/dag_placement.py`

- `_domain_scale(s) -> float` — std of the whole score matrix; non-finite or non-positive
  std returns `1.0`, so a constant or empty domain passes through its (already order-free)
  values instead of producing `inf`/`nan`. A zero-token domain scores all-zeros and
  therefore contributes nothing — the degenerate "inert domain" case, for free.
- `lr_domain_score_matrices(...) -> {m: [n_docs x n_nodes]}` — the per-domain score
  matrices, already normalized per `normalize`. Summing a subset's matrices *is*
  `lr_placement_scores_multidomain` over that subset, so a caller scoring many subsets
  computes each domain once.
- `lr_placement_scores_multidomain(..., normalize=None)` — delegates to
  `lr_domain_score_matrices` and sums. `normalize` accepts `None`, `"std"`, `"length"`,
  `"length+std"`; anything else raises `ValueError`.
- `lr_auc_sweep_multidomain(..., normalize=None)` — forwards the rule.

`normalize=None` must reproduce today's numbers exactly.

### 2. Readout — `analysis/cloud/multidomain_lr_readout.py`

- `--normalize {none,std,length,length+std}`, default `none`, governing the existing
  alpha-sweep / LR-AUC / PR-AUC / precision@recall tables.
- Compute per-domain matrices once via `lr_domain_score_matrices`; build each subset as the
  sum of its selected matrices, replacing the current per-subset recompute (which recomputed
  each domain roughly `n_dom + 2` times).
- New tables: **per-disease PR-AUC under each of the four rules**, printed as two stacked
  blocks of identical shape — one for subset `all`, one for subset `drop:<last domain>` —
  rather than paired columns in one table, so both readings the (corrected) acceptance
  criterion needs are visible: within-rule `drop:X` minus `all` is `drag(rule)`; across
  rules, the `drop:X` block alone shows what the rule does to the domains kept. This is the
  A/B, and it needs no re-fit — exp 0071/0072 already persist their held-out test sets.

### Acceptance criterion

**Correction of record.** This section originally defined a single cross-rule difference,
`gap(rule) = PR_AUC(drop:observation, none) − PR_AUC(all, rule)`, as the pass/fail measure.
**That criterion is REFUTED** — see the counterexample below — because it conflates two
independent quantities and can invert the ranking of rules relative to the property it
claims to measure. It is kept here, marked refuted, for the record; do not use it as a
gate.

Per disease and rule, define two **separate** quantities:

```
drag(rule)              = PR_AUC(drop:X, rule) − PR_AUC(all, rule)
PR_AUC(drop:X, rule)    (compared across rules)
```

`drag(rule)` — what keeping domain X costs under that rule — **is** the criterion: success
is `drag(rule)` shrinking toward zero relative to `drag(none)` (today's baseline, large:
0.351 absolute in the synthetic below; 0.100 absolute for Ehlers-Danlos in the original
condition+drug+observation readout). Judge on **PR, not ROC** (insight 0072, Finding 2),
because the damage is at the head of the ranking.

`PR_AUC(drop:X, rule)` compared across rules is the separate "would we be better off
curating domain X out entirely, under this rule?" baseline. **Both numbers are required**:
a rule can shrink `drag(rule)` while degrading the domains it keeps — i.e. while lowering
`PR_AUC(drop:X, rule)` relative to `PR_AUC(drop:X, none)` — because every non-`none` rule
also re-weights the retained domains against *each other*, not only against the dropped
domain.

**Why the original single-difference criterion is wrong.** It decomposes as

```
gap(rule) = PR_AUC(drop:X, none) − PR_AUC(all, rule)
          = drag(rule) − [PR_AUC(drop:X, rule) − PR_AUC(drop:X, none)]
                          \____________________________________________/
                             the rule's effect on the domains KEPT
```

The bracketed term is nonzero for every non-`none` rule, so `gap` mixes "does this rule
help domain X's drag" with "does this rule also hurt the domains we kept" into one number
— and can rank rules backwards relative to `drag` alone.

**Counterexample (synthetic, refuting the original criterion).** 3000 docs, 2% prevalence,
3 domains: domain 0 = strong signal, domain 1 = weak signal, domain 2 = high-volume pure
noise with lognormal per-doc volume heterogeneity and no class enrichment.

| rule | PR(all) | PR(drop:2) | drag(rule) | gap(rule) [refuted] |
|---|---|---|---|---|
| none | 0.365 | 0.716 | 0.351 | 0.351 |
| std | 0.539 | 0.631 | 0.092 | 0.177 |
| length | 0.337 | 0.337 | −0.001 | 0.378 |
| length+std | 0.337 | 0.410 | 0.073 | 0.379 |

`length` drives the noise domain's drag to (numerically) zero — the stated goal, fully
achieved — yet scores **worst** on `gap`, because it also collapses `PR(drop:2)` from 0.716
to 0.337 (Finding 2's evidence-quantity cost, incurred even though the noise domain isn't
in that column at all). `std` cuts drag 74% but `gap` shows only 50% of that improvement.
On `drag` alone the ranking is `length` ≈ `length+std` < `std` ≪ `none`; on the refuted
`gap`, `std` looks better than `length` — an inversion of the true drag ranking.

This is a measurement, not a threshold: report both `drag(rule)` and `PR_AUC(drop:X,
rule)` per candidate rule; a rule only wins if it improves (or does not much worsen) both.
If no rule does, that is itself the finding that pushes to reliability weighting.

## Testing

Library (`spark-vi/tests/test_lr_multidomain.py`, reusing its `_tiny()` fixture):

- `normalize=None` is identical to the current per-domain sum (regression).
- Mechanism, exact: with domain 1's BOW a 10× copy of domain 0's, `lr_domain_score_matrices`
  under `std` returns matrices whose std is 1.0 for both domains, and the un-normalized
  `σ` ratio is ~10. Scale equalization is what it claims to be.
- Invariance, exact: on a single domain, `std` preserves both `_auc` of the max-over-nodes
  score and the per-doc `argmax` over nodes versus `none`.
- `length` equals summing per-domain `lr_placement_scores(..., length_normalize=True)`.
- An unrecognized `normalize` value raises `ValueError`; `domains=[]` still raises.
- `lr_auc_sweep_multidomain(normalize=...)` agrees with a manual AUC of the correspondingly
  normalized score.

Readout (`analysis/cloud/tests/test_multidomain_lr_readout.py`):

- Parser default is `none`; a bad `--normalize` value is rejected.
- The rule-name → library-value mapping (`"none"` → `None`) round-trips.
- The comparison-table helper (`pr_by_normalization`) returns `{rule: {anchor: pr_auc}}`,
  and its `none` entry matches `per_disease_pr` called directly on the un-normalized subset
  sum, for both the `all` subset and an arbitrary `domains=` restriction, with the default
  `rules` returning all four rules (the capability the two-block `all` / `drop:X` readout
  depends on — the original single-block table only ever called it with `rules=("none",)`
  for the reference subset).

## Out of scope

- Per-domain *weights* (tuned or reliability-driven) — the follow-up this enables.
- Fitting-time ω. This is entirely a readout-side change; no re-fit.
- Changing the default rule. `none` stays the default until the measurement says otherwise.
