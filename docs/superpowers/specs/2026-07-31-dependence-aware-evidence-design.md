# Dependence-aware / bursty evidence for multidomain case-finding — design pass

**Date:** 2026-07-31
**Branch:** `hybrid-domain-reliability` (review branch
`claude/hybrid-domain-reliability-review-ckn2bq`)
**Status:** design options for approval — **not** an implementation plan
**Follows:** insight 0075 (domain-combination ceiling is small; λ reliability
does not identify it) and the 2026-07-30 hybrid-domain-reliability handoff
**Intended reader:** the user, deciding a next architecture before any plan

---

## Why this pass exists

Insight 0075 measured the diagnostic ceiling the hybrid domain-weighting
readout was built to measure. It is small: continuous, disease-specific,
out-of-fold domain weights beat the fixed condition+drug baseline by only
~6% (0073) and ~10% (0074) relative macro-AP, with inconsistent per-disease
movement, and no label-free λ statistic identified those weights. Under the
hybrid design's own pre-registered decision table, that is the "little
headroom → stop tuning domain combination; move to representation, observation
curation, or new information" branch. This pass takes that branch.

It targets the objection the handoff flagged as the single most important
unresolved modeling problem: **correlated evidence / conditional independence**
(the colleague's LIRICAL objection). It deliberately does **not** revisit the
shared hierarchical domain-placement mechanism, which pulls the same
near-exhausted lever and cannot be validated for pooling benefit at six anchors.

## The problem, stated precisely

The case-finding readout scores a patient against ontology node `u` with a
**naive-Bayes independent log-likelihood-ratio sum** over codes
(`spark-vi/spark_vi/models/topic/dag_placement.py`, `lr_placement_scores`):

```text
s(i, u) = Σ_w cnt(i, w) · log[ P(w | node u) / bg(w) ]
```

This is not merely *analogous* to LIRICAL — it is the same estimator. LIRICAL
(Robinson et al. 2020) multiplies per-HPO-term likelihood ratios as if the
terms were conditionally independent; we sum per-code log-LRs as if the codes
were. Robinson et al. name the assumption "naive" and note it "is almost never
true," and explicitly leave modeling correlations between terms to future work.

The generative model shares the assumption one level up:

```text
z_token ~ θ_patient          # topic assignment from the patient mixture
code_token ~ β_{domain,topic} # code emitted from that topic's multinomial
```

Given θ and the topic assignments, code emissions are conditionally independent
and exchangeable (multinomial). Shared θ induces marginal co-occurrence and lets
one topic absorb a common code bundle — so this is less naive than multiplying
independent feature LRs — but it does not represent dependence *among distinct
observations beyond what θ explains*. Three clinically routine patterns
therefore each add separate mass to the score:

1. **correlated conditions** (a syndrome's co-manifestations);
2. **a condition and its own treatment** (condition code + the drug that treats
   it — and these now live in *different domains*, so domain-weighting cannot
   discount them);
3. **many codes from one encounter** (a single clinical event described by
   several codes / a code repeated across a visit series).

Naive Bayes with correlated features is known to "count them twice," inflating
confidence and pushing scores toward the extremes. For a **ranking** objective
(AP / precision-at-recall) the harm is specific and worth stating exactly:
*a patient whose high score comes from a few redundant, correlated codes should
not outrank a patient with the same raw score built from diverse, independent
evidence.* Redundancy is a per-patient property, so the fix must act
non-uniformly across patients (see "A note on ranking" below).

### What is already partially mitigated — and what is not

The engine already blunts two of the easier sub-cases:

- **Same-code repetition:** `count_mode="log1p"` saturates repeated counts of
  the *same* code, so a code seen 10× does not add 10× its log-LR.
- **Background / comorbidity codes:** `explain_away_placement_scores` weights
  each code's contribution by its routing responsibility `r(u|w)`, so codes that
  belong to background or competing nodes are suppressed toward zero.

Neither touches the open case: **distinct, disease-relevant codes that are
correlated with each other** (patterns 1–3). `log1p` acts within a single code;
explain-away suppresses *off-topic* codes but leaves *on-topic* correlated codes
each contributing full mass. That residual is the target of this pass.

## Literature anchor

The handoff's "new literature" is the MixEHR/supervision lineage
(`docs/references.md`, "Topic hierarchies, supervision & gating" and
"Phenotyping / EHR topic models"). That lineage is about *supervision*, which is
the hierarchical-placement question — not this one. The load-bearing references
for dependence/burstiness are **not yet in `references.md`** (the handoff itself
predicted this pass would have to turn them up); they are catalogued at the end
of this doc and should be added.

Honestly scoped, the literature says:

- **Word burstiness via Dirichlet-compound-multinomial.** Madsen, Kauchak &
  Elkan (2005) model burstiness with the DCM; Doyle & Elkan (2009,
  "Accounting for Burstiness in Topic Models") replace LDA's multinomial
  topic-word emission with a DCM ("bag-of-bags-of-words") and get better
  held-out likelihood than LDA. This captures *within-topic* burstiness (a topic
  that emits a code becomes more likely to emit it/its neighbours again). It does
  **not** model dependence between codes on *different* topics.
- **Overdispersed counts via negative-binomial / Poisson factorization.** Zhou &
  Carin (2012/2015) and Gopalan, Hofman & Blei (2015) replace multinomial with
  Gamma-Poisson / NB emissions, which have a variance-inflating parameter that
  absorbs overdispersion. Same reach and same limit as DCM: within-component
  burstiness, not cross-component dependence.
- **Additive log-deviation emission (SAGE).** Eisenstein, Ahmed & Xing (2011)
  model each topic as a sparse additive log-deviation from a shared background —
  already in `references.md` as an alternative to gating. Relevant because the
  readout already works in log-deviation-from-background space
  (`log[P(w|u)/bg(w)]`), so a SAGE-style emission is the generative object whose
  natural readout *is* the current LR.
- **Prediction-focused / prediction-constrained topic models.** Ren, Kunes &
  Doshi-Velez (2020) and Hughes et al. (2018) use supervision to suppress
  task-irrelevant vocabulary. Adjacent but orthogonal: they decide *which* codes
  matter, not how to discount *correlated* ones.
- **Naive-Bayes-under-dependence.** The classic result: correlated features are
  effectively counted twice, producing overconfident and miscalibrated scores.
  The mitigation family is effective-count / dependence-aware discounting, which
  is exactly what LIRICAL names as its own future work.

None of this literature offers a general, *learned* mechanism for arbitrary
cross-observation dependence without either (a) heavy per-feature engineering or
(b) new latent grouping variables — the two directions the user already ruled
out. That is the honest gap this pass has to navigate around.

## A note on ranking (the crux)

A global, monotone rescaling of the total score changes nothing for AP or
precision-at-recall — those are rank statistics, and a global temperature
preserves rank order. So "calibrate the naive-Bayes overconfidence" is, by
itself, useless here. What can move ranking is a **redundancy-dependent**
discount: two patients with equal raw log-LR sums must be separated when one's
evidence is redundant (all from one encounter, or a condition+its-drug pair) and
the other's is diverse. Every architecture below is judged on whether it
produces a *non-uniform, redundancy-sensitive* adjustment, not merely a
better-calibrated absolute score.

## Three architectures

Ordered by cost and by where the fix lives. This ordering is deliberate: it
mirrors the methodology that produced insight 0075 — prove value at the readout
first, and only move into variational inference if a readout-level diagnostic
shows durable, redundancy-driven headroom.

### D1 — Readout-level effective-evidence discount (cheapest; do first)

**Idea.** Keep the topic model frozen. Replace the plain per-patient sum with a
sum that discounts codes by how much they are redundant with the other codes the
same patient already contributes to node `u`. Concretely, weight each code's
log-LR contribution by an effective-independence factor derived from corpus
code–code co-occurrence (a redundancy statistic estimated on the training fold),
so a condition+its-drug pair, or a tightly co-occurring code cluster, contributes
closer to one effective observation than to several.

**What dependence it captures.** Patterns 1 and 2 (correlated conditions;
condition+treatment across domains), because the redundancy statistic is
computed across the full code vocabulary including cross-domain pairs. Pattern 3
only insofar as same-encounter codes are also corpus-correlated.

**Inference / cost.** No refit. Reuses existing 0073/0074 artifacts exactly like
the domain-weighting readout did. The redundancy statistic and any discount
parameters are fit fold-locally under the ADR 0038 nested-CV / one-row-per-person
attestation contract; every reported score stays out-of-fold. Can be delivered
as a *diagnostic ceiling* first: does redundancy-aware discounting improve
out-of-fold AP at all, and on which diseases?

**Interpretability.** High. `lr_decompose` already itemizes per-code
contributions; the discount is one extra column ("this code was down-weighted ×
because it co-occurs with …"). Clinically legible.

**Leakage / PU.** Same posture as the current readout: labels never touch the
redundancy statistic (it is label-free corpus structure); only the (small)
discount hyperparameters, if any, are chosen out-of-fold.

**Risk / honesty.** This is the direction closest to the rejected "learned
evidence groups." The line it must not cross: it estimates a *redundancy
weighting over the existing representation*, not a *new set of latent variables
that compete with topics*. If it grows into learning explicit groups, it has
become the rejected approach and should stop. It also cannot recover signal from
genuinely independent evidence — it can only remove double-counting.

### D2 — Encounter-structured effective counts (medium; native structure)

**Idea.** A concrete, data-supervised instance of D1: use the EHR's own grouping
— visit / encounter / N3C macrovisit (Pfaff et al. 2023) — as the unit within
which evidence saturates. Codes co-occurring in one encounter are combined with a
saturating (soft-OR-like) combiner before summation, so "a single clinical event
described by five codes" contributes bounded, not additive, evidence. A
"bag-of-encounters" replaces "bag-of-codes."

**What dependence it captures.** Pattern 3 directly and by construction, and part
of 1–2 when correlated codes are also encounter-clustered. Its dependence
structure is *given by the data*, not learned, so it sidesteps the
evidence-groups objection entirely.

**Inference / cost.** No topic refit, but a real catch: the persisted rare6
artifact is a patient-level one-year-lookback BOW that has **already collapsed
encounter structure**. D2 needs re-assembly that preserves encounter ids — which
ties it to the multidomain-assembly / caching arc the handoff flagged (§4). So D2
is operationally heavier than D1 despite also being "readout-level," and it makes
the content-addressed assembly cache a near-prerequisite.

**Interpretability.** Very high and clinically natural ("counted this admission
once, not once per code").

**Leakage / PU.** As D1; the grouping is label-free.

**Risk / honesty.** Encounter boundaries in OMOP are noisy (visit granularity
varies by site/source); the saturation combiner needs a defensible, predeclared
form or it becomes a tuning surface. It also does nothing for correlations that
span encounters (a chronic condition and its maintenance drug recorded on
different days).

### A — Overdispersed / compound emission inside the model (most principled; most expensive)

**Idea.** Change the generative emission so the *model itself* expects
burstiness, then let the LR readout inherit the corrected likelihood. Two
literature-backed forms: (a) DCM / Pólya topic-word emission (Doyle & Elkan
2009), whose marginal likelihood for a repeated/correlated draw saturates; or
(b) NB / Gamma-Poisson emission (Zhou & Carin; Gopalan et al.), whose dispersion
parameter absorbs overcounting. The readout becomes the compound-distribution LR
rather than the multinomial LR.

**What dependence it captures.** Within-topic burstiness — i.e. a disease
topic's own co-manifesting codes (much of patterns 1–2 *when those codes load on
the same topic*). It is the generatively consistent version of D1/D2. It does
**not** capture dependence between codes on *different* topics (the residual that
would need care-process latents, which the user is skeptical of).

**Inference / cost.** High. Touches variational inference: DCM/NB emissions break
multinomial conjugacy and change the E-step and the λ sufficient statistics; this
is the SVI engine's core. Requires a fresh fit under the 0073/0074 attestation
contract. Justified only if D1/D2 show that redundancy discounting has durable,
material AP headroom that a frozen-model readout cannot fully capture.

**Interpretability.** Moderate. Dispersion parameters are interpretable per
topic/domain, but the change is inside the engine rather than in a legible
per-code readout column.

**Leakage / PU.** Unsupervised generative change; label handling unchanged. The
usual refit attestation applies.

**Risk / honesty.** Largest engineering surface and the one most able to
destabilize the existing spectral-init / gated-SVI stack. Its reach (within-topic
only) is narrower than its cost suggests, which is the main argument for proving
the readout versions first.

## What none of these solve

Genuinely distinct evidence on *different* topics that is clinically correlated
through a cause not represented as a topic (e.g. two disease processes sharing an
upstream driver). Representing that faithfully is the care-process/manifestation
latent-variable territory the user judged unlikely to be learned reliably. This
pass explicitly scopes it out; if D1/D2 leave a large redundancy residual that is
provably cross-topic, that finding — not a speculative latent — is the trigger to
revisit it.

## Recommended sequence

1. **D1 as a diagnostic ceiling**, on the existing 0073/0074 artifacts, exactly
   as the domain-weighting readout was run: does redundancy-aware discounting
   improve out-of-fold AP, for which diseases, and by how much? This is cheap,
   label-safe, and directly comparable to insight 0075.
2. If D1 shows redundancy-driven headroom, **decide D2 vs. A**: D2 if the
   residual is encounter-clustered (and accept coupling to the assembly-cache
   arc); A if the residual is within-topic but not encounter-bounded and the
   headroom justifies an engine change.
3. Only move dependence modeling *inside* VI (architecture A) once a readout
   diagnostic has earned it — same gate the hybrid design applied to hierarchical
   supervision.

Measurement stays a later arc, but note the ordering rationale: adding a
value-aware measurement domain *before* any of D1/D2/A would worsen the
double-counting (labs are drawn in correlated bursts), so settling dependence
handling first is a genuine prerequisite for measurement, not merely adjacent.

## Open decisions for the user

- Is D1 (frozen-model redundancy discount) the right first diagnostic, or do you
  want D2's encounter structure from the start despite its assembly-cache
  coupling?
- For D1's redundancy statistic, is a corpus-co-occurrence effective-count
  acceptable, or does even that read too close to "learned evidence groups" for
  comfort?
- Is the within-topic-only reach of architecture A worth keeping on the table
  now, or should it wait until a readout diagnostic proves a within-topic
  residual specifically?

No implementation plan follows until one of these is chosen.

## References to add to `docs/references.md`

New "burstiness / overdispersion / dependence" theme:

- **Madsen, Kauchak & Elkan (2005).** Modeling Word Burstiness Using the
  Dirichlet Distribution. *ICML*. — DCM for burstiness (foundational).
- **Doyle & Elkan (2009).** Accounting for Burstiness in Topic Models. *ICML*,
  281–288. — DCM-LDA; the canonical topic-model burstiness paper.
  https://cseweb.ucsd.edu/~elkan/TopicBurstiness.pdf
- **Zhou & Carin (2015).** Negative Binomial Process Count and Mixture Modeling.
  *IEEE TPAMI* 37(2):307–320. (arXiv:1209.3442) — NB/overdispersed count
  factorization.
- **Zhou, Hannah, Dunson & Carin (2012).** Beta-Negative Binomial Process and
  Poisson Factor Analysis. *AISTATS*, PMLR 22:1462–1471.
- **Gopalan, Hofman & Blei (2015).** Scalable Recommendation with Hierarchical
  Poisson Factorization. *UAI*. — Gamma-Poisson factorization at scale.
- **Robinson, Ravanmehr, Jacobsen, Danis, et al. (2020).** Interpretable Clinical
  Genomics with a Likelihood Ratio Paradigm (LIRICAL). *Am J Hum Genet*
  107(3):403–417. — the naive-Bayes phenotype-LR framework whose independence
  assumption this pass addresses; names correlated-term modeling as future work.

Already present and reused here: Eisenstein, Ahmed & Xing (2011, SAGE); Ren,
Kunes & Doshi-Velez (2020, prediction-focused); Hughes et al. (2018,
prediction-constrained); Pfaff et al. (2023, N3C macrovisits); Li, Nair, Lu et
al. (2020, MixEHR).
