# 0076 — At 41 anchors the weighting readout replicates 0075: domain combination is a flat lever, optimal weights are condition-primary, and case-finding is information-limited — supervised/pooled domain placement is not worth deploying

**Date:** 2026-07-31
**Topic:** case-finding | multidomain | supervision | pooling | scaling | decision
**Status:** Confirmed on exp 0076 (expanded rare-disease anchor set)

Exp 0076 fit the expanded `rare_priority` anchor set (6 rare6 + 35 prioritised
Monarch dismech #1079 diseases mapped MONDO→OMOP; 41 anchors, K=230 = 40 bg + 95
nodes × 2 tpn; cond+drug+obs, 1yr lookback, PPI stripped, spectral scalable init,
full-batch, seed 42). The generalized nested-CV weighting readout scored 39
anchors (2 skipped for too few held-out cases: Churg-Strauss variant,
Huntington's chorea; 1 dead node: transposition of great vessels).

The result is a clean replication of insight 0075 at 6.5× the anchors, plus a
sharper conclusion the larger, more diverse set makes possible.

## Domain combination is a flat lever

Macro median AP: **fixed condition+drug 0.032, continuous 0.034 (~6%),** discrete
0.026, and the three λ-derived model weights 0.023–0.024. So continuous
per-disease weighting barely beats the fixed baseline (same ~6–10% band as
0075), the discrete selector is unstable and worse, and label-free λ reliability
is worse still — and its domain *ordering* almost never matches the supervised
ordering (`same_domain_order_frequency` ≈ 0 across anchors). 0075 was not a
six-disease artifact.

## Optimal weights are condition-primary, with idiosyncratic (not clusterable) exceptions

~28 of 39 anchors select continuous median weights of `condition`=1.0,
`drug`=`observation`=0. Drug earns weight only where a genuine signature drug
exists — myasthenia gravis (0.6/0.4, pyridostigmine), familial
hypercholesterolemia (statins), POTS. **Observation earns weight essentially only
for Long QT syndrome (0.4 → the QT-interval measurement).** Every other
non-condition weight sits on a near-zero-AP anchor (Marfan, Takayasu,
thromboangiitis, cerebral amyloid angiopathy…) and is noise.

This kills the pooling rationale directly (the reason the anchor set was
expanded, per 0075's "prefer one shared partially-pooled mechanism"): the
vasculitis and neuroimmune clusters are uniformly condition-only (the deviations
within them are the noise anchors), which is *identical to the fixed baseline*.
Partial pooling toward a cluster mean would just relearn "use condition." There
is no borrowable cross-disease structure to exploit, so a shared/hierarchical
domain-weighting mechanism is **not worth building.**

## Placement is not supervised for domains, and should not become so

The deployed placement (gated multidomain fit + α→∞ LR readout) combines domains
with a *fixed* rule; no labels set the weights. This readout is a supervised
*diagnostic* of whether making that supervised/per-disease would help. Its
verdict: a fixed condition-primary combination (add drug where a signature drug
exists) is already at the ceiling, so supervised or pooled domain weighting
should not be moved into placement. This is not a "condition-only policy" to
hard-code — MG (drug) and Long QT (measurement) show domain relevance is real —
but it is disease-specific and small, and better served by richer information
than by weighting tuning.

## Case-finding here is information-limited, not weighting-limited

Absolute utility is bimodal and diagnostic. Diseases with distinctive
condition-code signatures are genuinely findable (P@10%-recall: EDS 0.52, SLE
0.40, scleroderma 0.30, congenital heart disease 0.28, thoracic aortic aneurysm
0.22, sarcoidosis 0.14). Diseases whose distinguishing evidence lives in labs,
biopsy, autoantibodies or imaging are near-useless from cond+drug+obs
(vasculitides GPA/MPA/Takayasu/temporal arteritis, most neuro, Behçet, Marfan,
GBS all P@10% < 0.05). The single case where observation earned weight — Long
QT's QT interval — is a *value/measurement* signal.

This completes the through-line: condition scoring exhausted (0062) → multidomain
added but small (0071–0074) → domain-weighting flat *at scale* and pooling
not worth it (0076). The model keeps running out of **information**, not
cleverness.

**Implication / decision.** Stop tuning domain combination (including pooling).
Keep placement's fixed condition-primary combination. The next lever is
**information — value-aware measurement (labs)**, the direction 0062 identified
and the one place observation earned its weight here (Long QT). Anchor expansion
did its job as a scale test (the engine held at K=230; topics recovered textbook
disease signatures — MG/pyridostigmine, Behçet/colchicine, cardiomyopathy/HF
meds) and as the diagnostic that closes the domain-weighting/pooling question.

**Setting context.** Exp 0076, `rare_priority` (ADR 0039), condition V=5000, drug
V=1274, observation V=1500, K=230, seed 42. Readout: 5 repeats × 5 outer × 4
inner nested stratified CV over the held-out set, α→∞ LR, fold-local
backgrounds/scales, nonnegative 3-domain simplex (grid 0.05), tie-collapsing AP.
See ADR 0039 (anchor selection), insight 0075 (rare6 ceiling), insight 0062
(information constraint).
