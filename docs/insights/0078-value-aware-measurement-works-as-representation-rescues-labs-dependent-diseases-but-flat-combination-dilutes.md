# 0078 — Value-aware measurement works as a representation and rescues labs-dependent diseases, but flat equal-weight domain combination dilutes it; the combination rule now matters in a way it did not for cond+drug+obs

**Date:** 2026-07-31
**Topic:** measurement | representation | multidomain | combination | case-finding | decision
**Status:** Confirmed on exp 0078 (rare_priority, cond+drug+measurement)

Exp 0078 is the first value-aware **measurement** fit (measurement arc Step 2;
insight 0077). It holds exp 0076 fixed except the domain set — **drops
observation, adds measurement**, keeps drug — using the range→coded→presence
cascade emitted as per-document binary synthetic tokens
(`charmpheno.omop.measurement_tokens`). K=230, 41 anchors, 0 dead nodes, 0
starved topics. Readout: the fast parameter-free `--fixed-only` mode (per-domain
+ fixed-inclusive AP; the supervised search is closed per 0076).

## The representation works (the primary goal of Step 2)

Value-aware tokens surface correctly and semantically in the fitted topics:
diabetes background carries **Glucose [high]** and **Hemoglobin A1c [high]**;
peripartum cardiomyopathy carries **Hemoglobin [low]** (anemia); Moyamoya carries
**Bilirubin [high] / Bicarbonate [low]**; Behçet carries **ESR / CRP / RDW**
states; Guillain-Barré surfaces **CSF color/volume** studies. The cascade's state
suffixes are meaningful, and the health is perfect. The bare-token representation
problem flagged in 0071/0076 is fixed.

## Measurement is a *specialist* domain — it rescues labs/imaging-dependent diseases

Per-disease, measurement-alone genuinely finds diseases that were near-useless
from cond+drug in 0076, several of them explicitly called out there:

| anchor | condition AP | measurement AP | fixed-inclusive AP |
|---|---|---|---|
| Guillain-Barré | 0.003 | **0.027** | 0.015 |
| Marfan | 0.001 | **0.016** | 0.013 |
| Osler telangiectasia | 0.007 | **0.020** | 0.009 |
| Scleroderma | 0.088 | 0.053 | 0.092 |
| Sarcoidosis | 0.042 | 0.042 | **0.054** |
| SLE | 0.112 | 0.081 | **0.125** |
| Long QT (control) | 0.047 | 0.036 | 0.045 |

GBS and Marfan are rescued almost entirely by measurement (condition is ~0). SLE
and sarcoidosis are lifted above condition-alone. This validates the arc thesis
directly: **the missing information for labs-dependent rare diseases is in the
measurement values**, and the value-aware representation captures it.

## But flat equal-weight combination dilutes it (the catch)

Macro median AP: **condition-alone 0.020, drug-alone 0.005, measurement-alone
0.007, fixed-inclusive (equal-weight cond+drug+meas) 0.017.** The fixed inclusive
combination is *below condition-alone*: summing the weak-on-average drug and
measurement domains at equal weight helps the labs-dependent diseases but hurts
the condition-dominant ones (thoracic aortic aneurysm 0.094→0.046, EDS
0.041→0.024, temporal arteritis 0.027→0.017). Measurement is strong for a subset
and noise for the rest, so a flat sum pays the noise tax on every disease.

This is the key structural finding, and it **partially reopens the combination
question insight 0076 closed.** 0076 concluded supervised/pooled per-disease
domain weighting was not worth building — but that was for cond+drug+**obs**,
where every extra domain was uniformly weak, so a flat combination was ~as good
as any weighting. Measurement is different: it carries genuinely *complementary,
disease-specific* signal. When a domain helps a distinct subset and hurts the
rest, the flat equal-weight rule is demonstrably suboptimal, and the combination
rule starts to matter — not necessarily via the full supervised pooling 0076
rejected, but at minimum via a non-diluting combine.

## Caveats / what is not yet established

- **Cross-experiment comparison is confounded.** 0076's reported cond+drug macro
  was 0.032 via the *old* readout; this readout reports condition-alone 0.020 on a
  *different joint fit* (measurement is in the shared-θ inference and reshapes
  condition emissions). Whether adding high-volume measurement tokens degraded the
  condition domain is not yet known. **Next: run `--fixed-only` on exp 0076** for
  per-domain + fixed numbers on the identical scale — that isolates (a) condition
  degradation, (b) measurement vs observation head-to-head.
- Much measurement mass is still `[measured]` (presence) where labs lack ranges
  (survey 0077: ~40% range coverage), so a fraction of the value signal is not yet
  captured; binned-numeric (representation option 4, deferred) is the escalation.

## Implication / decision

1. **Representation: ship it.** Value-aware measurement is validated; keep it.
2. **Combination: the lever is now the combine rule, not the representation.** A
   flat equal-weight sum leaves measurement's complementary signal on the table.
   Candidate parameter-free fix: **max across per-domain LR scores** instead of
   sum (a strong measurement signal wins for GBS without diluting condition for
   thoracic aneurysm) — testable cheaply on the same fit. If that is insufficient,
   revisit a *light* per-disease selector, whose ceiling may now be larger than
   0076 found precisely because measurement is complementary.
3. **Confirm on 0076** with the fast readout before drawing the cross-experiment
   conclusion.

**Setting context.** Exp 0078, rare_priority (ADR 0039), condition V=5000, drug
V=1274, measurement V=2500, K=230, seed 42, full-batch, spectral scalable init.
Readout: `--fixed-only`, 5 repeats × 5 outer folds, fold-local backgrounds, α→∞
LR, tie-collapsing AP. See insight 0077 (measurement survey), 0076 (domain
weighting closed for cond+drug+obs), 0062 (information constraint).
