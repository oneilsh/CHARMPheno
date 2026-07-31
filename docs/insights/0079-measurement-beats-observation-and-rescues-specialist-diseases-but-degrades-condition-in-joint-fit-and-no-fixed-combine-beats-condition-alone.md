# 0079 — On the identical fast readout, measurement beats observation as the third domain and rescues ~6 labs-dependent diseases, but it degrades the condition domain in the joint fit, and no fixed combination rule (sum, max, max-scaled) beats condition-alone at macro

**Date:** 2026-07-31
**Topic:** measurement | observation | multidomain | combination | modality-interference | decision
**Status:** Confirmed — exp 0076 (cond+drug+obs) vs exp 0078 (cond+drug+measurement), both run through the fast `--fixed-only` readout on the identical scale

The `--fixed-only` readout run on **both** 0076 and 0078 gives the apples-to-apples
comparison insight 0078 was missing (0078's earlier "0.017 vs 0076's 0.032" was
confounded — different readout, different fit). Macro median AP, same 5×5 folds,
same α→∞ LR, fold-local backgrounds:

| macro | condition | drug | 3rd (obs / meas) | fixed:sum | max:raw | max:scaled |
|---|---|---|---|---|---|---|
| **0076** cond+drug+**obs** | **0.024** | 0.005 | 0.003 | 0.013 | 0.012 | 0.007 |
| **0078** cond+drug+**meas** | **0.020** | 0.005 | 0.007 | 0.017 | 0.017 | 0.017 |

## 1. Measurement is the better third domain than observation

Measurement-alone (0.007) more than doubles observation-alone (0.003), and every
0078 combination (0.017) beats its 0076 counterpart (0.013 / 0.007). And it
uniquely **rescues labs-dependent diseases condition cannot find** — the "top:
measurement" rows in 0078: Guillain-Barré (condition 0.003 → measurement 0.027 →
max:scaled 0.024), Marfan (0.001 → 0.016 → 0.015), Osler telangiectasia (0.007 →
0.020), sarcoidosis, CIDP, familial hypercholesterolemia. Observation rescued
none of these. The arc premise — measurement carries the missing information for
labs-dependent rare disease — is confirmed.

## 2. But measurement degrades the condition domain in the shared-θ fit

Condition-**alone** macro fell **0.024 → 0.020** when the third domain was swapped
obs → measurement (same seed, same everything else). Per-disease the condition
column drops across most anchors — EDS 0.102 → 0.041, Long QT 0.067 → 0.047, SLE
0.136 → 0.112, scleroderma 0.112 → 0.088, thoracic aneurysm 0.099 → 0.094. This
is **modality interference**: measurement is high-coverage (labs are ordered for
nearly everyone; its near-universal `[measured]` presence tokens are the volume
analogue of observation's PPI problem, insight 0071/0077), so it pulls the shared
patient-topic mixture θ away from condition structure. This is the single most
actionable cost — measurement is helping at readout while quietly hurting the
domain that does most of the work.

## 3. No fixed combination beats condition-alone at macro — in either fit

condition-alone is the macro champion both times (0.024, then 0.020); sum, max,
and max-scaled all sit below it. The extra domains, combined by any fixed rule,
dilute the condition-dominant majority more than they add. This extends the
0062/0076 through-line: **condition scoring is the workhorse and the aggregate
ceiling is information-bound**, not combination-bound.

## 4. No single fixed rule wins — the diseases split into "additive" vs "specialist"

`max:scaled` recovers specialist-domain diseases that the equal-weight sum dilutes
(EDS 0.024 → 0.037, ALS 0.025 → 0.033, GBS 0.015 → 0.024, Marfan 0.013 → 0.015),
but *loses* diseases where two domains genuinely stack additively (SLE 0.125 →
0.100, MS 0.071 → 0.045, thoracic aneurysm 0.046 → 0.030). Net: sum and max-scaled
tie at macro (both 0.017). Some diseases want ADD (condition+measurement both
contribute — SLE, MS, sarcoidosis), others want MAX (one specialist domain —
GBS, Marfan, EDS). The per-anchor `top` attribution makes this legible
(measurement-dominant: sarcoidosis/GBS/CIDP/Marfan/FH/Osler; drug-dominant:
MG/POTS/Takayasu; condition-dominant: the rest). No parameter-free rule captures
both regimes — which is exactly the per-disease selection the supervised readout
targeted, now with real specialist signal to select (unlike the uniformly-weak
obs case 0076 rejected).

## Implication / decision

Measurement delivered on its promise **for the labs-dependent diseases** (a
genuine, keepable per-disease result: GBS/Marfan/EDS/Osler/CIDP/sarcoidosis go
from near-zero to findable via `max:scaled`), but at the **aggregate** it neither
beats condition-alone nor lifts the macro ceiling, and it dents condition via
shared-θ interference. Two things follow, in priority order:

1. **Protect the condition domain from measurement interference — the decisive
   next test.** Per-modality tempering exists in the driver (`--omega`): down-weight
   measurement's contribution to θ (e.g. `omega: 1.0,1.0,0.4–0.5`) and re-run the
   fast readout. If condition-alone recovers toward 0.024 while measurement keeps
   its specialist rescues, a combine could finally clear condition-alone. The
   targeted alternative is pruning the near-universal `[measured]` presence tokens
   (the measurement analogue of the PPI strip). This is the one cheap lever left.
2. **If tempering does not clear condition-alone**, the honest read is that
   aggregate case-finding is information-limited (0062) and measurement's value is
   real but disease-specific — ship it as a **specialist channel** (use it, via
   max-scaled or per-disease selection, only for the diseases where it dominates)
   rather than as a domain that improves the average, and close the "make the
   average better" line of the arc.

**Setting context.** Exp 0076 (rare_priority, cond+drug+obs, obs PPI-stripped) and
exp 0078 (cond+drug+measurement, value-aware), both K=230, seed 42, full-batch,
spectral scalable, ADR 0039 anchors. Readout: `--fixed-only`, 5 repeats × 5 outer,
fold-local backgrounds, α→∞ LR, counts <20 suppressed. See insight 0078
(representation validated), 0077 (survey), 0071 (observation drag), 0062
(information constraint).
