# 0111 scouting: label-window and depth, off the 0110 saved fit — 2026-09-02

Cheap analyses (no re-fit, no re-assembly; persisted heads + E4 sidecar) to decide
what exp 0111 should be. Tool: `diag-conversion-analysis --by-depth --horizon-eval`
(commit ae8db9a). Egress: pooled/bucket figures only.

## Finding 1 — the eval-side HORIZON SWEEP REFUTES window-widening

1y-fit scores vs longer-horizon incident labels, shared node set (1,622 nodes,
scoreable at all horizons), R2.2 discipline:

| horizon | shared AUC | shared AP |
|---|---|---|
| 365d  | 0.6003 | 0.0199 |
| 730d  | 0.5899 | 0.0289 |
| 1095d | 0.5857 | 0.0369 |

Pre-registered rule: rising AUC ⇒ the 1y label under-credits the model ⇒ widen the
training window. **AUC FALLS** (monotone), so the rule is NOT triggered — a wider
window slightly DILUTES discrimination. The AP rise is a base-rate artifact
(widening W raises prevalence; AP climbs with prevalence at fixed ranking), not
skill. Upside of a window re-fit is bounded tiny: 1y features already rank 3y
converters at 0.586, ~0.015 below the 1y number. **Decision: 0111 is NOT a
longer-window re-fit.**

Reconciliation with the decile gradient (which GROWS with horizon, +0.014→+0.032):
not a contradiction. The decile spread grows because the model's top-scored
negatives keep converting at longer horizons (ranking of eventual cases holds up);
the overall within-W AUC falls because the added year-2/3 converters are harder to
rank from year-earlier features. Case-finding validation (deciles) survives; the
"train wider for better AUC" story does not.

## Finding 2 — DEPTH: case-finding concentrates shallow, present everywhere

Decile case-finding table by DAG depth (banded shallow 1-3 / mid 4-6 / deep 7+;
per-depth too thin in the deep tail):

| bucket | nodes | 1y conv rate | 1y decile spread (top−bottom) |
|---|---|---|---|
| shallow (1-3) | 36 | 0.0493 | +0.0600 |
| mid (4-6) | 1126 | 0.0090 | +0.0138 |
| deep (7+) | 1373 | 0.0044 | +0.0071 |

Signal is strongest at shallow categories and attenuates monotonically with depth,
but is present (and monotone across deciles) at EVERY depth. Depth×horizon macro AUC
falls with horizon in every bucket (shallow 0.626→0.610, deep 0.584→0.569). This is
a reporting/deployment characterization (case-finding claims strongest for broad
categories), not a new-experiment motivator.

## What this redirects 0111 toward

The incident readout dropped **923 nodes as "small" (<20 incident positives)** — the
incident cohort is STARVED because a random population index catches only a small
fraction of cases in their pre-onset year. That is the quantified bottleneck. It is
exactly what **episode-anchored index sampling (the original 0111 / plan §E5)**
fixes: index every case just before its onset episode, capturing every case as
incident and multiplying the thin incident-positive counts. The two cheap
alternatives (window, depth) turned out to be a negative and a characterization;
episodes remain the real lever, now with a sharp motivation.

**Recommendation: 0111 = episode-anchored sampling.** Longer-window labeling is
demoted to a possible small validation/reporting side-axis (the sidecar already
supports longer-horizon deciles with no re-fit); the window is not a training-target
change. Survival/time-to-event labeling stays the named long-term endgame.
