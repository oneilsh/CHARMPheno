# 0080 — Roll-up hierarchical pooling hurts case-finding: SNOMED class nodes flood with common diseases, dragging rare anchors toward common-disease means

**Date:** 2026-08-01
**Topic:** hierarchy | pooling | rollup | case-finding | multidomain | decision
**Status:** Confirmed on exp 0080 (SNOMED class hierarchy + roll-up attestation)

Exp 0080 is the first fit with a real hierarchy above the anchors: the compact
SNOMED class hierarchy (`root → Disorder-class → anchor → descendants`, 60 classes
via restrict-under-Disease) **plus roll-up attestation** — each patient's
condition codes routed up to the nearest DAG node via `concept_ancestor`, so class
nodes gather their full descendant population. Random init, K=424 (40 bg + 192
nodes × 2).

## The mechanism worked — and that's the problem

The class nodes came alive with genuine, interpretable pooled topics:
`Disorder of immune function` = monoclonal gammopathy / cytopenias
(leukocytes/platelets/neutrophils [low]); `Hereditary disease` = thrombophilia /
hemochromatosis / Ig panel; `Disorder of cardiovascular system` = hypertension /
T2DM / obesity / BP[high] / BMI[high]; `Disorder of nervous system` = **low back
pain / lumbar radiculopathy / spinal stenosis**.

That last one is the tell. Roll-up gives a class node **every** patient with any
descendant disorder — and common diseases outnumber the rare anchors by orders of
magnitude. So "Disorder of nervous system" is dominated by back pain, not MS;
"Disorder of cardiovascular system" is hypertension, not the rare cardiomyopathies.
The class topics are common-disease topics.

## Result: macro case-finding AP collapsed ~3×

Macro median AP (fast readout, vs the flat random-init 0080 baseline in
parentheses): condition **0.006 (was 0.020)**, drug 0.001, measurement 0.003,
fixed 0.006, max:scaled 0.005. A 3× drop.

It is not uniform — it is the **mid-tier collapsing** while distinctive anchors
hold:
- **Held / improved:** Congenital heart 0.127, SLE fixed 0.142 (was 0.125),
  Sarcoidosis fixed 0.068 (was 0.054), EDS 0.059; a few measurement-driven ones
  jumped (Marfan measurement **0.055** was 0.016; GBS measurement 0.029).
- **Collapsed:** Multiple sclerosis 0.070 → 0.023, myasthenia gravis 0.023 →
  0.003, CIDP 0.020 → 0.003, ALS 0.032 → 0.016.

The collapsed anchors are exactly those whose class got flooded with a huge common
population (MS/MG/CIDP under "Disorder of nervous system" ← back pain). Pooling
drags the rare anchor toward the class mean, and the class mean is the common
disease. Distinctive/large anchors (congenital heart, SLE) resist because their
own signal dominates their local structure.

## Interpretation

**"All patients with a disorder of X" is dominated by common X disease, so the
class is a bad pooling prior for rare X.** The user's intuition (a class should
capture its whole population) is mechanically right and gives clean class
phenotypes — but for *rare-disease case-finding* it pools the target toward the
wrong center. This is a direct, mechanistic confirmation of insight 0076's
verdict (shared/pooled structure is not worth it), now demonstrated rather than
inferred.

## Caveats

- **Confounded with fit quality.** K doubled (230 → 424) and random init left
  81/424 starved topics + 22/192 dead nodes — a worse-conditioned fit that drags
  everything down somewhat. But the class-topic *content* (back pain, hypertension)
  and the class-specific mid-tier collapse show the flooding is a real structural
  effect, not just init noise.
- **Not yet isolated from roll-up.** This tested hierarchy **with** roll-up. The
  hierarchy **without** roll-up (classes pool only the anchor patients — a
  rare-flavored class mean, not flooded) is the clean isolator and is untested. It
  may help, be neutral, or reveal dead class nodes (if the gating does not flow
  anchor mass up to ancestors) — exp 0081.
- **The conditional use-case is separate.** The colleague's "rank within class"
  readout uses the hierarchy as a *scoring* structure at evaluation time; it does
  not require fit-time pooling and is unaffected by this negative result.

## Decision

Roll-up hierarchical pooling is **net-negative for aggregate case-finding** — do
not ship it. Next, run exp 0081 (same hierarchy, `rollup_attestation: false`) to
isolate whether pooling over *anchors only* is neutral/helpful or the class nodes
go dead. If 0081 is also flat-or-worse, the hierarchical-pooling direction is
closed for case-finding (0076 confirmed by construction), and the hierarchy's
remaining value is the conditional within-class readout, not the generative fit.

**Setting context.** Exp 0080, rare_priority, cond+drug+measurement, SNOMED
hierarchy (restrict-under 4274025, max_class_fraction 0.6, 60 classes) + roll-up,
random init, K=424, seed 42. Compared to flat random-init 0080 (condition macro
0.020). Fast `--fixed-only` readout. See insight 0079 (measurement specialist),
0076 (pooling not worth it), 0077 (measurement survey).
