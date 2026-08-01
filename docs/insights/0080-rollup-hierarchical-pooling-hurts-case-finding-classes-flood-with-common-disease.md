# 0080 — Roll-up hierarchical pooling hurts case-finding: SNOMED class nodes flood with common diseases, dragging rare anchors toward common-disease means

**Date:** 2026-08-01
**Topic:** hierarchy | pooling | rollup | case-finding | multidomain | decision
**Status:** Confirmed on exp 0080 (random init) AND exp 0082 (spectral init) —
the under-fit confound is now CLEARED: a well-conditioned spectral fit (5/424
starved, 2/192 dead) reproduces the collapse exactly (condition macro 0.006),
so the collapse is structural, not degenerate fitting. Roll-up-off (exp 0081)
still pending as the roll-up-vs-hierarchy isolator.

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

## exp 0082 update — the fit-quality confound is cleared (spectral init)

Exp 0082 reran the identical hierarchy+roll-up layout with **spectral init**. The
fit is now well-conditioned — **5/424 starved topics** (was 81) and **2/192 dead
nodes** (was 22) — yet case-finding **collapsed identically**: condition macro AP
**0.006** (byte-for-byte the random-init number), fixed 0.006, max:scaled 0.005,
vs flat ~0.020. The per-anchor pattern is unchanged: distinctive/large anchors
hold (Congenital heart 0.127, SLE 0.112, Sarcoidosis 0.066, EDS 0.059) while the
mid-tier neuro anchors sharing a flooded class collapse (MS 0.017, MG 0.004, CIDP
0.003, ALS 0.011). The spectral node topics confirm the flooding verbatim:
*Disorder of nervous system* = type-2 diabetes / diabetic neuropathy;
*Disorder of connective tissue* = fracture / low back pain / OA;
*Disorder of cardiovascular system* = SVT / CABG / arrhythmia.

**Conclusion: the collapse is structural, not under-fitting.** Random init and
spectral init give the *same* macro (0.006), so init is proven irrelevant here —
which also means downstream isolators (0081) need not spend spectral time.

## Caveats

- **Fit quality is ruled out (was the main caveat, now closed).** The original
  81/424 starved / 22/192 dead under random init is gone under spectral (5/2), and
  the AP is unchanged. So the collapse is not a degenerate fit.
- **Not yet isolated from roll-up.** This tested hierarchy **with** roll-up. The
  hierarchy **without** roll-up (classes pool only the anchor patients — a
  rare-flavored class mean, not flooded) is the clean isolator and is untested. It
  may help, be neutral, or reveal dead class nodes (if the gating does not flow
  anchor mass up to ancestors) — exp 0081.
- **The conditional use-case is separate.** The colleague's "rank within class"
  readout uses the hierarchy as a *scoring* structure at evaluation time; it does
  not require fit-time pooling and is unaffected by this negative result.

## Decision — roll-up hierarchical pooling hurts rare-disease case-finding (fit-quality ruled out)

The criterion set in the prior revision — *"only if the fit is well-conditioned
AND the hierarchy still underperforms flat do we conclude pooling is closed"* — is
now **met** by exp 0082: a clean spectral fit (5/424 starved) still gives condition
macro 0.006 vs flat 0.020. Roll-up hierarchical pooling **structurally
underperforms** flat for rare-disease case-finding. The mechanism is the flooding
above: "all patients with a disorder of X" is dominated by common X disease, so the
class is a bad pooling prior for rare X.

**Two hypotheses remain open** (0082 held tpn=2, so it only ruled out *global*
under-fitting, not these):

1. **Roll-up flooding is the specific villain** (not the hierarchy itself). Test:
   **exp 0081** (`rollup_attestation: false`) — classes then pool only the
   anchor-routed patients (a rare-flavored class mean, not flooded). Random init is
   fine (0080==0082 proved init-independence). **Run next — cheapest cut.**
   - If 0081 recovers toward flat → drop roll-up, keep the hierarchy (it's still
     wanted as the *eval-time* scoring structure for the within-class readout).
   - If 0081 stays low or classes go dead → hierarchy pooling doesn't help
     case-finding regardless of roll-up.
2. **Class under-capacity** — tpn=2 can't carve rare sub-populations (MS) out from
   the common center (diabetes) inside a class block. Test only if 0081 is also
   low: raise class-node capacity (data-driven fixed per-node block sizes, not
   per-node stick-breaking — see the HDP-rejection discussion). This is the last
   lever before closing fit-time pooling and using the hierarchy purely as an
   eval-time scoring structure.

**What is decided now:** do not ship roll-up hierarchical pooling for case-finding.
The hierarchy as an *eval-time* scoring structure (the colleague's "rank within
class" use-case) is untouched by this and remains the higher-value path.

**Setting context.** Exp 0080, rare_priority, cond+drug+measurement, SNOMED
hierarchy (restrict-under 4274025, max_class_fraction 0.6, 60 classes) + roll-up,
random init, K=424, seed 42. Compared to flat random-init 0080 (condition macro
0.020). Fast `--fixed-only` readout. See insight 0079 (measurement specialist),
0076 (pooling not worth it), 0077 (measurement survey).
