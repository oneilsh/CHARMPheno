# Case-finding vs. diagnosis, the right metrics, and value-of-information

**Date:** 2026-08-14 · **Audience:** whoever is driving the conditional-diagnosis /
closure-mask arc (exp 0078 → 0079) and the eventual VOI capability. Grounded in the 0078
conditional-sharpening readout. Companion to the rare-disease-diagnosis lit review + the
new bibliography sections (Rare-disease diagnosis & case-finding; Diagnostic utility & VOI).

## 1. Case-finding and diagnosis are the same model, conditioned differently

Two tasks, two conditional distributions over the *same* fitted representation:

- **Case-finding / screening:** `P(disease d | x)` at **population** prior. Base rate
  π_d ≈ 0.02–0.05. "Is this one of the rare cases among *everyone*?"
- **Diagnosis / sharpening:** `P(d | x, d ∈ C)` where C is a candidate set (e.g. the children
  of a parent node — "given a connective-tissue disorder, which one?"). Base rate is now the
  *within-C* prevalence, often 0.2–0.5.

They factor: `P(d | x) = P(d | x, x∈C) · P(x∈C | x)`. Diagnosis is screening with the
population-detection step factored out. This is *why* one representation gives detection
AP ≈ 0.147 and conditional AP ≈ 0.27–0.58 — the base-rate tax lifts when you condition. The
5–29× "lift over marginal" in the 0078 readout is mostly this factorization, not new skill
(see §2). That's still the whole point: **the base-rate tax is the thing making case-finding
look mediocre, and conditioning removes it.** But read the lift honestly.

## 2. Read the 0078 conditional numbers honestly — AUC is the sober column

The `cond_AP` and `lift` columns conflate two effects: (a) the base rate mechanically rising
when you condition (not model skill), and (b) the model's actual within-parent discrimination
(the skill). **`cond_AUC` isolates (b)** — it's prevalence-independent — and it tells a much
soberer story than `cond_AP`:

| depth | edges | cond_AUC | cond_AP | marg_AP | what it really says |
|---|---|---|---|---|---|
| 0 | 6 | **0.626** | 0.274 | 0.051 | modest within-parent discrimination; the 5× AP is ~3× base-rate-rise + ~1.6× skill |
| 1 | 7 | **0.622** | 0.219 | 0.014 | same — AUC ~0.62 is weak-to-moderate |
| 2 | 1 | **0.575** | 0.577 | 0.020 | **the cautionary case**: cond_AP 0.58 looks huge, AUC 0.575 is near-chance — the "29× lift" is almost entirely base rate |

So: **within-parent discrimination is AUC ≈ 0.58–0.63 — real but modest.** The impressive
`cond_AP` numbers are mostly the base-rate gift. Don't headline the 5–29× lift without the AUC
next to it, or the readout oversells.

**The multiclass top-1 numbers need their baselines too.** SLE→{glomerulonephritis,
drug-induced} top1 = 1.000 (n=99), Sarcoidosis 0.857, Amyloidosis 0.75–0.79 — but these are
**2-child** parents where random = 0.50 and majority-class can be 0.6–0.7. The forest root
(6 children) top1 = 0.509 vs random ≈ 0.17. All are above chance, but the depth-averaged
`cond_AUC 0.62` hides big per-parent variation: SLE at top1 = 1.0 is genuinely easy, while
other depth-1 edges must be near-chance to average to 0.62. **Report per-parent, not
depth-averaged**, and show the majority-class baseline beside every top-1.

**What to add to the readout** (cheap, decisive):
1. Per-parent **within-parent prevalence** (base rate) of each child → so `cond_AP`/top-1 are
   judged against the *within-parent* random baseline, not the marginal (population) one.
2. Per-parent **majority-class rate** beside top-1, and **balanced accuracy** or **macro-F1**
   (or **MRR**, §5) for parents with >2 children.
3. Keep `cond_AUC` as the headline discrimination number; demote `lift-over-marginal` to
   "context."
4. **Calibration** of `P(child|parent)` (reliability / ECE) — needed for a real diagnostic aid
   *and* mandatory for VOI (§4).

The A/B itself (supervised vs unsup) you already read correctly: conditional AP 0.268 vs 0.281
(Δ−0.013), top1 0.778 vs 0.787 (Δ−0.008) — **supervision does not help sharpening**, and the
hierarchy-aligned λ-specialization is a property of the gated representation, present with or
without the label. Good catch on the over-attribution. §3 says why this is expected.

## 3. Why 0079 (closure-mask) is exactly the right next test

The dichotomy is an **objective↔task mismatch**, and it's clean:

- `label_mask_mode = full` trains every node **against background** → a *one-vs-rest detection*
  objective. It optimizes `P(d | x)` vs the population. It never sees the sibling contrast, and
  by pulling every subtype of a parent toward "parent vs background" it can **blur** the
  within-parent distinctions subtyping needs. → helps detection (+0.025 AP), not sharpening.
- `label_mask_mode = closure` trains each node **against its siblings** → a *within-parent
  multinomial* (softmax) objective. It optimizes exactly `P(d | x, d∈C)` — the conditional
  metric. This is "train the objective you'll be evaluated on."

**Prediction for 0079:** the sign flips — closure-mask improves `cond_AUC`/`cond_AP`/top-1 over
both the full-mask supervised arm and the unsup twin, likely at some cost to detection AP
(background docs contribute nothing under closure). A clean trade — **closure buys sharpening,
full buys detection** — makes `label_mask_mode` the *knob that selects the clinical task*
(screen de-novo vs. sharpen a referral). That's the sharpest possible framing of the arc.

Two watch-items: (1) closure with tiny within-parent N (amyloid subtypes, n=75 over 2+
children) is a `p≫n` softmax — expect variance; report the per-parent N. (2) closure-trained β
are **sibling-contrastive by construction**, which is precisely what VOI wants (§4) — so 0079
is also the model that makes the VOI weights meaningful, not just the sharpening metric.

## 4. Value of information — the capability that falls out next, and that nobody else has

Once you're doing conditional diagnosis, the natural next question is the clinician's:
**"given this patient is ambiguous between children {A, B, …} of a parent, what unobserved
feature (code / lab / measurement) would most sharpen the call?"** This falls out of the
generative β and is a lightweight post-hoc computation — no retraining.

For candidate children with phenotype distributions β_A, β_B over the vocabulary, observing
feature v contributes a per-feature log-odds (naive-Bayes LLR):

> **w_v(A:B) = log( β_A(v) / β_B(v) )**

The *most informative not-yet-observed* feature is the one maximizing **expected information
gain** under the patient's current posterior:

> **EIG(v) = H(p) − 𝔼_{x_v ~ P(x_v | patient)}[ H(p | x_v) ]**

where p is the current posterior over the candidate set C (from the conditional model) and
`P(x_v | patient)` is the model's own predictive for observing v. "Informed by patient data"
= both p and the predictive are conditioned on the patient's current θ and observed codes.
Rank features by EIG → the ranked "next-best-test" list.

Why this is the differentiator: **discriminative case-finders (RarePT, PheNet) and
semantic-similarity rankers (Phenomizer) can't do this** — they don't carry calibrated
`P(feature | disease)` distributions. Your β *are* exactly those distributions. Even the LLM
diagnosis wave (RareBench, and per EJHG 2026 still trailing Exomiser on R@1) has no principled
VOI. This is open ground with a clean generative story. (Grounding in the bibliography's new
"Diagnostic utility & VOI" section: Nelson 2005 for the EIG-is-the-right-norm argument;
Westover 2012 for MI-of-a-test; Vickers 2006 for the net-benefit view of acting on it.)

**Two prerequisites** VOI makes non-negotiable, both already on the table: (1) **calibrated**
`P(child|parent)` (so H(p) is real, not just a ranking) — hence the calibration ask in §2; and
(2) **sibling-contrastive β** (so w_v discriminates the right thing) — hence closure-mask (§3).
The arc is self-consistent: closure-mask → calibrated conditional posteriors → VOI.

**How to evaluate VOI** (when you build it): simulate feature acquisition on held-out patients
— hide their codes, reveal them in EIG-ranked order vs. random vs. information-content order,
and measure how fast the conditional posterior concentrates (entropy reduction / steps-to-
correct-top-1). "EIG beats random and IC ordering" is the claim.

## 5. Metric taxonomy — pick the metric that matches the deployment

| Regime | Question | Right metric | Wrong/misleading here |
|---|---|---|---|
| **Screening / case-finding** | "find the rare cases in the population" | **AUPRC / detection-AP, P@R, R@FDR** (imbalance-honest) | plain accuracy; AUROC (base-rate-blind) |
| **Conditional diagnosis** | "which disease, given a candidate set" | **R@k, MRR, per-parent AUC + balanced-acc, top-1 vs majority baseline** | marginal P@R; `cond_AP` without a within-parent baseline; depth-averaged AUC |
| **Utility / deployment** | "does acting on this help" | **net benefit / decision-curve analysis** (Vickers) at the operating point | F-score (ranks inconsistently with utility); AUROC |
| **Value of information** | "what test next" | **expected information gain / posterior-entropy reduction**; sim-acquisition curves vs random/IC | — |
| **cross-cutting** | "are the probabilities usable" | **calibration (reliability, ECE)** | ranking metrics alone |

Notes: **MRR within the candidate set** is the compact diagnostic-utility scalar and makes you
**directly comparable to PhenoBrain / RareBench** (they report R@k on the same shape of task) —
worth adding to the conditional readout. AUROC is the *least* appropriate screening metric
(base-rate-blind) but the *right* discrimination summary *within* a conditioned set (where
prevalence is balanced-ish) — which is exactly why `cond_AUC` is the honest column in §2.

## 6. The through-line

The mask mode is the task selector, and the metric must match it. Reported on the screening
metric at population prevalence, this representation looks mediocre (det AP 0.147, ~3.5× lift,
the information-limited ceiling). Reported on the **conditional** metric it's a usable
diagnostic aid (SLE subtyping top1 = 1.0; sarcoid 0.86) — *and* that's the framing where the
base-rate tax lifts, where the confusable-sibling structure is the point rather than the
ceiling, and where **VOI gives you a capability no case-finder, semantic ranker, or LLM in the
landscape has.** The honest caveat from §2 rides along: within-parent *discrimination* is still
only AUC ≈ 0.6, so the win is "conditioning + VOI on an interpretable generative model," not
"we discriminate siblings better than anyone." 0079 is the test of whether the right objective
lifts that 0.6.
