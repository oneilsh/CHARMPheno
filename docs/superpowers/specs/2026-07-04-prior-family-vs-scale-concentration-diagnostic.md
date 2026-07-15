# Per-document concentration in a gated logistic-normal topic model: diagnostic evidence and a solution space

**For a reviewer.** This is a follow-up to the earlier open-problem brief (a gated
Structural Topic Model whose generative tool produces over-diffuse synthetic documents; the
concentration lives in the topic covariance Σ's diagonal scale, which the fit pins to a
unit diagonal for stability and thereby discards). It reports a new diagnostic that
separates two competing explanations, then lays out the solution space — including options
not yet on the table — and asks for a critique.
---

## 1. Recap of the problem (self-contained)

- **Model.** Gated STM: η_d ~ Normal(Γᵀx_d, Σ), θ_d = softmax(η_d), with topics partitioned
  into a shared background block plus per-group foreground blocks (a document expresses
  background ∪ its own group only). Fit by distributed online variational inference; the
  per-document E-step is a MAP/Laplace estimate of η.
- **Why Σ is unit-diagonal.** A free per-topic variance runs away at fit time in the gated
  setting — a document-scarce minority topic's variance blows up (empirically to ~1e8).
  Every variance prior/anchor was falsified. The shipped fit pins Σ_ii ≡ 1 (Σ is a
  correlation matrix), which is stable but **discards the variance scale**.
- **Why that breaks generation.** The generative tool draws η ~ Normal(Γᵀx, Σ). At unit
  scale the draws are over-diffuse (a synthetic document spreads across ~15–20 topics; a
  short seed lands on the wrong topic). Generation wants Σ at the corpus's true η-scale.
- **The open question.** Recover a bounded, proper, non-unit **generative** scale in the
  gated setting, data-driven (no hand-set constant), without reintroducing the fitting
  runaway. A pooled scalar c (Σ_gen = c·R) estimated at export with β/R frozen is
  runaway-safe but under-corrects (Laplace bias); it converged to c = 3.67, which fixes the
  wrong-topic failure but is still soft.

The reviewer previously proposed **indirect inference** (calibrate the biased export
estimator by simulation). A concern was raised that this is **self-referential** to the
model. Before building it, we ran a diagnostic to answer the prior question: is the
over-diffuseness a **scale** deficit (fixable by any scale-recovery method) or a
**prior-family** deficit (the logistic-normal is inherently smearier than a Dirichlet/LDA,
in which case scale calibration chases a mirage)?

---

## 2. Diagnostic: inference bias of each prior family

**Method (uses the real fitted topics; needs no corpus).** Plant documents at a *known*
true concentration — η_true ~ Normal(0, s·I), θ_true = softmax, s from diffuse (1) to peaked
(9) — sample the real mean length (44 tokens) from θ_true·β with the **real fitted β**, then
infer θ̂ back and measure how faithfully each prior recovers the planted concentration. STM
inference = MAP mode under Normal(0, c·I) across scales c; LDA inference = mean-field
variational under symmetric Dirichlet(α) across α. Metric: mean top-topic mass / mean
effective-number-of-topics over 300 documents. (Σ = c·I isolates the diagonal scale;
covariate mean 0.)

| inference | s=1 | s=3 | s=5 | s=7 | s=9 |
|---|---|---|---|---|---|
| **TRUE planted** | 0.11 / 38.2 | 0.26 / 18.6 | 0.35 / 12.2 | 0.44 / 8.5 | 0.50 / 6.7 |
| STM Σ = 1·I (unit — as fit) | 0.16 / 37.1 | 0.29 / **25.4** | 0.35 / **21.1** | 0.41 / **17.2** | 0.47 / **14.0** |
| STM Σ = 3·I | 0.21 / 23.1 | 0.32 / 15.7 | 0.39 / 12.0 | 0.46 / 9.8 | 0.53 / 7.9 |
| STM Σ = 5·I | 0.22 / 19.3 | 0.34 / 12.4 | 0.43 / 9.4 | 0.46 / 8.4 | 0.52 / 6.9 |
| STM Σ = 7·I | 0.22 / 17.3 | 0.34 / 11.6 | 0.41 / 9.0 | 0.49 / 7.2 | 0.52 / 6.1 |
| LDA α = 0.02 | 0.25 / **9.6** | 0.36 / **7.0** | 0.44 / 5.6 | 0.50 / 4.7 | 0.54 / 4.3 |
| LDA α = 0.10 | 0.22 / 14.1 | 0.34 / 10.3 | 0.40 / 8.7 | 0.45 / 7.5 | 0.51 / 6.5 |
| LDA α = 0.50 | 0.14 / 34.3 | 0.21 / 28.5 | 0.26 / 25.1 | 0.30 / 22.5 | 0.34 / 20.4 |

**Findings.**
1. **Over-diffuseness is a SCALE problem, not a prior-family one.** STM at unit scale
   systematically *smears* (true ~6.7-effective document inferred as ~14), but at Σ ≈ 5–7 it
   *recovers the truth faithfully* across the range. The logistic-normal is not inherently
   smeary; at the right scale it is unbiased. Raising the scale is the correct fix. (c = 3.67
   sits between unit and the faithful band, hence partial de-smearing.)
2. **"LDA is peaky" is substantially a Dirichlet ARTIFACT.** Small α over-sharpens
   *everything*: α=0.02 reports ~5–10 topics even for a genuinely diffuse document (true ~38).
   The Dirichlet manufactures the peak; α is just a concentration dial (0.5 smears, 0.02
   over-sharpens). An LDA fit looking peaky is weak evidence the documents are peaky.
3. **Both families are comparably capable once their concentration hyperparameter is set
   right.** The entire game is estimating that hyperparameter — STM's Σ scale, LDA's α.

**The operational asymmetry (why LDA "just works").** LDA estimates α **in-band, at fit
time, cheaply and stably** (α-optimization is a well-behaved corpus-level Newton/fixed-point
step). The logistic-normal's concentration lives in Σ's diagonal, which is **not stably
estimable at fit time** in the gated setting (per-topic free variances run away). So the gap
is not capability — it is that STM's concentration parameter has no stable in-band
estimator, while LDA's does.

**On the self-reference worry:** finding 1 largely dissolves it. Calibrating the STM scale
is not reproducing a prior-family smear (there is none to reproduce — STM is faithful at the
right scale); it is recovering a real, well-defined scale. The estimator is a
well-conditioned function of that scale, which is what makes any calibration sound.

---

## 3. Solution space

Framed by *where* the concentration is estimated and *what* it is matched to. Every option
below is data-driven (no hand-set constant) and runaway-safe unless noted; they are not
mutually exclusive.

### Class A — estimate the concentration IN-BAND (fit time), stably

**A1. A single global concentration scalar (softmax "temperature") — the direct
logistic-normal analog of α-optimization.** Keep the fit's unit-diagonal correlation R (all
of its stability), and estimate ONE global scale τ, Σ = τ²·R (equivalently θ = softmax(τ·η)
with η ~ Normal(μ, R)), by ML at fit time. Rationale: finding 3 says the whole game is the
concentration hyperparameter; LDA gets α stably because it is a **single well-identified
scalar**; the STM runaway (exp with a free diagonal) was **per-topic** — a document-scarce
topic drags its *own* variance to blow-up. A **single pooled** scalar cannot be dragged by
one topic (its residual is 1/K-weighted against every other topic's), so the runaway mode is
structurally absent — the same pooling argument that makes the export scalar safe, applied
in-band. This would recover concentration at fit time, with no export round-trip, no
Laplace-EM bias, and no borrowed number. **This is the option the diagnostic most directly
motivates and it has not been tried** (prior attempts estimated per-topic free variances or
a full covariance, never a single global scale with the correlation held unit-diagonal).
Risk: a global τ could in principle co-adapt with β during fitting (though pooling should
prevent the topic-scarcity runaway); it needs one fit-time experiment to confirm stability.

### Class B — estimate OUT-OF-BAND (export time), calibrated

**B1. Indirect inference / parametric-bootstrap bias inversion (the reviewer's proposal).**
Treat the biased export EM as an auxiliary statistic ĉ(·); simulate a corpus at candidate
c\*, run the estimator, invert ĉ(c\*) = ĉ(real). Corrects all pipeline biases end-to-end.
Self-referential to the model (mitigated by finding 1); recovers the pseudo-true scale under
misspecification.

**B2. External-target moment matching (the de-circularized calibration).** Same simulation
machinery, but match a **model-free observable** of real documents' concentration rather
than the model's own estimator — e.g. the empirical distribution of a raw per-document
token-diversity statistic (unique-token fraction, repeat rate), or a nonparametric summary
of real θ̂. Targets a property of the data, not of STM's inference — which is exactly the
move that breaks the self-reference.

**B3. Cross-model concentration transfer (borrow LDA's α as the thermometer).** Finding 3
says both families read the same concentration once calibrated. LDA estimates it stably
in-band; STM cannot. So: fit an (already-planned) gated LDA on the same corpus, take its
α-optimized per-document concentration as the corpus's concentration reading, and choose the
STM scale c that reproduces that same per-document concentration distribution. Uses the
stable estimator (LDA α) to set the unstable one (STM Σ scale). External to STM, so no
self-reference; reuses a fit the project intends to build anyway.

### Class C — target the USE, not the prior scale

**C1. Conditional-concentration calibration.** The generative tool always conditions on a
seed (it completes partial documents), so the prior scale mostly governs the tail. Calibrate
c directly to a **conditional** acceptance statistic — top-topic mass under a representative
seed panel — matched to real documents' equivalent θ̂ top-mass. Targets the actual use,
sidesteps prior-scale estimation and most of the misspecification worry. Cheapest to run.

### Class D — richer concentration, not one number

**D1. Per-block / covariate-dependent scale.** Finding 4 below (a single c over-sharpens the
mid-range while under-sharpening the diffuse end) suggests one scalar cannot be faithful to a
*distribution* of true concentrations. Use two scales (background vs foreground), or make the
scale depend on the covariate mean or document length, so the generated concentration tracks
the document's expected concentration. Composes with A1/B1/B2. Still pooled within each
scale, so still runaway-safe.

**D2. Hierarchical / per-document latent concentration.** Model the concentration itself as a
per-document latent (e.g. a length- or covariate-linked scale with a hyperprior) rather than
a corpus constant — the most faithful to a real spread of document concentrations, and the
largest modeling change.

### Class E — non-parametric (considered, limited)

**E1. Empirical resampling of fitted positions.** Generate by resampling/perturbing the real
documents' fitted η̂ instead of a parametric Σ. Sidesteps scale estimation — but the fitted
η̂ from a unit-diagonal fit is itself compressed, so the empirical distribution is too tight;
it needs the same de-compression it was meant to avoid. A dead end alone; could combine with
A1 (resample from de-compressed positions).

### Evaluation

| option | in/out of band | matches | data-driven | runaway-safe | avoids self-reference | cost | novelty |
|---|---|---|---|---|---|---|---|
| A1 global temperature | in (fit) | ML likelihood | yes | likely (pooled) — untested | yes | a re-fit + small M-step | high, untried |
| B1 indirect inference | out (export) | model's own ĉ | yes | yes | partial | simulation loop | high |
| B2 external moment match | out (export) | raw data statistic | yes | yes | yes | one statistic + match | medium |
| B3 LDA-α transfer | out | LDA α reading | yes | yes | yes | one gated-LDA fit | med-high |
| C1 conditional calibration | out | seed-panel top-mass | yes | yes | partial | low | medium |
| D1 per-block/covariate scale | either | (composes) | yes | yes | (inherits) | +1 dim | medium |
| D2 hierarchical latent scale | in | ML | yes | needs care | yes | model change | high |

---

## 4. Questions for the reviewer

1. **Is A1 (a single global fit-time temperature) actually stable, or does it co-adapt with
   β?** The pooling argument says one scalar cannot be dragged by a document-scarce topic —
   but is there a *different* instability (e.g. a global temperature and the topic-word
   parameters trading off) that per-topic runaway diagnostics wouldn't have surfaced? If A1
   is stable, it dominates the export options (in-band, unbiased, no round-trip, no
   self-reference) — is there a reason it was never the design?
2. **Given finding 1, does the self-reference in B1 still matter?** If STM is a faithful
   estimator at the correct scale, is indirect inference's "pseudo-true = self-consistent"
   caveat empty in practice, or is there a residual failure mode (e.g. the *shape* of the
   real θ̂ distribution differing from any single-scale model) that B1 would miss and B2/B3
   would catch?
3. **Is α-optimization on SHORT, sparse documents itself biased?** B3 leans on LDA's α being
   a trustworthy concentration reading. Does α-optimization systematically over- or
   under-concentrate as document length shrinks (44 tokens over a 5000-token vocabulary)? If
   biased, by how much and in which direction — and does that bias transfer when we use α to
   set STM's scale?
4. **One scale or a distribution (D1/D2)?** Finding 4: at a single c, STM over-sharpens the
   mid-range and under-sharpens the diffuse end. Real corpora have a *distribution* of
   document concentrations. Is a single scalar the right target at all, or should the
   generative concentration depend on the draw (covariate mean, length), and if so what is
   the minimal such dependence worth modeling?
5. **Which is the right *target quantity*** — the prior η-variance scale (A1/B1), a raw
   data-side concentration observable (B2), a cross-model reading (B3), or the conditional
   top-mass the tool actually produces (C1)? They are not the same under misspecification;
   which best serves a generative tool whose job is to produce documents that pass through
   the model's own inference the way real ones do?

**Finding 4 (referenced above):** at a single fixed scale STM is faithful only in a band
(Σ ≈ 5–7 here) — it slightly over-sharpens mid-range concentrations and under-sharpens the
diffuse end. A corpus with a spread of true document concentrations may therefore not be
well served by any single scalar, motivating Class D.

## 5. Recommended next steps (independent of the reviewer's answer)

- **Cheap and decisive:** run the definitive real-corpus comparison — LDA-α-optimized vs the
  STM fit on the *same* documents — and compare the per-document θ̂ concentration
  distributions. This pins the actual target concentration independent of prior family
  (settling question 2/3 empirically), and directly feeds B2/B3/C1.
- **Highest-value experiment:** test A1 (a single global fit-time temperature over the
  unit-diagonal correlation) for stability and faithfulness. If it holds, it is the cleanest
  solution and makes the export calibration unnecessary.

## Method notes / caveats

- The diagnostic measures **inference bias** (recovery of a *known* concentration) using the
  real β; it does not measure the real documents' actual concentration (that needs the
  corpus). It uses fixed α (not α-optimization) and Σ = c·I (correlations set aside, covariate
  mean 0); the production covariance is c·R with a covariate-dependent mean, which may add
  modest concentration at a given c.
- References: online variational inference (Hoffman, Blei & Bach 2010); logistic-normal /
  correlated topic model + Laplace posterior (Blei & Lafferty 2007); Structural Topic Model
  (Roberts, Stewart & Airoldi 2016); Dirichlet α-optimization (Blei, Ng & Jordan 2003 /
  Minka 2000); anchor-word spectral init (Arora et al. 2013).
