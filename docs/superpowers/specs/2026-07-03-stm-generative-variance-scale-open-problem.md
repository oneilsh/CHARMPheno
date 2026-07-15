# STM generative variance-scale in the gated setting — open problem, history, and candidate directions

**Purpose.** A self-contained review brief for a fresh reviewer with no prior project
context. It states an open modeling problem, the full history of what has been tried (so
the reviewer does not re-suggest falsified paths), and candidate directions to weigh. No
LaTeX; Unicode Greek (η, θ, Σ, Γ, β, λ, ν, α) throughout.

---

## 0. The question

A gated Structural Topic Model (STM) is fit with its covariance Σ pinned to a **unit
diagonal** (a correlation matrix, Σ_ii ≡ 1) — the only parameterization that keeps the
**gated** fit from a variance runaway. A downstream generative simulator draws synthetic
patients η ~ Normal(Γᵀx, Σ), θ = softmax(η), and needs Σ at the data's **true η-scale**
(empirically ≈ 7.6), which the unit-diagonal pin discards. Draws at unit scale are
over-diffuse ("rainbow": each patient spread over ~15–20 topics; a 2-code type-2-diabetes
seed lands on the *wrong* topic).

**How does one obtain a bounded, proper, non-unit *generative* variance scale in the
*gated* setting — data-driven (no hand-set constant), without reintroducing the fitting
runaway?**

---

## 1. Project context

CHARMPheno does unsupervised phenotype discovery from OMOP electronic-health-record
condition data using topic models (LDA/HDP/STM). A **document** is a patient's set of
condition codes over a window; **topics** are phenotypes (comorbidity clusters).

- **STM** (Roberts, Stewart & Airoldi 2016): topic proportions come from a
  covariate-dependent logistic-normal prior, η_d ~ Normal(Γᵀx_d, Σ), θ_d = softmax(η_d).
  Σ is the between-topic correlation/covariance (Blei & Lafferty 2007, correlated topic
  model).
- **Gated STM** (this project's core method for rare-subgroup discovery): topics are
  partitioned into a shared **background** block plus per-group **foreground** blocks; a
  document may only express background ∪ its own group's foreground (hard topic masking).
  This surfaces rare-subgroup phenotypes a majority-dominated shared model washes out.
- **The generative simulator** (dashboard): draws synthetic patients from the fitted STM
  for a "complete this patient record" tool and cohort exploration. It needs a generative
  covariance Σ_gen. This is the consumer that exposed the scale problem.

**Σ plays two roles that have been conflated:** (a) a **fitting prior** regularizing η
during inference, and (b) a **generation covariance** setting the spread of simulated
patients. The central realization below is that these two roles may require different
treatment.

---

## 2. The fitting-stability arc — why Σ is pinned to a unit diagonal

Numbers below: "collapse" ≈ Σ ≈ 1 (η near-uniform, topics clone the corpus marginal);
"runaway"/"blowup" ≈ Σ_kk or max eigenvalue → 1e5–1e10 (softmax saturation); "proper" ≈
bounded O(1–10), natural η-scale ≈ 7.6.

1. **σ_init knife-edge (non-gated).** Default σ_init=1 → collapse; σ_init=5/10 → escapes
   collapse but Σ → 1e10 (softmax-saturation boundary, not a real covariance). exps
   0008–0011.
2. **Three stabilizers (insight 0029):** (i) a **reference topic** (pin one topic η≡0,
   work in K−1 dims — removes the softmax translation degeneracy); (ii) **Σ-shrinkage**
   (a convex blend toward the diagonal) — explicitly the *minor* guard, "aimed at
   over-correlation rather than blowup," never the scale control; (iii) **spectral
   (anchor-word) init** — "the decisive one."
3. **The one proper non-unit Σ ever achieved (non-gated):** reference + **dense** spectral
   + σ_init=1 → Σ max **7.56**, all topics resolved, reference alive (exp 0015). This is
   the "natural η-scale ≈ 7.6." It requires *dense* spectral (a random-projection/JL
   variant split one topic to 8.3e5, exp 0017), a *small* σ_init inside a Goldilocks
   window (σ=20 → 1.8e8 blowup, exp 0016; σ≈0.01 → re-collapse), and the reference topic
   (reference alone at σ=1 → Σ ≈ 5e10, exp 0012). Full covariance under the same stack is
   also well-conditioned **non-gated** (cond 13.3, exp 0020).
4. **The gated runaway — the crux.** The **same** stabilizer stack **runs away in the
   gated setting**, every time a free variance is allowed: exp 0021 (cond 3.28e7,
   min-eig floored), 0023 (2.19e7), 0024 (6.08e8), 0025 (2.71e5, diverging), 0026 (×2600
   in 7 iterations). **Root cause (insight 0033):** the gated minority arm contains a
   **rare-but-coherent, document-scarce** topic (a real sub-phenotype, crisp β, but few
   documents). Its η is weakly constrained by data, so a *free* prior variance grows
   without bound (softmax-saturation feedback). "Better spectral init cannot help — the β
   is already crisp; no initialization adds documents to a rare phenotype." This is a
   data-scarcity property of gating, absent non-gated.
5. **Variance priors were tried on the gated cohort and falsified.**
   - Inverse-Wishart prior (exp 0022, ν=100, scale=2): moved the diagonal (1.0→1.98) but
     did **not** control conditioning or the runaway. N-weighted → only bites thin cells.
   - IW variance-anchor *as a runaway cap* (exp 0024, ν=2000, scale=2, + diagonal-shrink):
     made the runaway **worse** (2.2e7 → 6.08e8). A well-supported runaway topic outruns
     any usable ν; and scale=2 anchors variance *up* (loosening prior precision — the
     blowup-basin direction).
   - Diagonal inverse-gamma prior (exp 0018, scale=10, count=2000): never validated —
     superseded before a result was logged.
   - Diagonal-shrink (exp 0023) fixed the min-eigenvalue end but *triggered* the variance
     runaway by removing the off-diagonal coupling that had been stabilizing a
     weakly-identified topic.
6. **Shipped decision (ADR 0034): block-wise unit-diagonal correlation Σ** (Σ_ii ≡ 1 every
   M-step; empirical variance used only to standardize off-diagonals to correlations, per
   entry clipped to [−1, 1]). This "severs the softmax-saturation feedback loop by
   construction — a weakly-identified topic's prior variance can no longer grow." It is
   framed as "the ν→∞, scale=1 limit of a variance anchor at the load-bearing scale"
   (LKJ-style correlation modeling rather than full inverse-Wishart). Cluster-validated
   (exp 0027: Σ_var ≡ 1 every iteration, dementia sub-phenotypes crisp, no topic-quality
   cost). **The variance scale is discarded on purpose.**

**Takeaway for the reviewer:** unit-diagonal is not overcaution — every free-variance and
every variance-prior/anchor path was empirically falsified in the gated setting, and the
runaway is a fundamental consequence of document-scarce minority topics, which gating (the
project's whole point) necessarily creates.

---

## 3. The generation problem

The simulator draws η ~ Normal(Γᵀx, Σ) with Σ = the exported unit-diagonal correlation R.
Consequences observed:

- **Over-diffuse draws.** Prior/cohort draws spread over ~15–20 effective topics; a real
  patient concentrates on a few. A 2-code type-2-diabetes seed under the record-completion
  posterior lands on the **wrong** topic (a high-baseline background topic), not the clean
  diabetes/metabolic topic — because the small (unit) prior variance is too *stiff* for a
  short prefix's likelihood to overcome, so the posterior stays near the diffuse mean.
- **The topics are fine.** The diabetes code has 0.835 responsibility on a single clean
  "diabetes + hyperlipidemia + hypertension + obesity" topic; conditioning *should*
  concentrate there. With Σ scaled to ≈ 7.6 it does (top-topic mass 0.63). So the deficit
  is purely the **scale**, not topic quality.
- **The scale is unrecoverable post-hoc from a unit-diagonal fit (proven).** The empirical
  between-document variance of the fitted posterior modes η̂ is ~10× too small (the unit
  prior shrinks η̂ during inference). A synthetic check (known true scale 7.6, fit under a
  unit prior): even the law-of-total-variance correction (variance-of-modes + mean
  posterior variance) recovers only ~1.6 — the unit prior *censors* the large-variance
  regime; the information is not in the fit's outputs. A fit under the *correct* prior
  recovers ~6.8. So one cannot de-shrink a unit-diagonal fit into the right scale.

**The tension in one line:** the unit diagonal that the gated fit *requires* for stability
is exactly what deprives generation of the scale it *needs*.

---

## 4. Attempts this session (and outcomes)

- **Per-topic empirical η-variance, exported (`eta_var`), Σ_gen = R·√(v_i v_j):**
  compressed (~0.5 background, ~1.8 foreground; median 0.6, below unit) → still
  over-diffuse. Abandoned.
- **A single hand-set scalar α ≈ 7.6** (the natural scale from exp 0015), Σ_gen = α·R:
  works — crisp, correct topic (top mass 0.63) — but a borrowed constant from a different
  experiment. Rejected as a magic number.
- **Estimate the free Σ diagonal at fit time (exp 0032, gated):** **blew up** (Σ_var → 1.8e8
  by iter 124, driven by a topic with effective sample size 15) — an empirical
  confirmation of insight 0033's prediction. (The exp-0032 record still reads
  `status: pending`; the blowup was observed but not yet logged.)
- **Pooled single generative scale, estimated at export with β and R FROZEN
  (`corpus_eta_scale_gated`, Σ_gen = c·R):** the one **novel, runaway-safe** result. It
  **decouples** the two roles of Σ — the fit keeps its unit diagonal, and a single scalar
  c is estimated separately at export by an iterated law-of-total-variance EM (each round
  runs the per-doc E-step under prior c·R, pools Var_d(η̂_k) + mean_d(posterior variance)
  over free topics to one scalar, updates c). Frozen β breaks the fit-time co-adaptation
  loop; a single pooled scalar cannot be run away by one document-scarce topic (its noise
  is averaged against all others). On the production cohort it converged to **c = 3.67**
  (bounded, no runaway, ~5 iterations). Result: the diabetes seed now lands on the
  **correct** topic (top mass 0.38) — the core failure is fixed — but the concentration is
  **softer** than the α ≈ 7.6 scalar (0.38 vs 0.63), because the export EM **under-corrects**
  (Laplace/Gauss-Newton posterior-variance bias, regime-dependent: synthetics recovered
  60–88% of the true scale). Whether 3.67 is "crisp enough" or reads as realistic
  multimorbidity is a judgment call the project is weighing.

---

## 5. The open question, sharpened

Accept as established: (a) the **gated fit requires a unit diagonal** — free variance and
every variance-prior/anchor were falsified; (b) **generation requires a non-unit scale**;
(c) the unit-diagonal scale is **unrecoverable post-hoc**; (d) a **decoupled, export-time,
frozen-β, pooled scalar** is runaway-safe but **conservatively under-corrects**.

**Is the decoupling (unit-diagonal fit + a separately-estimated export-time generative
scale) the right architecture — and if so, how does one make the export estimate reach the
true generative scale, data-drivenly, without a hand-set constant?**

Sub-questions for the reviewer:

1. Is a single **pooled** scalar the right target, or should it be **per-topic** (the
   foreground topics genuinely vary more than background)? Per-topic is *stable at export*
   with β frozen (unlike at fit time), but a document-scarce topic's per-topic estimate is
   noisy — argue for robust pooling vs. per-topic shape.
2. The export EM's under-correction comes from the **Laplace posterior-variance bias**
   (Gauss-Newton curvature overestimates precision). Is there a principled, low-bias
   estimator of the single generative scale with β/R frozen (higher-order Laplace,
   importance-sampled marginal likelihood, or an EM that does not lean on the Laplace
   within-term)?
3. **Moment-matching / indirect inference:** rather than estimate the latent η-variance
   (which is biased), calibrate c so the *generated patient concentration* matches the
   *observed corpus concentration*. Real documents (~44 codes each) have a
   likelihood-dominated, hence un-compressed, fitted θ̂ concentration — a legitimate,
   dataset-specific target with no borrowed constant. Open subtlety: prior/cohort draws
   (no codes) vs. real documents (conditioned on ~44 codes) are not directly comparable;
   the matching target must be chosen carefully (e.g. match the record-completion draw
   under a representative prefix to real documents).
4. Should the "true generative scale" even be a single fit-derived number, or is the right
   object the **conditional** concentration the simulator actually needs (which the
   record-completion likelihood largely supplies once the prior is not over-stiff)?

---

## 6. Candidate directions (to evaluate or extend — not mutually exclusive)

- **A. Accept c = 3.67.** Honest, data-driven, runaway-safe, correct topic; softer
  concentration arguably reads as realistic multimorbidity (a dominant phenotype plus real
  comorbidities, not an artificial single-topic spike). Zero further risk.
- **B. Moment-matching calibration** (sub-question 3). Most-promising untried idea: no
  borrowed constant, targets the observable, dataset-specific. Needs a careful choice of
  what concentration statistic to match and under what conditioning.
- **C. Lower-bias export estimator** (sub-question 2). More rigorous, more work; may only
  move 3.67 toward ~5–6, still short of 7.6.
- **D. Per-topic export scale with robust pooling** (sub-question 1): recover the
  foreground>background shape while excluding/winsorizing document-scarce topics.
- **E. Fit-time natural-scale anchor** — the genuinely-untried variance-prior middle
  ground: a *strong* shrinkage anchoring the diagonal at the **natural** scale (~7.6), not
  the falsified scale=2. But it is fit-time (runaway-exposed if the anchor is finite and a
  well-supported topic outruns it) and re-introduces a hand-set scale. Lowest priority
  given the falsification history, listed for completeness.

---

## 7. Key experiment cells (appendix)

| exp | setting | Σ mechanism | Σ outcome | note |
|----|----|----|----|----|
| 0008 | non-gated | full-K, σ=1 | collapse ≈1 | baseline degenerate |
| 0012 | non-gated | reference, σ=1 | ~5e10 | reference alone insufficient |
| **0015** | **non-gated** | **reference + dense spectral, σ=1** | **7.56 proper** | the only proper non-unit Σ |
| 0016 | non-gated | reference + spectral, σ=20 | 1.8e8 | Goldilocks: small σ needed |
| 0020 | non-gated | full-Σ | cond 13.3 | full-Σ fine non-gated |
| 0021 | **gated** | full-Σ | cond 3.28e7 | first gated ill-conditioning |
| 0022 | gated | full-Σ + IW prior | unchanged | variance anchor falsified |
| 0024 | gated | full-Σ + IW + diag-shrink | **6.08e8 (worse)** | anchor cannot cap runaway |
| 0025/26 | gated | pd-completion | 2.7e5, ×2600/7 iters | runaway = doc-scarce minority |
| **0027** | **gated** | **unit-diagonal (ADR 0034)** | **Σ_var ≡ 1** | shipped, stable, no quality cost |
| 0028 | gated | unit-diagonal | ≡1 | production cohort (population_cancer) |
| **0032** | **gated** | **estimate free diagonal** | **1.8e8 blowup** | confirms insight 0033 (record still says pending) |
| — | gated, export | **pooled scalar c, β frozen** | **c=3.67 bounded** | runaway-safe, under-corrects |

## References

- Roberts, Stewart & Airoldi (2016) — Structural Topic Model.
- Blei & Lafferty (2007), *Annals of Applied Statistics* 1(1) — correlated topic model /
  logistic-normal prior + Laplace posterior.
- Chan, Golub & LeVeque (1979) — pairwise/parallel variance (Welford) combination.
- Project: insights 0029 (three stabilizers), 0030 (spectral → 7.56 non-gated), 0031
  (scalable spectral single-topic runaway), 0032 (gated variance-prior falsification),
  0033 (runaway = document-scarce minority), 0034 (unit-diagonal confirmation); decisions
  0033 (full-Σ), 0034 (unit-diagonal correlation), 0036 (record-completion + generative
  scale, incl. the eta_scale addendum); experiments 0015, 0020, 0022, 0024, 0027, 0028,
  0032.
