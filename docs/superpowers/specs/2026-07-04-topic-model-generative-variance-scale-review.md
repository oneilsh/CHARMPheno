# Generative variance-scale in a gated Structural Topic Model on short text — open problem, engine context, and history

**Purpose.** A self-contained review brief for a fresh reviewer with no prior context. It
describes a distributed topic-modeling engine, states an open modeling problem, gives the
full history of what has been tried (so the reviewer does not re-suggest falsified paths),
and lists candidate directions. The setting is **unsupervised topic modeling of short
social-media text** (tweets). No LaTeX; Unicode Greek (η, θ, Σ, Γ, β, λ, ν, α) throughout.

---

## 0. The question

A **gated** Structural Topic Model (STM) is fit with its topic covariance Σ pinned to a
**unit diagonal** (a correlation matrix, Σ_ii ≡ 1) — the only parameterization that keeps
the gated fit from a variance runaway. A downstream **generative** tool draws synthetic
users, η ~ Normal(Γᵀx, Σ), θ = softmax(η), and needs Σ at the corpus's **true η-scale**
(empirically ≈ 7.6), which the unit-diagonal pin discards. Draws at unit scale are
over-diffuse: each synthetic user is spread across ~15–20 topics, and a 2-token seed
(two hashtags strongly tied to one theme) lands on the *wrong* topic.

**How does one obtain a bounded, proper, non-unit *generative* variance scale in the
*gated* setting — data-driven (no hand-set constant), without reintroducing the fitting
runaway?**

---

## 1. Context: the task and the corpus

We do unsupervised **topic modeling of short-text social-media data** (tweets and short
posts) to discover latent discussion themes and community structure.

- A **document** is one user's bag of tokens (hashtags + words aggregated over their recent
  posts). Documents are **short** — a mean of ≈ 44 tokens per user, most vocabulary entries
  absent from any given document. (Short-text sparsity is the defining property; it is why
  per-document topic inference is weakly constrained.)
- **Topics** are coherent token clusters — discussion themes / communities of discourse.
- **STM** (Roberts, Stewart & Airoldi 2016) makes each document's topic proportions depend
  on document-level **covariates** through a logistic-normal prior:
  η_d ~ Normal(Γᵀx_d, Σ), θ_d = softmax(η_d). Covariates x here are user/post metadata
  (e.g. account-age bucket, region). Σ is the between-topic covariance (Blei & Lafferty
  2007, correlated topic model).
- **Gated STM** — the core method here for **rare-community discovery**: topics are
  partitioned into a shared **background** block (general-discourse themes present across
  the whole corpus) plus per-**group** **foreground** blocks (themes specific to a
  community/segment), where "group" is a categorical such as community membership,
  platform segment, or language. A document may only express background ∪ its own group's
  foreground (hard topic masking). This surfaces the distinctive themes of a **small niche
  community** that a majority-dominated shared model would wash out (the large general
  stream drowns the niche signal otherwise).
- **The generative tool** draws synthetic users from the fitted model — either
  unconditioned (a synthetic cohort for exploring the topic space) or **completing a
  partial profile** (given a few seed tokens, sample a plausible full token-set). It is the
  consumer that exposed the variance-scale problem below.

**Σ plays two roles that have been conflated:** (a) a **fitting prior** regularizing η
during inference, and (b) a **generation covariance** setting the spread of synthetic
users. The central realization is that these two roles may require different treatment.

---

## 2. The engine: spark-vi

`spark-vi` is an in-house distributed **online variational inference** library for topic
models, built on Apache Spark, with an MLlib-style estimator/transformer API. It implements
online LDA (Hoffman, Blei & Bach 2010), HDP (Teh et al. / Wang), and STM.

**Fitting (stochastic variational inference).** The corpus is an RDD of documents. Each
minibatch: a `mapPartitions` **local E-step** computes per-document variational parameters;
sufficient statistics are aggregated to the driver by `treeReduce`; the driver performs a
**global M-step** natural-gradient update. Global parameters — the topic-token parameters
λ/β, the covariate coefficients Γ, and the topic covariance Σ — live on the driver and are
broadcast to executors each minibatch. Learning rate follows a Robbins-Monro schedule
(τ0, κ). Fit runs as a Spark application on a YARN cluster.

**STM specifics.**
- The per-document **E-step** is a MAP/Laplace estimate of the logistic-normal η: Newton /
  Fisher-scoring to the posterior mode η̂ (prior precision Σ⁻¹ plus the multinomial
  observation curvature), returning both the mode η̂ and the inverse-Hessian at the mode
  (the **Laplace posterior covariance** ν_d; its diagonal is the per-topic posterior
  variance). This is the exact routine the generative tool's "complete a profile" posterior
  also uses.
- The **M-step** updates λ (topic-token Dirichlet parameters, SVI natural gradient), Γ
  (block-aware ridge regression on aggregated cross-products), and Σ.
- **Gating** is a topic-block partition object: the per-document E-step masks η to the
  document's allowed block set (background ∪ its group), and the Σ M-step is block-wise
  (each supported topic-pair's covariance estimated only from the documents where both are
  allowed; cross-group pairs with too few co-occurring documents are lazy-kept).
- **Stabilizers** (see §4): a **reference topic** (pin one topic η ≡ 0, work in K−1
  dimensions — removes the softmax translation degeneracy), and **spectral (anchor-word)
  initialization** of β (Arora et al.), in a dense and a random-projection variant.

**Export.** After fitting, a **separate Spark application** ("bundle build") runs over the
corpus with the converged β/Σ **frozen** to produce downstream artifacts — topic-token
tables, topic correlations, coherence, and (the subject here) the **generative scale**. The
new pooled-scale estimator (§4, §5) runs here: an iterated E-step over the corpus with β and
Σ frozen, aggregated by the same `treeReduce` idiom. Because export freezes β and Σ, it is a
different regime from fitting — this distinction is load-bearing below.

---

## 3. The model and the two roles of Σ

η_d ~ Normal(Γᵀx_d, Σ), θ_d = softmax(η_d), gated to the document's allowed topic blocks.
Σ is the between-topic covariance. Its **diagonal** is the per-topic variance of η — the
quantity that sets how *peaked* a draw's θ is (larger η-variance → softmax of more-spread
logits → more concentrated θ; smaller → more uniform θ). This diagonal scale is what
generation needs and what the fit, as shipped, discards.

---

## 4. The fitting-stability arc — why Σ is pinned to a unit diagonal

Numbers: "collapse" ≈ Σ ≈ 1 (η near-uniform, topics clone the corpus marginal);
"runaway"/"blowup" ≈ Σ_kk or max eigenvalue → 1e5–1e10 (softmax saturation); "proper" ≈
bounded O(1–10), natural η-scale ≈ 7.6.

1. **σ_init knife-edge (non-gated).** Default σ_init=1 → collapse; σ_init=5/10 → escapes
   collapse but Σ → 1e10 (a softmax-saturation boundary, not a real covariance). exps
   0008–0011.
2. **Three stabilizers (insight 0029):** (i) a **reference topic**; (ii) **Σ-shrinkage** (a
   convex blend of Σ toward its diagonal) — explicitly the *minor* guard, "aimed at
   over-correlation rather than blowup," never the scale control; (iii) **spectral
   (anchor-word) init** — "the decisive one."
3. **The one proper non-unit Σ ever achieved (non-gated):** reference + **dense** spectral
   + σ_init=1 → Σ max **7.56**, all topics resolved, reference alive (exp 0015). This is the
   "natural η-scale ≈ 7.6." It requires *dense* spectral (a random-projection/JL spectral
   variant split one dominant topic to 8.3e5, exp 0017), a *small* σ_init inside a
   Goldilocks window (σ=20 → 1.8e8 blowup, exp 0016; σ≈0.01 → re-collapse), and the
   reference topic (reference alone at σ=1 → Σ ≈ 5e10, exp 0012). Full covariance under the
   same stack is also well-conditioned **non-gated** (cond 13.3, exp 0020).
4. **The gated runaway — the crux.** The **same** stabilizer stack **runs away in the gated
   setting**, every time a free variance is allowed: exp 0021 (cond 3.28e7, min-eig
   floored), 0023 (2.19e7), 0024 (6.08e8), 0025 (2.71e5, diverging), 0026 (×2600 in 7
   iterations). **Root cause (insight 0033):** the gated minority arm contains a
   **rare-but-coherent, document-scarce** topic — a real niche theme with a crisp token
   signature but discussed by few users. Its η is weakly constrained by data, so a *free*
   prior variance grows without bound (softmax-saturation feedback). "Better spectral init
   cannot help — the token signature is already crisp; no initialization adds documents to
   a rare theme." This is a data-scarcity property of gating, absent non-gated.
5. **Variance priors were tried on the gated corpus and falsified.**
   - Inverse-Wishart prior (exp 0022, ν=100, scale=2): moved the diagonal (1.0→1.98) but
     did **not** control conditioning or the runaway. N-weighted → only bites thin cells.
   - IW variance-anchor *as a runaway cap* (exp 0024, ν=2000, scale=2, + diagonal-shrink):
     made the runaway **worse** (2.2e7 → 6.08e8). A well-supported runaway topic outruns any
     usable ν; scale=2 anchors variance *up* (loosening prior precision — the blowup-basin
     direction).
   - Diagonal inverse-gamma prior (exp 0018, scale=10, count=2000): never validated —
     superseded before a result was logged.
   - Diagonal-shrink (exp 0023) fixed the min-eigenvalue end but *triggered* the variance
     runaway by removing the off-diagonal coupling that had been stabilizing a
     weakly-identified topic.
6. **Shipped decision (ADR 0034): block-wise unit-diagonal correlation Σ** (Σ_ii ≡ 1 every
   M-step; empirical variance used only to standardize the off-diagonals to correlations,
   per entry clipped to [−1, 1]). This "severs the softmax-saturation feedback loop by
   construction — a weakly-identified topic's prior variance can no longer grow." It is "the
   ν→∞, scale=1 limit of a variance anchor at the load-bearing scale" (LKJ-style correlation
   modeling rather than full inverse-Wishart). Cluster-validated (exp 0027: Σ_var ≡ 1 every
   iteration, niche sub-themes crisp, no topic-quality cost). **The variance scale is
   discarded on purpose.**

**Takeaway:** unit-diagonal is not overcaution — every free-variance and every
variance-prior/anchor path was empirically falsified in the gated setting, and the runaway
is a fundamental consequence of document-scarce minority topics, which gating (the whole
point of the method) necessarily creates.

---

## 5. The generation problem

The generative tool draws η ~ Normal(Γᵀx, Σ) with Σ = the exported unit-diagonal
correlation R. Consequences observed:

- **Over-diffuse draws.** Unconditioned draws spread over ~15–20 effective topics; a real
  user concentrates on a few themes. Under the "complete a profile" posterior, a 2-token
  seed (two hashtags strongly tied to one theme) lands on the **wrong** topic (a
  high-baseline background theme), not the theme those tokens belong to — because the small
  (unit) prior variance is too *stiff* for a short seed's likelihood to overcome, so the
  posterior stays near the diffuse mean.
- **The topics are fine.** The seed token has 0.835 responsibility on a single clean topic
  (whose top tokens are a coherent theme); conditioning *should* concentrate there. With Σ
  scaled to ≈ 7.6 it does (top-topic mass 0.63). So the deficit is purely the **scale**, not
  topic quality.
- **The scale is unrecoverable post-hoc from a unit-diagonal fit (proven).** The empirical
  between-document variance of the fitted posterior modes η̂ is ~10× too small (the unit
  prior shrinks η̂ during inference). A synthetic check (known true scale 7.6, fit under a
  unit prior): even the law-of-total-variance correction (variance-of-modes + mean posterior
  variance) recovers only ~1.6 — the unit prior *censors* the large-variance regime; the
  information is not in the fit's outputs. A fit under the *correct* prior recovers ~6.8. So
  one cannot de-shrink a unit-diagonal fit into the right scale.

**The tension in one line:** the unit diagonal that the gated fit *requires* for stability
is exactly what deprives generation of the scale it *needs*.

---

## 6. Attempts (and outcomes)

- **Per-topic empirical η-variance, exported, Σ_gen = R·√(v_i v_j):** compressed (median
  ~0.6, below unit) → still over-diffuse. Abandoned.
- **A single hand-set scalar α ≈ 7.6** (the natural scale from exp 0015), Σ_gen = α·R: works
  — crisp, correct topic (top mass 0.63) — but a borrowed constant from a different fit.
  Rejected as a magic number.
- **Estimate the free Σ diagonal at fit time (exp 0032, gated):** **blew up** (Σ_var → 1.8e8
  by iter 124, driven by a topic with effective sample size 15) — an empirical confirmation
  of insight 0033's prediction that the gated runaway is fundamental.
- **Pooled single generative scale, estimated at export with β and Σ FROZEN
  (Σ_gen = c·R):** the one **novel, runaway-safe** result. It **decouples** the two roles of
  Σ — the fit keeps its unit diagonal, and a single scalar c is estimated separately at
  export by an iterated law-of-total-variance EM (each round runs the frozen-β per-doc
  E-step under prior c·R, pools Var_d(η̂_k) + mean_d(posterior variance) over free topics to
  one scalar, updates c). Frozen β breaks the fit-time co-adaptation loop; a single pooled
  scalar cannot be run away by one document-scarce topic (its noise is averaged against all
  others). On the production corpus it converged to **c = 3.67** in ~5 iterations (bounded,
  no runaway). Result: the 2-token seed now lands on the **correct** topic (top mass 0.38) —
  the core failure is fixed — but the concentration is **softer** than the α ≈ 7.6 scalar
  (0.38 vs 0.63), because the export EM **under-corrects** (Laplace/Gauss-Newton
  posterior-variance bias, regime-dependent: synthetics recovered 60–88% of the true scale).

---

## 7. The open question, sharpened

Established: (a) the **gated fit requires a unit diagonal** — free variance and every
variance-prior/anchor were falsified; (b) **generation requires a non-unit scale**; (c) the
unit-diagonal scale is **unrecoverable post-hoc**; (d) a **decoupled, export-time, frozen-β,
pooled scalar** is runaway-safe but **conservatively under-corrects**.

**Is the decoupling (unit-diagonal fit + a separately-estimated export-time generative
scale) the right architecture — and if so, how does one make the export estimate reach the
true generative scale, data-drivenly, without a hand-set constant?**

Sub-questions:

1. Is a single **pooled** scalar the right target, or should it be **per-topic** (foreground
   niche topics genuinely vary more than background)? Per-topic is *stable at export* with β
   frozen (unlike at fit time), but a document-scarce topic's per-topic estimate is noisy —
   robust pooling vs. per-topic shape.
2. The export EM's under-correction comes from the **Laplace posterior-variance bias**
   (Gauss-Newton curvature overestimates precision). Is there a principled, low-bias
   estimator of the single generative scale with β/Σ frozen (higher-order Laplace,
   importance-sampled marginal likelihood, or an EM not leaning on the Laplace within-term)?
3. **Moment-matching / indirect inference:** rather than estimate the latent η-variance
   (biased), calibrate c so the *generated user concentration* matches the *observed corpus
   concentration*. Real documents (~44 tokens) have a likelihood-dominated, hence
   un-compressed, fitted θ̂ concentration — a legitimate, corpus-specific target with no
   borrowed constant. Open subtlety: unconditioned draws (no tokens) vs. real documents
   (conditioned on ~44 tokens) are not directly comparable; the matching target must be
   chosen carefully (e.g. match a completed-profile draw under a representative seed to real
   documents).
4. Is the right object a single fit-derived η-variance number at all, or the **conditional**
   concentration the tool actually needs (which the profile-completion likelihood largely
   supplies once the prior is not over-stiff)?

---

## 8. Candidate directions (to evaluate or extend — not mutually exclusive)

- **A. Accept c = 3.67.** Honest, data-driven, runaway-safe, correct topic; the softer
  concentration arguably reads as a *realistically mixed* user (a dominant theme plus genuine
  secondary interests, not an artificial single-topic spike). Zero further risk.
- **B. Moment-matching calibration** (sub-question 3). Most-promising untried idea: no
  borrowed constant, targets the observable, corpus-specific. Needs a careful choice of what
  concentration statistic to match and under what conditioning.
- **C. Lower-bias export estimator** (sub-question 2). More rigorous, more work; may only
  move 3.67 toward ~5–6, still short of 7.6.
- **D. Per-topic export scale with robust pooling** (sub-question 1): recover the
  foreground>background shape while excluding/winsorizing document-scarce topics.
- **E. Fit-time natural-scale anchor** — the genuinely-untried variance-prior middle ground:
  a *strong* shrinkage anchoring the diagonal at the **natural** scale (~7.6), not the
  falsified scale=2. But it is fit-time (runaway-exposed if the anchor is finite and a
  well-supported topic outruns it) and re-introduces a hand-set scale. Lowest priority given
  the falsification history, listed for completeness.

---

## 9. Key experiment cells (appendix)

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
| 0028 | gated | unit-diagonal | ≡1 | production corpus |
| **0032** | **gated** | **estimate free diagonal** | **1.8e8 blowup** | confirms insight 0033 (ess≈15 topic) |
| — | gated, export | **pooled scalar c, β frozen** | **c=3.67 bounded** | runaway-safe, under-corrects |

## References

- Roberts, Stewart & Airoldi (2016) — Structural Topic Model.
- Hoffman, Blei & Bach (2010) — online variational inference for LDA (the SVI engine).
- Blei & Lafferty (2007), *Annals of Applied Statistics* 1(1) — correlated topic model /
  logistic-normal prior + Laplace posterior.
- Arora et al. (2013) — anchor-word / spectral topic recovery (the spectral init).
- Chan, Golub & LeVeque (1979) — pairwise/parallel variance (Welford) combination (the
  pooled EM's distributed reduce).
- Internal record (short-text corpus): insights 0029 (three stabilizers), 0030 (spectral →
  7.56 non-gated), 0031 (scalable spectral single-topic runaway), 0032 (gated variance-prior
  falsification), 0033 (runaway = document-scarce minority), 0034 (unit-diagonal
  confirmation), 0036 (fit-vs-export decoupling); decisions 0033 (full-Σ), 0034
  (unit-diagonal correlation), 0036 (generative scale / pooled export); experiments 0015,
  0020, 0022, 0024, 0027, 0028, 0032.
