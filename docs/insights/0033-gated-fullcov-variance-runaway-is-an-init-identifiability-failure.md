# 0033 — The gated full-Σ variance runaway is an initialization / identifiability failure, not an inherent thin-minority or model failure; unit-diagonal Σ is structural insurance

**Date:** 2026-07-01
**Topic:** stm | svi | conditioning | diagnostics | gating
**Status:** Confirmed (controlled local reproduction + real-cohort attribution via exp
0026 — the runaway topic is a rare-but-COHERENT minority dementia sub-phenotype,
under-constrained by low document count, not mis-identified)

exp 0025 (gated full-Σ, `cancer_or_dementia`, K=50, pd_complete M-step) developed a
max-eigenvalue **variance runaway** on the cluster: by iter 28 one topic's Σ_kk reached
2.71e5 (sigma_eig_max 3.21e5) with ELBO degraded to −1.56e7 (vs the ≈−1.59e6 target) and
still climbing, while |Γ| stayed bounded (3.21) — a *pure variance* blowup. This is the
insight [0029](0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md) /
[0032](0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md)-Finding-5
failure mode, appearing *even though* `pd_complete` had resolved the min-eigenvalue end.
A controlled local reproduction isolates the cause.

**Setting context.** Synthetic *gated logistic-normal* corpus generated from a KNOWN
unit-diagonal Σ_true (every topic's true prior variance is exactly 1, so any variance
blowup in the fit is provably a spurious estimation artifact, not in the data), with an
imbalanced minority arm (group weights 0.85/0.15, down to 0.05). Fit the current full-Σ
STM (**A**, free variances, pd_complete) against a unit-diagonal-projected variant
(**B** — standardize Σ ← D^-1/2 Σ D^-1/2 each M-step, "Option B") across three init
qualities (random / noisy-approximate-spectral / oracle-perfect-β), online SVI, K=14.

## Finding 1 — the runaway needs TWO ingredients; killing either one prevents it

1. a **weakly-identified topic** (a topic the likelihood does not pin, so its η can drift
   to the softmax-saturation boundary — [stm.py:422-425](../../spark-vi/spark_vi/models/topic/stm.py#L422-L425));
2. a **prior variance free to grow** (the M-step diagonal is pure MLE S/N with no anchor
   since the Σ-prior was removed, ADR 0033 decision 6).

Ingredient 1 is controlled by initialization. Random init → **A runs away** (max var
1.37e6, cond 1e13 — out of unit-variance data). Good init (noisy or oracle) → **A is
stable** (max var ≈1.5, cond ≈25–160), *even at a 5% minority arm* (83/1500 docs). So the
runaway is an **initialization / identifiability** failure, not something inherent to the
gated model or to thin minority arms.

**Two routes to "weakly identified."** The likelihood fails to pin a topic's η either
because its β is *diffuse* (never recovered — the random-init synthetic case) OR because
the topic is *rare* (β is crisp but too few documents constrain its variance — the
real-cohort case, Finding 4). Both leave the variance free to run; only the first is an
initialization problem.

## Finding 2 — this explains exp 0025 on the real cohort

Real spectral init is *approximate* on real data (anchor-word recovery for rare
sub-phenotypes is imperfect), so it leaves some topic weakly identified — worse than the
synthetic "noisy" init, bad enough to trip the runaway. The insight 0029/0030 stabilizer
stack (reference + spectral + σ_init) is partially failing on this cohort, not on the
model in general.

## Finding 3 — unit-diagonal Σ (Option B) is structural insurance, but not a quality fix

Constraining Σ to a correlation matrix each M-step (pin the diagonal to 1) removes
ingredient 2 **unconditionally**: max variance stayed exactly 1 across all init qualities,
including the random-init regime where A blew up. The prior variance can never loosen, so
the softmax-saturation feedback loop is severed by construction — no tuning knob (unlike
`sigma_diag_shrink`/IW, which Finding 4-6 of insight 0032 falsified). When both models are
stable (good init), A and B recover the planted correlations **equally** (supported-pair
MAE 0.28 vs 0.29), so Option B costs nothing on correlation quality in the good regime.

BUT Option B does **not** rescue correlation *quality* under bad init: with random init B
stays variance-stable (max var 1) yet its correlation readout is still poor (supported MAE
0.62, cond 8e8) because the min-eigenvalue block-arrow near-singularity persists.
**Correlation quality is gated by topic identification, not by the Σ representation.**

## Finding 4 — real-cohort attribution (exp 0026): the runaway topic is rare-but-COHERENT

exp 0026 (exp 0025's config re-run with the per-iter `maxvar[...]` attribution added to
`iteration_summary`) caught the runaway live and named it. Iter 16: `Σ_var max=3.45`,
`maxvar[topic=46 dementia peak=0.075 ess=106]`, ELBO −1.66e6 (healthy). Iter 23:
`Σ_var max=9.1e3`, `maxvar[topic=45 dementia peak=0.100 ess=114]`, ELBO −2.78e6 — a clean
in-progress blowup (variance ×2600 in 7 iters). The variance leader shifts among the
**rarest dementia sub-phenotypes** (46 = PTSD/alcohol/depression; 45 = post-concussion/TBI
— both E(β) ≈ 0.003–0.005, the lowest-mass topics in the model).

Decisively, those topics are **COHERENT, not diffuse** (β peak ≈ 0.10, effective support ≈
110 terms — legitimate, recognizable phenotypes). So this runaway is Finding 1's *second*
route: the topic is well-identified but *under-constrained* by document scarcity. This has
a sharp consequence for the fix — **better spectral init cannot help** (the β is already
crisp; no initialization adds documents to a rare phenotype). The two post-M-step variance
levers are also out: the inverse-Wishart prior was falsified (insight 0032 Finding 6) and
its `scale=2` anchors variance the wrong way (insight 0030: the load-bearing scale is 1).
The indicated fix is the unit-diagonal Σ (Finding 3) — pin every variance at 1 regardless
of document count, keeping the rare phenotypes intact while removing the runaway degree of
freedom. It is effectively the ν→∞, scale=1 limit of a variance anchor: the tuning-free
version at the correct scale.

Note the fit's *topics* stay crisp through the runaway (Alzheimer's/amnestic topic 41,
vascular dementia topic 49, and the full dementia tail all resolve at iter 16), consistent
with insight 0032 Finding 5 — it is Σ, the ELBO, and the correlation readout that degrade,
not the phenotype discovery. The runaway is also *stochastic*: exp 0026's early iters (max
var 3.45 at iter 16) were healthier than exp 0025's (~20–200 by the same point), because
the SVI minibatch sequence that kicks a rare topic into saturation differs run to run.

## Implications

1. **Variance runaway and correlation quality are separable problems.** The runaway is
   cured by fixing *either* identification (better init) *or* the growable variance
   (unit-diagonal). Correlation quality needs good identification regardless.
2. **Recommended posture:** adopt unit-diagonal Σ as cheap, tuning-knob-free structural
   insurance (guarantees the fit never explodes even when init is imperfect — which real
   data always is, removing the diverging-ELBO + long-Dykstra-iter failure mode), AND
   separately investigate why the real run leaves a weakly-identified topic (spectral-init
   quality, K sizing). With good identification, plain full-Σ is already stable and gives
   correlations as good as the unit-diagonal variant.
3. **Amends insight 0032's Resolution.** That Resolution held that `pd_complete`
   "preserves the stabilizing off-diagonal coupling exactly — so it cannot trigger the
   runaway." Correct that pd_complete does not *decorrelate* (the way `sigma_diag_shrink`
   did), but its implication that the max-eigenvalue runaway is thereby resolved is
   incomplete: the runaway has a *separate* init/identifiability cause that pd_complete
   neither triggers nor prevents (it is neutral on the diagonal it preserves exactly).

## Relationship to prior insights

Extends insight [0029](0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md)
(σ-init collapse/blowup and the missing stabilizers) and
[0030](0030-spectral-init-closes-stm-sigma-blowup-on-real-data.md) (spectral init +
small σ_init as the stabilizer that keeps Σ proper — here shown to be *identification*
quality, and load-bearing). Amends the Resolution of insight
[0032](0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md)
(pd_complete resolved the min-eigenvalue end but not the max-eigenvalue runaway).
Governed by ADR [0033](../decisions/0033-stm-full-covariance-sigma.md) (full-covariance Σ;
decision 6 removed the post-M-step variance anchor).
