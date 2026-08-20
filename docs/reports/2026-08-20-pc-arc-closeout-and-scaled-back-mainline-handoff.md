# Handoff — closing the Prediction-Constrained (PC) arc; scaling back to the gated + readout mainline

**Date:** 2026-08-20
**From branch:** `claude/spectral-anchor-topic-k-200nqp` (the PC research; now the stash)
**To branch:** `claude/gated-conditional-voi` (this branch — the scaled-back mainline)
**Author:** working session with Claude

---

## TL;DR

Prediction-Constrained (PC) gated-LDA supervision, pushed from 41-anchor scale to a
whole-Mondo-shaped fit (cardiovascular branch, K≈444), **hurts the deliverable**: the
post-hoc readout on PC-shaped topics scores **0.681** vs **0.739** on the *unsupervised*
gated topics (Δ−0.058, and negative in every rarity/conditional slice). Root cause is
simple and robust: **the co-fit "unified" head (0.572) is a worse predictor than the
unsupervised topics already are (0.739)**, so shaping the topics toward it can only drag
them down.

**Decision (a deliberate stopgap, not a permanent abandonment):** make the model-of-use the
**unsupervised gated LDA + a post-hoc readout LR**. Keep the three things that matter —
**conditional diagnosis, calibrated probabilities, value-of-information (VOI)** — all of
which ride on the gated β + readout and **do not require PC shaping**. Demote PC to an
optional arm (it is inert at `weight_y=0`), don't delete it. Revive it only when the co-fit
head can match the gate (see §6).

This buys reliability and unblocks the visible-results path: whole-Mondo scale-up, exports,
dashboards — without sacrificing scalability (dropping PC removes the parts that fought scale).

---

## 1. Why we pursued PC (reconstructed motivation)

PC was not chosen a priori; the team backed into it through a forced chain (evidenced on
`claude/hybrid-domain-reliability-review-ckn2bq` and the current branch's insights):

1. **The clinical goal** — find **uncoded / undiagnosed rare-disease patients** at ontology
   scale from EHR condition codes; "the real payload is finding *uncoded* patients." Ambition:
   scale to *thousands* of anchors, replace the SNOMED stand-in with **Mondo↔OMOP / the Mondo
   is-a DAG**, using *one shared mechanism* — "without building hundreds of unrelated
   classifiers." (hybrid-branch handoff `2026-07-30-…`, specs `2026-07-15-anchor-first-…`,
   `2026-07-15-case-finding-assembly-…`.)

2. **Why a gated topic model** — a single *shared, partially-pooled, ontology-aware*
   representation. Gating is load-bearing: tying topic blocks to DAG nodes structurally
   "lifted deep-level AUC from 0.68 to 0.97" (pivot report `2026-07-15-…`); the is-a ontology
   "supplies the pooling structure for free" — the honest way to help data-scarce rare nodes.

3. **Why PC specifically** — two findings converged:
   - *Empirical (insight 0064 + `docs/reports/2026-07-23-case-finding-levers-retrospective.md`):*
     the case-finder plateaued on condition codes; the LR readout is a **ranker, not a
     discoverer** (0 FDR-controlled discoveries despite +0.12 ROC). Six pre-registered
     model-side nulls ⇒ the binding constraint is **information**, not the model.
   - *Literature (`docs/references.md`, Hughes 2017):* supervised topic models (sLDA) fail
     because "supervision is drowned out by the word likelihood." **PC fixes this by posing
     prediction as a *constraint*** — explain x *subject to* predicting y — with a **calibrated
     `P(y)` head**, and it is **semi-supervised** (works with sparse labels = the rare regime).

4. **The intellectual unification** — gating (hard) and PC (soft) are **the same mechanism**:
   posterior regularization (Ganchev 2010). "Hard gating is the degenerate special case; PC is
   the soft-prediction-quality instance." So **gated-PC is natural, not a bolt-on** — the
   theoretical appeal that made it worth building. (`docs/references.md` Ganchev entry.)

5. **The calibration payoff that licensed the scale-up (insight 0069):** at 41-anchor scale
   the *unified* co-fit head emitted `P(child|parent)` **better calibrated than post-hoc LR**
   (pooled ECE 0.0098 vs 0.0119), with the optimistic read "scale moved us toward the good
   regime, not away." That belief — that the unified head *improves* with scale — is what
   drove the whole-Mondo push. **It did not hold at K≈444** (see §3).

## 2. The newer angle PC opened: conditional diagnosis + VOI (lit review ONGOING)

The framings the scaled-back line keeps — **conditional diagnosis `P(specific | general)`**
and **value-of-information (next-best-test)** — are **not** in the original (July) case-finding
rationale; a grep of the hybrid branch finds neither term. They emerged **later**, from the PC
exploration + a **broad rare-disease-diagnosis lit review** (branch
`claude/rare-disease-diagnosis-lit-review-ojs4ms`, reports dated 2026-08-14) — **and that lit
review is not finished.** The realization: PC-constrained *generative* modeling opens angles in
the (crowded) diagnostics space that discriminative/semantic/LLM incumbents structurally can't
touch. Per the review:

- The honest contribution is **"conditioning + VOI on an interpretable generative model," not
  "we beat case-finding."** Prediction accuracy is explicitly *not* the winning argument.
- **VOI** — "the capability nobody else has" — is computed from the per-node **β**
  distributions (`log β_A/β_B`, expected information gain). **Those β come from the gate**, not
  from PC shaping.
- **Conditional diagnosis** = `P(d|x) = P(d|x, x∈C)·P(x∈C|x)` — "screening with the
  population-detection step factored out"; the `label_mask_mode` knob selects the clinical task.

**Implication for this handoff:** the three keeps are the *matured* value proposition, grounded
in the (ongoing) lit review, and independent of PC shaping. Finishing that lit review is a live
task, not a closed one.

## 3. How the PC arc closed (empirical)

The scale-up first hit **three serial scale bugs** (insight 0073), fixed by **ADR 0045**:
α-collapse (`docConcentration≥0.5` floor), head runaway (per-doc-mean Newton), and additive
λ-correction detonating the ELBO (exponentiated-gradient, mass-preserving correction — the
natural-gradient story is ADR 0044 / insight 0072). With those fixed, PC could finally *shape*
at scale — and the honest result is that it **hurts**:

- **exp 0102 (dev, cardiovascular, path_cousins_kids head, weight_y=16):** readout
  `pc_topics_lr` **0.681 vs unsup 0.739 (Δ−0.058)**; rare Δ−0.048, common Δ−0.067, conditional
  Δ−0.058, detection flat. PC drags the deliverable down everywhere.
- **Why:** the co-fit head **as trained is 0.572**, far below the unsup readout (0.739). The
  head-formulation ladder decomposes the gap to the full-K target: **0.572 → 0.637** (its own
  localized oracle: a converged, intercept+standardized solve on the *same* support — a **+0.065
  "solver/formulation" lever**) **→ 0.681** (full-K: a **+0.044 "support" lever**). So the
  co-fit head underperforms even its own localized ceiling, mostly because it is a 1-step Newton
  fit against a *moving* θ.
- **The unified head cannot be full-K at scale:** a dense full-K co-fit head's per-node Hessian
  is `O(C·K²)` — 657 MB at K=444; its `treeAggregate` collect hit the 4 GiB driver cap (4.3 GiB)
  and thrashed on preemptible executors (exp 0101, now `superseded`). Bounded-support structural
  expansion (siblings → path-cousins → +children) recovered almost none of the gap — the missing
  signal is **out-of-DAG-neighborhood** (comorbidity), which no structural rule reaches.
- **The headroom hypothesis (not yet fully separated from over-drive):** the unsup gate is
  already strong (0.739); a supervisor worse than the data's own extraction hurts *by
  construction*, and at weight_y=16 the shaping over-drove (`corr_relΔλ`=0.72, destructive).
  Whether a *good* supervisor + gentle shaping would reach neutral-or-help is **the one open PC
  question** — see §6. Consistent with insight 0066 ("PC-shaping helps only in proportion to
  what the unsupervised fit misses") and exp 0097 ("shaping ~neutral; the gate already does well").

**Caveat carried forward:** 0102 was a **30-iter DEV run** (`CHARM_DEV=1`), which undertrains
both θ and the head. A **full 100-iter 0102** was staged but not yet interpreted; it would tell
us whether 0.572 is a real ceiling or dev-mode undertraining, and (via the new **quartile rarity
split**) whether PC helps the *extreme* low-mass tail (Q1) where insight 0066 predicts its only
headroom. See §6/§7.

> **Update (2026-08-20, post-handoff): the caveat is resolved.** The full 100-iter 0102 landed
> and confirms the dev run: co-fit head **0.567** (ceiling real, not undertraining), readout
> **0.688 vs 0.7395 (Δ−0.052)**, and **negative in all four rarity quartiles including Q1**
> (Δ−0.035) — the rare-tail-rescue hypothesis is refuted at this head quality. The closeout now
> rests on full-run evidence; the dev profile is validated as a ranking loop. Full numbers in
> exp 0102's run log.

## 4. The scale-back — what it is

**Deliverable / model-of-use = unsupervised gated LDA (the gate) + a post-hoc readout LR.**
The gate produces DAG-aligned, per-node topic blocks (β) and per-doc θ; the readout LR on θ is
the calibrated predictor (0.739 today on the cardiovascular branch). Conditional diagnosis,
calibration, and VOI all sit on top of β + readout:

| keep | provided by | with PC | without PC (this mainline) |
|---|---|---|---|
| conditional diagnosis `P(child\|parent)` | readout / β on the gated topics | cond AUC 0.681 | cond AUC **0.739** (better) |
| calibrated, interpretable probabilities | post-hoc LR on gated θ + per-node ECE | ECE ~0.012 | same machinery on stronger topics; isotonic if needed |
| value-of-information (next-best-test) | per-node **β** (LLR, EIG) | available | available (β come from the gate) |

**Scalability is improved, not sacrificed:** dropping PC removes the per-node head Hessian
collect (the `O(C·K²)` wall), the shaping instability, and the whole co-fit-head apparatus. The
one thing to watch at whole-Mondo (K≈3,800) is the readout LR's `D×K` θ-collect on the driver —
post-hoc, one-time, samplable.

## 5. What's kept vs demoted/stashed

- **Kept & active (this branch):** unsupervised `GatedOnlineLDA` + gate; multi-domain
  (condition+measurement+drug) BOW; Mondo↔OMOP crosswalk + per-patient frontier; the
  `gated_pc_cloud.py` driver's readout, conditional-sharpening, per-node reliability/ECE, and the
  new quartile rarity split; the whole-Mondo DAG engine (`mondo_dag.py`) + attestation.
- **Demoted, not deleted:** the PC engine (`OnlinePCLDA`, the co-fit head, natural-gradient
  λ-correction, EG mass-preserving correction, localized/path-cousins/path-cousins-kids head
  support, the trust-region cap `head_trust_move`). All inert at `weight_y=0`. Leave in place.
- **Stashed (branch `claude/spectral-anchor-topic-k-200nqp`):** the in-progress head-quality
  exploration and its exp docs (0099–0102), plus the local `mondo_cv_harness.py` /
  `sparse_many_topic.py` scratch harnesses (in the session scratchpad; ephemeral — the CV-DAG
  extraction from mondo.obo is `scratchpad/extract_cv_dag.py` if needed again).

## 6. Revival condition for PC + unfinished threads

**Precise revival condition: revive PC only when the co-fit head can match the gate
(co-fit ≥ ~0.74).** Until then, shaping toward it hurts by construction, and "VOI from a single
calibrated head" cannot beat "VOI from gate + post-hoc readout."

Threads that were live when we stopped (all on the stash branch):
- **The one open empirical question:** does a *converged, gate-quality* co-fit head applied with
  *controlled* shaping (trust cap / low weight_y) reach **neutral-or-help**, or is the gate
  genuinely at ceiling (no headroom)? The staged **full-100-iter 0102** + its **Q1 quartile
  rarity** result is the cheapest read; a positive Q1 delta would reframe PC as a **rare-tail**
  tool even if the macro stays negative.
- **The head fix, if pursued:** the ladder says the *solver* is the bigger lever (+0.065 vs the
  +0.044 support lever). Options, in order of appeal: **matrix-free (amortized L-BFGS) full-K
  co-fit head** (targets 0.68–0.74 directly, `O(C·K)` shuffle, no Hessian — but non-stationary
  curvature risk), or **dynamic MI-selected bounded support + exact Newton** (top-m predictive
  topics per node, re-selected every iter from a ρ-smoothed `(C,K)` correlation stat — keeps
  exact Newton, `O(C·m²)` shuffle). The distributed-cost analysis and both prototypes' scaffolds
  are on the stash branch.
- **Stability:** whichever head, pair it with the **trust-region cap** (`head_trust_move`,
  already in the engine) or lowered `weight_y` — bigger/stronger heads over-drive (`corr_relΔλ`
  up to 0.72) without it.

## 7. Immediate next steps for the new (scaled-back) session

1. **Make the gate + readout the first-class deliverable** in the driver: unsupervised gated
   fit, post-hoc readout LR, conditional-sharpening, calibration (add isotonic if per-node ECE
   needs it), and VOI (LLR / EIG from β) as reported outputs — not PC-arm side-effects.
2. **Whole-Mondo scale-up** of the gate + readout line (K≈3,800; body-system decomposition per
   insight 0071). Watch the driver-side readout collect.
3. **Exports + dashboards** on the calibrated per-node posteriors (the visible-results goal).
4. **Finish the rare-disease-diagnosis lit review** (branch `…-lit-review-ojs4ms`) — it's the
   backbone of the conditional/VOI positioning and is incomplete.
5. **Run the fair baselines** the lit review named: **PheRS** (`phers` R package on phecodes,
   same cohort/task) and a **transposed PhenoBrain**; these are what a writeup must beat/match.
6. **(Optional, closes the PC arc cleanly)** interpret the full-100-iter 0102 so the stash has a
   final, non-dev data point on head-convergence + the Q1 tail.

## 8. Where things live

- **This branch (`claude/gated-conditional-voi`):** scaled-back mainline. All infrastructure
  present; PC inert.
- **`claude/spectral-anchor-topic-k-200nqp`:** the PC research stash (head-quality arc, exp
  0099–0102, engine head-support work).
- **`claude/hybrid-domain-reliability-review-ckn2bq`:** origin of the case-finding engine,
  Mondo↔SNOMED crosswalk, the levers retrospective, and the PC design plan
  (`docs/superpowers/plans/2026-08-07-faithful-flat-pc.md`).
- **`claude/rare-disease-diagnosis-lit-review-ojs4ms`:** the (ongoing) lit review — the
  conditional/VOI positioning (`docs/reports/2026-08-14-*`).
- **Key reads:** insights 0064 (ranker≠discoverer), 0066 (shaping marginal), 0069 (unified head
  calibrated at 41-anchor), 0072/0073 (natural-gradient / three scale bugs); decisions 0038–0045
  (the PC engine lineage); the 2026-07-23 case-finding-levers retrospective.

---

*Bottom line: PC was a principled, elegant bet (constrained-posterior unification of gating +
supervision) that delivered a calibrated unified head at small scale but does not survive the
whole-Mondo scale-up as the model-of-use — the unsupervised gate already extracts the signal the
supervisor is worse at. The defensible contribution — an interpretable, hierarchical, calibrated
generative model enabling conditional diagnosis and VOI — lives on the gate + readout and is what
this mainline delivers. PC is stashed with a precise revival condition, not abandoned.*
