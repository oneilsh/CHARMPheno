# 0068 — The co-fit head's topics-quality gap was head-L2 miscalibration (×n_docs), NOT joint-vs-alternating: a reference ALTERNATING fit reaches the joint quality

**Date:** 2026-08-13
**Topic:** svi
**Status:** Confirmed (corrects insight 0067's "residual gap" attribution)

Insight 0067 / ADR 0040 fixed the co-fit head's runaway |w| with a fixed L2 but concluded
the *remaining* gap — online co-fit topics-LR ≈ 0.53 vs the faithful full-batch L-BFGS
reference's ≈ 0.87 — was **online/alternating vs joint optimization** plus a
**shape-vs-regularize tension**. Two experiments overturned that.

**Primary sources first.** Reading Hughes (arXiv:1707.07341 §3.4, 1712.00499 §3) showed
PC training is **joint** (Adam over (topics, head), or L-BFGS in our reference), with a
label-free `T≈100` exponentiated-gradient MAP and a deliberately-weak **fixed**
`lambda_w = 0.001`. Two settings we violated became confound suspects: our
`grad_cavi_iters = 10` (vs T≈100) and `weight_y = 20` (vs "≈ tokens/doc").

**Confound isolation (`manual_pc_hughes_settings_experiment`) — both refuted.** Bumping
π-iters 10→50 changed *nothing* (byte-identical fits): our coordinate-ascent CAVI
converges far faster than Hughes's tiny-step (ν≈0.005) EG, so 10 CAVI ≈ his 100, and at
the (then-mis-scaled) L2 the shaping correction was too weak to be θ-sensitive anyway.
weight_y 20→160 gave only +0.036 topics-LR. The gap survived both.

**Joint-vs-alternating de-risk (`manual_pc_joint_vs_alternating`) — jointness refuted.**
Added `fit_mode="alternating"` to the reference: block-coordinate L-BFGS (topics with
head fixed, then head with topics fixed, repeat), holding objective / π-MAP / L2 / init /
full-batch / solver **identical** to `joint` so only the topic↔head coupling differs. On
the same corpus:

    joint         HEAD=0.802  topics-LR=0.862  |w|~5.5
    alternating   HEAD=0.967  topics-LR=0.874  |w|~105
    OnlinePCLDA   HEAD~0.52   topics-LR~0.53   |w|~4.76   (online, mis-scaled L2)

**Alternation does not collapse** — it matches joint's topics-LR and beats its head. So
jointness is not the mechanism. The discriminating signal is |w|: reference alternating
grows |w| to ≈105; the online model was pinned at ≈4.76.

**The real cause: an ≈840× = n_docs over-regularization.** OnlinePCLDA applied `head_l2`
**per-doc, ×n_docs** (`ridge = head_l2·n_docs`), so `head_l2 = 1e-3` acted like Hughes's
`lambda_w ≈ 0.84`. Hughes's `lambda_w` is instead a ridge on the **corpus-summed** head
gradient (the `weight_y` and `÷n_tokens` cancel), i.e. **absolute**. Making `head_l2`
absolute (drop the `n_docs` factor) and defaulting it to `1e-3` recalibrates the head.
The online recalibration sweep (`manual_pc_head_l2_recalibration`):

    head_l2=0     topics-LR=0.948  |w|=3.4e11 (blowup)
    head_l2=1e-4  topics-LR=0.947  |w|=1.4e5
    head_l2=1e-3  topics-LR=0.957  HEAD=0.849  |w|=1.3e4   <- lambda_w, sweet spot
    head_l2=1e-2  topics-LR=0.914  |w|=1581
    head_l2=1e-1  topics-LR=0.503  |w|=6.65 (over-regularized)

The online one-step Newton head now shapes to topics-LR ≈ 0.96 (**≥ the reference's
0.87**) with a finite, readable head — **no joint step, Newton retained**. The good basin
is wide (≈1e-4…1e-2), centered on Hughes's canonical 1e-3. The "shape-vs-regularize
tension" of 0067 was an artifact of the over-regularized regime, not fundamental.

**Implications.** (1) A per-parameter regularizer meant to match a published absolute
prior must be scaled to the SAME quantity the reference penalizes — here the corpus-summed
gradient, so absolute (not per-doc). A plausible-looking "scale by n_docs to track the
Fisher" was the whole bug and produced a wrong, higher-level conclusion (needs joint
optimization). (2) When an online model underperforms a batch reference, isolate the axis
INSIDE the reference (add the online scheme's structural feature — here alternation — as a
toggle) before attributing the gap to that axis. The alternating toggle turned a suspected
architecture rewrite into a one-line ridge-scaling fix. (3) Residual: the co-fit head
still under-reads its own shaping (HEAD 0.85 vs topics-LR 0.96) — the one-step Newton
direction at large |w| is slightly off the converged classifier; minor, and off the
critical path since case-finding reads post-hoc LR on the shaped θ.

**Setting context.** Realistic Mondo-DAG benchmark on real EHR β, K_fit=20, C=7,
weight_y=20, 30% semi-supervised, `OnlinePCLDA`/`VIRunner`; reference `analysis/pc`
PCTopicModel with the new `fit_mode`. Recorded as ADR 0041 (which supersedes ADR 0040's
residual-gap analysis).
