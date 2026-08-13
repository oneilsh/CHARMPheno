# 0067 — The co-fit head's runaway `|w|` is a vanishing relative ridge on PC-separated topics; a fixed L2 fixes it, and an inner Newton loop is fixed-point-redundant

**Date:** 2026-08-13
**Topic:** svi
**Status:** Confirmed

Follow-on to insight 0065 / ADR 0039. On a **realistic** Mondo-DAG case-finding
benchmark — the archetype planted on real EHR β (300 cross-site LDA topics × 29,003 OMOP
concepts, `manual_pc_dag_case_finding_realistic.py`) — the one-step Newton head from 0039
reproduced the "shapes but can't predict" signature in an extreme form: the shaped topics
carry the signal (post-hoc LR on θ reaches **AUC 0.965**) but the co-fit head itself
reads **0.646**, with **`|w_CK| = 3.4e11`**.

**Mechanism — the relative ridge vanishes exactly where it is needed.** PC's supervised
shaping makes the topics **separable**, so `p(1−p) → 0`. ADR 0039's ridge is *relative*
(`head_newton_ridge · mean(diag H)`), scaled by the very Fisher information that is
collapsing — so the ridge vanishes with it. The logistic MLE on separable data is at
infinity; `|w|` runs away and the single damped Newton step oscillates and **misaims**
(direction cos to the batch-LR solution 0.637, from the diagnostic). 0039 called this a
"known limit… head chases a moving target"; on separable topics it is sharper than that —
it is a *degenerate objective*, not a lagging one.

**The fix is a fixed L2, not more steps.** A **non-vanishing** per-doc L2
(`head_l2`, scaled by `n_docs`) keeps `|w|` finite — precisely Hughes's deliberately-weak
`lambda_w = 0.001`. With `head_l2 = 1e-3` the head goes `|w| 3.4e11 → 4.76`. The faithful
full-batch L-BFGS reference (`analysis/pc`, ADR 0038) independently lands there: fixed
`lambda_w=0.001` → `|w|=5.97`, head 0.812 ≈ its topics-LR ceiling 0.868.

**An inner Newton loop is fixed-point-redundant with the fixed L2.** The obvious
"converge the head each iteration, Hughes-style" (`head_inner_iters`, driver-side Newton
on a bounded θ/y/obs subsample) was built and verified to *run* — but it lands on the
one-step result **byte-for-byte** (`INNER k=10 == one-step`, `|w|=4.76`, topics-LR=0.525).
Reason: with a finite ridge, **one Newton step per SVI iteration accumulates**, over the
~60 iterations the topics take to settle, to the *same* regularized fixed point the inner
loop reaches within one iteration. Verified two ways: on a frozen design, 1-step×60 lands
`‖·‖ = 0` from a 60-step inner loop; and the discriminating `INNER l2=0` config (internal
1e-3 fallback → `|w|=4.76`, unlike one-step `l2=0`'s 3.4e11) proves the branch fires yet
converges to the same place. **The lever is the ridge TYPE (fixed vs relative), not the
step count.** (My initial read — "the inner loop isn't executing" — was wrong; it
executes and is simply redundant.)

**The residual gap is joint-vs-alternating, not head convergence.** The finite L2 that
finitizes `w` also **damps the shaping gradient** (∝ `|w_CK|`): topics-LR falls 0.965 →
0.525 as `head_l2` goes 0 → 1e-3. This shape-vs-regularize tension is intrinsic to the
online/alternating scheme (natural-grad λ step + separate head Newton, coupled only
through the topic correction). The reference's full-batch L-BFGS optimizes (topics, head)
**jointly** to convergence and finds a basin (topics-LR 0.868) the alternating scheme does
not (0.525). So: the head's *own* under-prediction is a head-optimizer problem (fixed by
`head_l2`); the *topics-quality* gap to the reference is an **optimization-regime**
problem (online alternating vs full-batch joint), not something more head iterations can
close.

**Implications.** (1) Any *relative* ridge/conditioner on a parameter whose objective the
model itself drives toward separability is unsafe — it evaporates under the very
conditions that create the singularity; back it with a fixed floor. (2) "Converge the
inner parameter each iteration" and "take one aggregatable step per iteration" are the
**same fixed point** once the outer loop and the ridge are fixed — reach for the inner
loop only when the outer representation never stabilizes (aggressive minibatching / few
iters), and pay for it in raw-θ-to-driver (0039's aggregatable/scale-invariant property
is lost). (3) Closing the topics gap points at a joint/second-order (topics, head) step or
Hughes's **two-stage** fit (shape with weak L2, then re-fit the head on frozen topics),
not at head iteration count.

**Setting context.** `OnlinePCLDA`/`VIRunner`, K_fit=20 ≪ 300, C=7 Mondo-like DAG
(rare leaves 5,6 at prevalence 0.13/0.09), weight_y=20, 30% semi-supervised labels, 60
iters, real β. Configs in `manual_pc_dag_case_finding_realistic.py`; diagnostic in
`manual_pc_head_shaping_diagnostic.py`; reference head-to-head in
`manual_pc_reference_comparison.py`. Recorded as ADR 0040.
