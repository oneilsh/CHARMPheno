# PC supervised-head seam + DAG-closure head — design

**Date:** 2026-08-12
**Status:** Steps 1–2 SHIPPED (flat extraction + `DagClosureHead`, engine-level, tested green).
Step 3 (Gated-PC composition) and the shim exposure of the DAG head are follow-ons.
**Oracle:** the in-memory `analysis/pc/` reference + the existing PC pyspark suite validate the
extraction as behavior-preserving; new flavors get their own FD/oracle checks.

## Bottom line

- The entire supervised head of `OnlinePCLDA` is **one per-document loss function**
  (`_per_doc_sup_nll`, `models/topic/pc.py:153`). Everything downstream — the batch
  gradient accumulation, the digamma-Jacobian topic transform, the `update_global`
  corrections — is **head-agnostic** and stays exactly as it is.
- So a **head *flavor* = the per-doc loss body.** Extract it behind a small
  `SupervisedHead` protocol and PC gains a plug point with **zero change to the SVI
  math** and the increment-1 fidelity gate untouched (the head is inert at
  `weight_y == 0`).
- Two flavors motivate the seam: **`FlatLogisticHead`** (today's model, one independent
  logistic per label) and **`DagClosureHead`** — the Mondo case. **Mondo is a DAG, not a
  tree**, so the HSLDA (Perotte 2011, ICD-9 tree) parent-gating generalizes to a
  **closure product** that counts each ancestor once (diamond-safe).
- This label-side seam is **orthogonal** to the topic-side DAG gate (`GatedOnlineLDA`),
  giving a clean 2×2 (ungated/gated topics × flat/DAG head) where each axis is one plug
  point. Neither this spec nor step 1 touches the gate.

## What is flavor-specific vs generic (file:line)

**Flavor-specific (the seam):**
- `_per_doc_sup_nll` (`pc.py:153`) — the differentiable per-doc NLL
  `loss_y_d = −Σ_c obs_c · log σ(s_c · w_c·π_d)`; π_d is the label-free CAVI mean
  (`_cavi_theta_anp`), differentiated through.
- `_supervised_head_hessian` (`pc.py:276`) — the closed-form logistic Fisher info
  `H_c = Σ_d obs·p(1−p)·ππᵀ`, used only by the Newton/IRLS head (ADR 0039). This is the
  *only* second-order object that is flavor-specific; it exists because the flat logistic
  has a clean closed form.

**Generic (head-agnostic — unchanged by the seam):**
- `_supervised_batch_value_and_grad` (`pc.py:206`) — sums per-doc
  `autograd.value_and_grad` over the batch, scattering each doc's `∂/∂eb_d` into the dense
  `(K,V)`. Works for *any* differentiable per-doc loss.
- `_grad_topics_to_lambda` (`pc.py:176`) — pure Dirichlet geometry (digamma-Jacobian),
  independent of the loss.
- `update_global` consumers of `grad_wCK_stat` / `head_hess_stat` / `grad_topics_stat`
  (`pc.py:636-699`) — read stats, don't know the flavor.
- `local_update` emission points (`pc.py:543`, `pc.py:552`) — call the batch accumulators.

## The seam

```python
class SupervisedHead:
    """A per-doc supervised NLL loss_y(eb_d, W, doc) plus generic autograd batch
    accumulators built on top of it. A FLAVOR subclasses per_doc_nll (and MAY provide
    a closed-form batch_hessian; the base returns None → Newton falls back to autograd
    or the caller uses SGD)."""
    def per_doc_nll(self, eb_d, W, counts, s, obs, alpha_vec, K, n_iters): ...   # anp-traced
    def batch_value_and_grad(self, topics_repr, W, rows, alpha_vec, K, n_iters): ...  # generic sum
    def batch_value(self, topics_repr, W, rows, alpha_vec, K, n_iters): ...           # generic (FD target)
    def batch_hessian(self, topics_repr, W, rows, alpha_vec, K, n_iters): return None # closed-form iff known

class FlatLogisticHead(SupervisedHead):   # == today: _per_doc_sup_nll + _supervised_head_hessian, verbatim
class DagClosureHead(SupervisedHead):     # closure-product loss; batch_hessian = local-logistic Fisher (quasi-Newton)
    def __init__(self, layout: DagLayout): ...   # reuses the SAME DagLayout as the topic gate
```

`OnlinePCLDA` holds `self._head` (default `FlatLogisticHead()`); `local_update` calls
`self._head.batch_value_and_grad(...)` and, for the Newton head, `self._head.batch_hessian(...)`.
The existing module-level free functions are retained as thin wrappers over a default
`FlatLogisticHead` instance so the FD grad-check tests and any external callers are untouched.

### Fidelity invariants (must hold after step 1)
- `weight_y == 0` path is byte-for-byte `OnlineLDA` — the head is never constructed-into
  the hot path there (`pc.py:526`). Increment-1 equivalence test stays green.
- Flat-head numbers are identical: the two FD grad-checks and the Newton-direction test
  pass unchanged; toy-bars AUC (PC 1.0 vs two-stage ~0.51) unchanged.

## The DAG-closure head (Mondo, follow-on)

HSLDA on a tree: `P(child)=P(parent)·σ(w·π)`. On a DAG a node has multiple parents and the
naive recursion **double-counts shared ancestors** (the diamond problem). The diamond-safe
form reads the closure **set** (each ancestor once) — exactly what `DagLayout.closure` /
`DagGate.closure_indicator` already compute on the topic side:

```
log P(node_l = 1) = Σ_{a ∈ closure(l)} log σ(w_a · π)          # closure = {l ∪ all ancestors}, ancestor-once
loss_y_d          = −Σ_l obs_l [ y_l·logP(node_l) + (1−y_l)·log(1−P(node_l)) ]
```

This is the honest is-a generalization: a node fires only if its whole ancestral closure
fires. It is a smooth function of `(w, π)`, so the base `batch_value_and_grad` autograds it
with **no new derivation**.

**Head optimizer (must be Newton, not SGD).** ADR 0039 / insight 0065: one RM-SGD step
per SVI iteration does not converge a logistic head against a moving θ (AoU: AUC ≈ chance,
head ⊥ the batch-LR direction), and it starves the topic correction (whose gradient flows
through `w_CK`). So the DAG head must not run on SGD. The closure *product* couples head
rows, so the flat head's *exact* per-label Fisher `p(1−p)ππᵀ` is no longer the exact
curvature — but Newton only needs a **positive-definite** metric, not a closed form.
`DagClosureHead.batch_hessian` therefore returns the **local-logistic Fisher** (each node's
own `p_a(1−p_a)ππᵀ`, reused verbatim from the flat head) as a **quasi-Newton preconditioner**,
paired with the *exact closure-coupled gradient* (`grad_wCK`, autograd). It is PD (+ridge),
aggregatable `(C,K,K)`, and scale-invariant exactly like the flat Fisher, so `H_a⁻¹ g_a`
recovers Newton's convergence without the O((C·K)²)/doc cost of a full autograd Hessian.
The **exact** closure-coupled block Gauss-Newton is a documented refinement (not required to
beat SGD); a full `(CK,CK)` autograd Hessian is exact but only viable for small C.

## The 2×2 this unlocks

| | flat head | DAG-closure head |
|---|---|---|
| **ungated topics** | PC (today) | HSLDA-on-a-DAG |
| **DAG-gated topics** | Gated-PC | Gated-PC + DAG head (full) |

The head plugin is the **label-side** seam; the topic gate is the **topic-side** seam (it
swaps the base λ E/M step in `GatedOnlineLDA`). They compose because they touch different
seams — the shared head is what lets Gated-PC reuse this machinery rather than re-plumb it.

## Sequencing

1. **This step — pure refactor.** Extract `FlatLogisticHead` behind `SupervisedHead`;
   model holds `self._head`; free-function wrappers preserved. Re-verify the whole PC
   suite green. No behavior change, no new capability. Unblocks both follow-ons.
2. **`DagClosureHead`** (task #13) — **SHIPPED.** The closure-product flavor with the
   diamond-safe closure matrix, injected via a new optional `head=` constructor param
   (default → flat, so nothing changes for existing callers; the head's `C` is validated
   against the model's). Tested: closure matrix incl. a diamond, monotone `P(child) ≤
   P(parent)`, FD grad-check on both the topic-correction and the head (base autograd,
   no hand-derived gradient), the **quasi-Newton** head (local-logistic Fisher
   preconditioner + exact coupled gradient — the DAG head converges via Newton, NOT SGD),
   and the defensive guard (a flavor returning a `None` Hessian degrades to SGD without
   KeyError). The MLlib-shim exposure (threading a `closure_parents` structure through the
   Estimator so case-finding runs end-to-end on a DataFrame) is the immediate next step.
3. **Gated-PC composition** (task #14) — compose the shared head with the DAG-gated E/M
   step; resolve the plumbing fork (subclass-chain vs delegation) recorded earlier.

## What this supersedes

The Mondo case-finding "prediction machinery" that scores an *unsupervised* gated fit
post-hoc — likelihood-ratio placement (`plans/2026-07-21-likelihood-ratio-placement-readout.md`),
explain-away LR (`plans/2026-07-23-explain-away-lr-scorer.md`), LR-FDR, θ-mass affinity — is
replaced by a **trained** supervised head. PC's asymmetric prediction-constrained objective
also directly addresses HSLDA's reported weakness (recall over precision — more false
positives; `references.md:107`), which is the wrong failure mode for FDR-controlled
case-finding. Hence: PC first, DAG head as an additive flavor, not HSLDA-first.

## References
- Hughes et al. 2018 (PC-sLDA) — the objective and the flat head; `docs/hughes-comparison.md`.
- Perotte, Wood, Elhadad & Bartlett 2011 (HSLDA) — label-side hierarchy on ICD-9;
  `references.md:97`.
- Ganchev et al. 2010 (Posterior Regularization) — the umbrella over gating + PC + this head;
  `references.md:110`.
- ADR 0039 — Newton/IRLS head (scale-invariant `H⁻¹g`); the flat-head Fisher lives here.
