# Multi-domain Prediction-Constrained topic model — plan

**Date:** 2026-08-14
**Branch:** `claude/spectral-anchor-topic-k-200nqp`
**Status:** Plan (scoping done). The thesis: *supervised per-node shaping learns the
disease-specific domain balance that fixed/unsupervised readout weighting couldn't.*

## 1. The hypothesis (validated as UNTESTED, not refuted)

The hybrid branch (`hybrid-domain-reliability-review-ckn2bq`) exhausted **fixed and
post-hoc-supervised** ways to combine domains and hit a ~6% ceiling. But that ceiling
measures a *weaker* mechanism than PC:

- Every combination approach (normalization rules 0073, fixed ω weights 0071/0079,
  nested-CV per-disease weighting 0075/0076) operated on a **frozen unsupervised fit** —
  reweighting per-domain LR *scores*, never a gradient into λ (`multidomain_weighting.py:53-152`).
- The failure was structural: **no global rule fits a disease-specific truth** (0073:
  rules "trade off in opposite directions across diseases"; 0079: diseases are "additive"
  (want condition+measurement summed — SLE, MS, sarcoid) vs "specialist" (want ONE domain
  — GBS/Marfan/EDS via measurement, MG via drug)), and a **shared θ can't specialize per
  node**.

PC is a different mechanism: it shapes **the fit itself** (∂loss_y/∂λ), and the **per-node
gate** gives each disease node its own topic block that can specialize toward the domain
that predicts *it*. Nothing in the insights refutes that this can extract the
disease-specific balance; several motivate it (0073, 0075 "prefer one shared partially
pooled mechanism", 0079 "no parameter-free rule captures both regimes").

**Honest counter to hold:** 0076/0079 conclude aggregate case-finding is
INFORMATION-limited, not combination-limited — condition-alone is the macro champion
under *fixed readout*. But the specialist rescues (Marfan/GBS via measurement, MG via
drug) are demonstrably present; the open question is whether supervised per-node shaping
extracts them into the macro. Run 6 (this experiment) already showed PC does real
representational work when the unsupervised fit misses — the exact regime this targets.

## 2. What exists where (the cross-branch reality)

- **THIS branch:** OnlinePCLDA + Gated-PC (injectable `GatedOnlineLDA` topic engine) +
  the gated_pc case-finding driver + `case_finding_assembly.emit_labels` (per-node
  label/labelMask). SINGLE fused vocab.
- **Hybrid branch:** multi-domain `GatedOnlineLDA` (per-domain λ dict `{m:(K,V_m)}`,
  fuse/split bridge `_assemble_expElogbeta`/`_split_to_domains`, per-domain η, ω), the
  value-aware measurement tokenizer (`measurement_tokens.py`: `concept_id*100+state`,
  low/normal/high/coded/presence), the multi-domain assembly
  (`multi_domain.py::assemble_multidomain_from_events` — a thin N-domain wrapper reusing
  `assemble_from_events`'s split/frontier/prune/strip verbatim), `domains.py`
  (domain_bounds), `multidomain_cloud.py` (per-domain vocab specs).

So multi-domain PC = **bring the hybrid multi-domain infra onto this branch + compose
with our OnlinePCLDA + make the topic correction domain-aware.**

## 3. Design (concrete)

**Corpus — multi-domain LABELED case-finding.** `multi_domain.py::assemble_multidomain`
already wraps `assemble_from_events` (which now has `emit_labels`), so the per-node
label/labelMask should come along for free — verify + thread the flag. Emits per-domain
`features_0..features_{N-1}` columns + `domain_bounds`. Domain 0 = conditions (the gate
axis). Start with **conditions + value-aware measurement** (the ONE non-condition domain
that carries signal, 0078/0079); add drug as a 3rd only after.

**Engine — inject the multi-domain gated engine + domain-aware correction.**
- Inject the multi-domain `GatedOnlineLDA` (per-domain λ dict) as OnlinePCLDA's
  `topic_engine` — the injection seam already exists (ADR 0042). The head reads K-dim θ,
  UNCHANGED (domain-agnostic — the one clean thing here).
- The PC **topic correction** (∂loss_y/∂λ) is a `(K,V)` gradient over the *concatenated*
  vocab. Make it domain-aware by scattering it through `_split_to_domains` before the
  per-block `(1-ρ)·old + ρ·target` blend — mechanically ~one call, but it's the real new
  engine work (verify the correction path handles dict-λ, not a single array).
- ω is NOT needed — PC replaces the tuned per-domain weight with a *learned* per-node
  domain emphasis (the whole point). Keep ω=1 (untempered).

**MLlib shim + driver.** Extend the gated-PC shim to accept multi-domain features
(featuresCols + domain sizes → domain_bounds), mirroring `mllib/topic/gated_lda.py`'s
`_concat_domain_features`. New/extended cloud driver arm.

## 4. The test (what would prove it)

Multi-domain Gated-PC vs three baselines, on rare6 (or the labs-dependent subset):
1. **condition-only Gated-PC** (our runs 4/7) — does adding meds/labs *under supervision*
   beat condition-alone? (The bar fixed readout couldn't clear.)
2. **unsupervised multi-domain gated** — does supervision beat the unsupervised
   multi-domain fit? (Isolates the PC shaping from the extra information.)
3. **fixed readout weighting** (the hybrid branch's ~6%) — does shaping beat reweighting?
Read: detection AP + per-node — DOES Marfan's/GBS's topic go measurement-heavy and EDS's
stay condition-heavy? (Inspect the per-domain λ mass per node — the direct test of
"per-node specialization.") The specialist-disease per-node story is the headline, more
than the macro.

## 5. Steps + open decisions

- **A. Merge direction (GATING decision).** Bring the hybrid multi-domain modules onto
  this branch (cherry-pick `multi_domain.py`, `measurement_tokens.py`, `domains.py`,
  multi-domain `GatedOnlineLDA` path, `multidomain_cloud.py` helpers) vs merge the
  branches vs port PC onto hybrid. Recommend: cherry-pick the multi-domain modules onto
  THIS branch (keeps our PC + gated_pc driver + emit_labels; the multi-domain engine path
  is additive to GatedOnlineLDA). Assess conflict surface first.
- **B.** Multi-domain labeled corpus (verify emit_labels flows through
  assemble_multidomain; condition + measurement).
- **C.** Domain-aware topic correction in OnlinePCLDA (the scatter) + shim multi-domain
  features. Unit-test the scatter (a 2-domain synthetic: correction lands in the right block).
- **D.** Run: multi-domain Gated-PC vs the 3 baselines; per-node λ-mass inspection.

**First move:** assess the merge/cherry-pick conflict surface (A), since everything
depends on getting the hybrid multi-domain engine path onto this branch cleanly.
