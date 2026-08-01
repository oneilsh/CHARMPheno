# 0081 — Data-driven per-node topic count (K_v) via effective rank of the spectral sketch

**Date:** 2026-08-01
**Topic:** capacity | per-node-K | spectral | hierarchy | HDP | method | decision
**Status:** Method built + validated on synthetic; awaiting real numbers (opt-in
probe on a spectral fit). Motivated by insight 0080's open capacity hypothesis.

## The problem

Uniform `tpn` is wrong in both directions. A tight leaf anchor (a few dominating
concepts) wants ~2 topics; a broad class node — especially a roll-up-flooded one
like "Disorder of nervous system" spanning diabetes, back pain, MS, ... — wants
many (insight 0080). And looking forward to hundreds/thousands of anchors,
uniform `tpn` gives `node_count × tpn`, which blows up to tens of thousands of
topics regardless of whether the corpus actually contains that much distinct
phenotype structure.

Patient **count** is the obvious but wrong allocator: it tracks *volume*, not
*diversity*. 100k patients with one phenotype still need one topic. As a cap it
never binds (effective rank tops out orders below patient count), so it has no
useful role here.

## Why not HDP / per-node stick-breaking

The "truly Bayesian" answer is a per-node stick-breaking prior that learns K_v.
We rejected HDP long ago, and per-node HDP is *worse*: it reimports the
prior-dominance/instability that sank it (insight 0017: replacing K with γ made
the result MORE sensitive, γ is harder to set than K), now once per node.
Insight 0019 is the key release valve: LDA at large K gracefully under-uses
excess capacity — it does not over-fragment. So over-provisioning is *safe*; the
only cost of a fixed generous K is compute, not quality. That reframes the whole
problem as **budget allocation**, not model selection — a parametric, bounded,
stable lever, not a new prior.

## The method: the effective rank is already computed, unused, in the init

`spectral_init_scalable.find_anchors_projected` is greedy pivoted-QR: each step
picks the word whose co-occurrence-sketch row has the largest residual norm after
projecting out the basis of already-chosen directions. Because projecting out
more directions only shrinks residuals and we take the max each step, the selected
residual norms form a **non-increasing sequence — the rank-revealing spectrum** of
that node's normalized co-occurrence sketch. Anchor finding stops at a fixed n;
run it out and the collapse point is the numerical rank = the number of distinct
phenotype directions the node holds = **K_v**. The per-node sketches
(`group_QR[g]` / `group_p_w[g]`) already exist in every spectral init — nothing
new is computed distributed; the estimate is a driver-side read of the same pass.

**Estimators** (`spark_vi.models.topic.effective_rank`): participation ratio
`(Σλ)²/Σλ²` is the default — the only one that is both parameter-free AND
scale-invariant (residual magnitudes differ across nodes, so a scale-invariant
rule applies uniformly). Threshold(τ) and eigengap are reported as cross-checks;
`n_probed` (where the greedy collapses below eps) is the hard rank. The pivoted-QR
is vectorized (in-place residual deflation, one matmul per step) so the
O(#nodes)×V driver probe is BLAS-fast.

**The scaling prize.** Summed over nodes, `Σ round(K_v)` grows with the corpus's
intrinsic phenotype dimensionality, not with node count. Once the phenotype space
is covered, adding anchors stops adding topics — the exact property uniform `tpn`
lacks, and the thing that makes this matter at thousands of anchors.

## Validation (synthetic)

Planted-rank row-sets recover cleanly: rank-2 → PR 1.9 / thr 2; rank-6 → PR 5.2 /
thr 6; rank-25 → PR 20.6 / thr 25. Scale-invariance and non-increasing spectrum
pinned by unit tests (`spark-vi/tests/test_effective_rank.py`). A driver-side
probe over 3 synthetic nodes of differing diversity produced a diversity-driven
foreground K of 28 vs uniform tpn×nodes of 6 — i.e. it reallocates, giving the
flooded class ~20 and the tight leaf ~2.

## How to get real numbers (opt-in, rides an existing spectral fit)

`scalable_spectral_init_beta` prints a per-node effective-rank table when
`CHARM_PROBE_EFFRANK` is set (`CHARM_PROBE_EFFRANK_MAX` overrides max_probe=40).
No effect on the fit — it logs and returns the same beta. Run it on a spectral
fit of the SNOMED-hierarchy layout (e.g. re-init exp 0082) and read the
`[effrank]` lines: does "Disorder of nervous system" show a large PR (confirming
it needs more capacity) while tight leaves show ~2? And does `Σround(PR)` come out
near today's K (~a few hundred) or "crazy large"? If many nodes saturate at
n_probed=max_probe, raise `CHARM_PROBE_EFFRANK_MAX`.

## Decision / sequencing

Build first, wire into layout only if needed. Per insight 0080, per-node capacity
matters **only if exp 0081 (roll-up off) shows classes-alive-but-AP-still-low**.
If roll-up-off recovers AP, flooding was the villain and this estimator is not
needed for case-finding (still valuable for the generative "broad class = broad
phenotype" model). If capacity IS the wall, `allocate_topics` feeds per-node block
sizes straight into the DagLayout (which already indexes per-node blocks) — a
parametric, bounded, stable replacement for uniform `tpn`, no new prior.

**Setting context.** Follows insight 0080 (roll-up pooling hurts, capacity is the
open hypothesis), 0019 (large-K graceful underuse), 0017 (γ prior-dominance sank
HDP), 0070 (spectral init non-critical for quality but seed-fragile). Code:
`spark_vi/models/topic/effective_rank.py`,
`spark_vi/models/topic/spectral_init_scalable.py` (probe hook).
