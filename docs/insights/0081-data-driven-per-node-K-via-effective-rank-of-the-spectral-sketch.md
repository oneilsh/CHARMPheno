# 0081 — Data-driven per-node topic count (K_v) via effective rank of the spectral sketch

**Date:** 2026-08-01
**Topic:** capacity | per-node-K | spectral | hierarchy | HDP | method | decision
**Status:** Method built + validated; real numbers (exp 0084 raw, exp 0083 re-probe
with parent-deflation) yield a NEGATIVE RESULT — effrank-of-co-occurrence is not a
usable K_v CEILING (EHR co-occurrence is intrinsically ~100-dim/node; parent-
deflation does not collapse it). It survives as a FLOOR/prune detector (thin nodes
-> tpn=1) and the tooling is reusable. See the NEGATIVE RESULT section below.
Motivated by insight 0080's capacity hypothesis; the phenotype-profiling K should
be chosen for interpretability, not data dimensionality.

## Real-data finding (exp 0084) and the parent-deflation refinement

The first real probe (exp 0084, no-roll-up spectral, background-only deflation)
came back SATURATED: ~140/155 nodes hit thr=40 and n_probed=40 (max_probe ceiling),
PR spanned 2.3–36, and Σround(PR) ~2800 vs the current foreground K of 775 —
"crazy large", the opposite of the hoped-for efficient bound. Diagnosis: the RAW
per-node co-occurrence rank measures the node population's ENTIRE comorbidity load
(every rare-disease patient also carries hypertension/diabetes/labs/drugs), which
is high-dimensional EHR structure shared with background and ancestors — so raw
rank is closer to a VOLUME proxy than a diversity one (the labeled readout reports
corr(PR, log10 n_docs) precisely to confirm this).

**Fix — progressive (parent-)deflation.** The meaningful K_v is a node's phenotype
INCREMENT over its ancestors, not its whole load. The probe now deflates each node
against background + its already-recovered ANCESTOR anchors (the SAME seed_rows the
fit uses for that node, so it mirrors the forward topo-order deflation), collapsing
the shared structure and leaving the node's own increment. Implemented in
`gated_init.scalable_block_aligned_lambda` (the gated multidomain init — NOT the
STM `scalable_spectral_init_beta`, which was the earlier wrong home).

**Labeled sidecar readout.** The probe writes `effrank.json` (node -> {PR, thr,
gap, n, n_docs}) to the run dir (path via CHARM_PROBE_EFFRANK_OUT, set by
multidomain_cloud). `analysis/cloud/effrank_readout.py` (`make effrank-readout
ID=N`) joins it with manifest names + each node's training-doc count (n_docs, now
surfaced on ProjectedCoocResult) into a labeled table sorted by PR, plus the
PR-vs-volume correlation and a saturation warning. n_docs is captured free (the
per-node projected pass already counts docs).

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

## NEGATIVE RESULT (exp 0083 re-probe): effrank-of-co-occurrence is NOT a usable K_v ceiling

Parent-deflation did NOT collapse the ranks. At max_probe=100 the parent-deflated
Σround(PR) came to **4708** (vs ~2800 background-only at max_probe=40), with 50+
nodes still SATURATING at 100. Confound: we changed deflation method AND ceiling
(40->100) together, so most of the jump is the ceiling revealing previously-clipped
directions — but the magnitude settles it regardless: parent-deflation removes only
a handful of ancestor-anchor directions (~2-10), negligible against a co-occurrence
rank of 100+. Deflation cannot rescue it.

**Root cause:** EHR co-occurrence is intrinsically ~100-dimensional per populous
node — every patient carries a rich, distinct comorbidity/lab/drug profile — so the
effective rank measures the RICHNESS OF THE DATA, not the number of clinically
meaningful phenotype topics. There is essentially NO natural data-driven ceiling.
exp 0084 corroborates from the other side: at tpn=5 only 15/815 topics starved, so
the fit uses about as many topics as it is given. "How many topics does the data
support" has no useful finite answer here.

**What survives:**
- **effrank as a FLOOR/prune detector, not a ceiling.** The low end is clean and
  meaningful: nodes 102 (PR 1.6), 133 (2.6), 78 (3.6), 32 (4.4), 8 (6.0), 28 (6.7)
  are exactly the thin/degenerate nodes that floored/starved in the fit. effrank
  reliably flags nodes that genuinely want only ~1-2 topics -> use it to prune thin
  branches to tpn=1, NOT to set a large per-node ceiling.
- **The tooling is sound and reusable** (probe, sidecar, labeled readout,
  PR-vs-volume correlation) — it did its job by producing a clear negative result.

**Reframe — choose K per node for the OBJECTIVE, not data dimensionality:**
case-finding -> ~2 (exp 0084: more dilutes/hurts); phenotype profiling -> a small
human-interpretable number (2-5 per class) with effrank floor-pruning of thin
nodes. A principled larger per-node K, if ever wanted, needs a MODEL-SELECTION
criterion (per-node held-out predictive gain / coherence drop-off), not linear
co-occurrence rank.

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
