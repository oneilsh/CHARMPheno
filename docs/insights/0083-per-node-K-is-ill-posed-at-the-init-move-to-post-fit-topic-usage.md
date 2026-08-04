# 0083 — Per-node K is ill-posed at the init (p≫n); the right question is post-fit topic usage

**Date:** 2026-08-04
**Topic:** capacity | per-node-K | spectral | parallel-analysis | pca | post-fit | discovery | decision
**Status:** CLOSING the init-side per-node-K arc (continues 0081's negative result
with the parallel-analysis estimator of exp 0087). The pa_k estimator + recurrence
floor is the best init-side K we can build, and it still trades one error for
another for a **fundamental** reason (p≫n). Recommendation: stop tuning init-side
thresholds; the well-posed per-node K — and the within-class discovery deliverable —
lives **post-fit** (θ-usage restricted to a node's patients). That readout is **not
built**. This doc is a handoff for building it in a fresh thread.

---

## TL;DR for whoever picks this up

- **Fit is locked.** Multidomain gated fit is stable/validated (insight 0082).
  Case-finding champion = **condition-only flat**; the hierarchy's value is
  **profiling / phenotype discovery**, not identification. Do not re-litigate this.
- **Capacity is uniform, and none of the per-node-K work touches the fit yet.**
  Every node gets a fixed `tpn` block (`dag_placement.py`; default `tpn=5`, exp
  0087 used `tpn=2, n_bg=40`). The whole effrank / pa_k line is a **pure diagnostic
  sidecar** — `allocate_topics` is never called by the fit path. We have been
  *characterizing* per-node K, not *changing* any fit.
- **Init-side K estimation is ill-posed at p≫n.** exp 0087's parallel-analysis
  estimator (pa_k) with a recurrence floor is the cleanest attempt yet. It kills the
  small-node inflation but is **too blunt** — it also zeroes decently-supported but
  *diffuse* phenotypes (POTS, eosinophilia), and it **does not decouple K from
  volume** (corr unchanged at +0.51). Every threshold trades small-node over-count
  for mid-node under-count. This is not a tuning failure; it is the wall.
- **The clean answer is post-fit.** Over-provision `tpn`, fit once (β learned
  globally where n≫p), then per node measure **which of its block's topics its
  patients actually load on** (θ-mass restricted to the node's docs) + top-tokens.
  That single readout gives (a) the real per-node K, (b) the phenotype profiles, and
  (c) *is* the within-class discovery deliverable. It makes pa_k optional (a cheap
  init sizing hint, nothing more). **This readout does not exist — build it next.**

---

## What we already have post-fit (and what we do NOT)

Easy to conflate these — they are all λ / word-side, not θ / usage-side:

1. **Health calls (λ-based).** `dead_node_report` and `starved_topic_report`
   (`multidomain_cloud.py:80-169`), surfaced as the "Health: dead nodes N/#nodes ·
   starved topics N/K" line in the fit card. A node is "alive" if any block topic
   has λ max/mean ≥ 5; a topic is "starved" if its λ row stayed ~flat (got ≈zero
   data). These measure whether *allocated* topics **peaked** — an over-provisioning
   signal read off the topic-word rows.
2. **Per-node top-tokens** (`summarize_fit.py` fit card) — the human-readable "what
   phenotype is this block," top terms per node per domain.

**Not built:** per-node **θ-usage** — restrict to a node's patients, ask which of
the global/block topics they actually load on and how much. This is the p≫n-immune
readout and the recommended next build (see below). Do not mistake the health line +
top-tokens for it.

---

## exp 0087 result (parallel-analysis pa_k, leading-run + recurrence floor)

Layout identical to exp 0085 (no-roll-up SNOMED hierarchy, umbrella root at
max_class_fraction 1.0), so pa_k reads on the same nodes we have effrank numbers
for. Estimator = **parallel analysis (Horn's method) with a per-node null** built at
the node's own sample size (see exp 0087 doc for the method and the two corrections
vs the old effrank probe).

Columns in the readout: `pa_k` = leading-run + τ=0.01 (no df floor); **`rec`
(=pa_k_rec)** = same + per-node recurrence df floor (min_df=3); `p_all` = count-all
diagnostic; `PR` = raw participation ratio; then depth, n_docs, node id, name.

**Headline numbers**
- nodes: 153 | Σpa_k (leading-run) = **597** vs current foreground K = 314
  (count-all diagnostic Σ = 1215; 6 nodes had pa_k_all > n_docs — the tail-noise
  inflation the leading-run rule removes).
- corr(pa_k, log10 n_docs) = **+0.51** (vs raw corr(PR, log n_docs) = +0.70).
- recurrence-floored **Σpa_k_rec = 244**; corr(pa_k_rec, log10 n_docs) = **+0.51**
  (UNCHANGED).
- by support: tiny <50 (n=61) −0.02 · small 50–300 (n=51) +0.35 · big ≥300 (n=41)
  +0.35.
- NOTE: 153/153 nodes saturate at n_probed=300 — raise `CHARM_PROBE_EFFRANK_MAX` to
  see their true rank.

**What the recurrence floor did — decisive, in two directions**
- ✓ **Kills the small-node inflation, as hoped.** Bicuspid cardiac valve (10 docs)
  9→**0**; Immune-mediated neuropathy (3) 2→0; Congenital pulmonary valve (27)
  14→8. The private-rare-word idiosyncrasy collapses.
- ✗ **Too blunt — also zeroes well-supported but *diffuse* nodes.** Not small:
  Disorder characterized by eosinophilia (131 docs) 21→**0**; Postural orthostatic
  tachycardia syndrome (257) 13→**0**; Neuromyelitis optica (79) 9→0; Vascular
  disorder (268) 8→0; Congenital anomaly of coronary artery (312) 7→0; Structural
  disorder of heart (140) 2→0.
- Meanwhile *coherent* big diseases keep their signal: SLE (2225) 6→6; Disorder of
  soft tissue (5042) 6→5; Disorder of brain (2461) 47→25.

**Interpretation.** The df floor is not separating **signal from noise** — it is
separating **coherent** phenotypes (codes many patients share) from **diffuse** ones
(POTS, eosinophilia — real syndromes whose codes are each rare, spread thinly across
patients). It nukes the diffuse ones as collateral. The tell: **corr(rec, log
n_docs) = +0.51, unchanged** — the floor didn't decouple K from volume, it just
scaled everything down and zeroed diffuse nodes across all sizes. So it trades
small-node over-count for mid-node under-count. Not "the right answer."

---

## Why there is no clean init-side answer (the p≫n wall, stated finally)

Every init-side lever trades one error for another for a fundamental reason:
- Raise the df / recurrence floor → kill idiosyncrasy **and** diffuse real
  phenotypes (this run, min_df=3).
- Lower it → keep diffuse phenotypes **and** the single-patient idiosyncrasy
  (earlier runs; min_df=2 barely moved anything).

No threshold separates them, because **at p≫n "a rare code in one patient" and "a
rare code that is part of a thinly-spread real phenotype" are the same object** until
you have enough patients for the second to recur. This is the same wall insight 0081
hit from the effrank side (rank measures word-space token richness, bounded by
~min(#words, d), not patient count). pa_k + per-node null is a genuine improvement
(it *is* sample-size aware — small nodes collapse), but the residual diffuse-vs-
idiosyncratic ambiguity is irreducible on the init side.

**Conclusion:** the init sketch can give a capacity *hint* (τ + a min_k floor), not
a per-node K. The well-posed K only appears **after** the global fit.

---

## The recommended next build: post-fit per-node θ-usage

The fit does its dimensionality reduction **globally** — β is learned over all
patients, where n≫p — so "which topics does this node's population use" becomes a
low-dimensional, well-posed question even for a 10-patient node. Concretely:

1. **Over-provision** `tpn` (generous per-node block; insight 0019: LDA gracefully
   under-uses excess capacity, so over-provisioning is safe — cost is compute, not
   quality).
2. **Fit once** (the locked multidomain gated fit; nothing new in the fit path).
3. **Per node, restrict θ to that node's patients** and measure the topic-usage
   distribution: which of its block's (and ancestors'/background) topics carry real
   θ-mass, how concentrated, with top-tokens attached.

This one readout yields:
- **(a) the real per-node K** — how many topics survived *use*, not a fragile init
  guess (robust to p≫n because it rides the global β);
- **(b) the phenotype profiles** themselves (top-tokens of the used topics);
- **(c) the within-class discovery deliverable** — essentially the same object.

And it demotes pa_k to an **optional init sizing hint** (how big to make the
over-provisioned block), sidestepping the small-node wall entirely. Over-provision
symptoms are already measurable via the existing **dead/starved health line**
(`multidomain_cloud.py`), which becomes the natural companion signal (surplus topics
show up as dead/starved).

**Caveats to design around (learned from 0081/0085):** a parent's gated block is
trained on its **whole descendant subtree** (a patient attests its frontier's
closure, so the parent block is active for all descendants' docs). This is why an
over-large parent block *can* grab a descendant-specific phenotype (0084's
likelihood-degeneracy). The θ-usage readout should attribute usage per node with the
gating scope in mind, and reverse-topo attribution (exp 0086) may still matter for
"phenotypes not correlated to known labels." Depth-driven attribution bit us in 0085
(forward order gave K to shallow nodes, ~0 to deep anchors) — post-fit usage should
be less prone to this since it reads actual loadings, but verify.

---

## Aside: can the spectral init be read through a PCA lens?

Asked in-thread; the answer splits cleanly and is worth recording:

- **The K-counting probe (pa_k / effrank) *is* PCA.** It is the squared singular
  values (eigenvalues) of the co-occurrence sketch — "how much variance per
  direction, how many directions matter." That half is PCA through and through.
- **The β-recovery (anchor words) is PCA's non-negative cousin, not PCA.** Same
  input object (word×word co-occurrence, ≈ XᵀX), different decomposition:
  - **PCA** finds *orthogonal axes* (eigenvectors) — signed, abstract directions of
    max variance. A PC mixes all codes with ± loadings → **not** a valid topic (a
    topic must be a non-negative distribution over codes).
  - **Anchor-word recovery** (Arora et al., separable NMF) finds the *vertices of
    the data simplex*: row-normalize each word's co-occurrence profile; the rows sit
    in a simplex whose **corners** are "anchor" words (codes ~unique to one topic);
    recovery expresses each code as a **non-negative** convex combination of those
    corners → interpretable topics.
  - Hold it as: **PCA finds the ellipsoid's axes; anchor-word finds the simplex's
    corners.** They span the *same* K-dim subspace — so "number of significant PCA
    eigenvalues" ≈ "number of anchors needed," which is exactly why pa_k
    (PCA-side counting) is a legitimate "how many topics" for a method whose recovery
    is NMF-side.
  - The greedy anchor search ("take the word farthest from the span of the anchors
    so far") is *precisely* the pivoted-QR our effrank probe runs — same
    rank-revealing geometry, one used to *select interpretable rows*, the other to
    *count eigenvalues*. Not a coincidence; the code is shared.

If topics-as-directions-of-variance are ever wanted, PCA on a node's patients gives
the subspace but loses the one-topic-per-anchor interpretability that makes β
readable.

---

## Pointers / code

- Estimator + probe: `spark_vi/models/topic/effective_rank.py`,
  `spark_vi/models/topic/spectral_init_scalable.py` (probe hook),
  gated init in `gated_init.scalable_block_aligned_lambda`.
- Layout / capacity: `dag_placement.py` (uniform `tpn` blocks; `allocate_topics`
  exists but is **not** wired into the fit).
- Post-fit health (companion to the proposed θ-usage readout):
  `multidomain_cloud.py:80-169` (`dead_node_report`, `starved_topic_report`);
  fit card + top-tokens in `summarize_fit.py`.
- Readout tooling: `analysis/cloud/effrank_readout.py` (`make effrank-readout
  ID=N`), corpus-bundle cache, `CHARM_MAX_ITER=1` effrank-only fast mode.
- Experiment: `docs/experiments/0087-multidomain-parallel-analysis-per-node-K-probe.md`.
- Lineage: insights 0080 (roll-up pooling hurts case-finding), 0081 (effrank NOT a
  valid K — the first negative), 0082 (no-roll-up recovers case-finding; flooding
  was the villain), 0019 (large-K graceful underuse → over-provisioning is safe),
  0017 (γ prior-dominance sank HDP → no per-node stick-breaking).

## Decision

Close the **init-side** per-node-K arc: the spectral sketch gives a capacity hint
(τ + min_k floor), not a per-node K, and no df/recurrence threshold fixes the
diffuse-vs-idiosyncratic ambiguity at p≫n. **Next step: build the post-fit per-node
θ-usage readout** (over-provision → fit → per-node topic-usage + top-tokens), which
answers the K question where it is well-posed and doubles as the within-class
discovery deliverable. Bank τ + min_k as the init sizing hint in the meantime.
