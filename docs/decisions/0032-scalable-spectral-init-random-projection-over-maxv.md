# 0032 — Scalable spectral init scales by random projection, not vocabulary capping (maxV)

**Status:** Accepted (governs the not-yet-built scalable spectral-init arc; the current `spectral_init.py` is a small-V prototype)
**Date:** 2026-06-29

## Context

Spectral (anchor-word) initialization is the third STM stabilizer (insight 0029):
a deterministic, data-driven β seed via the Arora et al. anchor-word algorithm,
which cures the random-init collapse/blow-up. The current implementation
([`spark-vi/spark_vi/models/topic/spectral_init.py`](../../spark-vi/spark_vi/models/topic/spectral_init.py))
materializes a dense V×V word co-occurrence matrix Q on the driver — O(V²)
memory. That is fine at the cancer-demo scale (V≈3691 → ~109 MB) but a
non-starter at the vocabulary sizes spark-vi targets as a general library: a
dense V×V is ~80 GB at V=100k and ~320 GB at V=200k, on a single driver, which
is exactly where Spark's scale-out does not help. So before spectral init can be
wired to the cluster (ADR-pending arc), the anchor-word algorithm must be put in
a memory-scalable form.

There are two documented ways to scale anchor-word init to large V:

1. **Random projection of the word rows** (Arora et al. 2013, "A Practical
   Algorithm for Topic Modeling with Provable Guarantees"; and Mimno's dedicated
   "Random Projections for Anchor-based Topic Inference"). The convex hull of the
   V word-rows lives in a K-dimensional subspace, so a Johnson-Lindenstrauss
   projection to d ≈ max(K, ε⁻²·log V) dimensions preserves the hull geometry.
   Anchor selection runs on a V×d matrix, never V×V.

2. **Vocabulary capping** (the `stm` R package's `maxV` parameter). The
   reference STM implementation's default `init.type="Spectral"` keeps only the
   most-frequent `maxV` words for the initialization and reintroduces the rest
   afterward, bounding the gram matrix by capping V.

## Decision

The scalable spectral-init arc will scale by **random projection**, not `maxV`
vocabulary capping — a deliberate divergence from the reference `stm` default.

The candidate-frequency restriction (the anchor-word fragility cure) will be an
**absolute document-frequency floor** (a word may anchor only if it appears in
at least M documents, computed in the distributed co-occurrence pass), replacing
the current mean-relative `min_marginal_frac` floor.

## Alternatives considered

- **`maxV` vocabulary capping (the reference STM default).** Rejected for our
  use case. CHARMPheno's purpose is rare-subgroup phenotype discovery — surfacing
  the conditions that define minority arms (a rare cancer subtype, a gated
  foreground block). `maxV` keeps the *most frequent* words and drops the rest
  from anchor candidacy, which is precisely the set of rare-but-pure phenotype
  words we are trying to recover as anchors. This is the same failure mode as a
  mean-relative `min_marginal_frac` floor on a heavy-tailed vocabulary
  (measured: at frac=1.0, only ~13% of words clear the bar; mean/median ≈ 4×).
  Frequency capping and rare-phenotype discovery are in direct tension.
- **Dense V×V on the driver (status quo).** Rejected at scale: O(V²) memory,
  ~80 GB at V=100k. Acceptable only as the small-V prototype.
- **Sparse Q.** Rejected as unreliable: the head of a Zipfian vocabulary
  co-occurs broadly, so Q's high-frequency rows are dense even when the tail is
  sparse; the greedy projection geometry on a partially-dense sparse Q is awkward
  and still risks large intermediate memory.

## Consequences

- **Memory scales as O(V·d) with d ≈ max(K, ε⁻²·log V), i.e. ≈ O(V·K)** — about
  1–2 GB at V=100k, K=1000 (the projected rows V×d), versus 80 GB dense. The
  recovery step (NNLS per word against the K anchor rows) is per-word independent
  and distributes as a Spark map needing only the K anchor rows broadcast; the
  V×K recovery input and the K×V β output are ~0.8 GB each and never collected
  whole. Nothing is ever V×V.
- **Random projection keeps ALL V words as anchor candidates** — only the
  *dimension* is compressed, not the vocabulary — so rare-but-pure phenotype
  words remain eligible to anchor. The absolute document-frequency floor excludes
  only degenerate words (e.g. present in 1–2 documents, whose near-pure Q row is
  a spurious hull vertex), which is the genuine fragility the floor exists to fix,
  without the heavy-tail collateral damage of a frequency cap or a mean-relative
  floor.
- **Deliberate divergence from the reference implementation.** We follow the
  Arora/Mimno random-projection branch over `stm`'s `maxV` capping on purpose,
  because our objective (rare phenotypes) is the opposite of "use the frequent
  words." This is a documented, principled choice, not an oversight — recorded
  here so a future reader does not "fix" it back toward the reference default.
- **Time is the cheaper resource than memory** for this project's scale targets,
  so the extra distributed passes random projection buys (distributed
  co-occurrence accumulation, distributed NNLS) are an acceptable trade.
- **No current code changes to the algorithm.** This ADR is the design direction
  for the future scalable spectral-init arc; the prototype in `spectral_init.py`
  is unchanged. The `min_marginal_frac` → document-frequency-floor change lands
  in that arc, not now. Note: the existing dense spectral init IS wired into the
  cluster fit path (threaded through `StreamingSTM` and both drivers, default ON,
  validated by exp 0015 / insight 0030) — only the *scalable random-projection
  rewrite* governed by this ADR remains pending.
- **Dirichlet-family models are unaffected** — spectral init is an STM concern.

## References

- Arora, Ge, Halpern, Mimno, Moitra, Sontag, Wu, Zhu (2013). "A Practical
  Algorithm for Topic Modeling with Provable Guarantees." ICML.
  https://arxiv.org/pdf/1212.4777
- Mimno. "Random Projections for Anchor-based Topic Inference."
  https://mimno.infosci.cornell.edu/papers/sparse-anchors.pdf ; and
  "Low-dimensional Embeddings for Interpretable Anchor-based Topic Inference,"
  https://arxiv.org/pdf/1711.06826
- `stm` R package — `init.type="Spectral"` and the `maxV` parameter (CRAN
  vignette https://cran.r-project.org/web/packages/stm/vignettes/stmVignette.pdf
  ; source https://github.com/bstewart/stm/blob/master/R/spectral.R )
- insight 0029 (the three missing stabilizers); ADR 0031 (K−1 reference).
