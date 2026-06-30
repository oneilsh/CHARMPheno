# 0032 — Scalable spectral init scales by random projection, not vocabulary capping (maxV)

**Status:** Accepted + Implemented (the scalable random-projection module
[`spectral_init_scalable.py`](../../spark-vi/spark_vi/models/topic/spectral_init_scalable.py)
is built and synthetic-equivalence-validated, and is threaded behind the opt-in
`spectral_method="scalable"` knob; the dense `spectral_init.py` remains the
exact, validated default for small V. Real-data topic-equivalence is CONFIRMED by
exp 0017: all 40 cancer phenotypes recovered, NPMI on par with dense 0015 (+0.166
vs +0.173). Σ-equivalence is PARTIAL — one dominant topic (hypertension/
cholesterol) escaped to Σ~8e5 vs dense's fully-bounded 7.56, attributed to JL
approximation pressure on the most-dominant co-occurrence geometry; see insight
0031. The scalable path is validated for phenotype discovery; Σ stability deferred
to the Σ-prior lever (exp 0018) or a larger projection dimension (exp 0019).)
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

- **Memory scales as O(V·d)** — about 1–2 GB at V=100k with the default d=1000,
  versus 80 GB dense. The recovery step (NNLS per word against the K anchor rows)
  is per-word independent and distributes as a Spark map needing only the K anchor
  rows broadcast; the V×K recovery input and the K×V β output are ~0.8 GB each
  and never collected whole. Nothing is ever V×V.
- **The projection dimension default is a FIXED ~1000**, not a V-scaled eps
  formula. d = min(V, max(K, 1000)). This is established practice across the
  anchor-word literature AND every reference implementation: Arora et al. 2013
  (ICML) projects to 4·log(V)/ε² with a footnote that "1000 works well"; Mimno's
  anchor-words study recommends "around 1000 random projections"; and three
  independent implementations hardcode project_dim = 1000 — ankura
  (byu-aml-lab/jefflund), anchor-topic (forest-snow tandem-anchors), and
  mimno/anchor (Java/MALLET, `--num-random-projections 1000`). Rationale: the
  greedy farthest-point anchor search needs only JL-preserved pairwise distances
  among the V word rows, and O(log V) dimensions suffice for that — a flat 1000
  is a safe, V-independent margin even at V=100k. At V=3000 this gives d=1000 (a
  3x reduction), well above the measured rare-arm cliff at d=89 (33x reduction).
  Cap at V (projecting beyond V is pointless); floor at K (need at least K
  dimensions for the K-simplex). (A future optimization could scale d with K
  rather than V — only the K anchor vertices must be preserved, not all V rows;
  Damle & Sun 2017 prove d=K+1 suffices w.p. 1 — but that diverges from the
  reference convention and is deferred.)
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
- **Implemented as a separate module; the dense path is untouched.**
  `spectral_init_scalable.py` implements this design — a single distributed
  projected co-occurrence pass (rank-1 per-doc accumulation into a V×d sketch),
  greedy anchor selection and per-word NNLS recovery in projected space, and the
  block-aware orchestrator `scalable_spectral_init_beta` mirroring the dense
  `spectral_init_beta`. The `min_marginal_frac` → absolute document-frequency
  floor (`min_doc_freq`, default 5) lands here. The dense `spectral_init.py`
  prototype is byte-for-byte unchanged and remains the exact small-V default;
  the scalable path is selected by the opt-in `spectral_method="scalable"` knob
  (threaded through `StreamingSTM` and both drivers, Task 7). The dense spectral
  init is the validated default-ON stack (exp 0015 / insight 0030); the scalable
  path is synthetic-equivalence-validated (planted rare-arm recovery matches
  dense at d≈1000; survives down to d=201 at V=3000). Real-data topic-equivalence is
  confirmed (exp 0017: all 40 cancer phenotypes, NPMI +0.166 vs dense’s +0.173);
  Σ-equivalence is partial — one dominant topic escaped to Σ~8e5 (see insight 0031);
  Σ stability follow-ups are exps 0018 (Σ-prior) and 0019 (larger d).
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
- Reference anchor-word implementations that hardcode project_dim = 1000:
  ankura (https://github.com/byu-aml-lab/ankura), forest-snow tandem-anchors /
  anchor-topic (https://github.com/forest-snow/tandem-anchor), and mimno/anchor
  (Java/MALLET, `--num-random-projections 1000`,
  https://github.com/mimno/anchor ).
- Damle & Sun (2017), "A geometric approach to archetypal analysis and
  nonnegative matrix factorization," Technometrics
  (https://arxiv.org/abs/1405.4275) — d=K+1 preserves all K extreme points w.p. 1
  (the deferred K-scaling future optimization).
- insight 0029 (the three missing stabilizers); ADR 0031 (K−1 reference);
  exp 0017 (the scalable real-data equivalence re-run).
