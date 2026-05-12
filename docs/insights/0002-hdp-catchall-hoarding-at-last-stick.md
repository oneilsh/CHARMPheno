# 0002 — HDP catch-all hoarding at the last stick
**Date:** 2026-05-12
**Topic:** hdp
**Status:** Observed

In our patient-year HDP run, the catch-all background topic concentrated in
*topic 149* — the **last** stick in a T=150 GEM truncation — and its
E[β] climbed monotonically (0.149 → 0.224 over iters 3→12) while the
effective-topic count dropped (83 → 62). The first ~14 sticks (topics
0–13) became "vestigial" variants of the same catch-all, all leading
with Essential hypertension at peak ~0.005.

This is unusual: in Wang/Paisley/Blei's reference HDP, the catch-all
typically lands at *topic 0* (first stick), because GEM stick-breaking
puts the largest weights at the start. The last-stick catch-all suggests
an initialization pathology — slot 149 grabbed early-batch variance
during warm-up and ran away with it, the GEM dynamics never redistributed.

**Implications.** A monotonically-climbing catch-all E[β] past ~0.20 is
pathological hoarding, distinct from healthy "background absorbs into one
topic, frees others for phenotypes" behavior. The diagnostic to watch
is not just "is there a big topic" but "is the big topic *still growing*
late in training."

**Mitigations to try next time.**
- Warm-start λ from an LDA fit's topic-word distributions instead of
  random init.
- Smaller T (e.g. 50) to force redistribution among fewer slots.
- Inspect whether shuffling stick order at init changes which slot wins.

**Setting context.** Online HDP, patient-year docs from condition_era,
T=150, γ₀=50, η=0.01, K_doc=15. Compared against a separate K=25 LDA run
on the same docs, which produced clean phenotype topics without this
failure mode (see [0005](0005-lda-decomposes-background-into-flavors.md)).
