# 0091 — α is closed as a lever in both directions: learned α collapses to ~1/K and moves the starvation cliff UP a level; a HELD children-first α (equal total prior mass per block, ~1000× leaf/ancestor spread) feeds no extra depth either. The word competition is decided on β, not θ — which is why spectral init (a sharp β start) is the only lever that ever fed deep nodes

**Date:** 2026-09-11
**Topic:** gated-pc, alpha, empirical-bayes, deflation, starvation, spectral-init, exp 0121, exp 0122
**Status:** Confirmed on the CV branch at tpn=5, same bundle/budget as 0113 (fixed uniform α = 0.5), via exps 0121 (equalized init + learned α) and 0122 (equalized α held, optimizer off); depth tables from `inspect-topics --digest`.

## Observation

| fit | α policy | starved | fed through depth | depth-3 median ev | p90 ev | macro AUC @100 |
|---|---|--:|--:|--:|--:|--:|
| 0113 | uniform 0.5, fixed | 72% | 3 | 160 | 5.83e3 | 0.8087 |
| 0121 | equalized init → learned (≈0.001 everywhere) | 79% | 2 | 65.8 | 1.28e3 | 0.7895 |
| 0122 | equalized, HELD (leaves ~0.6, ancestors ~0.001) | 77% | 2 | 66.8 | 1.51e3 | — |
| 0114 (0082) | uniform 0.5 + **spectral init** | **1%** | **all** | (d4 1000) | — | not yet at a converged head |

1. **The optimizer erases any init in a few Newton steps and lands at α ≈ 1/K.** The
   ELBO wants sparse θ. At that α the E-step is winner-take-all among a document's allowed
   topics, the winners are the ancestor/background blocks, and the cliff moves UP a level.
   Ancestor capture is the marginal-likelihood optimum on this corpus, not an optimization
   failure — every fit-side lever that lets the model optimize ratifies it.
2. **Holding the tilt changes nothing.** 0122 gives a leaf document's own block ~600× the
   prior weight of each ancestor block, and the depth table is 0121's, not 0113's. The
   prior on θ cannot route a word to a block whose β is flat: φ ∝ E[θ]·E[β], and a
   uniform-over-5,000-words β loses every word to a sharp ancestor topic whatever θ says.
   The child never gets a word, so its β never sharpens — the flat-start trap (0079/0082),
   restated from the θ side.
3. **So the lever is β.** The one fit that fed every depth (0114: 72% → 1% starved,
   depth-4 evidence 62 → 1000, coherent AF / cardiomyopathy / valve topics per 0082) changed
   only the β START. Its recorded cost — "spectral costs case-finding at every depth"
   (0083: 0.7813 vs 0.7567) — was measured with the ridge-1, non-converged readout that
   insight 0089 showed is worth ±0.03 by itself, so it is NOT established.

## Why it matters

- **Close the α thread.** Uniform, learned, and held-asymmetric are all now measured at
  this scale; none feeds depth. Do not spend another fit on α.
- **The legibility goal ("fed, legible per-node topics", the thing the small-DAG era
  delivered) was already met once on this branch, by 0114,** and was set aside on an
  AUC cost read through a broken instrument. Re-reading spectral at a converged head is
  the cheapest route back to it. Note the run dir name is fixed per experiment, so the
  MAX_ITER=2 bootstrap re-run of 0114 overwrote its spectral λ: use 0115 (spectral +
  binary counts, which also fixed 0114's pregnancy-by-volume anchor) if its dir is intact,
  else refit.
- **The structural levers that act on β directly** — document-credit weighting in the λ
  update (a leaf-attested document counts fully toward its frontier block and only
  fractionally toward each ancestor block), frontier-only gating, or the SAGE cascade —
  are the ones left. Shallow DAGs (0071's per-system cascade; the dismech ribbon) sidestep
  the competition rather than win it, and are the regime where the engine already fed
  every node.

## Reproduce

```bash
make -C analysis/cloud inspect-topics ID=121 INSPECT_ARGS="--digest"   # and ID=122, ID=113
```
