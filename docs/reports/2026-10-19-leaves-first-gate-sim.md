# Leaves-first gate schedule: local Spark simulation (step 1)

**Date:** 2026-10-19 · **Script:** `analysis/local/sim_leaves_first.py` · **Feeds:** exp 0152 read 3b

## Question

On the whole-Mondo nested DAG (0152), some parent blocks hold a child's vocabulary and
some children lose their words to comorbidity. Does fitting in rounds help? The gate opens
by node height (leaves first), and each round adds the next level up. The test here is a
planted tree, before any cluster fit.

## Setup

- **Model and runner.** The production `GatedOnlineLDA` through `VIRunner` on local Spark,
  with the production mini-batch regime: subsample 0.1, tau0 64, kappa 0.51, 100
  iterations. The user flagged that full-batch EM can land elsewhere, so it is not used.
- **Plant.** Two branches. P has a dominant child (its own subtype under it), a minor child
  and a rare child; Q has two children. 40% of documents are background. Every document
  draws a common pool plus two random comorbidity clusters; labeled documents add
  signatures along the closure, with the node's own signature at 65% and its ancestors
  splitting 35%. There are 2 background topics, so node blocks are tempted by
  comorbidity, as at scale.
- **Schedule.** The gate state rides `global_params`, the per-iteration broadcast. Closed
  blocks are frozen. `warm` iterations run per height level, then the whole tree is open.
- **Metrics.** Per block: mass on its own signature (`own`), on its children's signatures
  (`kids`), and on comorbidity (`comorb`). Each value is the mean over 3 seeds.

## Results (own · kids · comorb)

| node (height) | single round | leaves, unseeded | leaves + seed, 25/level | leaves + seed, 10/level |
|---|---|---|---|---|
| P (2) | 0.22 · 0.20 · 0.16 | 0.02 · 0.06 · 0.20 | 0.25 · 0.14 · 0.17 | 0.21 · 0.09 · 0.18 |
| Q (1) | 0.16 · 0.14 · 0.17 | 0.01 · 0.03 · 0.43 | 0.33 · 0.06 · 0.17 | 0.30 · 0.03 · 0.17 |
| P.dominant (1) | 0.24 · 0.13 · 0.20 | 0.02 · 0.02 · 0.20 | 0.28 · 0.00 · 0.17 | 0.28 · 0.00 · 0.17 |
| P.minor (0) | 0.02 · 0 · 0.20 | 0.22 · 0 · 0.16 | 0.25 · 0 · 0.15 | 0.29 · 0 · 0.13 |
| P.rare (0) | 0.55 · 0 · 0.05 | 0.48 · 0 · 0.08 | 0.52 · 0 · 0.06 | 0.53 · 0 · 0.06 |
| Q.a (0) | 0.20 · 0 · 0.17 | 0.28 · 0 · 0.17 | 0.28 · 0 · 0.17 | 0.28 · 0 · 0.17 |
| Q.b (0) | 0.32 · 0 · 0.12 | 0.35 · 0 · 0.14 | 0.34 · 0 · 0.14 | 0.34 · 0 · 0.14 |
| P.dominant.sub (0) | 0.25 · 0 · 0.16 | 0.48 · 0 · 0.13 | 0.48 · 0 · 0.12 | 0.47 · 0 · 0.12 |

## Reads

1. **The plant reproduces both 0152 failures under the single-round gate.** Parents hold
   about as much of their children's words as their own (P kids 0.20 against own 0.22;
   Q 0.14 against 0.16). One child lost its words entirely: P.minor has own 0.02 and is
   comorbidity-led.
2. **Plain leaves-first kills every interior block.** A block opened late starts flat
   against already-sharp children and background, so it wins no responsibility and decays
   to the prior. The result did not depend on when the block opened: the 5 and 25
   per-level runs gave identical tables, because a dead block leaves the leaves'
   trajectory untouched.
3. **Seeding a block when it opens fixes both failures.** The seed is the word
   distribution of its subtree's documents, at a typical open topic's mass.
   - Children's words leave the parents: P kids 0.20 falls to 0.14 or 0.09, Q 0.14 to
     0.06 or 0.03, P.dominant 0.13 to 0.00.
   - Parents keep or gain their own stratum: P 0.21 to 0.25, Q about double.
   - Leaves recover: P.minor 0.02 to 0.25 or 0.29, P.dominant.sub 0.25 to 0.48.
   - The shorter warmup (10 per level) leaks less.
4. **Comorbidity absorption is unchanged** (about 0.12 to 0.18 everywhere). The schedule
   orders who claims node vocabulary; it does nothing about background capacity.
5. **The per-block step clock is moot at tau0 64.** rho only falls from 0.12 to 0.095
   over 100 iterations.

## Caveats and next

- **Small plant, 3 seeds, means only.** Before a cluster fit, check the per-seed spread
  and vary the parent share, the background count and the depth.
- **Production analogue of the seed.** Per-node subtree word counts come from one pass
  over the corpus; the spectral seed already gathers per-node closure counts. The
  schedule must ride the broadcast `global_params`, as here. Both are changes in spark-vi
  plus a driver flag; no hashed module moves.
- **Step 2:** the cardiovascular branch on the cluster, single round against leaves +
  seed, with the same seed and the read-3b digest on the cardiomyopathy and heart-failure
  nodes.
