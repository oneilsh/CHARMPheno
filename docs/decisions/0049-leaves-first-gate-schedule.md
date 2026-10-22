# ADR 0049 — Leaves-first gate schedule with seeded opening (opt-in)

**Date:** 2026-10-20
**Status:** Accepted (experimental; default off)

## Context

On the nested whole-Mondo gated fit (exp 0152, read 3b) some parent blocks hold a
child's vocabulary and some children lose their words to comorbidity. A single-round
gate lets parent and child compete for the same documents from iteration 0. The local
Spark simulation (report 2026-10-19-leaves-first-gate-sim) reproduced both failures. Two
results followed: opening blocks by node height (leaves first) kills every late block,
because it starts flat against sharp competitors; seeding each block when it opens fixes
both failures.

## Decision

`GatedOnlineLDA.set_gate_schedule(warm, open_seed)` (spark-vi), exposed as the
`gateWarm` / `gateOpenSeed` estimator Params and the driver's `--gate-warm N` /
`--gate-no-open-seed`:

- **Rounds by height** (`DagLayout.height`, longest path down, so on a multi-parent DAG
  every child opens before any parent). The gate admits closure nodes with height ≤ the
  current level; after `warm × max height` iterations it is the production gate.
- **Closed blocks are frozen.** The round rides `global_params` (`gate_iter`,
  `gate_open_at`), so executors see it every iteration.
- **Seeded opening.** A block's row at opening is the word counts of the documents its
  gate admits (`gated_init.gate_open_seed_counts`: one pass, the E-step's scatter without
  CAVI), scaled to a typical open topic's mass per domain. The seed is driver-only:
  `__getstate__` drops it (ADR 0047).
- **Profile seed, optional** (`--gate-seed-profile TSV`, `gateSeedProfile`; added
  2026-10-21): a node with an external phenotype profile (HPO) in the vocabulary has its
  condition-domain seed replaced by the profile concepts, IDF-weighted over the run's
  profiled nodes (`profile_eta.build_profile_seed`). Leaves get it at iteration 0, later
  blocks at opening; other nodes and other domains keep the histogram seed or the init.
- **Balanced seed, optional** (`--gate-seed-mode balanced`, `gateSeedMode`; added
  2026-10-22): a late block's seed is the equal-weight mean of each child's normalized
  subtree distribution plus one share for the documents coded at the node itself, with
  every proper descendant's own attesting codes removed (`gateSeedOwnCodes`, from the
  native code map). It targets the dominant-child start. One pass, (nodes, V) per
  partition: branch-scale.
- **Default off.** `warm = 0` is byte-identical. It is a fit parameter: no corpus or
  bundle cache key moves.

## Consequences

The schedule changes optimization only, not the model; a converged fit is still a
gated LDA. The seed is the cluster mean, so very broad parents (near-root groupings)
may seed close to the background distribution; watch them. Validation starts on the
cardiomyopathy subtree (exp 0153 single round, exp 0154 leaves-first). Supersede this
ADR if the default changes.
