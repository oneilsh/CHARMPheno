# Stacked closure readout + an HSLDA-like flat-topic arm — design

**Status:** draft for build (2026-10-06, branch `claude/gated-conditional-voi`). Follows
the 2026-10-06 handoff (`docs/reports/2026-10-06-guided-anchors-and-stratum-capture-handoff.md`).
Nothing built yet.

## Why (two sentences)

Ten runs (0123–0131) show a gated per-node block is a five-way split of the node's
patients into co-occurrence strata — one topic is the disease signature, four are
strata — and nothing done to anchors or deflation changes that. HSLDA (Perotte et al.
2011, `docs/references.md:97-109`) never put the hierarchy on the topic side: flat
shared topics, one head per label, a label can fire only if its parent fires, so
P(node) = Π over the closure of per-node conditionals. We built that head
(`DagClosureHead`, `spark-vi/spark_vi/models/topic/pc.py:573`; ADR 0042 "inject the
hierarchy once") and never read the product on real data.

## Part A — the stacked arm on the readout (no fit; build first)

The readout head is already HSLDA's conditional: each node's logistic is trained only
on rows inside the parent's closure with siblings as negatives
(`label_mask_mode: closure`), so σ(z_c) ≈ P(c | parent(c)). What is missing is the
product. Add to `analysis/cloud/gated_pc_readout.py` / `distributed_readout.py` a
**stacked score** per (doc d, node c):

    log P_stack(c | d) = Σ_{a ∈ closure(c) \ {root}} log σ(z_a(d)),   z_a = V[a]·θ_d + b[a]

(closure, not path: diamond-safe, counts each ancestor once — the `DagClosureHead`
convention). The root's head is the one trained against the background, so the product
carries the population prior down the tree. Evaluate, per node and pooled, next to the
existing arms:

| read | existing (`gated_pc`) | new (`gated_pc_stacked`) |
|---|---|---|
| per-node ranking AUC (within parent's closure) | σ(z_c) | P_stack(c) — identical ordering within a parent's cohort only if all ancestors' scores are constant there; they are not, so this is a real test |
| detection AUC (case vs background, all persons) | σ(z_c) — weak, 0.60–0.63, the head never saw the background | **P_stack(c)** — the number stacking exists to fix |
| calibration (ECE by depth, `conditional_readout`) | per-node | stacked |

Seams: the per-doc per-node scores are already formed in `distributed_readout`'s stats
kernel (`_stats_kernel`, raw-θ scoring `V @ θ + b_raw`); the closure sets are
`DagLayout.closure` from the bundle's `parent_int`. One extra pass over the scored rows
that accumulates the closure-sum of log σ before the AUC/AP reducers (ADR 0047: nothing
array-shaped on a closure; treeAggregate zeros None-sentinel). Output: a third arm in
`results_readout*.json` and the log, same egress rules (pooled figures, counts of nodes).
Flag: `--readout-stacked` (default off → byte-identical results for every existing run).

Run on the saved 0123 and 0124 fits (`gated-pc-readout ID=123 GPR_ARGS="--readout-mode
distributed --readout-l2 100 --readout-stacked"`), ~15 min each, no refit. Acceptance:
detection AUC of the stacked arm materially above 0.63 with per-node ranking not worse.
If detection does not move, the heads' conditionals are the limit and Part B is where
the test continues.

## Part B — the HSLDA-like arm: flat topics + stacked heads

Same readout (Part A), applied to an UNGATED fit: the existing ungated path
(`skip_unsup_gated: false` / the ungated arm in `gated_pc_cloud.py`; no `gateParent`),
K fixed at the gated run's 1498 so the comparison is like for like (K is a knob in flat
LDA; matching the gated K is the non-arbitrary choice), same bundle, same split, spectral
or random init as 0113 (random; spectral's anchors are per-node and do not apply). Exp
doc: 0132 = 0113's config with the gate off, K=1498, readout with `--readout-stacked`.
Reads: stacked detection and per-node ranking vs 0123's; the digest (flat topics are
strata by construction — the question is whether the heads map them to nodes as well as
the gated blocks do); cost (a flat fit at K=1498 is cheaper than the gated one: no gate,
no spectral seed).

Where this leaves tpn=1: if Part A/B say the stacked head is the right decoder, the
topic side only has to supply a signature per node, and tpn=1 (gated) vs flat-K (ungated)
becomes a two-arm comparison under the same head — the first time tpn is asked with a
head that needs exactly one thing from each block.

## Out of scope

Co-fitting the closure head (ADR 0042: never with the gate); SAGE regime (b); changing
the label mask; any strength knob.
