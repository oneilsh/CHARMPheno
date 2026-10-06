# Stacked closure readout + an HSLDA-like flat-topic arm — design

**Status:** Parts A and B built 2026-10-06 (branch `claude/gated-conditional-voi`);
awaiting the cluster runs (Part A on 0123/0124, Part B = exp 0132). Follows the 2026-10-06 handoff
(`docs/reports/2026-10-06-guided-anchors-and-stratum-capture-handoff.md`).

## Why (two sentences)

Ten runs (0123–0131) show a gated per-node block is a five-way split of the node's
patients into co-occurrence strata — one topic is the disease signature, four are
strata — and nothing done to anchors or deflation changes that. HSLDA (Perotte et al.
2011, `docs/references.md:97-109`) never put the hierarchy on the topic side: flat
shared topics, one head per label, a label can fire only if its parent fires, so
P(node) = Π over the closure of per-node conditionals. We built that head
(`DagClosureHead`, `spark-vi/spark_vi/models/topic/pc.py:573`; ADR 0042 "inject the
hierarchy once") and never read the product on real data.

## Part A — the stacked arm on the readout (no fit) — BUILT 2026-10-06

The readout head is already HSLDA's conditional: each node's logistic is trained only
on rows inside the parent's closure with siblings as negatives
(`label_mask_mode: closure`), so σ(z_c) ≈ P(c | parent(c)). What is missing is the
product. `--readout-stacked` on `gated_pc_readout.py` adds a **stacked score** per
(doc d, node c):

    log P_stack(c | d) = Σ_{a ∈ closure(c)} log σ(z_a(d)),   z_a = V[a]·θ_d + b[a]

(closure = c plus every ancestor INCLUDING the root, each counted once — diamond-safe,
the `DagClosureHead` convention; `gated_pc_cloud.closure_matrix` is asserted equal to
the head's matrix). An earlier draft of this spec wrote `closure(c) \ {root}`; that was
wrong, and for a reason that decides the design:

**The root head under the closure mask never saw the background.** The root is observed
only on foreground rows (`frontier_to_label`, closure mode: a background doc observes
nothing), all of them positive, so the readout fits it as the degenerate constant 1.0.
A product over such heads changes nothing at depth 1 and only shrinks deeper nodes — it
cannot move detection. So the stacked arm makes ONE change to the solve: the root is
observed on every train row (`gated_pc_readout.observe_root_everywhere`, `[1.0] ++
mask[1:]` on the train split only). The batched solve is per-node independent
(`solve_batched_lr`), so every head of a node ≥ 1 is byte-identical to the prevalent
arm's, and the root head becomes HSLDA's root conditional: a case-vs-background
logistic on θ. The product then carries that prior down the tree.

Reads, per node and pooled, next to the existing arm (block `gated_pc_stacked`,
`gated_pc_cloud.stacked_readout`):

| read | flat (`gated_pc`) | stacked (`gated_pc_stacked`) |
|---|---|---|
| detection AUC (case vs background, persons) | max_c σ(z_c), c ≥ 1 — the 0.60–0.63 number | **max_c P_stack(c)**; and, beside it, the **root head alone** (if stacked ≈ root-only, the product adds nothing a detection head would not) |
| per-node ranking AUC (within parent's cohort, same mask) | σ(z_c) | P_stack(c) — a real change even at depth 1, since the root factor varies per doc; reported as a paired per-node delta, pooled and by depth (counts of nodes only) |
| calibration | marginal ECE of σ(z_c) over all test docs by depth (expected bad: it is a conditional) | marginal ECE of P_stack(c) by depth; depth 0 = the root head's own |

Outputs are TAGGED so the record is untouched: `results_readout_stacked.json` (carrying
both `gated_pc` — the prevalent arm re-solved with the root head, nodes ≥ 1 identical —
and `gated_pc_stacked`), `readout_heads_gated_pc_stacked.npz`,
`readout_ckpt_gated_pc_stacked.npz`. Distributed readout + driver eval path only (the
product is formed on the collected (D,C) proba; no executor-side change — ADR 0047
untouched). Flag off → byte-identical.

Run on the saved 0123 and 0124 fits (`gated-pc-readout ID=123 GPR_ARGS="--readout-mode
distributed --readout-l2 100 --readout-stacked"`), ~15 min each, no refit. Acceptance:
stacked detection AUC materially above 0.63 AND above the root head alone, with per-node
ranking not worse (paired median ≥ 0). If stacked ≈ root-only, the heads' conditionals
add nothing to detection and the question moves to Part B.

## Part B — the HSLDA-like arm: flat topics + stacked heads — BUILT 2026-10-06, exp 0132

Same readout (Part A), applied to a FLAT fit. There was no usable existing path: the
driver's only ungated arm (`--with-dag-head`, the co-fit `DagClosureHead`) never saves its
globals, `skip_unsup_gated` governs a GATED twin, and the estimator refuses multi-domain
feature columns without a gate (`featuresCols ... require gateParent`) — so a true
"gate off" switch would have needed a fused-features path through the fit, the save, the
drift gate and the readout. Instead the flat model is the gated engine with a FLAT LAYOUT:
`tpn: 0` (every node's block is empty) and `n_bg: 1498` (the background IS the topic
range). Every document's allowed set is then all K topics — plain online LDA — on the
same multi-domain corpus, saved in the same format, re-read by the same tools with no
driver change. Pinned by `spark-vi/tests/test_flat_layout_tpn0.py` (layout K and blocks,
the E-step touching every topic for every doc, the equalized-alpha init returning
uniform instead of dividing by tpn — the one engine edit).

Exp doc: `0132` = 0113's config with `tpn: 0`, `n_bg: 1498` (the gated K, 8 + 298×5),
`init: random`, `optimize_doc_concentration: false` (live since 0121), `diag_only`; then
the ridge-100 readout (record + ABs vs 0123/0113) and the stacked readout. Reads: stacked
detection and per-node ranking vs 0123's stacked block (Part A on the gated fit — the
paired control, same tool); the flat heads' macro vs 0123's; cost. The digest has no
node blocks to report (every topic is background); interpretability on a flat fit lives
in the HEAD loadings (`--top-loadings`): which strata a node's conditional reads. Cost
note: with empty blocks the E-step touches all K per document where a gated document
sees only its closure's blocks, so an iteration is slower than 0123's; there is no
spectral seed.

Where this leaves tpn=1: if Part A/B say the stacked head is the right decoder, the
topic side only has to supply a signature per node, and tpn=1 (gated) vs flat-K (ungated)
becomes a two-arm comparison under the same head — the first time tpn is asked with a
head that needs exactly one thing from each block.

## Out of scope

Co-fitting the closure head (ADR 0042: never with the gate); SAGE regime (b); changing
the label mask; any strength knob.
