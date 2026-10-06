# Handoff — guided anchors work; blocks are patient strata; frontier scope is the lever (2026-10-06)

Supersedes `2026-09-11-instrument-strip-alpha-and-the-beta-lever-handoff.md` as the entry
point (its §1–§3, §6 and cluster facts still hold; read its §7 table for 0113–0122).
Branch `claude/gated-conditional-voi`; cluster shape n2-standard-8 master (AGENTS.md).

## 1. What was settled, in order (exps 0123–0131; docs under `docs/experiments/`)

1. **Raw counts beat binary** (0123 vs 0115): +0.005 macro, detection 0.630 vs 0.604.
   Binary is dropped. Spectral's remaining tax vs random init (0113) is −0.016 median,
   concentrated at d5.
2. **HPO-guided anchors are built and free** (0124; spec
   `docs/superpowers/specs/2026-10-05-hpo-guided-spectral-anchors.md`): preferred-set
   two-stage search (`preferred=`), estimator Param `spectralAnchorCandidates`, driver
   `--spectral-anchor-profile`, front matter `spectral_anchor_profile`. 594/1490 anchors
   from profiles, paired −0.001 vs unguided. Legibility: cardiomyopathy d3 went from
   generic symptoms to a textbook block; intrinsic CM left the gestation-week stratum.
3. **Anchors are vertex labels, not topic words** (0125 anchor dump,
   `<run>/spectral_anchors.json` + `analysis/cloud/name_spectral_anchors.py`): the greedy
   search picks the purest rows = rare codes under the 5-doc floor; β recovery assigns the
   common words to the nearest vertex. Raising the floor to the node's mean word frequency
   (`spectral_marginal_floor: guided|all`, built) makes the anchors common words and
   changes NOTHING in the blocks (0129/0130).
4. **A node's block is a five-way split of its seed documents into co-occurrence strata.**
   The pregnancy stratum in dilated cardiomyopathy (0114–0130) is peripartum
   cardiomyopathy's patients training DCM's block under closure scope. Not an anchor
   effect (0125: anchors changed, block didn't), not fixable by leaves-first deflation
   (0128: deflation removes only what the child's anchors span; it SPREAD the stratum
   and thinned the parents), not by the prenatal-HPO exclusion (0125, kept; right in
   principle, 3 candidates).
5. **`anchor_scope: frontier` removes it** (0131, cardiomyopathy subtree): DCM's block
   loses the pregnancy stratum and gains a real DCM topic; pregnancy sits in peripartum.
   Cost: a grouping class nobody is coded with (intrinsic CM) has no frontier docs → no
   seed → junk block; readout −0.015 on 7 nodes, at that node's depth.

## 2. The subtree probe harness (use it)

`mondo_branch: MONDO:0004994` (cardiomyopathy, 17 nodes, K=93, 7 scored) runs seed+fit+
readout+digest+anchors in ~25 min (bundle cached per cluster). 0127 (forward, closure)
is the control; pairs 0128/0129/0130/0131 each changed one line. The compact read is in
0130's / 0131's Run sections. Numbers do not pair with the CV branch; mechanisms do.

## 3. Next — DECIDED 2026-10-06 (user): revisit HSLDA's output-side hierarchy

Spec: `docs/superpowers/specs/2026-10-06-stacked-closure-readout-and-hslda-arm.md`.
**Both parts are BUILT (2026-10-06); three cluster runs are queued, nothing has run.**

- **Part A — `--readout-stacked`** on `gated_pc_readout.py`: P_stack(c) = Π over
  closure(c) (root included) of σ(z_a). The finding that shaped it: under the closure
  mask the ROOT head is the degenerate constant 1.0 (observed only on foreground rows),
  so a product over the saved heads alone cannot move detection — the arm observes the
  root on every TRAIN row (`observe_root_everywhere`), the per-node-independent solve
  gives a case-vs-background root head and the prevalent arm's own heads for nodes ≥ 1.
  Block `gated_pc_stacked`: detection three ways (flat max / root head alone / stacked
  max), paired per-node ranking delta by depth, marginal ECE by depth. Outputs tagged
  `*_stacked`; record untouched. Commands are in the 0123 and 0124 docs ("Stacked
  readout" section), ~15 min each on a warm bundle.
- **Part B — a FLAT layout through the gated engine** (`tpn: 0`, `n_bg: 1498`): every
  node's block is empty and the background is the whole topic range, so every document
  sees all K — plain LDA on the same corpus, same saved format, same tools, no driver
  change (a real gate-off switch would need a fused-features path: the estimator refuses
  multi-domain columns without a gate). Exp **0132** = 0113 so configured + the ridge-100
  readout + the stacked readout. Command in the 0132 doc.
- **Read them together:** 0123-stacked (gated topics + stacked heads) vs 0132-stacked
  (flat topics + stacked heads), same tool, same corpus. Acceptance for Part A is in the
  spec: stacked detection above 0.63 AND above the root head alone, ranking not worse.

The per-node scope rule below is NOT being built (user: same kind of trick, tried many
ways); tpn=1 is the user's lean for the gated side, to be asked under the stacked head
(caveat: the signature was the minority topic in every 0127–0131 block).

Superseded options, kept for the record:

- **Per-node scope:** seed from frontier docs when the node has ≥ `min_positives` of them
  (the existing label floor), else from its closure. Keeps DCM clean, re-seeds intrinsic
  CM. Small change in the scalable seed's per-node doc filter (`gated_init`,
  `_anchor_node_set` / `_NodeGroups`); needs per-node frontier doc counts (one
  countByValue). Run as 0132 on the subtree, paired vs 0127 and 0131.
- **The representation question (user's call):** with tpn=5 a block IS five patient
  strata (acute inpatients, cardiac-risk men, hypothyroid women…), and only one of the
  five is the disease signature. Evidence: every digest from 0127–0131. Whether a block
  should be strata or a signature is not a run; it is the design question the handoff of
  09-11 §5 was circling (tpn was closed as a strength knob; this is a different framing).
- The CV-branch full run of the winning scope (0126 is on file for reverse order and
  should NOT be run; write a new doc for frontier/per-node scope when the subtree says so).

## 4. Operational notes added this session

- Chained readouts wedged on Dataproc's metrics-publisher JVM thread: fixed
  (`_driver_common.hard_exit`, `spark.dataproc.listeners=` on the readout submit).
- `optimize_doc_concentration: true` in 0113–0120 was inert and is LIVE since 0121: copy
  with `false`.
- `INSPECT_ARGS` is unquoted in the recipe: a `--grep` with `|` needs its own quotes.
- `CREDITED=1` needs the eta TSV (`~`, wiped per cluster); the survey now drops the
  prenatal HPO subtree by default (`--keep-prenatal` to reproduce 0124's table).
- Master: n2-standard-2 OOM-kills every bundle rebuild; n2-standard-8 is the shape.
