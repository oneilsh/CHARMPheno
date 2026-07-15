# The wall was the likelihood, not the estimator — and the metric was the wrong one

Two turns since the basin-trapping adjudication, and both were course corrections rather than
refinements. Here's where it went and where it's going.

## The pivot: one wall in five costumes

Reading the whole arc end to end, the calibration failure was never an estimator problem. Five
estimator families — Laplace (compressed), mean-field VI (attenuated, sign-flipped), EM (ran away),
the marginalized read-out (saturated), and exact Gibbs (basin-trapped) — all hit the *same* object
in the *same* regime (short items, scarce gated blocks). When point-estimation through exact
sampling all fail identically, the verdict isn't "find the right estimator"; it's that the
correlated-Gaussian-through-softmax likelihood geometry is multimodal and weakly identified exactly
where the use case lives. Continuing meant open-ended inference research with no guaranteed endpoint.

So the correlated-Gaussian latent is retired, and the work moved to the Dirichlet family. The
validated library came along unchanged — the identifiability compiler, the gating/masking, the
plant-and-recover discipline, the true-state oracle, the held-out calibration harness — because none
of it was ever about the Gaussian. The retired branch stands as a thorough negative result: a map of
where every estimator class fails on scarce gated blocks.

## The sharper correction: we were grading the wrong thing

The bigger realization: calibrated coverage was never the right success metric for the actual goal.
The goal is **placement / retrieval** — take an item with a known hierarchical label, hide it, and
place it in the label DAG from its features alone, scored by how close the placement lands
(DAG-distance, per-node precision/recall under heavy imbalance). That needs *monotone discriminative
scores*, not calibrated intervals. So the coverage failures that consumed the last several turns are
largely **orthogonal** to the objective — a genuinely liberating finding, because coverage in this
regime looks regime-hard for any family, and we no longer need to win it.

And on the metric that *does* matter, the same "separate what's correct from where the residual
lives" instrument that made the true-state oracle so clean paid off again. My first read-out forced
the affinity profile down to a single hard node (argmax + threshold), and placement accuracy was
barely above chance. But that collapse was the wrong read-out: scoring the **whole affinity profile
over the DAG** instead of one node, the signal was strong — per-node case-finding AUC ~0.96 at the
shallow level, degrading gracefully with depth, with the true node in the top-2 most of the time.
The hard placement was throwing away a good profile. Present the range of information, not one label.

## What made it work: structural gating, and refusing to collapse early

Two design moves carried it:

1. **Gated training ties topics to nodes structurally.** A labeled training item is masked to write
   only to the blocks along its label's closure, so each node's topic is anchored to its subtree by
   construction — no post-hoc alignment. That single change lifted deep-level AUC from 0.68 to 0.97
   (family 0.99, MRR 0.89, steady across seeds). The cross-branch bleed I'd been fighting *was* the
   post-hoc alignment.
2. **Don't collapse inseparable nodes early — make the confusion observable.** The tempting move is
   to run the identifiability compiler as a pre-fit editor and merge nodes it can't separate. The
   better move, and the one that matches your "measure, don't guess" reflex: keep the full DAG, and
   repurpose the compiler as a *post-fit diagnostic annotation* that labels which node-pairs are
   genuinely inseparable — so split affinity mass reads as "real ambiguity" vs "we lack the
   contrasts," and you collapse *later*, data-informed and reversibly, only if the profiles tell you
   to. Baking the merge in throws away the very observation that would justify it. (Merges, if ever
   made, stay within the structure — parent↔child or sibling — never across branches; cross-branch
   similarity is a reporting fact, not a structural one.)

## The design, converged

Anchor-first: pick anchor categories, build the DAG from the label hierarchy beneath them at
whatever irregular depth it has, keep every permissively-attested node, no baked-in separability
collapse. One windowed document per item; label = the most-specific in-window node, with sibling
ambiguity resolved to the lowest common ancestor (and the LCA-collapse rate *instrumented* so
starvation of deep nodes is visible, not assumed). Leakage handled asymmetrically — documents left
intact at fit time (the label codes anchor the topics, and the real targets lack them anyway, so
fit-with/score-without mirrors deployment), label-matching codes stripped only at evaluation.
Gated-train, spectral (greedy-anchor) init, no external-profile seeding required. Output is the
graded affinity profile; evaluation is per-node AUC + DAG-distance + MRR, imbalance-aware.

One knob I'd flag as more than a knob: multiple topics per node. It lets the model discover
distinct presentation clusters *within* a node that the label hierarchy doesn't encode, and —
across nodes — recurring presentation patterns. That's a discovery capability sitting orthogonal to
the given structure, and I suspect it's where something genuinely new could show up.

## Next

Validation so far is synthetic and model-matched — it proves the mechanics, not real-world accuracy,
and the real test is a held-out labeled corpus (imagine placing held-out items — tweets, say — into a
learned topic hierarchy) where the leakage line is where the result lives or dies. First real target
is a common category with clean sub-structure (power before rarity). The engine is domain-agnostic —
`(docs, labels, dag)` as integer ids — so the same code that scored 0.97 on the plant is what points
at the real corpus once it's assembled. Spec is written; plan is next.

The throughline I'd underline for you: the two best moves this whole arc — the true-state oracle and
the affinity-profile-vs-hard-placement split — are the same instrument, *isolate what is provably
correct so the entire residual has one named home.* That, and the discipline of keeping problems
observable rather than assuming them away, are what turned a dead correlated-Gaussian into a live
placement method.
