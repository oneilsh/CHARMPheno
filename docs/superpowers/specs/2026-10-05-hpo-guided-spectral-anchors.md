# HPO-guided spectral anchors — design

**Status:** draft for review (2026-10-05, branch `claude/gated-conditional-voi`).
Implements handoff 2026-09-11 §5.1. No code has been written against this yet.

## Problem, in one paragraph

Spectral init fills every node's topic block (starvation 72% → 1%, exp 0114/0115/0123),
and a filled block carries its own node's signal (own-bg 0.731 / 0.730 vs 0.669 unfed,
exps 0115/0123). But the anchor words spectral chooses are picked by co-occurrence
geometry alone, which on an EHR corpus tracks *who gets coded a lot together*, not *what
the disease is*: intrinsic / dilated cardiomyopathy came back anchored on gestation-week
codes (exp 0114, insight 0082), because the pregnant-young-female stratum is a sharp,
pure direction in the co-occurrence matrix. The cost is real and uniform: −0.014 macro
AUC vs the random-init baseline at every depth (0123 paired AB), with the ladder saying
the block unit and the feeding are fine. The thing to fix is **which words anchor which
block**, and nothing else in the model.

## Idea

Each node's anchor search gets a *preferred candidate set*: the node's own HPO-profile
terms, realized to OMOP condition concepts, that survive the vocabulary and clear the
document-frequency floor **in that node's own training documents**. The greedy
farthest-point search runs over the preferred set first; if it cannot fill the block from
there, it falls back to the full candidate set for the remainder. Everything downstream is
untouched: deflation against background and ancestors still applies, β is still recovered
for every word against the chosen anchors, the fit still runs from the seed.

"HPO decides WHICH words may anchor; the data decides WHAT the topic is." There is no
strength knob. The only inputs are a set membership and the floor that already exists.

Nodes without a profile, or whose profile has no surviving term, get exactly today's
search, byte-for-byte. Partial coverage therefore costs nothing (44% of the CV branch's
label space is profiled at median 0.87 coverage, report 2026-09-06).

## What a "preferred candidate" is (the derived set, no knobs)

For node u, the preferred set P_u is the set of condition-domain vocabulary rows i such that

1. i is the vocab index of an OMOP concept in u's profile table (the same `--emit-eta`
   TSV the profile-eta work uses: `mondo_id, concept_id, weight, neg, coverage`),
   mapped through the bundle's `vocab_maps[0]` — membership there is the survivorship
   test (leakage strip, min_df, cap), exactly as `profile_eta.build_profile_eta_boost`
   does it;
2. the row is **positive** (`neg == 0`): a NOT-annotated phenotype must never anchor;
3. the row has **`weight > 0`**: the emitted weight is freq × IDF with IDF over the
   credited label nodes, so a term every credited node shares has weight 0. This is the
   knob-free exclusion of profile-generic terms ("fatigue", "dyspnea") — it uses a
   quantity the survey already derives, not a threshold we choose;
4. `df_w[i] >= min_doc_freq` in u's own training documents — the existing candidate
   floor of `find_anchors_projected` (ADR 0032), unchanged, applied to P_u as to
   everyone else.

`coverage` and the magnitude of `weight` are NOT used: using them would be a strength
knob. The preferred set is a set.

**Which nodes have a set.** The emitted table is the `--rollup` one (report 2026-09-06,
stage 2b): a label node's profile is the union of its Mondo-descendants' annotations,
frequency-max-pooled, so an internal node without direct HPO terms inherits from BELOW.
On the CV branch that reaches 132 of 299 label nodes; the other 167 have no annotated
descendant and get today's open search unchanged. Profiles are deliberately NOT rolled
DOWN from parent to child: unannotated siblings would then share one preferred set,
anchor on the same words, and their blocks would converge — the worst case for the
sibling-contrast readout. If the 0124 digest shows unprofiled children of profiled
parents still misanchored, the derived fallback to try is "parent's set minus the
parent's chosen anchors", recorded here so it is not reinvented as a knob.

**Deflation order** stays `forward` (ancestors first), as in 0115/0123: the parent takes
its profile terms, the child takes its own minus what the parent claimed. `reverse`
(leaves first, spec 2026-07-23) remains an orthogonal knob for a later A/B.

## The search (what changes in `find_anchors_projected`)

Today (per node u, scalable path, `gated_init.scalable_block_aligned_lambda`):

    seed_rows  = bg_anchors + anchors of u's already-recovered relatives
    fg_anchors = find_anchors_projected(QR_u, p_w, df_w, tpn, seed_rows=seed_rows)
    β_u        = recover_beta_projected(QR_u, p_w, seed_rows + fg_anchors)[len(seed_rows):]

Proposed:

    fg_anchors = find_anchors_projected(QR_u, p_w, df_w, tpn, seed_rows=seed_rows,
                                        preferred=P_u)          # new kwarg, default None

inside which the greedy loop becomes two stages over the SAME basis:

    stage 1: candidate = eligible & in(P_u)   → pick by largest residual norm, up to tpn
    stage 2: candidate = eligible             → pick the remaining (tpn − |stage 1|)

Both stages share the Gram–Schmidt basis (seeds, then stage-1 picks, then stage-2 picks),
so deflation and the hull-vertex geometry are identical to today; only the pool a pick
may come from differs. `preferred=None` or an empty set skips stage 1 and reproduces
today's output exactly (this is a test). A preferred row that is already in `chosen`
(an ancestor anchored on it) is skipped as today; one whose residual is ~0 after
deflation (linearly dependent on the ancestors' anchors) is skipped as today — both
fall through to stage 2 naturally.

The dense `spectral_init.find_anchors` gets the same `preferred` kwarg for parity and
for the unit tests that run without Spark; `spectral_block_aligned_lambda` threads it the
same way. The background step never receives a preferred set.

Multi-domain note: profile concepts are condition-domain only, and condition is domain 0
of the concatenated vocabulary, so P_u indexes directly. Measurement and drug anchors can
only arrive through stage 2 — expected, and visible in the counts below.

## Components (the eta wiring, mirrored)

- **Engine** `spark_vi/models/topic/spectral_init_scalable.py`:
  `find_anchors_projected(..., preferred=None)`; `spectral_init.find_anchors(...,
  preferred=None)`. Pure, id-agnostic, no new dependency.
- **Engine** `spark_vi/models/topic/gated_init.py`: `scalable_block_aligned_lambda(...,
  anchor_candidates=None)` and `spectral_block_aligned_lambda(...,
  anchor_candidates=None)`, where `anchor_candidates: dict[node_id, sequence[int]]`.
  Looked up per node inside the existing batched loop; a missing node → `None`.
  Returns, alongside λ, per-node counts `(n_preferred_eligible, n_anchors_from_preferred)`
  for the log/manifest (counts of words, never patient counts).
- **Estimator** `spark_vi/mllib/topic/pc.py`: Param `spectralAnchorCandidates` (default
  `None`), passed straight through at the two `*_block_aligned_lambda` call sites.
- **Driver** `analysis/cloud/gated_pc_cloud.py`: `--spectral-anchor-profile PATH`
  (default `''`). When set, a driver-side builder reads the TSV and produces
  `{engine_node_id: [vocab_idx, ...]}` from rules 1–3 above (rule 4 is the engine's
  floor, applied where `df_w` lives). New function `build_anchor_candidates(eta_df, *,
  eid_by_mondo, vocab_index_by_concept)` in `analysis/cloud/profile_eta.py` next to its
  sibling — same inputs, same disclosure-safe `stats` dict (nodes profiled / nodes
  mapped into this DAG / concepts surviving / concepts dropped as neg / as weight-0).
  Requires `--init spectral`; raises otherwise. Manifest field `spectral_anchor_profile`
  (the path's basename + row count) and the pooled anchor counts after the seed.
- **Wrapper** `scripts/run_experiment.py`: front-matter key `spectral_anchor_profile`
  emitted as the flag only when set, so every existing arg string stays byte-identical
  (the spectral and profile-eta pattern).
- **Log line** after the seed, next to the existing spectral banner:
  `[pc] spectral anchor guide: nodes guided=N/C; anchors from profile=M/(C·tpn);
  nodes fully guided=F, partially=P, fallback-only=Z`. Counts of nodes and words only.

Nothing hashed is touched (no cohorts / assembler / DAG module edits); no cache key moves.
`--py-files` is unaffected: the anchor search runs driver-side on the collected sketch.

## Validation

**Unit (no Spark):**
- `find_anchors(preferred=None)` ≡ today, and `preferred=set()` ≡ today, on a fixed Q.
- With a preferred set containing ≥ tpn eligible, non-degenerate rows: all tpn anchors
  come from it, in farthest-point order within it.
- With a preferred set of size < tpn: stage 1 takes all usable ones, stage 2 fills the
  rest from the full pool, and the basis is one shared Gram–Schmidt sequence (assert the
  result equals a hand-run two-stage reference).
- Preferred rows that are seeds, below the floor, or linearly dependent on the seeds are
  skipped and the count of "anchors from preferred" reflects it.
- `build_anchor_candidates`: neg rows excluded; weight-0 rows excluded; concepts not in
  `vocab_index_by_concept` excluded and counted; nodes not in `eid_by_mondo` skipped and
  counted; a node whose surviving set is empty is absent from the dict (not an empty
  list), so the engine's `None` path is taken.
- Scalable/dense parity on the simulator fixture already used by the spectral tests: the
  same preferred sets produce the same anchors on both paths.

**Cluster (exp 0124, "0123 + guided anchors"):** 0123's config verbatim plus
`spectral_anchor_profile: data/ontology/profile_eta_MONDO_0004995.tsv` (regenerate the
TSV on a fresh cluster: `hpoa-profile-survey --emit-codes` then
`hpoa-stage2-probe ID=116 --emit-eta`; bundle HIT needed). Same bundle, K, seed.
Reads, in priority order:
1. **Legibility (primary, the thing this build is for).** `inspect-topics --digest
   --grep 'cardiomyopathy|pregnan|gestation|atrial fibrillation|heart failure|valve'`
   vs 0123: do intrinsic / dilated cardiomyopathy shed the gestation anchors; does
   peripartum CM keep them (it should — pregnancy IS that disease); do AF/HF/valve stay
   coherent. The seed log's counts say how much of the anchoring the profiles actually
   decided.
2. **Starvation** from the fit log: must stay ≈1%. A guided anchor that the data cannot
   support would show as a thin block; the floor (rule 4) is what prevents it.
3. **Case-finding guardrail** at ridge 100: full head vs 0123 (0.7946) and 0113 (0.8087);
   detection vs 0.630; own-bg vs 0.730; paired AB vs 0123 by depth. Acceptance is
   non-inferiority to 0123 (within ~0.005 macro). Recovery toward 0113 is the
   hypothesis, not the bar: the readout is the instrument, the topics are the deliverable.

Decision reads: anchors re-align AND AUC holds → spectral + guided anchors is the base;
proceed to whatever is next on the list with this as the default init. Anchors re-align
but AUC drops → the profile terms are legible but not discriminative in this corpus; look
at WHICH profile anchors the heads do not read (the digest + readout loadings) before
touching the rule. Anchors do NOT re-align (pregnancy persists) → the preferred set is
being exhausted by rule 4 or dependence on ancestors; the seed counts will say which, and
that is a floor/deflation question, not a strength one.

## Interactions and risks

- **Generic profile terms** are the main risk. Three things stand between them and a
  block: the weight-0 exclusion (shared by every credited node), deflation against the
  ancestors' anchors (an ancestor that anchored on "dyspnea" removes that direction from
  every descendant's search), and the residual-norm criterion itself (a generic row is a
  mixture, not a vertex, so it loses to a pure row even inside P_u). If the digest still
  shows generic anchors, the next derived quantity to consider is restricting P_u to
  terms whose IDF is above the profile's own median — still a derived set, flagged here
  so it is not reinvented as a knob.
- **Profiled nodes whose terms are all below the floor in their own docs** get stage 2
  only, i.e. today's behavior; the log's `fallback-only` count makes this visible rather
  than silent.
- **Multi-parent nodes**: unchanged; the relatives set and deflation are what they are
  today. The preferred set is per node, not per path.
- **The TSV is cluster-ephemeral** (it lives under `data/ontology/`, wiped with `~`
  on restart). The driver fails loudly on a missing path. Regeneration is two make
  targets and a bundle HIT.
- **Spectral + profile-eta together** (word prior on the block in addition to guided
  anchors) remains untested and is NOT part of this spec; if 0124 re-aligns the anchors,
  the prior has little left to do (insight 0084: nothing to hold without a fed block;
  with a fed, correctly-anchored block the question is moot until shown otherwise).

## Out of scope

Strength, coverage or weight-magnitude gating of the preferred set; using profiles to
decide tpn per node; guiding the background block; document-credit weighting (§5.2);
frontier-only gating (§5.3); any change to the readout head.
