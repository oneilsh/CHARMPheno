---
id: 60
slug: dag-placement-rare6-frontier-anchors
status: pending
model_class: dag_placement
cohort: population_rare6
cohort_def: population_rare6
disease: rare6
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
n_bg: 40
tpn: 5
print_topics_every: 10
holdout_frac: 0.2
init: spectral
node_alpha_scale: 1.0
spectral_max_vocab: 12000
spectral_method: scalable
anchor_scope: frontier
strip_mode: both
max_iter: 1000
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0060 — DAG-placement rare6 forest, FRONTIER-scoped spectral anchors

Differs from exp 0059 (sym α + strip_both + scalable-spectral init) in
`anchor_scope: frontier` (0059 is the default `closure`) and, as of the long
run, `max_iter: 1000`.

The frontier-vs-closure A/B is already sealed at `max_iter: 200` (both full-200,
both reproduced): frontier wins every metric — detection 0.709 vs 0.697, and
deep-node routing `auc_by_depth` d2 0.604 vs 0.575, d3 0.586 vs 0.552. At 200
iters the fit had effectively converged by ~150 (full-200 ≈ early-stopped-157).
`max_iter: 1000` is a long-run probe: does that plateau simply hold, or is there
slow late-stage drift the 200-iter budget missed? Now that mini-batch fits run
the full budget (no noisy-ELBO early stop), all 1000 iterations execute. The
Robbins-Monro step size keeps shrinking (rho ≈ 0.017 at iter 200 → ≈ 0.007 at
1000), so late updates are small — improvement, if any, will be gradual. NB: this
overwrites 0060's run dir; the sealed 200-iter A/B numbers above are the record.

## The hypothesis

Anchor selection is a max-residual / farthest-point search — it picks the word
that is most geometrically distinctive (loosely, highest "information content"),
NOT the most frequent. So a rare disease's single most-defining term is exactly
the kind of extreme word the BACKGROUND pass can grab first; once it is a
background anchor, the node's own pass deflates against it and never recovers it.
The same theft runs parent → child down the DAG.

`anchor_scope: frontier` closes this off by drawing each anchor set only from its
own docs:

- background anchors from ONLY empty-frontier (true non-case) docs, so no
  foreground term can seed a background topic;
- node u's anchors (and its recovered β) from ONLY docs where u is the
  most-specific attested node (u in the frontier, no attested descendant), so no
  descendant term can seed an ancestor's topic.

The forward-topological ancestor deflation is unchanged — a frontier-`{u}` doc
still carries background + ancestor signal generatively, so u is still deflated
against background + ancestor anchors to isolate its own increment. What frontier
scope removes is the descendant contamination (and the foreground contamination
of background), so the anchor-stealing cannot propagate at any depth.

## What to read

- Compare `metrics.detection` and `metrics.auc_by_depth` against 0059. The
  hypothesis predicts CLEANER node/subtype anchors (less background/parent
  bleed), which should help deep-node routing (`auc_by_depth` at depths 2–3) and
  possibly detection.
- Expected cost: an internal node whose patients ALWAYS carry a more-specific
  subtype gets few/no frontier-`{u}` docs, so its block may stay at the prior
  floor (Σλ ≈ η·V). That is informative, not a bug — it means the node has no
  distinct own-signal separate from its children. Watch how many node topics fall
  to the floor vs 0059.
- The topic dumps should show fewer foreground disease terms leaking into the
  background (bg) block than 0059's closure-scoped background.
