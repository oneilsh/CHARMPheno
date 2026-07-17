# 0059 — The SVI gated engine left background docs ungated, polluting the node topics

**Date:** 2026-07-17
**Context:** exp 0055 (rare-disease-forest case-finding, `dag_placement`), first cluster run.

## Observation

The first rare6 forest run produced clean background topics but a **strong negative on
detection**: case-vs-background `auc=0.532`, `ap=0.056` (= prevalence 0.050); at 90%
sensitivity the background false-positive rate was 0.876 (precision 0.051 — no lift over
random). `ap_macro` was 0.013. The per-iteration topic log showed the cause directly: the
140 disease-node topics were mostly **generic comorbidity** (hypertension, obesity, nausea,
chest pain), not disease-specific — only a few (notably SLE) captured their disease's
vocabulary. The new `bg_mass` diagnostic confirmed it: background docs put only ~40% of
their topic mass on the background block (foreground ~36%), i.e. everyone spent ~60% of
their mass on disease-node topics.

## Root cause

`GatedOnlineLDA.local_update` (the SVI training E-step) treated an **empty frontier** as
*unlabeled → full-K* (`allowed = np.arange(self.K)`), not as *labeled background →
background-only*. So the ~183k background documents ran ungated CAVI and scattered their
sufficient statistics into **all K topics, disease nodes included** — the large background
population literally trained the node topics, collapsing them into generic comorbidity and
destroying node specificity. At deployment the transform is (correctly) ungated, so a
background patient then lit up those non-specific disease topics, and detection collapsed to
chance.

This diverged from **both** references, which gate a labeled-background doc to the
background block only:
- the collapsed-Gibbs oracle `dag_placement.fit_gated` gates every training doc via
  `allowed_set(label)`, and `allowed_set(empty) == background-only`;
- the gated STM `TopicBlockPartition.allowed_indices`: "a group with no foreground block
  contributes nothing (background-only) — this is what lets a large 'common' cohort inform
  the background while only rare groups carry foreground topics."

The divergence was **deliberate and test-locked** (`test_local_update_empty_frontier_is_ungated`),
so it was not a typo — it encoded the wrong semantics for the case-finding setup, where an
empty frontier is a known negative, not missing data. The equivalence validation missed it
because the synthetic plants label every document, so the empty-frontier branch was never
exercised. It also silently degraded the diabetes arms (0052–0054, ⅔ background — the
"background grabs a diabetes topic / split signal" note in insight 0058).

## Takeaways

1. **A background/negative label must gate to background-only, not full-K.** Full-K on
   background lets the majority class define the minority-class topics. Fixed to match the
   oracle + STM (unconditional `allowed_set(frontier)`).
2. **Engine-equivalence gates only cover the plants' regime.** Fully-labeled synthetic
   plants never exercised the empty-frontier path; the defect only surfaced on real data
   with a large labeled-background arm. Equivalence plants should include the dominant
   real-data regime (here: mostly-background docs).
3. **The detection block (bg_mass, case-vs-background AUC) was the diagnostic that made the
   pollution legible** — per-node AUC alone (0.64–0.68, "modest") hid it, because with many
   nodes the max-over-nodes affinity is inflated for everyone. Report the deployment metric,
   not just per-node discrimination.

Whether the fix is *sufficient* (does "which of 6 distinct diseases" then separate, once the
node topics are trained only on disease patients) is the question the rerun answers; the
seeded-β Monarch layer remains the designed remedy if node topics are still too generic.
