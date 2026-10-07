# DisMech ribbon as the label space — design

**Status:** REVIEWED 2026-10-07 (decisions recorded at the end); WP-A next. Branch `claude/dismech-ribbon` (forked from
the tip of `claude/gated-conditional-voi`; voi is frozen until the exp 0134 readout is
in). Nothing here is built yet. Follows the 2026-10-07 arc review (this session) and the
2026-10-06 handoff
(`docs/reports/2026-10-06-guided-anchors-and-stratum-capture-handoff.md`).

## Why (the arc review in four sentences)

Every problem of the last two months — blocks that are patient strata not signatures
(exps 0123–0131), the peripartum stratum leaking into dilated cardiomyopathy through
closure scope, the open "same-vocabulary child gets a cohort" item (insight 0094), the
per-node-scope rule we declined to build, the depth-5 starvation cliff (insight 0079) —
is created by **nesting the label set on the topic side**: a parent and child compete
for the same documents and the same words, and every fix is a rule for who wins. The
hierarchy earned its keep in exactly one place: the **stacked closure read** (insight
0092, de-novo AUC 0.712 → 0.850, gain growing with depth), which needs the DAG only at
readout, over a set of heads. The project's founding aim was to *characterize diseases*
(exp 0030 / insight 0035: a gated foreground recovers clinically faithful EDS
sub-phenotypes against a population background), and DisMech
(<https://dismech.monarchinitiative.org/>, `monarch-initiative/dismech`) now offers a
curated, deliberately **flat** disease set keyed to Mondo, with HPO annotations and a
LinkML data model that can carry EHR-derived profiles. So: take the DAG out of the fit,
keep it in the read, and make DisMech the label space and the consumer.

What this is NOT a reversal of: the gate (insight 0093: +0.03 within-cohort, +0.05
de-novo AP over flat strata), spectral anchors (0124: free), the readout instrument
(ridge 100, insight 0089; the stacked head), the cache/sidecar/distributed-readout
infrastructure. All of it is label-space-agnostic and all of it is kept.

## Definitions

- **Ribbon.** The set R of Mondo ids of DisMech *disorders* (`kb/disorders/*.yaml`,
  one `disease_term` MONDO id each). **Pinned to DisMech commit `71cd0452`
  (2026-10-07):** 3,312 disorder files, 3,273 with a MONDO `disease_term`, 3,218
  distinct MONDO ids (a few disorders share a term; they collapse to one label node).
  Subtypes (`has_subtypes`, present on 1,095 disorders) are NOT in R. DisMech's
  selection is already a cross-cut of Mondo at roughly one depth; R is treated as flat
  even where Mondo nests two members (that is a DisMech curation question, surfaced as a
  receipt, not a modeling rule). The pin is recorded per experiment like `mondo_version`.
- **DisMech classifications (what they can and cannot carry).** Of 21 classification
  systems, two have real coverage: a coarse free-text `category` on nearly every
  disorder (Mendelian 1,925 / Complex 363 / Genetic 291 / Infectious 168 / ...) and
  `harrisons_chapter` on 1,159 (35%: GENETICS_ENVIRONMENT 252, NEUROLOGIC 246,
  ONCOLOGY_HEMATOLOGY 184, IMMUNE_RHEUMATOLOGIC 101, CARDIOVASCULAR 87, ...). The rest
  are niche nosologies (ICIMD 171, ISDS skeletal 259, IUIS 168, ICD-O 162, mechanistic
  122, channelopathy 35). None is a hierarchy over R, and none covers R. So DisMech
  classifications are **candidate sets S for conditional diagnosis** (§D4) where
  present — Harrison's chapters and the specialist nosologies are exactly the "we know
  it's a connective-tissue / immune / skeletal disorder" framing — and NOT the source of
  readout heads. Readout heads come from Mondo (below).
- **Label node.** A member of R with closure support ≥ `min_positives` (same rule as
  exp 0110's native path: distinct persons whose frontier attestation rolls to the node
  through Mondo's is-a closure). Everything a patient's codes resolve to that is *under*
  a label node rolls up to it (`roll_terms_to_kept`); anything under no label node is
  background-only. `population_rare6` (6 anchors) and `population_rare_priority` (6 + 35
  DisMech priority anchors, `cohorts.py:159`) are the ancestors of this construction; the
  ribbon is the same idea at R's size.
- **Readout node.** A label node, OR a Mondo ancestor of label nodes that the stacked
  read uses as a head. Ancestor labels are derived (union of descendant label-node
  attestations); they exist only in the readout, never in the fit.
- **Subtype head** (DEFERRED, decided 2026-10-07). A powered DisMech subtype could be a
  *child readout node* of its disorder (P(subtype) = P(subtype | disorder) · P(disorder)).
  Not a topic block, not in 0135–0137; an easy later add because the readout DAG is
  built from the full Mondo parent map anyway.
- **Profile.** One topic of a label node's block, exported as a per-domain weighted code
  list (`docs/proposals/dismech-profiles/profiles.yaml`, salvaged from the orphaned
  `linkml-phenotype-profiles` branch). A disease has `n_profiles(d)` ∈ [0, tpn_max].
- **Signature vs stratum.** A block topic is a *signature* when its word distribution is
  specific to the disease (far from every background topic and from the node's
  marginal); a *stratum* when it is a care-context split of the node's patients that a
  background topic already explains (exps 0127–0131 showed tpn=5 blocks were one
  signature + four strata). The distinction is a derived quantity (§D3), not a knob.

## Requirements

- **R1 — flat fit, hierarchical read.** No label-node block is nested under another on
  the topic side. The stacked read uses Mondo closures over readout nodes.
- **R2 — multiple profiles per disease are allowed and *estimated*.** tpn is a ceiling
  (`tpn_max`), and `n_profiles(d)` is read post-fit (§D3). The export carries only
  signatures. (Rare6/EDS at 20 topics per block is the precedent; 0133/0134's tpn=1 is a
  CV-branch result under nesting, not a law.)
- **R3 — DisMech is the eventual consumer, not the first deliverable.** The
  `ProfileSet` export and the frequency-band comparison (§D5.1, D5.3) are *designed
  for* here so nothing in the fit or census forecloses them, but they are NOT on the
  0135–0137 critical path (decided 2026-10-07). When built, profiles exist only for
  diseases with cohort ≥ 20 and codes with count ≥ 20.
- **R3b — multi-domain infrastructure is kept and exercised.** The per-domain λ
  engine, `multi_domain.py`, `measurement_tokens.py`, `--extra-domains` and the
  per-domain `CodeDistribution` factoring of the export contract all stay. Condition
  only in 0135 (one variable at a time against 0133/0134); drug + measurement enter at
  0137 or a 0137b, with the orphaned-branch findings (measurement rescues labs-dependent
  diseases but degrades condition in a joint fit; no fixed combination rule beats
  condition alone) salvaged into a report at that point.
- **R4 — HPO is incorporated three ways** (§D5): as an *alignment score* on each
  profile (now, in the census); as *observed-vs-literature frequency bands* per
  (disease, HP term) (later, with the export); and optionally as the word-side
  `profile-eta` prior (0116/0117: buys alignment, not AUC — which for characterization
  is the deliverable).
- **R5 — conditional diagnosis is a readout, not a fit property** (§D4): P(d | θ, d ∈ S)
  for a candidate set S (a Mondo ancestor's ribbon descendants, or a DisMech
  classification tag), and the code-level VOI from the per-disease β contrast.
- **R6 — no patient-level data leaves the workspace;** all exports are pooled, floor 20.

## Design

### D1 — Label-space construction (the only hashed-module edit)

`mondo_native_dag.build_mondo_native_fit_inputs` gains a `kept_filter: set[str] | None`
(Mondo curies). With it set, step 3 (powering) thresholds closure support over
`kept_filter` only, and step 4 builds the label DAG over the powered subset of
`kept_filter`. The ribbon file is a TSV of `(mondo_id, disorder_name, category,
harrisons_chapter)` derived from DisMech's `kb/disorders/` at the pinned commit (a small
`dismech_ribbon.py` reader over a shallow sparse clone; public data, no CDR; the TSV is
committed under `analysis/cloud/anchor_selection_data/` with the commit sha in its
header) and passed by the driver as `--label-set ribbon:<path>` (`--dag-source
mondo_native` stays). The classification columns ride along so §D4's candidate sets
need no second lookup. Receipts: |R|, powered count, members of R that Mondo nests under
another member (surfaced, not acted on), frontier terms that roll to no label node
(background-only mass).

This edits a **source-hashed module** (AGENTS.md "cache-key landmine"): one deliberate
commit that re-pins the tripwire hashes
(`tests/scripts/test_case_finding_cache_mondo.py`) and names the drop. Cost is low — the
HDFS caches die with the cluster daily — and the conversion sidecar is keyed
independently and survives. The hazard the landmine warns about (a poisoned cache under
a byte-identical key) does not arise, because the key moves.

The readout DAG (closure matrix for the stacked head) is built separately from the
*full* Mondo parent map over readout nodes = label nodes ∪ chosen ancestors
(`induced_hasse_parents`), so the fit sees a flat forest and the read sees the hierarchy.
Which ancestors are heads (decided 2026-10-07, after sizing DisMech's classifications):
every Mondo ancestor of a label node with ≥ 2 label-node descendants (that is where the
stacked product has something to multiply), from Mondo's own graph, up to and including
the root. DisMech classifications do not serve here (35% coverage, no hierarchy); they
serve as candidate sets in §D4. A later ablation can prune the ancestor set.

### D2 — The fit

Gated LDA through the existing `gated_pc_cloud` engine, unsupervised (PC stays off —
concluded three times: insights 0066, 0087, the 08-20 closeout). Starting config is
0134's with the label space swapped: spectral init with guided anchors (HPO profiles as
the preferred set, 2026-10-05 spec), raw counts (0123), wide shared background
(`n_bg` ≥ 1,200, pending 0134's verdict on whether width restores the root head),
`tpn_max` = 3 for the first run (R2 says a ceiling, not 1; 5 produced four strata on a
nested branch, 3 is the cheapest number that can still show ≥ 2 signatures), learned
α off (insight 0091: α is closed). Whole population (insight 0035: for a rare
foreground the background sample is the load-bearing knob).

### D3 — Estimating `n_profiles(d)` post-fit (new readout)

The orphaned branches found per-node K is ill-posed at init (p ≫ n) and should move to
post-fit topic usage; that readout was never built. Build it here, in `gated_pc_readout`
as `--profile-census`, per block topic k of node d:

1. **usage** — the node's θ-mass on k (pooled over the node's documents); below a floor
   the topic is *unused* (starved).
2. **specificity** — min over background topics b of a symmetric divergence between
   β_k and β_b, and the same against the node's pooled code marginal. A topic that a
   background topic already explains is a *stratum*.
3. **coherence** — held-out NPMI of its top codes, full-corpus reference (insight 0018).
4. **HPO alignment** — fraction of top-code mass realizable in the disease's HPOA
   profile via the frozen HP→SNOMED xref pin (v2026-06-23; `hpoa_profile_survey.py`).

`n_profiles(d)` = count of used, specific, coherent topics. Report the pooled
distribution of `n_profiles` over label nodes and the stratum rate; per-node rows stay in
the workspace. This also answers "what should tpn be" empirically: if `n_profiles` is ≤ 1
almost everywhere, `tpn_max` drops; if a tail sits at the ceiling, it rises.

### D4 — Readouts (hierarchy here, and only here)

- **Within-cohort and de-novo heads** as today: per-node logistic on θ (top-m), ridge
  100, closure mask with siblings as negatives; root observed on every row
  (`observe_root_everywhere`); `--readout-stacked` for the closure product. Closures come
  from the readout DAG of §D1, so the stacked read multiplies disorder × ancestors (and
  subtype × disorder × ancestors where a subtype head exists).
- **Conditional diagnosis.** For a candidate set S ⊆ readout nodes, P(d | θ, S) ∝
  σ(z_d(θ)) over d ∈ S (or the stacked score, when S spans depths). S comes from
  three sources, all available without a refit: the label-node descendants of a Mondo
  ancestor; a DisMech `harrisons_chapter` (1,159 disorders) or specialist nosology
  (ICIMD, ISDS, IUIS, ...); or a hand list. Report
  `cond_AUC` as the sober column (the 2026-08-14 VOI metrics report explains why
  `cond_AP`'s lift is mostly base-rate).
- **Value of information.** For a code w and candidate set S: expected posterior entropy
  reduction over S from observing w once, computed from the per-disease block β rows
  (signatures only) and the background β under the current θ. This is a β-contrast; it
  needs no DAG and no refit. First deliverable is the ranked-codes table for one or two
  hand-picked S (connective-tissue disorders; cardiomyopathies), pooled, floor 20.

### D5 — HPO and the DisMech exports

1. **ProfileSet export** (`analysis/cloud/export_dismech_profiles.py`): one
   `ProfileSet` per fit; one `Profile` per signature topic of each label node with
   cohort ≥ 20; one `CodeDistribution` per domain (condition now; drug/measurement when
   the multi-domain corpus is in the run — the schema is already factored per domain);
   `prevalence` = the topic's share of the node's block mass; `source` carries the exp
   id, CDR version, OMOP vocabulary version, `min_cohort_size: 20`. Validates with
   `linkml-validate`. Open naming questions for the DisMech devs are in the proposal's
   README and stay open.
2. **HPO alignment score** rides on each profile (D3 item 4) as `source.metadata`
   until the schema grows a slot; it splits a profile into literature-known and
   EHR-observed-but-unannotated manifestations, which is what a curator wants to see.
3. **Observed frequency bands.** A counting driver (no model): per (label node, HP
   term), the share of the node's cohort with ≥ 1 code realizing the term, binned to
   DisMech's `FrequencyEnum` (OBLIGATE / VERY_FREQUENT / FREQUENT / OCCASIONAL /
   VERY_RARE), suppressed below 20 persons. Delivered as a pooled table of
   (literature band, observed band) counts plus the per-disease rows inside the
   workspace. This reuses the usage-dashboard HPO axis plumbing (`mondo_usage_core`,
   the HPO xref parse) and is the one contribution DisMech cannot get from papers.
4. **profile-eta** stays an optional arm (`--profile-eta`), judged by D3 alignment and
   `n_profiles`, not by AUC.

### D6 — Dashboard

The Mondo + HPO dual-axis dashboard (main, `mondo-usage-dashboard/`) is keyed by Mondo
id, as DisMech pages are. Add a per-disease *profiles* panel fed by the ProfileSet
JSON (codes, weights, HPO-realizable marks, `n_profiles`), and the observed-vs-literature
band strip from D5.3. Payloads are already aggregated and suppressed; nothing new
crosses the floor. Embedding in DisMech pages by Mondo id is a later conversation with
the DisMech devs.

## Experiments (numbers continue the series; next free is 0135)

- **0135 — ribbon smoke.** Ribbon label space, `tpn_max` 3, `n_bg` per 0134's verdict,
  person_mod for a ~1-hour fit. Receipts of §D1; starvation rate; flat and stacked
  readouts at ridge 100. Comparison column: the label nodes shared with the CV branch
  (0133/0134) — within-cohort macro AUC and de-novo AUC/AP on the shared set.
  Pre-registered reads: (a) shared-node AUC within ±0.01 of 0134 → the hierarchy on the
  topic side was buying nothing; (b) clearly below → the gate's +0.03 needed the nested
  negatives, re-examine the closure mask on a flat forest; (c) above → nesting was a tax.
- **0136 — profile census.** `--profile-census` on 0135: the `n_profiles` distribution,
  stratum rate, HPO alignment; decides `tpn_max` for the record run.
- **0137 — ribbon record.** Whole population, `tpn_max` from 0136, both readouts, the
  conditional-diagnosis and VOI tables for two candidate sets (one Mondo-ancestor set,
  one Harrison's chapter). Export and frequency bands follow as their own experiments
  once the record is read; multi-domain (drug + measurement) as 0137b against 0137.

## Work packages and what each reuses

| WP | Builds | Reuses |
|---|---|---|
| A | `dismech_ribbon.py` reader + `--label-set` plumbing + `kept_filter` (hashed-module commit, tripwires re-pinned) | `mondo_native_dag`, `anchor_selection.py`'s DisMech seed parser |
| B | readout DAG from full Mondo over readout nodes; subtype heads | `induced_hasse_parents`, `closure_matrix`, `--readout-stacked` |
| C | `--profile-census` | `inspect_topics --profile-support`, NPMI eval, `hpoa_profile_survey` |
| D | conditional-diagnosis + VOI readout | `solve_batched_lr`, block β from `VIResult` |
| E | `export_dismech_profiles.py` + LinkML validation test | `docs/proposals/dismech-profiles/`, beta writer |
| F | frequency-band counting driver | `mondo_usage_core` HPO axis, usage export |
| G | dashboard profiles panel | `mondo-usage-dashboard/` |

A → 0135 → B/C → 0136 → D → 0137. A, B, C, D are the critical path. E, F, G are
designed for but deferred (decided 2026-10-07); they depend on nothing but a read 0137.

## Branching, salvage, and cruft (to become an ADR with WP-A)

- Branch from the tip of `claude/gated-conditional-voi`: all infrastructure lives
  there; `main` is 381 commits behind and holds nothing voi lacks except the dashboard
  payloads. voi stays frozen until 0134's readout is in; then voi → ribbon merge once.
- Salvaged in this branch's first commit, verbatim and dated (no renumbering needed):
  the DisMech LinkML proposal (`docs/proposals/dismech-profiles/`) and the three
  2026-08-14 lit-review reports (rare-disease diagnosis; sureLDA/PhenoBrain; VOI
  metrics). The orphaned multidomain findings (roll-up flooding, measurement vs
  observation, hybrid weighting) get a salvage *report* under a fresh date when the
  multi-domain corpus re-enters (D5.1), not before.
- Cruft is removed in a deliberate sweep *after* WP-A–C exist and 0135 has run, as
  `case-finding` did in 2026-07 (dead-code sweep + ADR): candidates are the SNOMED
  `--dag-source snomed` path, `--anchor-scope`, `--spectral-topo-order`, the PC head
  arms. Nothing is deleted at branch time; docs are never cruft.

## Not in scope

Co-fitting any head; changing the label mask or document unit; PC; per-node scope rules;
re-litigating α or the count transform; the whole-Mondo nested fit as a mainline
(remains runnable for comparison).

## Decisions taken at review (2026-10-07)

1. `tpn_max` = 3 for 0135; no side-by-side with 1.
2. Readout heads from Mondo ancestors ("≥ 2 label-node descendants" rule). DisMech
   classifications sized: `harrisons_chapter` covers 35%, the rest are niche; they are
   candidate sets, not heads.
3. Subtype heads deferred.
4. DisMech pinned at `71cd0452` (2026-10-07, "Regenerate pages, app data, dashboard, and
   schema docs (#13662)"); re-pin per experiment.
5. ProfileSet export and the frequency-band comparison deferred past 0137. Multi-domain
   infrastructure is kept (R3b) and re-enters at 0137b.

## Still open

- Whether a DisMech slot can take a weighted code list today, or `ProfileSet` is a new
  class proposed upstream — a question for the DisMech devs when E is picked up.
- The ancestor-head ablation (how far up the stacked product should reach).
