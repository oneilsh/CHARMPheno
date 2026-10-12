# Handoff — DisMech ribbon record (0137), code-map repair, and WP-D (2026-10-12)

**Branch:** `claude/dismech-ribbon` (all work committed and pushed; tree clean). Every
cluster command needs the AGENTS.md preamble:

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
```

Run dir of record: `/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap`
(survives cluster restarts; holds `gated_pc_result.npz`, `readout_heads_gated_pc_{mass99,stacked_mass99}.npz`,
`results_readout_{mass99,stacked_mass99}.json`, `readout_dag.json`, `bundle_meta.json`,
`code_map.tsv`, `profile_census.md`). HDFS caches die with the cluster; the first Spark
step on a fresh cluster rebuilds the bundle (~20 min).

## Where the line of work stands

1. **Label space (spec `docs/superpowers/specs/2026-10-07-dismech-ribbon-label-space-design.md`).**
   DisMech ribbon (3,226 Mondo ids at DisMech 71cd0452, `analysis/cloud/anchor_selection_data/dismech_ribbon.tsv`),
   flat fit DAG, R1a (a patient attests only the most specific member of a nested pair),
   hierarchy only in the READ (readout DAG of Mondo ancestors, ADR 0048).
2. **Code-map repair — the big finding (insight 0097).** The native-Mondo code map
   attested diseases through concepts far broader than the disease, three ways:
   (1) an ICD multi-map's context target (O90.3 → *Finding related to pregnancy*: every
   pregnant woman = peripartum CM, which under nesting was the DCM "young-women stratum"
   of 0124–0134); (2) an ICD xref whose only target is broad (tularemia ← GI-tract
   disorders; osteochondrosis ← vertebral-column disorders); (3) a wrong Mondo SNOMED xref
   (MONDO:0008559 *thrombophilia due to thrombin defect* → SCTID 111293003 *Venous
   thrombosis*: every DVT code). Fixed in `mondo_native_dag.py`, now **native-mondo-v6**:
   multi-map guard; hierarchy-consistency test (drop a non-SNOMED target SNOMED places
   above >= 10 Mondo terms Mondo does not place under the term); `EXCLUDED_MONDO_XREFS`
   (curated, one entry); build-time receipts (`multi-map guard`, `hierarchy test` with
   named drops and the kept 3–9 band, `SUSPECT Mondo SNOMED xrefs`, `attestation audit`).
   Withdrawn on their receipts: v3 anchor test (dropped DCM, Down syndrome, Lyme…), v4 at
   threshold 3 (dropped Down syndrome, AKI, ITP, acne). Main had documented the O90.3 case
   for the usage report in August (main's insight 0076, 1/m fractional attribution);
   the modeling path never took the lesson. What is NOT explained by the defect:
   starvation/deflation across hundreds of nodes, α, spectral init, every readout-
   instrument finding.
3. **0137 = the ribbon record (done).** v6 map, 389 diseases, tpn 5, K 3,145, fit 1.5 h
   (diag_only, unsupervised). Hierarchical readout 698 heads, 23.9M observed cells, dense θ
   (θ is diffuse even above α's floor: excess p10 0.39 at m=256 — top-m is not a lever
   here). Within-cohort 0.797; **root head 0.836 (best of series)**; stacked de-novo
   0.822 / AP 0.293; stacked − flat de novo +0.076, within-cohort −0.042; stacked marginal
   ECE <= 0.009 at every depth. Paired de-novo by Mondo id: vs 0135 −0.005 (581 nodes), vs
   0134 −0.026 (71 nodes; both baselines carry the v1 map, so part is the price of honest
   labels).
4. **Census (0136 tool, refined).** Own-code identity + thin floor + the usage
   dashboard's interpretation (relevance ranking λ 0.6, KL from the model word marginal,
   ANCHOR flag; strict own-code rule by relevance rank `--own-top 5`). On 0137: decidable
   n_profiles median 2, p90 4, 3% at the ceiling; Marfan / hEDS / DCM / thrombophilia
   read clinically. `tpn_max` 5 stands (0138 run log).
5. **WP-D (exp 0138, pending).** Built and unit-tested, not yet run on the cluster:
   * part 1 — `inspect_topics --voi REGEX` (`analysis/cloud/voi_tables.py`): per Mondo
     group (readout-DAG node), EIG (bits) of observing each code over the group's member
     diseases; profiles = census signatures; Poisson presence model with a background
     term; prior = scored test positives; own codes excluded and listed apart.
   * part 2 — `gated_pc_readout --conditional-from-heads stacked_mass99`: per readout-DAG
     edge, within-parent-cohort cond AUC / top-1 vs majority / ECE for flat and stacked,
     from the saved heads (test split only, no solve).
   * part 3 (not built) — simulated acquisition: hide held-out codes, reveal in EIG vs
     random vs information-content order, measure posterior concentration.

## Next actions (in order)

1. Run 0138 part 1 (minutes) and part 2 (~1–1.5 h incl. bundle rebuild); commands are in
   `docs/experiments/0138-wpd-conditional-diagnosis-voi-on-0137.md`. Candidate sets:
   `'^cardiomyopathy$|^connective tissue disorder$'` (exact Mondo names). Read: are the top
   EIG codes clinically discriminating; flat vs stacked cond AUC by depth; which parents
   beat majority.
2. If parts 1–2 look sensible, build part 3 (needs a per-patient θ recompute with codes
   hidden — the gated CAVI transform on a modified bag).
3. One modeling comparison against 0137 (pick one): **source-code-first attestation**
   (match `condition_source_concept_id` to Mondo's own source codes before any `Maps
   to`; main's usage-dashboard rung 1 — the cleaner long-term code-map fix) or the
   **fixed-α ladder** (α 0.1 / 0.02, maybe a narrower background: "are patients spiky?").
   Optionally a per-disease tpn from the census.
4. Housekeeping: file the Mondo issue (draft in the session: MONDO:0008559 same_as
   SCTID:111293003 *Venous thrombosis*); the curated-exclusion receipt counts only its
   second stage (cosmetic, hashed module — fold into the next deliberate cache drop);
   pre-existing failing test `test_readout_integration.py::test_the_fit_only_save_precedes_every_readout_in_main`;
   stale statuses on `claude/gated-conditional-voi` (0076–0099 "pending", 0111/0112
   "running", 0124/0126 "pending"); multi-domain 0137b and the export/frequency-band WPs
   (E/F/G) stay deferred; the usage dashboard (main) is out of scope for the ribbon.

## Constraints the user set (keep)

* No LLM calls on the cluster (the dashboard's `scripts/label_phenotypes.py` is not run).
* Egress floor: no cell < 20; never commit patient-level data; pooled figures only.
* Hashed modules (`mondo_native_dag.py` etc.): edits are deliberate, announced cache
  drops in one commit; the tripwire suite stays byte-identical.
* REVIEW_LOG is the human walkthrough sign-off trail only (docs/META_process.md §4);
  arc summaries go in experiment docs, insights, spec status, or a handoff like this.
* No LaTeX in chat; every cluster command carries the preamble.

## Tool cheat-sheet (all in the 0138 / 0137 docs)

* Build audit: `grep -E "multi-map guard|SUSPECT|attestation audit|label set dismech|label DAG \(" <run>/driver_log.md`.
* Readouts log to `<run>/readout_log.md` (not driver_log). Fit-only manifests are
  expected for diag_only fits.
* Census: `make -C analysis/cloud inspect-topics ID=137 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --relevance-lambda 0.6 --own-top 5 ..."`; reads `<run>/bundle_meta.json` (no HDFS needed).
* Paired AUC by Mondo id: `--readout-auc --results-file results_readout_stacked_mass99.json --compare-results-file <f> --readout-label gated_pc_stacked.marginal_ranking.stacked`, `COMPARE=<id>`.
* Fresh cluster, old run: `make -C analysis/cloud gated-pc-readout ID=<n> GPR_ARGS="--bundle-only"` writes `bundle_meta.json` + `code_map.tsv`.
