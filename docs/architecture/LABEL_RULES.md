# Label rules: what a case-finding row is, and what it may see

The rules every DAG-gated fit, readout and framing in this repo follows, in one place
(audit 2026-10-26, "Pre-index closure: cut ... to be written down in one place").
Each rule names where it is implemented; the code is authoritative where the two
disagree, and a change to a rule updates this page in the same commit.

## The row

1. **One document per person, one index per person.** The index date is the
   population index: a deterministic pick over the person's condition events
   (`min hash(person_id, event_date, salt)`, ties to the earliest date), gated on a
   fully observed label year after it (`cohorts.case_finding_population_index_table`
   via `_random_event_windows`). Not `F.rand()`: re-derivable, so the sidecars rebuild
   the same index (`conversion_sidecar.feature_window_index_table`).
2. **Features are pre-index.** Every domain (conditions, drugs, measurement result
   tokens) is read from the lookback window before the index
   (`multi_domain.lookback_feature_frames`, `lookback_days`). Post-index drugs and
   measurements never enter the bundle.
3. **Labels are the label year.** A node is positive when the person is attested at
   it, or below it (the is-a closure of the attested frontier), in
   `[index, index + label_window_days)`, conditions only (`attach_frontiers`,
   `attach_labels`; the native Mondo code ladder places codes on nodes,
   `native_dag.make_mondo_native_attested_provider`). `label_mask_mode="full"`: every
   node is observed for every row.
4. **The split is by person, first.** One salted hash of `person_id` splits every
   domain and the labels, so a person is on one side only; vocabularies and the DAG
   pruning are fit on train and frozen for test.
5. **Attesting codes are stripped from test features.** A node's own marker codes are
   removed from test documents (`strip_mode="test_only"`; `"both"` also strips train),
   so a test row is not scored on the code that defines its label.

## The fit and the readout

6. **The gate is train-only.** The fit admits node u's topic block on a training
   document only when u is in the closure of its frontier (the label-year frontier by
   default; all-ever attestation with `gate_attestation: ever`, exp 0146). The test
   transform is the deployment fold-in: ungated, full-K CAVI over every block
   (`DagGatedLDAModel._transform`), because at deployment the label is unknown.
7. **Heads train on all positives.** The per-node readout heads (`spark_vi.readout`)
   fit on every training row with its full label vector; nothing is held out by
   history.

## Evaluation

8. **Incident-only evaluation, from the whole-history sidecar.** A patient already
   diagnosed at a node (or anywhere in its closure) before the index is a tracking
   case, not a case-finding one. The incident arm drops those rows per node, using the
   first-attestation dates of the conversion sidecar, which reads the WHOLE history
   (not the lookback) with the code map built without the label-set filter
   (`prevalence_check.sidecar_nodes_to_readout`, the whole-history prior rule of exp
   0145; `--prevalence-check`). Reported metrics name their arm: all rows, or
   incident.
9. **Framing patients are incident within the group.** The differential framings
   (`framings.run_differential`) score test patients with no member or group
   diagnosis before the index and exactly one member first diagnosed in the label
   year; multi-positive patients are set aside (`framings._incident_test_frame`,
   `collect_set_patients`; exp 0151, insight 0103). The workup framing's window is `[index, t_dx)`, `t_dx` the
   member's first attestation (spec 2026-10-28 workup-voi-design).
10. **The egress floor.** No cell from fewer than `charmpheno.egress.EGRESS_FLOOR`
    (20) persons leaves the workspace; per-node tables stay internal, committed
    reports carry pooled figures (AGENTS.md, "Egress floor").

## What is not a rule any more

- **Pre-index closure `R_d`** (E1, a per-document closure of the pre-index frontier
  carried in the bundle): cut 2026-10-26, removed 2026-10-29; rule 8 answers the same
  question from the sidecar without a bundle column. Last at commit 29556f3d.
- **Fit-time label sets** (the ribbon as a fit-time filter): cut 2026-10-26; a ribbon
  is a post-hoc readout subset.
