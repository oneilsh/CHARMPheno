# Label redesign: all-ever topic gating, incident evaluation, staged rebuild

**Date:** 2026-10-13
**Status:** draft for review
**Builds on:** insight 0100 (strip no-op, random index mixes prevalent with incident),
exp 0143 (prevalence check), exp 0144 (incident-trained heads are worse), the ribbon
record 0137.

## Why

0143: once patients with a prior diagnosis of X are set aside, 0137's stacked macro AUC
falls 0.818 → 0.702 (flat 0.727 → 0.616). About half of every node's label-year positives
are prevalent re-codes. 0144: retraining the heads on incident labels makes this worse
(0.628 / 0.602) — prevalent positives teach the heads what X looks like. So the three
places a label enters the pipeline get three different rules:

| where | today | redesign |
|---|---|---|
| topic GATE (which label blocks a train doc may use) | X coded in the label year `[index, index+365d)` | X coded **ever** in the record |
| readout HEAD training label | X coded in the label year (mixed) | unchanged (mixed); ablation: ever-through-label-end |
| EVALUATION | X coded in the label year, all patients | X **newly** coded in the label year; patients with X **anywhere before the index** set aside |

The two leakage questions settle themselves under this split:

- **Own codes in the features stay.** Under incident evaluation every patient with an X
  code before the index is out of X's scoring, so X's own codes cannot inflate X's
  number; and they are the most informative tokens for topic learning. The native-path
  strip (a no-op, insight 0100) is left as it is and documented, not repaired.
- **Topic learning may use the future.** The gate shapes the topics on the TRAIN split
  only; test documents are folded in ungated (full-K CAVI) and never see a gate. Gating a
  train patient diagnosed three years after the label window into X's block teaches the
  block that patient's prodrome — wanted, not leakage.

## Facts this rests on (code, 2026-10-13)

- Gate = label today: one `frontier` column built from `label_events`
  (`multi_domain.py:235-289`), read as `frontierCol` by the estimator
  (`gated_pc_cloud.py:3117`); `label`/`labelMask` come from the same frontier
  (`multi_domain.py:309-314`).
- A driver-side frontier override already exists: `--gate-frontier-days N`
  (`gated_pc_cloud.py:4346-4361`) rewrites `frontier` only (`_overwrite_frontier`,
  `:3767-3810`) from the E4 first-attestation sidecar (`_gate_attested_frame`,
  `:3735-3764`), with its own cache-key token (`:3640-3663`). It is forward-only and
  runs only with `--index-arm`. **`gated_pc_cloud.py` is driver-owned: extending this
  seam moves no hashed module and drops no cache.**
- The E4 sidecar is `(person_id, node_cid, first_attested_date)` over the whole record,
  keyed independently of the bundle (survives a bundle-key move).
- The incident arm `preindex_closure` reads the 5-year lookback only (`preindex_closure.py:
  257-293`) and lives in the fit's node space; the readout skips it under the Mondo
  readout DAG (`gated_pc_readout.py:2227-2232`).
- The random index: deterministic hash pick over eligible condition-event dates
  (`cohorts.py:1087-1149`), 365-day prior-observation gate. Re-derivable by calling the
  assembler's own functions (calling a hashed module moves nothing).
- `[measured]` tokens (`STATE_PRESENCE = 0`, `measurement_tokens.py:42`) have no drop
  flag; the code map has an exclusion hook (`EXCLUDED_MONDO_XREFS`) but no addition hook.

## Requirements

### R1 — Incident evaluation v2 (readout-side; no fit)

- **R1.1** Per test document and readout node X: `prior[d, X] = 1` iff the sidecar has a
  first attestation of X, or of a label descendant of X, dated **before the document's
  index** (whole history, not the lookback). Node ids widen to the readout DAG exactly as
  `prevalence_check.vocab_to_nodes` does (via `pos_sources`).
- **R1.2** The index per document is re-derived by calling the assembler's own index
  builder (as `preindex_closure` does), and checked: a re-derived index set that does not
  reproduce the bundle's document count refuses to score.
- **R1.3** Report per node AUC over all patients vs prior set aside, as 0143; pooled
  only. Replaces 0143's lookback-codes rule; 0137 is re-scored under it first (the new
  baseline; expected at or below 0.702, since more prevalent patients are found).
- **R1.4** Driver-owned code only (`gated_pc_readout.py`, `prevalence_check.py`).

### R2 — All-ever topic gating (new fit; no cache drop)

- **R2.1** `--gate-attestation {label,ever}` (default `label` = today). `ever`: a train
  document's `frontier` = every label node with a sidecar first attestation anywhere in
  the person's record. `label`/`labelMask` untouched.
- **R2.2** Generalize `_overwrite_frontier` / `_gate_attested_frame` to the population
  lookback arm (lift the `--index-arm` requirement for this mode); cache-key token
  `gate-ever`. Train split only — the test frontier is never read at transform.
- **R2.3** Report at build: per node, gated train documents under `label` vs `ever`
  (pooled counts of nodes and the median ratio).
- **R2.4** Fit = 0137's configuration with only R2.1 changed (same K, α, tpn, seed,
  corpus). Readout: the record readout + stacked arm, then R1.

### R3 — Head-label ablation (readout-side; cheap, optional)

- **R3.1** `--readout-label ever-through-label`: train cells labelled positive when X is
  coded in the label year OR before the index (R1's prior); evaluation unchanged (R1).
  Tests whether still more positives help the heads (0144 suggests they might).

### R4 — Hashed rebuild (deliberate cache drop; after R2 reads)

- **R4.1** `[measured]` filter: drop `STATE_PRESENCE` measurement tokens at load, as an
  assembler parameter that enters the cache key (spec 2026-10-13 clinician framings D1).
- **R4.2** Code-map additions hook `EXTRA_STD_PAIRS` beside `EXCLUDED_MONDO_XREFS`, bump
  `MONDO_NATIVE_VERSION`; first entries from the 0141/0142 audits (HCM 40565500;
  Dupuytren's after a name check). Ancestor-group codes ("Cardiomyopathy" 321319,
  "Primary cardiomyopathy" 4232495) attest no label node and stay out — recorded, not
  forced onto a child.
- **R4.3** One commit names the drop and re-pins the tripwire keys in
  `test_case_finding_cache_mondo.py` (`_SNOMED_KEY`, `_SNOMED_LABELED_KEY`,
  `_SNOMED_LOOKBACK_KEY`, `_MONDO_KEY_NO_COLLAPSE`).

## Staging and reads

1. **0145 — R1 on 0137** (readout only). The honest baseline under the whole-history
   prior. Optional R3 in the same session.
2. **0146 — R2 fit.** Pre-registered read: incident macro AUC (R1) of the stacked
   readout vs 0145's baseline. A clear rise (> ~0.02) says label-year gating starved the
   topics of prevalent cases; flat says the topics are not the lever at this K and the
   next question is features/representation, not labels.
3. **R4 rebuild**, folded into the next fit after 0146 (it changes the vocabulary, so it
   is not mixed into the 0146 comparison).

## Cluster prerequisites

- The E4 sidecar for 0137's corpus (`make build-conversion-sidecar ID=137`), ideally on
  the persistent bucket (`GPR_SIDECAR_URI`) so R1 and R2 do not rebuild it per cluster.
  Ask for the current in-boundary bucket before writing.

## Open

- `ever` includes diagnoses after the label window. If 0146 helps, a `through-label-end`
  variant isolates how much came from future diagnoses (prodrome learning) vs prior ones.
- Whether incident evaluation should also require some minimum observation before the
  index beyond the 365-day gate (very short pre-index records make "no prior X" weak).
