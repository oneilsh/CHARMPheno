# Mondo + HPO dual-axis EHR-usage dashboard — design

Date: 2026-08-27 · Status: approved for planning · Branch: publish-main (mondo-usage-dashboard)

## Motivation

Mondo is a **disease** ontology, but a large, common class of EHR codes are **phenotypes /
findings / lab abnormalities** (headache, neck pain, nausea, dysphagia, hypomagnesemia).
They have no disease home, so the `source_climb` ladder climbs them to a general — or
occasionally *wrong* — Mondo term (headache → "vertebral artery occlusion").

The HPO phenotype-gap probe (exp 0108, 2026-08-27) sized it: of the standard SNOMED
concepts coded in the EHR that Mondo can only **climb** to or **drop**, HPO gives an exact
term to ~5–7% of *concepts* but ~**20% of climbed and ~34% of dropped person-mass** — the
missing phenotypes are common. That is a SNOMED-xref **lower bound** (4,371 HPO terms carry
a SNOMED xref; 12,816 more are UMLS-only). Verdict: a phenotype axis is worth adding.

## What we're building

A second ontology axis. One run attributes each EHR condition to its **single best home** —
Mondo disease term *or* HPO phenotype term — and the dashboard shows the two DAGs **side by
side**, Mondo growing rightward (as today), HPO mirrored growing leftward, unlinked.

## Decisions (locked)

1. **Routing = one extra rung in the existing ladder** (first hit wins, per condition):
   1. source-exact Mondo (ICD `same_as`)
   2. standard-exact Mondo (SNOMED → `Maps to`)
   3. **exact HPO** (the standard SNOMED concept — or source ICD — is an HPO xref) ← new
   4. climb Mondo (nearest mapped ancestor)
   5. drop
   So Mondo is preferred whenever it has a real term; HPO claims only what Mondo would
   otherwise climb/drop. Ambiguity (exact in both) → Mondo wins (earlier rung). Each
   condition lands in exactly ONE axis (best single home).
2. **HPO exact-only in v1** — no HPO-side climbing. Keeps the phenotype axis precise
   (no HPO overshoot). HPO climb is a possible v2.
3. **SNOMED-first matching.** The UMLS bridge (12,816 HPO UMLS xrefs → SNOMED via UMLS
   `MRCONSO`) is v2: free-but-gated UMLS license, used at build time only (never
   redistributed), staged onto the workbench. Design the xref table to accept UMLS rows
   later without a rewrite. Also match source ICD codes against HPO's (few) ICD xrefs.
4. **Full HPO DAG** — not anchored at Phenotypic abnormality; EHR codes may hit other
   branches (clinical course, past history, blood group, …). The existing "focus on what's
   used, allow full DAG browse" treatment prunes the visual noise, same as Mondo.
5. **Reversible.** Gated behind `--with-hpo`. Without it, `source_climb` behaves exactly as
   today (single Mondo payload); the dashboard falls back to the single Mondo view when no
   HPO payload is present. Going back to Mondo-only is a non-change.
6. **Disclosure unchanged** — per-axis small-cell suppression, bands are ranges, fractional
   1/m per axis; the safe summary gains an HPO-axis section. No new per-code patient counts.

## Architecture

### Driver (`analysis/cloud/mondo_usage_cloud.py`)

- **HPO ingest** (extend the probe loader): parse `hp.obo` into (a) an HPO **DAG** —
  `is_a` edges + `id`/`name` → `hpo_edges` / `hpo_nodes` frames analogous to Mondo's — and
  (b) the **code→HP xref** map (already have `parse_hpo_xrefs`). Cache the obo.
- **Extended ladder**: insert `t_hpo` between standard-exact and climb. `t_hpo` = conditions
  not in t1/t2 whose std SNOMED concept (or source ICD) is an HPO xref, attributed to the
  HP term(s). `t3` (climb) then operates on the remainder *after* HPO, so HPO-claimed
  conditions no longer inflate Mondo's climb tier (a fidelity win for Mondo too).
- **Per-axis assembly refactor**: factor "given an attribution DataFrame + an ontology's
  DAG structures (parent_adj / has_child / label_of / rare_of) → assemble payload" into a
  reusable step, invoked for Mondo (t1∪t2∪t3) and HPO (t_hpo). Bands, fractional 1/m,
  source-code catalog, collision handling all reused per axis. `rare_of` is Mondo-only
  (empty for HPO).
- **HPO multi-xref**: a SNOMED concept mapping to ≥1 HP term is handled exactly like a
  Mondo collision (fractional 1/m across the HP terms); expected to be rare.
- **Outputs**: `mondo_usage.json` (Mondo axis, unchanged schema — now minus HPO-claimed
  climbs) **+** `hpo_usage.json` (HPO axis, same schema, HPO DAG). Safe summary extended
  with an HPO-axis coverage block.

### Dashboard (`mondo-usage-dashboard/index.html`)

- **Load both** payloads (`mondo_usage.json` required; `hpo_usage.json` optional). If HPO
  absent → today's single Mondo view (no regression).
- **Dual DAG**: two browsers side by side. Parameterize the graph layout's growth
  **direction** (Mondo rightward, HPO leftward-mirrored). Shared toolbar / theme / search
  (search spans both axes). Selecting a node opens the drawer for its axis. Expect some UI
  iteration on the split layout / mirroring.
- The whole per-axis engine (bands, fractional roll-up, `≡`/`↑` provenance, drawer) is
  identical; only the payload and growth direction differ.

## Non-goals / later

- **UMLS bridge** (v2) — raises HPO recovery from the SNOMED lower bound.
- **HPO-side climbing** (v2) — currently exact-only.
- **Cross-ontology links** — the two axes stay unlinked (no disease↔phenotype edges).

## Open / tunable (surface as they arise, not blockers)

- HPO match key: SNOMED std concept is primary; also try the source ICD against HPO ICD
  xrefs. Confirm whether to match on std-only or std-or-source.
- Combined single payload vs two files — spec assumes **two files** (each a standard
  payload; cleanest for graceful fallback).
- Side-by-side vs mirror-flip details — a UI question to iterate on with the real data.
