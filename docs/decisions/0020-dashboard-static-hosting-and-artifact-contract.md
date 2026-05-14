# 0020 — Dashboard: static hosting and artifact contract

**Status:** Accepted
**Date:** 2026-05-14
**Spec:** docs/superpowers/specs/2026-05-13-dashboard-design.md
**Plan:** docs/superpowers/plans/2026-05-14-dashboard.md

## Context

CHARMPheno needs a salesmanship surface — a live, interactive demonstration of the trained
topic model for non-technical audiences (leadership, funders). Real patient data cannot be
exported from the secure environment, and we want hosting cheap enough that a researcher can
ship it without infra negotiation.

## Decision

1. **Static hosting on GitHub Pages.** No backend, no WASM. The dashboard is a Svelte 5 +
   Vite + D3 single-page app under `dashboard/`, deployed via a `gh-pages` branch on push
   to `main`.
2. **Synthetic-only patient framing.** Every patient-shaped artifact in the UI is
   generated *in the browser* from the exported model. The cloud side ships no patient
   data, synthetic or otherwise. The dashboard demonstrates the claim that trained
   models do not carry sensitive information, rather than apologizing for it.
3. **Minimal four-file bundle as the modeling ↔ UI contract:** `model.json`,
   `phenotypes.json`, `vocab.json` (trimmed to top-N=5000 codes by corpus frequency),
   `corpus_stats.json`. Schema in the spec.
4. **Model-class adapter pattern.** A small `charmpheno.export.model_adapter` normalizes
   LDA, HDP, and future model classes into a uniform `DashboardExport` shape, so the
   bundle writer and the dashboard contract are model-class-agnostic.
5. **No temporal axis in the Simulator.** Each sampled completion is one full year-of-life
   code bag; the visualization is a single-snapshot posterior over phenotype proportions.
   BOW exchangeability would make a finer time grain misleading.
6. **Re-implement pyLDAvis** rather than embed it. Static HTML output is fine for a
   one-off, but clinical-domain affordances (domain coloring, NPMI overlays, advanced
   view, linked code→topic highlights) want first-class control.
7. **Advanced-view toggle, not clinician-view.** Default is the simpler view; the toggle
   *reveals* technical affordances rather than hiding them.

## Alternatives considered

- **Embed pyLDAvis output.** Rejected: no clinical affordances, no synthetic patient,
  no model-class flexibility.
- **Server-rendered dashboard (Streamlit/Dash).** Rejected: requires hosting, complicates
  the "anyone can fork and run" story.
- **Python-side synthetic cohort export.** Rejected: ships a synthetic dataset and forces
  a Python build step; weakens the privacy story; doubles the contract surface.
- **Temporal Simulator (per-year bins or event prefix).** Rejected for v1: requires either
  a per-bin α covariate (parked in `docs/superpowers/specs/2026-05-13-per-bin-alpha-covariate.md`)
  or an event-level model. Future work.

## Consequences

- The export pipeline (`charmpheno.export.dashboard` + `model_adapter`) becomes a reusable
  artifact for any future dashboard variant; coupling to the Svelte app is the JSON
  schema only.
- The dashboard cannot show real-patient prevalence breakdowns, cohort comparisons, or
  trajectories. Those would require additional aggregate exports negotiated separately.
- Computing synthetic cohorts and topic-map MDS in JS removes a separate "local Python
  build" from the workflow and makes the privacy story tighter.
- Bundle size grows with K and trimmed-V. v1 targets <2 MB gzipped; if exceeded, gzip
  served by GH Pages or shrink top-N.
- New model classes (e.g. correlated topic models) only need a new adapter function, not
  bundle-schema or dashboard changes.
