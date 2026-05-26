# CHARMPheno Dashboard

Static, single-page Svelte 5 + Vite + D3 dashboard demonstrating a trained CHARMPheno topic model.
All patient-shaped artifacts are synthetic and generated in the browser; no real patient data is shipped.

## Development

    make dev      # vite dev server (base / for local)
    make build    # static output to dist/ (base /CHARMPheno/ for GH Pages)
    make preview  # build with base / and preview on localhost:4173
    make test     # vitest

## Data bundle

The dashboard reads four JSON files per cohort from `public/data/<cohort>/`:
`model.json`, `vocab.json`, `phenotypes.json`, `corpus_stats.json`. The set of
available cohorts is listed in `public/data/manifest.json`.

Bundles checked into the repo are sufficient for frontend development; no
regeneration is needed for UI work. To regenerate from a checkpoint on the
cloud side, see `analysis/cloud/build_dashboard_cloud.py`; for a local
checkpoint against a synthetic parquet, see `analysis/local/build_dashboard.py`.

## Deploy

Pushes to `main` that touch `dashboard/**` trigger `.github/workflows/dashboard.yml`,
which builds and deploys to the `gh-pages` branch.
