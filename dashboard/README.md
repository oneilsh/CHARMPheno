# CHARMPheno Dashboard

Static, single-page Svelte 5 + Vite + D3 dashboard demonstrating a trained CHARMPheno topic model.
All patient-shaped artifacts are synthetic and generated in the browser; no real patient data is shipped.

## Development

    make dev      # vite dev server (base / for local)
    make build    # static output to dist/ (base /CHARMPheno/ for GH Pages)
    make preview  # build with base / and preview on localhost:4173
    make test     # vitest

## Data bundle

The dashboard reads four JSON files from `public/data/`:
`model.json`, `vocab.json`, `phenotypes.json`, `corpus_stats.json`.

Regenerate from a checkpoint:

    poetry run python ../analysis/local/build_dashboard.py \
        --checkpoint ../data/runs/<checkpoint> \
        --input ../data/simulated/omop_N10000_seed42.parquet \
        --out-dir public/data \
        --vocab-top-n 5000

Or for a synthetic fixture bundle (no Spark needed):

    poetry run python ../scripts/make_dev_bundle.py --out-dir public/data --k 10 --v 200 --seed 0

See `docs/superpowers/specs/2026-05-13-dashboard-design.md` for the bundle schema.

## Deploy

Pushes to `main` that touch `dashboard/**` trigger `.github/workflows/dashboard.yml`,
which builds and deploys to the `gh-pages` branch.
