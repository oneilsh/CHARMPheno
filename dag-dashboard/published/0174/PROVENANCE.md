# Published export: run 0174 (model of record, v9 CDR)

The four `dag-gated-export/1` files in this directory are served by GitHub Pages at
`/CHARMPheno/dag-topics/` (copied to `data/` next to `dag-dashboard/index.html` by
`.github/workflows/dashboard.yml`). They were extracted from the reviewed self-contained
page with `python dag-dashboard/dev/unembed.py dag_view.html dag-dashboard/published/0174/`,
which writes them byte-identical to the run's `dag_export/`.

| | |
|---|---|
| run | `0174-model-of-record-v9-cdr` (experiment doc `docs/experiments/0174-*.md` on branch `claude/dismech-ribbon`) |
| CDR | _TBD (from manifest.json `cdr`)_ |
| Mondo version | _TBD_ |
| exported (UTC) | _TBD (manifest.json `created_utc`)_ |
| egress review | _TBD (date, reviewer)_ |

## What it holds

Model parameters (topic word weights, log-byte coded, top 40 by β plus 40 by lift per
domain), concept names, the label DAG, per-node support banded to 20-wide midpoints, and
NPMI averaged over term pairs with joint count >= 20. No patient-level data and no cell
below the egress floor of 20.

## Audit at export time

```
TBD: paste the audit output (min_docs_per_node, minimum shipped support band,
NPMI min_pair_count, corpus min_patient_count, file sizes)
```
