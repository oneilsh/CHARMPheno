# Published export: run 0174 (model of record, v9 CDR)

The four `dag-gated-export/1` files in this directory are served by GitHub Pages at
`/CHARMPheno/dag-topics/` (copied to `data/` next to `dag-dashboard/index.html` by
`.github/workflows/dashboard.yml`). They were extracted from the reviewed self-contained
page with `python dag-dashboard/dev/unembed.py dag_view.html dag-dashboard/published/0174/`,
which writes them byte-identical to the run's `dag_export/`.

| | |
|---|---|
| run | `0174-model-of-record-v9-cdr` (experiment doc `docs/experiments/0174-*.md` on branch `claude/dismech-ribbon`) |
| CDR | `wb-silky-artichoke-2408.C2025Q4R6` |
| Mondo version | `2026-06-02` |
| exported (UTC) | 2026-11-20T14:42:22Z |
| egress review | 2026-11-20, by the maintainer (workspace audit below); `dag_view.html` left the workspace as `dag_view_0174.zip`, sha256 `ba24c5e86ed2100e253920d5c392f93595df34dea10fec5ff6718da7007a2ae6` |

## What it holds

Model parameters (topic word weights, log-byte coded, top 40 by β plus 40 by lift per
domain), concept names, the label DAG, per-node support banded to 20-wide midpoints, and
NPMI averaged over term pairs with joint count >= 20. No patient-level data and no cell
below the egress floor of 20.

## Audit at export time

`export_audit.txt` from the zip (read from the manifest embedded in the reviewed page).
The page in the zip is byte-identical to `dag-dashboard/index.html` at this commit, and
the extracted files match the manifest's recorded sizes.

```
schema: "dag-gated-export/1"
run: "0174-model-of-record-v9-cdr"
created_utc: "2026-11-20T14:42:22Z"
cdr: "wb-silky-artichoke-2408.C2025Q4R6"
mondo_version: "2026-06-02"
K: 3253
n_bg: 200
min_docs_per_node: 100
disclosure: {"min_cell": 20, "support": "20-wide bands shown at the midpoint (1-20 -> 10 '<=20', 21-40 -> 30, ...)"}
selection: {"top_beta": 40, "top_lift": 40, "lift_pool": 200, "forced": "profile", "forced_terms": 16854}
npmi: {"top_n": 10, "split": "train", "min_pair_count": 20, "domains": [0, 2], "scale": 1000}
sizes: {"nodes.json": [950302, 214200], "vocab.json": [766498, 187256], "topics.json": [5053919, 1620822], "manifest.json": [2154, 1150]}
min shipped support band midpoint: 90
```
