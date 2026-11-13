# DAG-gated topic browser

A single-file, no-build page (`index.html`, vanilla JS; only Google Fonts is external) for
browsing a `dag-gated-export/1` directory: the Mondo label DAG a DAG-gated model was given,
with each node's topic in a detail drawer. The format is specified in
[`docs/superpowers/specs/2026-11-11-dag-gated-dashboard-export.md`](../docs/superpowers/specs/2026-11-11-dag-gated-dashboard-export.md).

The DAG engine and drawer are copied from `mondo-usage-dashboard/index.html` on `main`
(developed separately there, not shared); the relevance ranking is a port of the main
dashboard's CodePanel (`dashboard/src/lib/inference.ts`, `relevance`).

## Making an export (on the cluster, from a finished fit)

```bash
make -C analysis/cloud export-dag-gated ID=173 RESOLVE_NAMES=1   # writes <run>/dag_export/
```

It also writes **`<run>/dag_view.html`**: the page with the export embedded, one
self-contained file that opens without a server. Use it where the page cannot fetch its
data files: JupyterLab serves files in a sandbox (`Content-Security-Policy: sandbox`), so
"Open in New Browser Tab" on `index.html` shows the page but not the data, and
`file://` blocks the fetches too. In a Jupyter workbench, open `dag_view.html` with
"Open in New Browser Tab". (`--no-single-file` skips it.)

It needs no Spark (it reads `<run>/model/result`, `label_dag.json`, `bundle_meta.json`;
`RESOLVE_NAMES=1` names the terms through the CDR concept table) and prints each file's
raw and gzip size. The export holds model parameters, names, and banded per-node support
only; follow the workspace's egress rules before copying it out.

## Viewing an export locally

```bash
cp -r <run>/dag_export dag-dashboard/data     # manifest.json, nodes.json, vocab.json, topics.json
cd dag-dashboard && python -m http.server
# open http://localhost:8000/
```

Another directory can be named with `?data=<path>`, e.g.
`http://localhost:8000/?data=dev/sample/` for the synthetic sample.

**`dag-dashboard/data/` must never be committed** (it is in `.gitignore`). An export holds
model parameters and banded supports only, but it stays in the workspace until it has been
reviewed for release.

## What it shows

- **Header:** model run, Mondo version, CDR, topic counts, node counts by status, and the
  export's size (raw and gzip, per file on hover) from `manifest.json`.
- **Label DAG:** every node of the label subgraph with its given parents, coloured by
  status (topic / pruned / spliced / unsupported; dashed ring = no topic) and sized by
  banded support. Click expands one level; right-click or the hover dot opens the node menu
  (expand or collapse children / parents / whole subtree, focus here); shift-click
  multi-selects; hover highlights the lineage; search marks matches (capped at 60) and
  clicking a match lands there. "Only nodes with a topic" hides the rest; topic nodes
  below a hidden node stay reachable through faint shortcut edges.
- **Node drawer:** name, ID (Monarch link), status, depth, support as its 20-wide band
  ("≤20" for 1-20, else the midpoint), and `supportFraction` (0 sharp, 1 flat). For a
  node with a topic: the top 15 shipped terms per domain ranked by relevance
  λ·log β + (1 − λ)·log(β / p) with one λ slider (default 0.6), and "differs from parent":
  for each kept parent, the node's terms with the largest β / β_parent (a term missing from
  the parent's list gets the parent's smallest shipped weight, so this is approximate).
  For a node without a topic: why (pruned, spliced, unsupported) and links to the nodes
  its labels land on (or to the background topics).
- **Own codes and HPO profile** (when the export carries a `marks` block): in the
  condition column, an `own` badge on a term that is one of the node's own codes (the
  code ladder puts it on this node) and an `HPO` badge on one in its HPO phenotype
  profile (positive annotations only); a legend line above the columns and the hover
  tooltip say which. Below the columns, the node's own codes and profile concepts with
  their β rank in the topic's shipped list (or "not in the topic's list"), and how many
  are among the top 15 shown and among the shipped terms; long lists collapse after 10.
  The export picks up `<run>/code_map.tsv` (written by the fit) and
  `<run>/profile_marks.tsv` (written by `make -C analysis/cloud profile-marks ID=N`).
- **Background topics:** the shared background topics in the same three-column form.

## Synthetic sample

`dev/make_sample.py` writes `dev/sample/` (59 nodes, all four statuses, multi-parent,
3 domains, 5 background topics): seeded random names and weights, no model and no patient
data. Regenerate with `python dag-dashboard/dev/make_sample.py`.
