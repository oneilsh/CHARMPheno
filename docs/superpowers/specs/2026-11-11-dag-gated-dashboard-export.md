# A browsable export of a DAG-gated model: the Mondo browser with a per-node topic drawer

**Date:** 2026-11-11
**Status:** accepted (maintainer, 2026-11-11: 25 + 25 terms per domain, every size trick,
the minimum required information; the proposal below otherwise as discussed)
**First model:** exp 0173 (`<run>/model`: 2,695 node topics + 200 background, 3 domains).

## Why

The main dashboard ships a dense β and computes topic distances in the browser; at
K ≈ 2,900 and V = 11,601 that is ~34M weights (~200 MB of JSON). The Mondo usage
dashboard (`mondo-usage-dashboard/` on main) already browses a multi-parent Mondo DAG
lazily with a detail drawer. So: its DAG browser as the frame, a node's topic as the
drawer, and an export of the minimum needed to draw that. Target: about 2 MB gzipped in
total (measured 2026-11-11 on a synthetic export of this shape: topic weights compress
~2.5-3x, text ~7x).

## Contents (decided)

The browser shows the label subgraph the model was given (every node: kept with a topic,
pruned, spliced, unsupported), so it shows where every term went. The drawer, first
version:
1. the node's topic: per domain, its terms with an LDAvis relevance slider
   (relevance = λ log β + (1 − λ) log(β / p), λ in [0, 1]);
2. what differs from its parent(s): computed in the browser from the two shipped lists
   (approximate; terms outside a list count as absent);
3. health: support band, `supportFraction` (0 sharp, 1 flat);
4. for a node without a topic: its status and where its labels land (links).
A background-topics view lists the 200 shared topics in the same form. Later, not now:
own codes, background mixtures per node, readout AUC.

## Format `dag-gated-export/1` (a directory of minified JSON files)

All arrays are columnar (parallel arrays) and references are integer indices into
other arrays, never repeated strings.

`manifest.json`:
```
{schema: "dag-gated-export/1", run: str, created_utc: str, mondo_version: str, cdr: str,
 K: int, n_bg: int, tpn: int,
 domains: [{name: str, shipped_terms: int}],            # order = domain index
 selection: {top_beta: 25, top_lift: 25, lift_pool: 200},
 weight_codec: {kind: "log-byte", per_domain: [{lo: float, hi: float}]},
 freq_codec:   {kind: "log-byte", per_domain: [{lo: float, hi: float}]},
 min_docs_per_node: int,                                 # the fit's threshold (a pruned node's support is below it)
 disclosure: {min_cell: 20, support: "20-wide bands shown at the midpoint (1-20 -> 10 '<=20', 21-40 -> 30, ...)"},
 sizes: {"<file>": [raw_bytes, gzip_bytes]},
 notes: str}
```

`nodes.json` (one entry per subgraph node, index = position):
```
{id: [str],                  # Mondo CURIE (user id)
 name: [str],
 parents: [[int]],           # indices; [] = top level (the given DAG, not the kept one)
 kept_parents: [[int]|null], # the kept, reduced DAG for nodes with a topic; null otherwise
 depth: [int],               # longest path from the top in the given DAG, top level = 1
 status: [int],              # 0 topic, 1 pruned, 2 spliced, 3 unsupported
 landing: [[int]|null],      # status 1-3: the nodes its labels land on ([] = background)
 support: [int|null],        # banded (midpoint), see disclosure; null when not recorded:
                             # pruned nodes (only known to be < min_docs_per_node), or a
                             # spliced / unsupported node whose count cannot be derived
 support_frac: [int|null],   # supportFraction * 1000, rounded; null without a topic
 topic: [int|null]}          # index into topics.json; null without a topic
```

`vocab.json` (only terms that appear in some shipped list; index = term index):
```
{domain: [int], code: [str|int], name: [str], freq: [int]}   # freq: log-byte (freq_codec)
```
`freq` is the model's own term marginal within its domain, p(w) ∝ Σ_k λ_kw (no corpus data
needed), the p in the relevance formula.

`topics.json` (index = topic; 0 .. n_bg-1 background, then node topics):
```
{node: [int|null],           # index into nodes.json; null for background
 d: [[ [ids], [w] ], ...]}   # per topic: one [ids, w] pair per domain
```
`ids` are term indices into vocab.json, sorted ascending and DELTA-coded (first value
absolute, then gaps); `w` are log-byte weights aligned with the decoded ids: the topic's
E[β] within that domain (its λ row in the domain, normalized), b in 0..255,
β = exp(lo + (hi − lo) · b / 255) with the domain's (lo, hi). The browser re-sorts by β.

Selection per topic and domain: its top `top_beta` terms by β, plus up to `top_lift`
more by lift β / p among its top `lift_pool` terms by β (lift restricted to a pool so a
rare term's prior-level weight cannot rank first).

## Disclosure

Model parameters over a vocabulary floored at 20 patients per term; no document or
patient data. Node support is shown banded as the Mondo usage dashboard does. Nothing
per-node beyond the banded support and the topic.

## Code

- Exporter: `charmpheno/charmpheno/export/dag_gated.py` (pure: model result + node set +
  names -> the files) and `analysis/cloud/export_dag_gated.py` (driver: reads
  `<run>/model/result`, `label_dag.json`, `bundle_meta.json`, names via
  `resolve_vocab_names.py` or a names CSV and the Mondo labels; writes `<run>/dag_export/`;
  prints raw and gzip size per file and in total). Make target `export-dag-gated`.
- Browser: `dag-dashboard/index.html` (vanilla JS, no build step), adapted from the Mondo
  usage dashboard's DAG engine and drawer (copied, not shared: that dashboard is
  developed separately on main), plus a port of the main dashboard's relevance ranking.
  Deployment as a gh-pages subdirectory is a later step.
