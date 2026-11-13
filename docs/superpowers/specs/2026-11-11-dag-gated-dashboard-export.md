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
background mixtures per node, readout AUC. Own codes and HPO-profile terms were added
2026-11-13 (section "Marks").

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

`vocab.json` (only terms that appear in some shipped list, plus the marked terms of
section "Marks"; index = term index):
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
Optional (added 2026-11-13, maintainer): `npmi: {domains: [int], gated: [[int|null]],
all: [[int|null]]}`, per scored domain one value per topic, round(1000 * NPMI) of the
topic's top terms (gated: over the documents its gate admits; all: every document),
null when unscored; with `manifest.npmi = {top_n, split, min_pair_count, domains,
scale: 1000}`. From `analysis/cloud/topic_npmi.py`'s `<run>/npmi_top10.json`; only these
scalars ship, no count. A topic's NPMI averages the word pairs whose joint document
count is at least the result's `min_pair_count` (`topic_npmi.run_npmi`; the analysis
default is 3; the `floor` given to `score` is the gated reference-set size, not the pair
cells). So the scalars rest on cells at or above the egress floor only when
`min_pair_count >= 20`. Rule (corrected 2026-11-13, code review A1): the exporter ships
the block only when `min_pair_count >= EGRESS_FLOOR` and otherwise drops it with a
printed reason; an exportable NPMI comes from the readout's
`--npmi --npmi-min-pair-count 20`.

`ids` are term indices into vocab.json, sorted ascending and DELTA-coded (first value
absolute, then gaps); `w` are log-byte weights aligned with the decoded ids: the topic's
E[β] within that domain (its λ row in the domain, normalized), b in 0..255,
β = exp(lo + (hi − lo) · b / 255) with the domain's (lo, hi). The browser re-sorts by β.

Selection per topic and domain: its top `top_beta` terms by β, plus up to `top_lift`
more by lift β / p among its top `lift_pool` terms by β (lift restricted to a pool so a
rare term's prior-level weight cannot rank first).

### Marks (optional; added 2026-11-13)

Two optional blocks that let the browser badge a topic's terms: a node's OWN codes and
its HPO PROFILE concepts, both standard Condition concepts read against the condition
domain's vocabulary (`manifest.marks.domain`: the domain named `condition`, else 0).

- **Own codes**: the code ladder landed on the fitted node set, `<run>/code_map.tsv`
  (`std_cid`, `node_cid`; `dag_gated_fit.write_code_map` / `landed_code_map`). Each
  code sits on the kept node its Mondo term is, or on the nearest kept ancestors of a
  pruned / spliced term, so in practice only nodes with a topic have own codes.
  `node_cid` is the numeric part of the node's Mondo CURIE (the bridge's `node_cid`).
- **HPO profile**: `<run>/profile_marks.tsv` (`mondo_id`, `concept_id`, `neg`;
  `hpo_profile_marks.py`, from the stage-1 `hpoa_profile_survey.py --emit-codes`, made
  by `make profile-marks`). Only POSITIVE concepts mark: NOT-annotated rows are dropped
  and a concept listed both positive and NOT for the same node (contested) is dropped,
  as `profile_eta.build_profile_seed` does. The HPO release comes from the
  `profile_marks.json` sidecar the make target writes (or `--hpo-release`).

`nodes.json` gains, per block present:
```
 own:       [[int]|null],   # vocab.json indices of the node's own codes (in the vocabulary);
                            # null when it has none there
 own_n:     [int],          # how many of the node's own codes are in the model's vocabulary
 profile:   [[int]|null],   # vocab.json indices of its positive HPO-profile concepts; listed
                            # for nodes WITH A TOPIC only (null otherwise or when none)
 profile_n: [int],          # how many of its positive profile concepts are in the vocabulary
                            # (every node, with or without a topic)
```
Lists are sorted ascending (not delta-coded: they are short). `own_n` equals the list's
length where it is shipped; it is kept so a node without a list still says how many it
has. A concept outside the model's vocabulary has no term and is neither listed nor
counted. Profile lists are limited to topic nodes because a profile marks a topic's
terms, and the whole label DAG's profiles would outweigh the topics at full scale.

A marked term in the vocabulary that no shipped topic list holds is ADDED to vocab.json
(with its name and `freq`), appended after the marks domain's shipped terms in local
index order; `manifest.domains[marks domain].mark_terms` counts them (`shipped_terms`
stays the count of topic-shipped terms, so a topic's ids never point past them). The
freq codec's range covers the added terms.

`manifest.json` gains:
```
 marks: {domain: int,
         own?:     {source: "code_map.tsv", nodes: int},       # nodes with a non-null own
         profile?: {source: str, hpo_release?: str, nodes: int}} # nodes with a non-null profile
```
Driver flags: `--own-codes TSV` (default `<run>/code_map.tsv` when present) /
`--no-own-codes`; `--profile-marks TSV` (default `<run>/profile_marks.tsv` when present) /
`--no-profile`; `--hpo-release`. A block whose default table is absent is left out; an
explicit path that does not exist is an error.

Size (synthetic run at the real scale, `dag_export_synth.py --measure`: 2,895 topics,
7,000 given nodes, 11,421 own (code, node) pairs on 2,278 kept nodes, 27,538 profile rows
on 3,833 nodes): total 1,336,811 -> 1,384,404 bytes gzipped (+48 KB; nodes.json 157,736
-> 205,255), raw 4.61 -> 4.78 MB. At this scale every vocabulary term is already shipped
by some topic, so no term is added.

## Disclosure

Model parameters over a vocabulary floored at 20 patients per term; no document or
patient data. Checked, not assumed (code review A2, 2026-11-13): `build_export` refuses
a node set fitted with `min_docs_per_node < 20`, and the driver refuses a run whose
`corpus_manifest.min_patient_count` is below 20 or missing. Node support is shown banded as the Mondo usage dashboard does. Nothing
per-node beyond the banded support and the topic, and the marks.

The marks (section "Marks") are ontology and code-ladder mappings: which vocabulary
terms are a node's own codes (Mondo -> OMOP mapping landed on the fitted node set) and
which are in its HPO phenotype profile (public HPO annotations mapped through the CDR's
concept tables). They carry no count beyond counts of concepts (`own_n`, `profile_n`,
the manifest's node counts), no document or patient count; every listed term is a
vocabulary term, so floored at 20 patients like every other shipped term.

## Code

- Exporter: `charmpheno/charmpheno/export/dag_gated.py` (pure: model result + node set +
  names -> the files) and `analysis/cloud/export_dag_gated.py` (driver: reads
  `<run>/model/result`, `label_dag.json`, `bundle_meta.json`, names via
  `resolve_vocab_names.py` or a names CSV and the Mondo labels; writes `<run>/dag_export/`;
  prints raw and gzip size per file and in total). Make target `export-dag-gated`.
  The marks: own codes from `<run>/code_map.tsv`, the profile from
  `<run>/profile_marks.tsv`, which `make profile-marks ID=N` writes (the stage-1 HPOA
  survey over Mondo's root `MONDO:0000001` into `<run>/hpoa/`, then
  `hpo_profile_marks.py`'s targeted `bq` mapping; off-YARN).
- Browser: `dag-dashboard/index.html` (vanilla JS, no build step), adapted from the Mondo
  usage dashboard's DAG engine and drawer (copied, not shared: that dashboard is
  developed separately on main), plus a port of the main dashboard's relevance ranking.
  With a marks block, the drawer badges a topic's condition terms (`own`, `HPO`, both
  possible; a legend line above the columns, the tooltip says which) and lists, below
  the columns, the node's own codes and HPO-profile concepts with their β rank in the
  topic's shipped list ("not in the topic's list" otherwise) and how many are among the
  top terms shown and among the shipped terms; collapsed after 10. Without a marks
  block the page shows nothing of this.
  Deployment as a gh-pages subdirectory is a later step.
