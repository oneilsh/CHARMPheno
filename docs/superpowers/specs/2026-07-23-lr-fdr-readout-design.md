# LR-FDR readout — apply the empirical-null FDR to the LR / explain-away scores

**Status:** design (brainstormed 2026-07-23, branch `case-finding`)

## Problem

The Efron two-groups empirical-null per-node FDR (`per_node_discoveries` + the
`by_q` precision/recall report) is currently computed ONLY over theta-mass (the
node-block affinity `P`), inside `evaluate`. On this data it found ZERO discoveries
at every q (0.05/0.10/0.20) across exps 0067/0068/0069 — because theta-mass buries
the mass-starved node signal (the whole reason the LR readout exists). Meanwhile the
LR readout at the alpha->inf lift limit beats theta-mass by +0.11-0.13 ROC. The open
question: does LR's detection edge translate into actual FDR-controlled discoveries
where theta-mass got none?

`per_node_discoveries` is score-agnostic — it takes any `[n_docs x n_nodes]` score
matrix, uses the background docs as the per-node/per-length-bin empirical null, and
applies BH/BY per node. So we can run the identical FDR machinery on the LR and
explain-away score matrices and report the discoveries beside theta-mass.

## Idea

Add an FDR block to the LR readout (`lr_readout.py`) that runs the same FDR report
on the LR score matrix (at alpha->inf) and the explain-away score matrix, and prints
`by_q` (n_discoveries / precision / recall) for both, beside the theta-mass FDR
already carried in the run manifest. Post-hoc, no re-fit.

## Architectural decision (recorded from brainstorming)

The FDR report lives in the ENGINE, not the driver. Rationale: `evaluate` and every
FDR primitive (`per_node_discoveries`, `_fdr_reject`, `_empirical_right_tail_p`,
`_zib_empirical_gap`) already live in `spark_vi/models/topic/dag_placement.py`; the
FDR report on a score matrix is id-agnostic array math (integer node/doc space, no
concept ids) and is an evaluation primitive the engine already owns. Extracting the
shared reporter (a) keeps ONE id-agnostic FDR path, (b) guarantees the theta-mass vs
LR comparison is provably like-for-like (same code), and (c) matches the established
pattern. (Open architectural question — whether evaluation should live in the engine
at all — is deliberately out of scope; revisit only if evaluation is split from the
engine wholesale.)

## Design

### Engine — `spark_vi/models/topic/dag_placement.py` (id-agnostic)

Extract the FDR reporting that currently lives inline in `evaluate`
(the `by_q`, `multimorbidity`, `saturation_rate`, `zib_gap_*`,
`n_length_bins_effective` block, ~lines 779-833) into a reusable pure function:

    def fdr_discovery_report(P, is_fg, doc_lengths, truth, mm_rows, *,
                             q_grid=(0.05, 0.10, 0.20), n_length_bins=4,
                             method="bh") -> dict

- `P` [n_docs x n_nodes]: the score matrix (theta-mass, LR, or explain-away).
- `is_fg` [n_docs] bool: foreground (>=1 scoreable frontier node) vs background.
- `doc_lengths` [n_docs]: per-doc token counts (the length-conditioning axis).
- `truth` [n_docs x n_nodes] bool: subtree-membership truth (patient i is a true
  positive for node u iff its frontier intersects `subtree(u)`) — for precision/recall.
- `mm_rows` [n_docs] bool: which docs are truly multimorbid (>= 2 scoreable frontier
  nodes) — for the multimorbidity payoff. Computed by the caller from the frontiers
  (NOT derivable from `truth`, since a single deep frontier node makes `truth` true
  for all its ancestors too).
- Returns the exact `fdr_block` dict `evaluate` builds today.

`evaluate` is refactored to compute `truth` and `mm_rows` (as it already does) and
call `fdr_discovery_report`, so its output is byte-identical (guarded by the existing
`evaluate` tests). No behavior change to `evaluate`.

### Driver — `analysis/cloud/lr_readout.py`

- Build the truth + mm_rows from the held-out test frontiers (`build_test_bow`
  already returns per-row `frontier` in its meta): `truth[i,u] = bool(frontier_i &
  lay.subtree(u))`; `mm_rows[i] = len(frontier_i & set(lay.nodes)) >= 2`;
  `lengths = np.asarray(bow.sum(axis=1)).ravel()`; `is_fg` is the same
  foreground mask the detection block already computes.
- Compute `P_lr = lr_placement_scores(bow, lam, lay, alpha=inf, background, count_mode,
  length_normalize)` and `P_ea = explain_away_placement_scores(...)` (same alpha=inf,
  same count_mode/length_normalize as the detection block).
- Call `fdr_discovery_report` on `P_lr` and `P_ea`; print a labeled `by_q` block for
  each (n_discoveries / precision / recall), beside the manifest's theta-mass FDR
  `by_q` (from `manifest["metrics"]["fdr"]` / `manifest["fdr"]` if present).
- Gate behind the existing readout flow (always print, like the detection block); no
  new required flag. `count_mode`/`length_normalize` threaded consistently (as the
  detection block already is).

## The comparison this enables

A three-row table in one readout:

    fdr by_q (n_disc, precision, recall):
      theta-mass (manifest):  q=0.05 (0, nan, 0.00) ...   <- the buried-signal baseline
      LR @alpha=inf:          q=0.05 (?, ?, ?) ...          <- does LR surface discoveries?
      explain-away @inf:      q=0.05 (?, ?, ?) ...

If LR yields nonzero discoveries at controlled q with reasonable precision, that is
the first FDR-controlled case-finding signal on this task — a real result (update
insight 0063 / the retrospective). If LR also finds ~zero, the buried-signal problem
is not lens-specific and the information-limit conclusion strengthens.

## Validation

- **Engine unit test** (`test_dag_placement.py`): `fdr_discovery_report` on a small
  synthetic `P`/`truth`/`is_fg`/`lengths` returns the expected `by_q` structure
  (n_discoveries, precision in [0,1] or nan, recall in [0,1]); a planted case where a
  node's foreground scores sit clearly above the background null yields >=1 discovery
  at q=0.20 with precision 1.0; an all-null `P` yields zero discoveries (mirrors the
  theta-mass zero-discovery behavior).
- **Refactor guard:** the existing `evaluate` FDR tests stay green byte-for-byte
  (the extraction must not change `evaluate`'s output).
- **Driver test** (`test_lr_readout.py`): the truth/mm_rows/lengths construction from
  a synthetic frontier list is correct (truth = subtree membership; mm_rows = >=2
  scoreable frontier nodes); the FDR block renders for LR and explain-away without
  error on a tiny fixture.
- **Cluster (post-hoc, no re-fit):** `make lr-readout ID=69` (and 0067) prints the
  three-way FDR table; read LR/EA discoveries vs the theta-mass zero.

## Out of scope

- Running FDR across the whole alpha sweep (alpha->inf lift limit only, matching the
  detection block).
- A background-conditioned base rate (the FDR null already uses background docs; the
  LR evidence denominator stays the flat corpus rate, as elsewhere).
- Any change to the FDR math itself (`per_node_discoveries` / BH / empirical null are
  reused unchanged).
- Moving evaluation out of the engine (the recorded architectural question).
