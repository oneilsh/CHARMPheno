# Scalable gated spectral init — design

**Status:** approved (brainstorm 2026-07-17). Exploratory research build; no
production target.

## Problem

`gated_init.spectral_block_aligned_lambda` (the DENSE block-aligned spectral
init for `GatedOnlineLDA`) does not complete at the rare6 scale
(K=180 / V=10000 / person_mod=1): it collects the whole training corpus to the
driver and builds ~29 dense V×V co-occurrence matrices (pooled + one per node),
single-threaded, before iteration 1. This blocks the init-axis diagnostic
(exp 0059) and any real spectral run at that scale.

The scalable random-projection foundation already exists
(`spectral_init_scalable.py`, ADR 0032) but is FLAT — each group is deflated
against background only, with no DAG topology (no ancestor deflation, no
forward-topological ordering). We need the gated, DAG-topological analogue.

## Key insight (why this is mostly reuse)

The dense per-node co-occurrence `Q_u` is the co-occurrence over docs whose
frontier-closure contains `u`. That is exactly what
`projected_cooccurrence_rdd`'s per-group sketch `group_QR[u]` computes **if each
doc's `groups` is set to `closure(frontier) \ {0}`**. So ONE distributed pass
yields the projected image of every node's `Q_u`, no driver V×V. The
forward-topological ancestor deflation is then pure driver-side logic that
mirrors the dense function, reusing the projected primitives
`find_anchors_projected` / `recover_beta_projected` — both of which already
accept `seed_rows` for Gram–Schmidt deflation.

The projection preserves the anchor geometry: the greedy farthest-point search
depends only on residual-norm comparisons among word rows, which
Johnson–Lindenstrauss keeps under a ~1000-dim Gaussian sketch. Deflating node
`u`'s rows against its ancestors' anchor rows isolates `u`'s node-specific
phenotype from what it inherits through the closure — the same operation the
dense path performs, sketched.

## Components

### 1. `gated_init.scalable_block_aligned_lambda(rdd, lay, V, *, d=None, seed=0, min_doc_freq=5, scale=200.0) -> (K, V) lambda`

- Adapter: map the `GatedBOWDocument` RDD to the shape
  `projected_cooccurrence_rdd` consumes — an object per doc carrying
  `indices`, `counts`, and `groups = closure(frontier) \ {0}` — plus a minimal
  partition-like holder whose `.groups = tuple(lay.nodes)`. `projected_cooccurrence_rdd`
  stays untouched (shared with STM).
- One distributed pass → pooled sketch + per-node sketches + within-node marginals.
- Step 1 (background): `find_anchors_projected(pooled_QR, p_w, df_w, n_bg)` →
  `recover_beta_projected` → background block.
- Step 2 (each node `u`, ancestors-first by `lay.depth`):
  `seed = bg_anchors + [anchors of u's already-recovered proper ancestors]`;
  `fg = find_anchors_projected(group_QR[u], group_p_w[u], group_df_w[u], tpn, seed_rows=seed)`;
  `beta_u = recover_beta_projected(group_QR[u], group_p_w[u], seed + fg)[len(seed):]`
  → `block[u]`. Zero-doc / no-anchor nodes: warn, leave block at floor (matches
  dense).
- Returns `(beta + 1e-9) * scale` — same scale contract as the dense function,
  so it is a drop-in λ seed.

### 2. `GatedOnlineLDA.initialize_global` accepts a precomputed λ

If `data_summary` carries a precomputed λ (key `spectral_lambda`), use it
directly instead of running a dense `INIT_STRATEGIES` strategy. Dense path
unchanged (no `spectral_lambda` → run the strategy). Mirrors the STM shim, which
precomputes `spectral_beta` on the RDD and passes it via `data_summary`.

### 3. Shim `GatedLDAEstimator` routing

- New param `spectralMethod` ∈ {"auto", "dense", "scalable"}, default "auto"
  (explicit control, STM parity). Optional `spectralD`, `spectralMinDocFreq`
  params (defaults: None → `default_projection_dim`, 5).
- `spectralMaxVocab` is repurposed as the AUTO threshold: `auto` → dense if
  `V < spectralMaxVocab` else scalable (default 10000, STM's
  `SPECTRAL_AUTO_VOCAB_THRESHOLD`). The old `NotImplementedError` guard is
  removed — large-V dense is now an explicit user choice, and scalable handles
  large V.
- `_fit`: when `init == "spectral"` and resolved method is `scalable`,
  precompute λ via `scalable_block_aligned_lambda(rdd, lay, V, …)` and pass
  `data_summary = {"spectral_lambda": lambda0}`; dense path unchanged (collect →
  `{train_docs, train_labels}`).

### 4. Driver / config

- `dag_placement_cloud.py`: pass `spectralMethod` (+ optional d/min_doc_freq)
  through; record the RESOLVED method in the manifest/log so 0059 shows it ran
  scalable.
- `run_experiment.build_dag_placement_args`: emit `--spectral-method`.
- `_base.yaml`: `spectral_method: auto`.
- exp 0059: set `spectral_method: scalable` explicitly (force it for the
  diagnostic; no ambiguity).

## Validation

Structural unit tests (the real performance check is the 0059 cluster A/B vs
0058 — a synthetic gate is invalid here because the dense gated init is itself a
validated-negative on synthetic plants):

- block-aligned β: on a small parent/child plant, each node's block loads its
  planted anchor tokens; background block loads the shared tokens.
- topological deflation took effect: a child block differs from its parent block
  (ancestor deflation isolated the child's phenotype).
- determinism: same seed → identical λ.
- zero-doc / no-anchor node guard: block left at floor, warning logged, no NaN.
- `initialize_global` consumes a precomputed `spectral_lambda`.
- shim: `spectralMethod` resolves by vocab under "auto"; forced "scalable" at
  small V fits and yields a K-row λ; forced "dense" still runs the collect path.

`projected_cooccurrence_rdd`, `find_anchors_projected`, `recover_beta_projected`
are reused unmodified.

## Out of scope

- Distributed β-recovery (the projected recover loop stays driver-side, as in
  the flat scalable path; fine at these V).
- Any change to the dense path's behavior.
- Tuning d / min_doc_freq / scale — defaults ship; the 0059 A/B decides whether
  spectral matters before any tuning.
