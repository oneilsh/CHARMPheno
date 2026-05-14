# Salesmanship Dashboard — Design Spec

**Date:** 2026-05-13
**Status:** Draft, pending user review
**Scope:** A static, GitHub-Pages-hostable single-page web dashboard that demonstrates a trained CHARMPheno topic model to non-technical audiences (CHARM leadership, funders). Three tabs — **Phenotype Atlas**, **Patient Explorer**, **Simulator** — driven entirely by exported model artifacts and locally-generated synthetic patients. No real patient data ever leaves the cloud. Does **not** cover temporal modeling, demographic covariates, or any modeling-engine changes; those are separate workstreams.

A parallel design stub for an age-bin (or other patient-level) covariate extension lands as [`2026-05-13-per-bin-alpha-covariate.md`](2026-05-13-per-bin-alpha-covariate.md). The dashboard ships against the current covariate-free model and does not block on that work.

---

## Context

CHARMPheno has working topic models, an evaluation pipeline (NPMI coherence), and a striking static visual ([`profile_example.png`](../../../profile_example.png)) that funders and leadership have responded to. The next deliverable is "salesmanship" — turning the trained model into a live, interactive artifact people can play with in a browser.

Two hard constraints shape the design:

1. **No real patient data may be exported** from the secure environment, even at aggregate-by-individual grain. Sequences of per-patient θ vectors are explicitly out. Aggregate-over-corpus statistics (e.g., per-phenotype mean θ across the corpus, per-vocab marginal frequency) are exportable but should be reviewed against the deployment's data-egress rules before they ship. The cloud-side export footprint is targeted at kilobytes, not megabytes.
2. **Static hosting** (GitHub Pages). No backend, no server-side compute, no WASM. All interactivity is JavaScript over pre-built JSON artifacts.
3. **Cloud side exports the minimum required.** Anything that can be computed cheaply from the model itself (JSD/MDS for the topic map, synthetic cohorts for patient-shaped views) is computed client-side in JS at startup. The cloud emits only the trained model, vocabulary metadata, per-phenotype metrics, and a handful of corpus scalars.

These constraints turn out to be friendly. A trained LDA model is essentially β (K × V) + α + vocabulary + a handful of scalars — a few hundred kilobytes of JSON. Variational inference, Dirichlet/multinomial sampling, and JS-divergence MDS are all client-side-cheap. The dashboard is *read-only* at runtime (with respect to the model) but generates its own synthetic patients on the fly.

Every patient-shaped artifact in the UI is **synthetic**, generated in-browser from the exported model and a handful of corpus-aggregate scalars. The dashboard becomes a working demonstration of the README claim that "trained models do not contain sensitive information" — the privacy story is part of the pitch, not an apology.

## Goals

1. **Ship a polished, demo-grade single-page dashboard** at `https://<owner>.github.io/CHARMPheno/` (or similar) that loads in under three seconds and runs smoothly on a mid-range laptop.
2. **Showcase the model itself** (Phenotype Atlas) — a pyLDAvis-style explorer reimplemented from scratch with clinical-domain affordances (NPMI overlays, domain-colored codes, simple-by-default with an advanced-view toggle for technical affordances).
3. **Showcase a single synthetic patient** (Patient Explorer) — pick from an in-browser-generated cohort, see their phenotype profile and top contributing codes, with the "synthetic — generated from model" badge prominent.
4. **Showcase the generative model under conditioning** (Simulator) — given a prefix of codes, sample N complete year-of-life code bags from the model, render the resulting posterior over phenotype profiles as a high-K density carpet plus a top-expected-codes panel. No temporal axis.
5. **Establish a minimal export pipeline** (`charmpheno.export.dashboard`) that turns a saved `VIResult` into a four-file JSON bundle (model, phenotypes, vocab, corpus_stats). Reproducible, scriptable, CI-friendly.
6. **Land ADR 0020** documenting the artifact contract, the synthetic-only framing, and the static-hosting decision.

## Scope

### In v1 (`feat/dashboard`)

Two cooperating pieces. The cloud-side export pipeline lives in `charmpheno`; the dashboard app lives at the repo root in `dashboard/`.

#### Cloud / export side (deliberately minimal)

- `charmpheno/charmpheno/export/dashboard.py` — builder that consumes a `DashboardExport` (see "Model classes" below) and emits a four-file JSON bundle (schema below). No synthetic-patient logic; that's client-side.
- `charmpheno/charmpheno/export/model_adapter.py` — model-class adapter. Reads a `VIResult`, dispatches on `metadata["model_class"]`, returns a uniform `DashboardExport`. Supports LDA and HDP in v1; future model classes plug in by adding an `adapt_<class>` function.
- `analysis/local/build_dashboard.py` — driver script: loads a saved `VIResult`, calls the adapter, runs the export builder, materializes the bundle into `dashboard/public/data/`.
- A `corpus_stats.json` sidecar produced as part of the bundle: corpus_size_docs, mean_codes_per_doc, K, V. Small scalars only.
- **Vocab is trimmed at export time** to the top-N codes by corpus frequency (default N=5000) to keep the bundle small. β is trimmed and renormalized to the same N columns; `codes[i].id == i`, and column `i` of β corresponds to `codes[i]`. The driver exposes `--vocab-top-n` for tuning.

#### Model classes supported (v1)

The dashboard contract is "K topics with β rows + α prior on θ". The adapter normalizes each model class to this contract:

- **LDA (`OnlineLDA`).** Identity adapter. `beta` is `lambda_ / lambda_.sum(axis=1, keepdims=True)`. `alpha` is `global_params["alpha"]`. `corpus_prevalence` is per-topic mean of `gamma_d / sum(gamma_d)` across the corpus (gamma row-normalized, mean over rows). `topic_indices` is `0..K-1`.
- **HDP (`OnlineHDP`).** Filtering adapter. Selects the top-K-by-usage topics via the existing `top_k_used_topics(u, v, k)` helper (default K=50). `beta` is the filtered topic-term matrix. `alpha` is the *effective* per-displayed-topic prior derived from the GEM corpus-level sticks (`E[β_k]` from `u, v`, restricted to the displayed indices and renormalized). `corpus_prevalence` is the same GEM-derived mass for the displayed topics. `topic_indices` carries the original HDP truncation indices so the advanced view can display them.

The simulator's `θ ~ Dirichlet(α)` sampling is exact under LDA. Under HDP it is an approximation: the true HDP prior is GEM stick-breaking, but treating the displayed topics as a finite Dirichlet with the GEM-derived α gives the same first-moment behavior and is fine for visualization. A footnote in advanced view discloses the approximation.

Future model classes (CTM, STM, dynamic topic models) plug in via a new `adapt_<class>` function. No changes to the JSON bundle, the dashboard, or any of the client-side math.

#### Dashboard app

- `dashboard/` — Svelte 5 + Vite project, separate `package.json`, builds to `dashboard/dist/`.
- Three routes, hash-routed (`#/atlas`, `#/patient`, `#/simulator`) so deep links work on GitHub Pages.
- `dashboard/src/lib/inference.ts` — client-side math: variational E-step for prefix-conditioning, JSD between two distributions, λ-relevance reranking, classical MDS for the topic map. All vanilla TS; no `mathjs`.
- `dashboard/src/lib/sampling.ts` — seedable PRNG, Dirichlet/Categorical/Poisson samplers.
- `dashboard/src/lib/cohort.ts` — in-browser synthetic-cohort generator. Samples N=1000 patients from the loaded model at startup, precomputes cosine-θ neighbors. Re-runnable from the UI if the user wants a different seed or N.
- `dashboard/src/lib/store.ts` — Svelte stores holding the model bundle, the (in-browser-generated) synthetic cohort, the current selected patient, the current simulator prefix, and the advanced-view toggle.
- `dashboard/src/lib/atlas/` — topic map (D3 scatter on client-computed MDS coordinates) + code panel (top-N bars with λ-relevance slider).
- `dashboard/src/lib/patient/` — profile bar component (the "My Profile" hero, reused), top-contributing-codes panel, nearest-neighbor ribbon over the synthetic cohort.
- `dashboard/src/lib/simulator/` — prefix editor, sampler loop, density-strip carpet, top-expected-codes panel, N-slider.
- `dashboard/public/data/` — the JSON bundle (four files), committed at v1.
- GH Actions workflow (`.github/workflows/dashboard.yml`) that builds `dashboard/` and deploys `dashboard/dist/` to `gh-pages` on push to `main`.
- ADR 0020 documenting artifact contract, framing, hosting choice.

### Not in v1 (explicit follow-ons)

- **Cohort Map tab (UMAP of synthetic profiles).** Earlier-design candidate; cut from v1 to keep the dashboard tight at three tabs. Reasonable v2 if the demo benefits from showing the synthetic cohort's spread. The nearest-neighbor ribbon in the Patient Explorer covers the "patients-like-me" intuition for v1.
- **Delta-against-baseline overlays in the simulator.** Comparing two what-if simulations visually is compelling but doubles the rendering and the cognitive load. Defer; a single what-if edit re-samples in place.
- **Per-trajectory code timeline.** Fine-grain "which code at which time" is not modeled (BOW exchangeability); we don't render it.
- **Horizon-chart row encoding** for the simulator carpet. The flat density-strip is the v1 visual; horizon charts are a precision-density upgrade if the basic version undersells.
- **Time axis / Poisson-rate timing layer.** The model has no clock; "year-of-life scope" is the natural sample unit and is honest. The simulator displays a single posterior snapshot per sample, not an evolution.
- **Live model re-export from the dashboard.** Out of scope; the dashboard is read-only.
- **Authentication / private deployments.** Public static hosting only. A private-mirror variant is a separate workstream.
- **Cohort comparison views** ("here's a peds onc cohort vs. a rare-disease cohort"). Requires per-cohort aggregate exports, which is more data-export negotiation. Reserved for v2.

### Never (dropped from scope)

- **Any per-real-patient data in the bundle.** Profiles, code bags, year sequences, nearest-neighbor identifiers — all synthetic. Bundle review at export time should fail if any real `person_id` makes it through.
- **Server-side inference.** Even a small inference endpoint is out of scope for v1; static hosting is the contract.
- **Showing per-trajectory code-time pairs** as if they were realistic event sequences. BOW exchangeability makes the ordering meaningless and would invite a misreading.

## Architecture

Two layers, well-separated:

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 1: Cloud-side export                                  │
│  (Python; charmpheno.export.dashboard + training driver)     │
│                                                              │
│  Trained VIResult                                            │
│   → model.json, phenotypes.json, vocab.json,                 │
│     corpus_stats.json    (4 files; ~MB; only model           │
│                           parameters + aggregate scalars)    │
└──────────────────────────────┬───────────────────────────────┘
                               │ artifacts only; no patient data
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 2: Static SPA                                         │
│  (Svelte + Vite + D3 + vanilla TS)                           │
│                                                              │
│  Loads bundle once on startup. At runtime:                   │
│   - generates a synthetic cohort (N=1000) in JS              │
│   - computes JSD/MDS coords for the topic map in JS          │
│   - runs variational E-step + sampling for the simulator     │
└──────────────────────────────────────────────────────────────┘
```

The boundary between the two layers is the security boundary AND the schema in §"Data Bundle Schema" below. Layer 1 runs in the secure environment; layer 2 runs in any browser. The four-file bundle is the entire contract; everything else (synthetic patients, topic-map geometry, posterior inference) is regenerated client-side.

Pulling synthetic-cohort generation and MDS into the browser makes the privacy story tighter (the cloud emits only model parameters and aggregate scalars) and removes a separate "local Python build step" from the workflow.

## Data Bundle Schema

Four JSON files. Sizes are illustrative for K=80, V=5000 (trimmed from V_full=10000).

### `model.json` (~3 MB, gzip ~700 KB)

```jsonc
{
  "K": 80,
  "V": 10000,
  "alpha": [0.05, 0.05, ...],          // length K
  "beta": [[0.0001, 0.00002, ...], ...] // K × V row-stochastic
}
```

β is the trained `E[β]` matrix (`lambda_ / lambda_.sum(axis=1, keepdims=True)`), restricted to the top-N vocab columns by corpus frequency and renormalized so each row sums to 1 over the displayed vocab (default N=5000, configurable). V in `model.json` reflects this trimmed width; the original V_full is recorded in `corpus_stats.json` for transparency. Floating-point values are encoded at 6-decimal precision to keep the bundle compact; the inference loss is negligible.

### `phenotypes.json` (~10 KB)

```jsonc
{
  "phenotypes": [
    {
      "id": 0,
      "label": "Cardiac arrhythmia",          // hand-curated or empty string
      "npmi": 0.18,
      "corpus_prevalence": 0.064,             // mean θ_k over corpus
      "junk_flag": false                       // npmi < threshold
    },
    ...
  ],
  "npmi_threshold": 0.0
}
```

The `label` field is initially empty (the dashboard falls back to "Phenotype 0", etc.). A separate curation pass populates it; the dashboard re-deploys when labels change. No model retraining required.

### `vocab.json` (~500 KB at top-N=5000)

```jsonc
{
  "codes": [
    {
      "id": 0,                  // vocab index — matches β column index
      "code": "I48.91",
      "description": "Unspecified atrial fibrillation",
      "domain": "condition",    // condition | drug | procedure | measurement | observation
      "corpus_freq": 0.014       // marginal probability over corpus
    },
    ...
  ]
}
```

Code metadata (the human description and OMOP domain) is resolved at export time from a local concept-name cache so the cloud side doesn't need to ship an OMOP concept table. The `codes` array is trimmed to the top-N entries by `corpus_freq` (default N=5000); β in `model.json` is trimmed and renormalized to the same N columns. `codes[i].id == i`, and column `i` of β corresponds to `codes[i]`.

Codes whose `description` couldn't be resolved fall back to the integer concept_id as a string.

### `corpus_stats.json` (~80 bytes)

```jsonc
{
  "corpus_size_docs": 1234567,
  "mean_codes_per_doc": 18.4,
  "k": 80,
  "v": 5000,        // matches model.json: trimmed-vocab width
  "v_full": 10000   // model's original V; informational only
}
```

Calibration scalars for the client-side synthetic generator (just `mean_codes_per_doc`) and header-strip metadata. `v_full` is recorded for transparency about the trim.

### Computed client-side at runtime (not in the bundle)

- **Topic-map coordinates.** Classical MDS on the K × K JSD distance matrix between β rows. K=80 is small enough to compute in <100ms in JS; precomputing was YAGNI.
- **Synthetic cohort.** N=1000 patients drawn from the model's generative process in JS at app startup. For each patient: `θ ~ Dirichlet(α)`, `N_codes ~ Poisson(mean_codes_per_doc)`, then `N_codes` `(z, w)` pairs. Patient ids are `synth_NNNN` (zero-padded). Top-8 cosine-θ neighbors precomputed in JS once the cohort is sampled. Re-runnable from the UI with a different seed if the user wants fresh patients.
- **Per-patient posterior θ given a prefix.** Variational E-step on the trained β; the simulator uses this.

## Tab Designs

### Phenotype Atlas

Two-panel layout, pyLDAvis silhouette, reimplemented from scratch in D3 + Svelte.

**Left panel — topic map.**
- D3 scatter of K phenotypes at client-computed MDS coordinates (JSD on β rows).
- Bubble size: `corpus_prevalence`.
- Bubble color: NPMI (sequential) by default; toggle to prevalence (sequential) in the header.
- Hover: tooltip with label, prevalence, NPMI.
- Click: selects the phenotype, drives the right panel.
- A small "junk" badge on bubbles with `junk_flag: true`. Honestly displayed, not hidden.

**Right panel — code panel for the selected phenotype.**
- Top-N codes (N=20 by default) as horizontal bars.
- Each bar shows two stacked segments: red-tinted `p(code | phenotype)` and blue-tinted `lift = p(code | phenotype) / corpus_freq(code)`.
- λ-relevance slider (0 ≤ λ ≤ 1, shown only in **advanced view**): top codes re-rank live by `relevance(w | k) = λ·log p(w|k) + (1-λ)·log(p(w|k)/p(w))` (Sievert-Shirley). Recompute is cheap.
- Each code row: code ID, human description, domain chip (condition/drug/procedure/measurement/observation), and a small `corpus_freq` indicator bar.
- An **advanced view** toggle in the header reveals the λ slider, NPMI badges, and other technical affordances. Default is the simpler view: labels + top codes + co-occurring phenotypes only. Same data; the toggle adds, doesn't remove.

**Linked highlight (the pyLDAvis-beyond move).**
- Hover a code in the right panel → bubbles in the left panel highlight if that code is in their top-N. Reveals shared-code structure across phenotypes that's invisible in vanilla pyLDAvis.

### Patient Explorer

A single synthetic patient at a time. Three vertical sections.

**Patient picker.**
- Dropdown / search over the in-browser-generated cohort by id.
- A "shuffle" button picks one at random.
- A "regenerate cohort" affordance (advanced view only) re-runs the synthetic generator with a new seed.
- A persistent "synthetic — generated from model" badge sits next to the picker; this is the privacy-story surface.

**Profile section.**
- The "My Profile" hero ([`profile_example.png`](../../../profile_example.png)) realized as a live component: stacked horizontal bar of θ percentages, colored by phenotype, with labels and the percentages.
- "Other (XX%)" bucket for the long tail below a threshold.
- Click a band → expands the contributing-codes panel below for that phenotype.

**Top contributing codes.**
- For the selected phenotype (or all, when none selected): the codes from the patient's `code_bag` that contributed most to that phenotype's mass. Computed as `count(w in bag) × p(z=k | w)` where `p(z=k | w) ∝ β_{k,w} · θ̂_k`. Sorted descending.
- Rows: code, description, count in this patient's bag, contribution score.

**Nearest neighbors ribbon.**
- A horizontal strip of 5-8 other synthetic patients with the closest cosine similarity to the selected patient's θ.
- Each neighbor renders as a tiny profile bar (the same component, miniaturized).
- Hover: see the neighbor's id; click: jump to that patient.
- Similarity computed once when the synthetic cohort is generated and held in memory as a Map<patientId, neighborIds[]>. N=1000 × N=1000 cosine similarities is a few ms in JS.

### Simulator

Conditioning-only. No time, no event axis, no λ. Each draw is a complete year-of-life code bag.

**Prefix editor (left rail).**
- Three input modes:
  1. **Start from a synthetic patient** — populates the prefix with their full code bag.
  2. **Start blank** — empty prefix.
  3. **Free composition** — autocomplete-search the vocabulary, add codes one by one.
- Editable list view: each code has a remove button. A "force phenotype" affordance: pick a phenotype, the editor injects its top-M codes (M=5) into the prefix.
- "Re-sample" button triggers the N-sample loop.

**N-slider.**
- Slider for N (number of sampled completions): 10 → 2000, default 1000.
- As the user drags, the density strips tighten visibly. This is the Bayesian-touch dynamism that earns its keep without claiming time.

**Density-strip carpet (main view).**
- One row per phenotype (all K shown — the K-is-big visual punch).
- Each row: horizontal density of θ_k across the N sampled completions, rendered as a smoothed kernel density or a P10/P25/P50/P75/P90 box-glyph. Implementation choice: start with the 5-quantile box-glyph (cheap, readable); evaluate KDE if it looks better in practice.
- Row label: phenotype name + median θ value at right.
- Sort by: median θ descending (default), spread (P90 − P10), NPMI, alphabetical. Toggle in the header.
- Click a row → expanded inline panel showing the top expected codes for that phenotype across the sampled completions (cousin of the Patient Explorer's contributing-codes panel, aggregated over the N draws).

**Top expected codes panel (right rail).**
- Bar chart of the top 30 codes by expected count across the N completions.
- Each bar: median count, plus a thin range bar for P10–P90.
- Sorted by median descending.
- Click a code: highlights phenotypes in the carpet for which this code is high-weight (mirrors the Atlas linked-highlight).

**Methods footnote (always visible, small text below the chart).**
- "Year-of-life scope; code ordering and timing not modeled. N=<current>, K=<K>, prefix length=<len>."

## Math

### Prefix conditioning (per-doc variational E-step)

Given a prefix bag of codes `w_{1..n}` with counts `n_w`, run the standard LDA variational E-step to obtain the variational posterior over θ:

```
γ_k = α_k + Σ_w n_w · φ_{w,k}
φ_{w,k} ∝ E[β_{k,w}] · exp(ψ(γ_k))
```

Iterate the two updates until γ converges (or for a fixed 20 iterations — typically converges in fewer). The posterior mean is `θ̂_k = γ_k / Σ_j γ_j`. This is identical to the inference the model already does at training time, just run on one new bag with the trained β frozen.

Implementation: ~30 lines of TypeScript; β is loaded once, the loop is dense and vectorizable, ~10 ms for a typical prefix.

### Generative sampling (one completion)

Given `θ̂`:

```
Sample N_codes ~ Poisson(mean_codes_per_doc)
For n in 1..N_codes:
  Sample z ~ Categorical(θ̂)
  Sample w ~ Categorical(β_z)
  Append w to the bag
```

After the completion, recompute θ̂ on the prefix-plus-completion to get the posterior θ for *this* completion. (Alternative: skip the recompute and use the original θ̂. Tradeoff: the recompute slightly increases between-sample variance and is more faithful to the generative process given the new evidence. We do the recompute; it's cheap.)

### N-sample density

Repeat the completion N times → N θ posterior vectors. Per-phenotype density / quantiles computed driver-side at render time.

### JSD-MDS for the topic map (client-side, at load time)

Computed in JS once when the bundle finishes loading:

```
For each pair (i, j) of phenotypes:
  M = (β_i + β_j) / 2
  JSD(i,j) = (KL(β_i || M) + KL(β_j || M)) / 2

D = sqrt(JSD)         # metric form
coords = classical MDS on D, 2 components
```

Classical MDS = double-center `D²`, eigendecompose, take the top-2 eigenvectors scaled by sqrt of their eigenvalues. K=80 → an 80×80 symmetric eigendecomposition, well under 100ms in JS using a small hand-rolled Jacobi or power-iteration routine. Deterministic given β.

### Cohort generation (client-side, at app startup)

Run once when the bundle loads:

```
For n in 1..N_synth:
  theta_n ~ Dirichlet(alpha)
  N_codes ~ Poisson(mean_codes_per_doc)
  bag_n = []
  For c in 1..N_codes:
    z ~ Categorical(theta_n)
    w ~ Categorical(beta[z])
    bag_n.append(w)

# Cosine-θ neighbors
sim = (theta @ theta.T) normalized per row
neighbors[n] = top-K rows of sim with diagonal masked
```

Deterministic given the user-controllable seed. The cohort is held in a Svelte store and consumed by the Patient Explorer + the Simulator's "load from patient" affordance.

## Build & Deploy

- `dashboard/package.json` declares Svelte 5, Vite, D3, and `@types/d3`. No other runtime dependencies. Dev dependencies include TypeScript and Vitest.
- `dashboard/Makefile` targets:
  - `make dev` — Vite dev server (default base `/`)
  - `make build` — production build with `VITE_BASE=/CHARMPheno/`
  - `make build-local` — local-only build with `VITE_BASE=/` so `dist/` can be served at the root
  - `make preview` — `vite preview` against the production build; surfaces base-URL / asset-path issues the dev server hides
  - `make test` — Vitest
- Two ways to produce a data bundle:
  - **From a real checkpoint:** `poetry run python analysis/local/build_dashboard.py --checkpoint … --out-dir dashboard/public/data` (Spark-using; slow but real).
  - **Dev fixture:** `python scripts/make_dev_bundle.py --out-dir dashboard/public/data --k 10 --v 200` (no Spark, no checkpoint, ~1 second). Generates a small synthetic-but-schema-conformant bundle for dashboard work without the rest of the stack.
- `.github/workflows/dashboard.yml`: on push to `main`, install Node, install `dashboard/` deps, run `make build`, deploy `dashboard/dist/` to `gh-pages` branch via `peaceiris/actions-gh-pages`. Optional: a separate `make data` step that runs against a committed `data/` directory — for v1, the data bundle is committed under `dashboard/public/data/`, so the CI build is purely the Svelte build.
- The deployed URL: `https://<owner>.github.io/CHARMPheno/`. The Vite `base` config is set accordingly.

## Testing

### Unit (Vitest, fast tier)

- **Variational E-step.** Hand-built K=3, V=5 β matrix, hand-built prefix bag, hand-computed expected γ. Tolerance 1e-6 on `θ̂` after convergence.
- **Sampling reproducibility.** Same seed + same prefix → identical completion. (Use a seedable PRNG, not `Math.random()`.)
- **λ-relevance reranking.** Known β row, known corpus_freq, hand-computed expected ordering at λ ∈ {0, 0.5, 1}.
- **JSD computation.** Symmetric, in `[0, log 2]`, zero on identical distributions.
- **Classical MDS.** Hand-built 4×4 distance matrix with known 2D embedding (e.g., points on a square); verify reconstructed coords match within a rotation/reflection.
- **Cohort generation.** Determinism (same seed → same cohort), theta-rows-on-simplex, neighbor invariants (no self, all distinct, |neighbors| == K).
- **Density / quantile aggregation.** N=100 known-distribution samples, verify P10/P50/P90 match a known reference.

### Visual / interaction (Playwright, slow tier — optional for v1)

- Load the dashboard, switch tabs, change a phenotype, edit a prefix, re-sample. Just smoke tests; visual regression is overkill for v1.

### Python-side

- **Export builder.** Round-trip a small `VIResult` through `charmpheno.export.dashboard.build` and verify the emitted JSON conforms to the schema (one schema-validation test per file).
- **Vocab trim.** Top-N selection by `corpus_freq` is deterministic; β rows are re-normalized after column-trimming so each row sums to 1 over the trimmed columns.
- **End-to-end smoke.** `analysis/local/build_dashboard.py` against a tiny fixture VIResult + simulated parquet emits exactly the four expected files with conformant schemas.

## Migration / Compatibility

- The export pipeline depends on a sidecar `corpus_stats.json` next to the saved `VIResult` checkpoint. The training drivers gain a small extension to compute and write this at save time. Existing checkpoints can be enriched retroactively by a one-shot script that re-reads the BOW and writes the sidecar — additive, no changes to `VIResult` itself.
- The dashboard app is its own subproject and does not touch any existing Python code.

## ADR

ADR 0020 lands with this branch and records:

- Static-hosting + GH Pages decision (constraint and consequence).
- Synthetic-only patient framing (privacy story as feature).
- Three-tab scope (Atlas / Patient / Simulator); deferred Cohort Map.
- The artifact contract (the four JSON files above) as the boundary between modeling and dashboard.
- The decision to compute synthetic cohorts and topic-map MDS coordinates client-side rather than ship them — minimizes cloud-side egress and makes the privacy story tighter.
- The decision to drop the temporal/event axis from the simulator (BOW exchangeability + Poisson-rate calibration risk).
- The decision to reimplement pyLDAvis rather than embed/extend it.

## Open Questions

The brainstorming session settled the major forks; a few items remain for the implementation plan to nail down but they do not require user input:

- **Density-strip glyph choice** (5-quantile box vs. KDE). Settled by visual evaluation during implementation.
- **GH Pages workflow specifics** (single-branch deploy vs. orphan `gh-pages`). Pick the project's existing convention if any.
- **Default vocab top-N**. Spec defaults to 5000; revisit if the bundle is still uncomfortably large or if too many simulator outputs end up "unresolved".
- **Domain-tagging of phenotypes for a categorical color mode** (v2). Requires a one-shot offline computation (top-domain per phenotype's top-N codes). Not in v1; v1 supports NPMI and prevalence coloring.
