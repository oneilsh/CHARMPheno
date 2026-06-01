<script lang="ts">
  import { onMount } from 'svelte'
  import { get } from 'svelte/store'
  import {
    bundle, advancedView, cohort, manifest, patientProjection,
    prevalenceReader,
    searchedConditionIdx, searchedPhenotypeForPatients,
    selectedCohort, selectedPatientId, selectedPhenotypeId,
    simulatorPrefix,
  } from './lib/store'
  import type { DashboardBundle } from './lib/types'
  import { loadBundle, loadManifest } from './lib/bundle'
  import { route, type Route } from './lib/router'
  import { copy } from './lib/copy'
  import { generateCohort } from './lib/cohort'
  import { ensurePatientProjection } from './lib/patient/projection'
  import CohortSelector from './lib/CohortSelector.svelte'
  import Tabs from './lib/Tabs.svelte'
  import Atlas from './lib/tabs/Atlas.svelte'
  import Patient from './lib/tabs/Patient.svelte'
  import Simulator from './lib/tabs/Simulator.svelte'
  import { installTooltips } from './lib/tooltip'
  import { startTour } from './lib/tour'

  // Mapping from route id to its top-level tab component. Paired with
  // router.ts's TABS list, adding a new tab is then a two-line change:
  // append to TABS, add an entry here.
  const TAB_COMPONENTS: Record<Route, ConstructorOfATypedSvelteComponent> = {
    atlas: Atlas,
    patient: Patient,
    simulator: Simulator,
  }

  // Initial batch size. Adaptive sizing in cohort.ts may push the final
  // count higher (rounded to multiples of 100) until at least one of the
  // clean / messy buckets reaches 1000. Final counts depend on the
  // model's clean/messy phenotype mix.
  const DEFAULT_COHORT_N = 1500
  const DEFAULT_COHORT_SEED = 42
  const DEFAULT_NEIGHBORS = 8

  let error: string | null = null

  // Pick a phenotype to highlight on first load so the detail panel and
  // bubble selection ring are populated immediately. First-paint emptiness
  // is a worse onboarding than an arbitrary-but-relevant default. Prefer
  // the highest-prevalence "real" phenotype (quality in {phenotype, anchor}
  // or unknown null); fall back to id 0 if nothing qualifies.
  function pickDefaultPhenotype(b: DashboardBundle | null): number | null {
    if (!b) return null
    const reader = get(prevalenceReader)
    const ps = b.phenotypes.phenotypes
    const good = ps.filter((p) => p.quality === 'phenotype' || p.quality === 'anchor' || p.quality == null)
    const pool = good.length ? good : ps
    let best = pool[0]
    // Tiebreak by corpus_prevalence desc so phenotypes tied at 0 (e.g. when τ
    // is set high enough that no patient clears it) still get a stable choice.
    for (const p of pool) {
      const a = reader(p), b_ = reader(best)
      if (a > b_ || (a === b_ && p.corpus_prevalence > best.corpus_prevalence)) best = p
    }
    return best ? best.id : null
  }

  // Seed the simulator with three related inflammatory conditions
  // (atopic-airway + skin) so the default Simulate run has something
  // interesting to chew on instead of drawing from the bare prior.
  // Looked up by exact description so we keep working if vocab ids
  // shift between bundles. Falls back silently if any of these aren't
  // in the vocab.
  const SIMULATOR_SEED_CONDITIONS = ['Asthma', 'Psoriasis', 'Atopic dermatitis']
  function pickSimulatorSeedPrefix(b: typeof $bundle): number[] {
    if (!b) return []
    const out: number[] = []
    for (const desc of SIMULATOR_SEED_CONDITIONS) {
      const c = b.vocab.codes.find((x) => x.description === desc)
      if (c) out.push(c.id)
    }
    return out
  }

  // Token guarding against stale loads: if the user changes cohorts twice
  // in quick succession, the first fetch may resolve after the second.
  // We bump `loadToken` on each load and bail out of any in-flight load
  // whose token doesn't match the current one.
  let loadToken = 0

  async function loadCohortBundle(cohortId: string) {
    const token = ++loadToken
    // Hard reset: clear bundle + everything derived from it so the UI
    // shows the "loading" state and stale views (selected phenotype IDs
    // from the old K, the cohort's UMAP, etc.) don't briefly render
    // against the new model's data.
    bundle.set(null)
    cohort.set(null)
    patientProjection.set(null)
    selectedPhenotypeId.set(null)
    selectedPatientId.set(null)
    searchedConditionIdx.set(null)
    searchedPhenotypeForPatients.set(null)
    try {
      const b = await loadBundle(import.meta.env.BASE_URL, cohortId)
      if (token !== loadToken) return  // a newer load has started; abandon
      bundle.set(b)
      selectedPhenotypeId.set(pickDefaultPhenotype(b))
      simulatorPrefix.set(pickSimulatorSeedPrefix(b))
      const c = generateCohort({
        model: b.model,
        meanCodesPerDoc: b.corpusStats.mean_codes_per_doc,
        n: DEFAULT_COHORT_N,
        seed: DEFAULT_COHORT_SEED,
        nNeighbors: DEFAULT_NEIGHBORS,
        // Adaptive sizing: oversample until one bucket hits 1000, then
        // truncate both to round-100 counts. See cohort.ts.
        qualityByPhenotype: b.phenotypes.phenotypes.map((p) => p.quality),
      })
      if (token !== loadToken) return
      cohort.set(c)
      // Start the patient-atlas UMAP fit now, on load, rather than lazily when
      // the Patient/Simulator tab first mounts. fitAsync runs in the background
      // without blocking the UI, so the layout is ready by the time the user
      // (or the guided tour) reaches the Patient tab instead of freezing it.
      ensurePatientProjection()
    } catch (e) {
      if (token !== loadToken) return
      error = (e as Error).message
    }
  }

  onMount(async () => {
    installTooltips()
    try {
      const m = await loadManifest(import.meta.env.BASE_URL)
      manifest.set(m)
      // Resolve which cohort to load: prior session's pick (from
      // localStorage) if it still exists in the manifest, else the
      // manifest's `default`. Bad/orphaned ids in localStorage are
      // silently ignored — better than a broken first paint.
      const persisted = ($selectedCohort && m.cohorts.some((c) => c.id === $selectedCohort))
        ? $selectedCohort
        : m.default
      selectedCohort.set(persisted)
      await loadCohortBundle(persisted)
    } catch (e) { error = (e as Error).message }
  })

  // React to user-driven cohort changes (CohortSelector writes to the
  // store). Skipped while the manifest hasn't loaded yet, and skipped
  // for the initial selection because onMount handles that explicitly
  // (preventing a double-load race on first paint).
  let firstSelection = true
  $: if ($manifest && $selectedCohort) {
    if (firstSelection) { firstSelection = false } else { loadCohortBundle($selectedCohort) }
  }
</script>

<main>
  <header class="masthead">
    <div class="brand">
      <svg class="mark" viewBox="0 0 24 24" aria-hidden="true">
        <!-- Abstract mark: two circles, one ink-stroked, one cyan-filled, offset.
             Reads as "a data point in a coordinate system" without literalism. -->
        <circle cx="9" cy="9" r="5" fill="none" stroke="#0a0a0a" stroke-width="1.5" />
        <circle cx="16" cy="16" r="2.5" fill="#06b6d4" />
      </svg>
      <div class="lockup">
        <span class="title">CHARMPheno</span>
        <span class="subtitle">{copy.masthead.subtitle}</span>
      </div>
    </div>

    <div class="cohort-slot">
      <div class="cohort-wrap" data-tour="cohort">
        <CohortSelector />
      </div>
      {#if $bundle && $advancedView}
        <dl class="metadata" data-numeric data-tour="metrics">
          <div title={copy.masthead.meta.k}>
            <dt>K</dt><dd>{$bundle.model.K}</dd>
          </div>
          <div title={copy.masthead.meta.v}>
            <dt>V</dt><dd>{$bundle.model.V.toLocaleString()}<span class="of">/{$bundle.corpusStats.v_full.toLocaleString()}</span></dd>
          </div>
          <div title={copy.masthead.meta.n}>
            <dt>n</dt><dd>{($bundle.corpusStats.corpus_size_docs / 1000).toFixed(0)}<span class="of">k</span></dd>
          </div>
        </dl>
      {/if}
    </div>

    <div class="controls">
      {#if $bundle}
        <button
          class="tour-link"
          on:click={() => startTour($advancedView ? 'advanced' : 'basic')}
        >{copy.tour.startLabel}</button>
      {/if}
      <div class="seg" role="group" aria-label="View density" data-tour="view-toggle">
        <button class="seg-btn" class:active={!$advancedView} on:click={() => advancedView.set(false)}>basic</button>
        <button class="seg-btn" class:active={$advancedView} on:click={() => advancedView.set(true)}>advanced</button>
      </div>
    </div>
  </header>

  <hr class="rule" />

  {#if error}
    <p class="error">Failed to load bundle: {error}</p>
  {:else if !$bundle}
    <p class="loading"><span class="loading-dot"></span> loading model bundle</p>
  {:else}
    <Tabs />
    <svelte:component this={TAB_COMPONENTS[$route]} />
  {/if}
</main>

<style>
  main {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 2rem;
  }

  .masthead {
    display: grid;
    grid-template-columns: auto auto 1fr auto;
    align-items: center;
    gap: 2rem;
    padding: 1.5rem 0 1.25rem;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
  }
  .mark {
    width: 26px;
    height: 26px;
    flex-shrink: 0;
  }
  .lockup {
    display: flex;
    flex-direction: column;
    line-height: 1.1;
  }
  .title {
    font-family: var(--font-body);
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--ink);
  }
  .subtitle {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--ink-faint);
    margin-top: 0.15rem;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    /* Pin to the fourth grid column explicitly so the segmented control
       stays right-justified. */
    grid-column: 4;
  }

  /* Cohort selector slot: middle column of the masthead grid so the
     dropdown sits between the brand and the basic/advanced toggle.
     Lets the selector be the visual anchor users grab to swap models.
     The model-size readout (advanced only) sits to its right. */
  .cohort-slot {
    grid-column: 2;
    display: flex;
    align-items: center;
    gap: 1rem;
    justify-content: flex-start;
  }

  /* Model-size readout (instrument-style), advanced mode only. Sits beside
     the cohort selector. It is shorter than the brand lockup / cohort
     selector that fix the masthead row height, so toggling advanced on and
     off never changes the masthead height — no vertical jump. */
  .metadata {
    display: flex;
    gap: 0.85rem;
    margin: 0;
    padding: 0.2rem 0.65rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
  }
  .metadata div {
    display: flex;
    align-items: baseline;
    gap: 0.35rem;
  }
  .metadata dt {
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-faint);
    margin: 0;
    font-weight: 500;
  }
  .metadata dd {
    margin: 0;
    font-size: var(--fs-small);
    font-weight: 500;
    color: var(--ink);
  }
  .metadata .of {
    color: var(--ink-faint);
    font-weight: 400;
    font-size: var(--fs-micro);
    margin-left: 1px;
  }
  /* "Take the tour" link: subtle dotted-underline link matching the
     "what is this?" disclosures, sitting just left of the view toggle. */
  .tour-link {
    border: 0;
    background: transparent;
    padding: 0;
    color: var(--accent);
    cursor: pointer;
    font-family: var(--font-body);
    font-size: var(--fs-small);
    border-bottom: 1px dotted var(--accent);
    text-underline-offset: 2px;
    white-space: nowrap;
  }
  .tour-link:hover { color: var(--ink); border-bottom-color: var(--ink); }

  /* Segmented control */
  .seg {
    display: inline-flex;
    background: var(--surface);
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius-sm);
    padding: 2px;
    gap: 1px;
  }
  .seg-btn {
    border: 0;
    background: transparent;
    padding: 0.28rem 0.7rem;
    font-family: var(--font-body);
    font-size: var(--fs-small);
    font-weight: 500;
    color: var(--ink-muted);
    cursor: pointer;
    border-radius: 3px;
    letter-spacing: -0.005em;
    transition: all 0.12s ease;
  }
  .seg-btn:hover { color: var(--ink); }
  .seg-btn.active {
    background: var(--ink);
    color: var(--surface);
  }

  /* Loading / error */
  .loading, .error {
    padding: 3rem 0;
    color: var(--ink-muted);
    font-size: var(--fs-small);
  }
  .error { color: var(--danger); }
  .loading {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: var(--font-mono);
    font-size: var(--fs-small);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .loading-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 0.3; transform: scale(0.85); }
    50% { opacity: 1; transform: scale(1.15); }
  }
</style>
