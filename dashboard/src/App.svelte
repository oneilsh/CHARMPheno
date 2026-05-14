<script lang="ts">
  import { onMount } from 'svelte'
  import { bundle, advancedView, cohort } from './lib/store'
  import { loadBundle } from './lib/bundle'
  import { route } from './lib/router'
  import { generateCohort } from './lib/cohort'
  import Tabs from './lib/Tabs.svelte'
  import Atlas from './lib/tabs/Atlas.svelte'
  import Patient from './lib/tabs/Patient.svelte'
  import Simulator from './lib/tabs/Simulator.svelte'

  const DEFAULT_COHORT_N = 1000
  const DEFAULT_COHORT_SEED = 42
  const DEFAULT_NEIGHBORS = 8

  let error: string | null = null
  onMount(async () => {
    try {
      const b = await loadBundle(import.meta.env.BASE_URL)
      bundle.set(b)
      const c = generateCohort({
        model: b.model,
        meanCodesPerDoc: b.corpusStats.mean_codes_per_doc,
        n: DEFAULT_COHORT_N,
        seed: DEFAULT_COHORT_SEED,
        nNeighbors: DEFAULT_NEIGHBORS,
      })
      cohort.set(c)
    } catch (e) { error = (e as Error).message }
  })
</script>

<main>
  <header>
    <h1>CharmPheno</h1>
    <span class="badge">demo · synthetic patients</span>
    {#if $bundle}
      <span class="meta">K = {$bundle.model.K} · V = {$bundle.model.V} (of {$bundle.corpusStats.v_full}) · corpus ≈ {($bundle.corpusStats.corpus_size_docs / 1000).toFixed(0)}k docs</span>
    {/if}
    <span class="spacer"></span>
    <label class="toggle"><input type="checkbox" bind:checked={$advancedView} /> Advanced view</label>
  </header>

  {#if error}<p class="error">Failed to load bundle: {error}</p>
  {:else if !$bundle}<p>Loading model bundle…</p>
  {:else}
    <Tabs />
    {#if $route === 'atlas'}<Atlas />{:else if $route === 'patient'}<Patient />{:else}<Simulator />{/if}
  {/if}
</main>

<style>
  main { font-family: system-ui, sans-serif; max-width: 1400px; margin: 0 auto; }
  header { display: flex; align-items: center; gap: 1rem; padding: 1rem; border-bottom: 1px solid #ddd; }
  header h1 { margin: 0; font-size: 1.25rem; }
  .badge { font-size: 0.75rem; padding: 0.15rem 0.5rem; background: #fff3cd; border: 1px solid #d4a017; border-radius: 4px; }
  .meta { font-size: 0.75rem; color: #555; }
  .spacer { flex: 1; }
  .toggle { font-size: 0.85rem; display: flex; align-items: center; gap: 0.25rem; }
  .error { color: #b00020; padding: 1rem; }
</style>
