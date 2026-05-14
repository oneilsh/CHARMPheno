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
  let cohortSeed = DEFAULT_COHORT_SEED

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

  function regenCohort() {
    if (!$bundle) return
    cohortSeed += 1
    cohort.set(generateCohort({
      model: $bundle.model,
      meanCodesPerDoc: $bundle.corpusStats.mean_codes_per_doc,
      n: DEFAULT_COHORT_N,
      seed: cohortSeed,
      nNeighbors: DEFAULT_NEIGHBORS,
    }))
  }
</script>

<main>
  <header class="masthead">
    <div class="brand">
      <svg class="mark" viewBox="0 0 32 32" aria-hidden="true">
        <circle cx="16" cy="16" r="11" fill="none" stroke="#1f1b16" stroke-width="1.25" />
        <circle cx="16" cy="16" r="4.5" fill="#b25b2c" />
        <line x1="16" y1="2" x2="16" y2="6" stroke="#1f1b16" stroke-width="1" />
        <line x1="16" y1="26" x2="16" y2="30" stroke="#1f1b16" stroke-width="1" />
        <line x1="2" y1="16" x2="6" y2="16" stroke="#1f1b16" stroke-width="1" />
        <line x1="26" y1="16" x2="30" y2="16" stroke="#1f1b16" stroke-width="1" />
      </svg>
      <div class="lockup">
        <span class="serif title">CharmPheno</span>
        <span class="eyebrow subtitle">Phenotype Atlas · synthetic demonstration</span>
      </div>
    </div>

    {#if $bundle}
      <dl class="metadata" data-numeric>
        <div><dt>K</dt><dd>{$bundle.model.K}</dd></div>
        <div><dt>V</dt><dd>{$bundle.model.V.toLocaleString()}<span class="of"> / {$bundle.corpusStats.v_full.toLocaleString()}</span></dd></div>
        <div><dt>Corpus</dt><dd>{($bundle.corpusStats.corpus_size_docs / 1000).toFixed(0)}<span class="of">k docs</span></dd></div>
      </dl>
    {/if}

    <div class="controls">
      {#if $bundle && $advancedView}
        <button class="btn btn-ghost regen" on:click={regenCohort} title="Re-roll synthetic cohort with a new seed">
          ↻ regenerate cohort
        </button>
      {/if}
      <label class="view-toggle">
        <input type="checkbox" bind:checked={$advancedView} />
        <span class="view-label" class:active={!$advancedView}>simple</span>
        <span class="view-sep">/</span>
        <span class="view-label" class:active={$advancedView}>advanced</span>
      </label>
    </div>
  </header>

  <hr class="rule" />

  {#if error}
    <p class="error">Failed to load bundle: {error}</p>
  {:else if !$bundle}
    <p class="loading"><span class="loading-dot"></span> Loading model bundle…</p>
  {:else}
    <Tabs />
    {#if $route === 'atlas'}<Atlas />{:else if $route === 'patient'}<Patient />{:else}<Simulator />{/if}
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
    grid-template-columns: auto 1fr auto;
    align-items: center;
    gap: 2rem;
    padding: 1.75rem 0 1.5rem;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 0.85rem;
  }
  .mark {
    width: 30px;
    height: 30px;
    flex-shrink: 0;
  }
  .lockup {
    display: flex;
    flex-direction: column;
    line-height: 1;
  }
  .title {
    font-size: 1.55rem;
    font-weight: 500;
    font-style: italic;
    letter-spacing: var(--tracking-tight);
    color: var(--ink);
  }
  .subtitle {
    margin-top: 0.35rem;
    font-size: 0.65rem;
  }

  .metadata {
    display: flex;
    gap: 1.5rem;
    margin: 0;
    justify-self: center;
    padding: 0.5rem 1.2rem;
    background: var(--paper-elevated);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
  }
  .metadata div {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.05rem;
  }
  .metadata dt {
    font-family: var(--font-body);
    font-size: var(--fs-micro);
    letter-spacing: var(--tracking-eyebrow);
    text-transform: uppercase;
    color: var(--ink-faint);
    margin: 0;
  }
  .metadata dd {
    margin: 0;
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--ink);
  }
  .metadata .of {
    color: var(--ink-faint);
    font-weight: 400;
    font-size: 0.78rem;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 1.5rem;
  }
  .regen {
    font-family: var(--font-body);
    font-size: var(--fs-small);
    letter-spacing: 0.01em;
  }
  .view-toggle {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: var(--fs-small);
    color: var(--ink-faint);
    cursor: pointer;
    user-select: none;
    position: relative;
  }
  .view-toggle input {
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }
  .view-label {
    transition: color 0.15s ease;
    padding-bottom: 1px;
    border-bottom: 1px solid transparent;
  }
  .view-label.active {
    color: var(--ink);
    border-bottom-color: var(--terracotta);
  }
  .view-sep {
    color: var(--ink-faint);
  }

  .loading,
  .error {
    padding: 3rem 0;
    color: var(--ink-muted);
    font-style: italic;
  }
  .error {
    color: var(--brick);
  }
  .loading-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--terracotta);
    margin-right: 0.5rem;
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 0.3; transform: scale(0.85); }
    50% { opacity: 1; transform: scale(1.15); }
  }
</style>
