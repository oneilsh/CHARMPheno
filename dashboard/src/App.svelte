<script lang="ts">
  import { onMount } from 'svelte'
  import { bundle, advancedView, cohort, selectedPhenotypeId } from './lib/store'
  import { loadBundle } from './lib/bundle'
  import { route } from './lib/router'
  import { generateCohort } from './lib/cohort'
  import Tabs from './lib/Tabs.svelte'
  import Atlas from './lib/tabs/Atlas.svelte'
  import Patient from './lib/tabs/Patient.svelte'
  import Simulator from './lib/tabs/Simulator.svelte'
  import { installTooltips } from './lib/tooltip'

  const DEFAULT_COHORT_N = 1000
  const DEFAULT_COHORT_SEED = 42
  const DEFAULT_NEIGHBORS = 8

  let error: string | null = null
  let cohortSeed = DEFAULT_COHORT_SEED

  // Pick a phenotype to highlight on first load so the detail panel and
  // bubble selection ring are populated immediately. First-paint emptiness
  // is a worse onboarding than an arbitrary-but-relevant default. Prefer
  // the highest-prevalence "real" phenotype (quality in {phenotype, anchor}
  // or unknown null); fall back to id 0 if nothing qualifies.
  function pickDefaultPhenotype(b: typeof $bundle): number | null {
    if (!b) return null
    const ps = b.phenotypes.phenotypes
    const good = ps.filter((p) => p.quality === 'phenotype' || p.quality === 'anchor' || p.quality == null)
    const pool = good.length ? good : ps
    let best = pool[0]
    for (const p of pool) if (p.corpus_prevalence > best.corpus_prevalence) best = p
    return best ? best.id : null
  }

  onMount(async () => {
    installTooltips()
    try {
      const b = await loadBundle(import.meta.env.BASE_URL)
      bundle.set(b)
      selectedPhenotypeId.set(pickDefaultPhenotype(b))
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
      <svg class="mark" viewBox="0 0 24 24" aria-hidden="true">
        <!-- Abstract mark: two circles, one ink-stroked, one cyan-filled, offset.
             Reads as "a data point in a coordinate system" without literalism. -->
        <circle cx="9" cy="9" r="5" fill="none" stroke="#0a0a0a" stroke-width="1.5" />
        <circle cx="16" cy="16" r="2.5" fill="#06b6d4" />
      </svg>
      <div class="lockup">
        <span class="title">CHARMPheno</span>
        <span class="subtitle">exploring latent phenotypes</span>
      </div>
    </div>

    {#if $bundle && $advancedView}
      <dl class="metadata" data-numeric>
        <div title="K: the number of phenotypes (topics) the model was asked to learn from the dataset.">
          <dt>K</dt><dd>{$bundle.model.K}</dd>
        </div>
        <div title="V: distinct conditions displayed in the dashboard, over total distinct conditions in the source dataset. Low-count conditions are suppressed for patient privacy.">
          <dt>V</dt><dd>{$bundle.model.V.toLocaleString()}<span class="of">/{$bundle.corpusStats.v_full.toLocaleString()}</span></dd>
        </div>
        <div title="n: number of patient records the model was fit on (in thousands).">
          <dt>n</dt><dd>{($bundle.corpusStats.corpus_size_docs / 1000).toFixed(0)}<span class="of">k</span></dd>
        </div>
      </dl>
    {/if}

    <div class="controls">
      {#if $bundle && $advancedView && $route !== 'atlas'}
        <button class="btn-ghost regen" on:click={regenCohort} title="Re-roll synthetic cohort with a new seed">
          ↻ regenerate cohort
        </button>
      {/if}
      <div class="seg" role="group" aria-label="View density">
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

  /* Metadata strip (instrument-readout) */
  .metadata {
    display: flex;
    gap: 1.25rem;
    margin: 0;
    justify-self: center;
    padding: 0.45rem 0.95rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
  }
  .metadata div {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
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
    font-size: 0.92rem;
    font-weight: 500;
    color: var(--ink);
  }
  .metadata .of {
    color: var(--ink-faint);
    font-weight: 400;
    font-size: 0.78rem;
    margin-left: 1px;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 1rem;
    /* Pin to the third grid column explicitly so toggling between simple
       and advanced (which hides/shows the metadata block in column 2)
       doesn't shift the segmented control across the masthead. */
    grid-column: 3;
  }
  .regen {
    font-family: var(--font-body);
    font-size: var(--fs-small);
  }
  .regen:hover { color: var(--accent); }

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
