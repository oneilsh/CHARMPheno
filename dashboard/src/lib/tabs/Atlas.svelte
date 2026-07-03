<script lang="ts">
  import { bundle, atlasConditioning } from '../store'
  import TopicMap from '../atlas/TopicMap.svelte'
  import CodePanel from '../atlas/CodePanel.svelte'
  import ConditionSearch from '../atlas/ConditionSearch.svelte'
  import PhenotypeBrowser from '../atlas/PhenotypeBrowser.svelte'
  import CorrelationHeatmap from '../atlas/CorrelationHeatmap.svelte'
  import ConditioningBar from '../conditioning/ConditioningBar.svelte'
  import { copy } from '../copy'

  // (Conditioning reset on cohort change is handled centrally in App.svelte via
  // resetConditioningForCohort(); no per-panel reset needed here.)

  // The main viz area shows EITHER the bubble map or the topic-correlation
  // heatmap (correlation is a static Sigma artifact — covariate-independent — so
  // it lives as an alternate view, not a conditioned panel).
  let view: 'map' | 'correlation' = 'map'

  // Background-click closes the disclosure popover so the user doesn't
  // have to find and click the same link again to dismiss it.
  let whatIsEl: HTMLDetailsElement
  let whatIsOpen = false
</script>

<svelte:window on:click={(e) => {
  if (whatIsOpen && whatIsEl && !whatIsEl.contains(e.target as Node)) {
    whatIsOpen = false
  }
}} />

<section class="atlas">
  <header class="section-head">
    <div class="title-block">
      <div class="title-row">
        <h1>{copy.atlas.title}</h1>
        <details class="what-is" bind:this={whatIsEl} bind:open={whatIsOpen}>
          <summary>{copy.atlas.whatIsSummary}</summary>
          <div class="what-is-body popover">
            {#each copy.atlas.whatIs($bundle?.model.K ?? '~80') as para}
              <p>{@html para}</p>
            {/each}
          </div>
        </details>
      </div>
      <p class="kicker">{copy.atlas.kicker}</p>
    </div>
    <div class="controls">
      <ConditionSearch />
    </div>
  </header>

  <div class="grid">
    <div class="left-col">
      {#if $bundle?.correlation}
        <div class="viz-switch" role="tablist" aria-label="Atlas view">
          <button
            type="button"
            role="tab"
            aria-selected={view === 'map'}
            class:active={view === 'map'}
            on:click={() => (view = 'map')}
          >Bubbles</button>
          <button
            type="button"
            role="tab"
            aria-selected={view === 'correlation'}
            class:active={view === 'correlation'}
            on:click={() => (view = 'correlation')}
          >Correlations</button>
        </div>
      {/if}

      {#if view === 'correlation' && $bundle?.correlation}
        <div class="corr-wrap">
          <p class="corr-kicker">{copy.correlation.kicker}</p>
          <CorrelationHeatmap correlation={$bundle.correlation} />
        </div>
      {:else}
        <TopicMap>
          <ConditioningBar store={atlasConditioning} showGroup={false} inlineControls />
        </TopicMap>
      {/if}

      <PhenotypeBrowser />
    </div>
    <CodePanel />
  </div>
</section>

<style>
  .atlas {
    padding: 0.25rem 0 3rem;
  }

  .section-head {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    gap: 2rem;
    padding-bottom: 1.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--rule);
  }
  .title-block {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
  }
  .title-block h1 { margin: 0.1rem 0 0; }
  .title-row {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    flex-wrap: wrap;
    position: relative;
  }
  .kicker {
    margin: 0.25rem 0 0;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    max-width: 62ch;
    line-height: 1.55;
  }

  /* "What's a phenotype?" disclosure. Body floats as a popover so it
     doesn't shove the kicker and the rest of the page down when opened. */
  .what-is {
    position: relative;
  }
  .what-is summary {
    cursor: pointer;
    color: var(--accent);
    font-size: var(--fs-small);
    list-style: none;
    display: inline-block;
    border-bottom: 1px dotted var(--accent);
    text-underline-offset: 2px;
  }
  .what-is summary::-webkit-details-marker { display: none; }
  .what-is summary::marker { display: none; }
  .what-is summary:hover {
    color: var(--ink);
    border-bottom-color: var(--ink);
  }
  .what-is[open] summary {
    color: var(--ink);
    border-bottom-color: transparent;
  }
  .what-is-body {
    margin-top: 0.6rem;
    max-width: 62ch;
    padding: 0.85rem 1rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-sm);
  }
  /* Popover variant: anchored under the inline summary, floats over the
     content below so the masthead doesn't grow when expanded. */
  .what-is-body.popover {
    position: absolute;
    top: 1.6rem;
    left: 0;
    z-index: 5;
    width: 62ch;
    max-width: min(62ch, calc(100vw - 4rem));
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
  }
  .what-is-body p {
    margin: 0 0 0.55rem;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    line-height: 1.6;
  }
  .what-is-body p:last-child { margin-bottom: 0; }
  /* :global because the popover paragraphs are injected via {@html} from
     copy.ts, so their <em>/<strong> don't receive Svelte's scoping hash. */
  .what-is-body :global(em) {
    font-style: italic;
    color: var(--ink);
  }

  .controls {
    display: flex;
    align-items: end;
    gap: 1.25rem;
  }

  .grid {
    display: grid;
    grid-template-columns: 1.1fr 1fr;
    gap: 1.5rem;
    align-items: start;
  }
  .left-col {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    min-width: 0;
  }

  /* Segmented control: switch the main viz between the bubble map and the
     topic-correlation heatmap (only shown when the bundle carries correlation
     data — see bundle.correlation in types.ts). */
  .viz-switch {
    display: inline-flex;
    align-self: flex-start;
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius-sm);
    overflow: hidden;
  }
  .viz-switch button {
    border: 0;
    background: var(--surface);
    color: var(--ink-muted);
    padding: 0.3rem 0.85rem;
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    cursor: pointer;
    transition: color 0.12s ease, background 0.12s ease;
  }
  .viz-switch button + button {
    border-left: 1px solid var(--rule-strong);
  }
  .viz-switch button:hover {
    color: var(--ink);
  }
  .viz-switch button.active {
    background: var(--accent-faint);
    color: var(--accent);
  }

  .corr-wrap {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
  .corr-kicker {
    margin: 0;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    max-width: 52ch;
    line-height: 1.5;
  }
</style>
