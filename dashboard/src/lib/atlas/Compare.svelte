<script lang="ts">
  import { bundle, comparePair } from '../store'
  import { copy } from '../copy'
  import CorrelationHeatmap from './CorrelationHeatmap.svelte'
  import DifferencePane from './DifferencePane.svelte'

  // Row / column block pickers live here (Compare's header control slot,
  // mirroring Explore's ConditionSearch placement) instead of inside the
  // heatmap card, so both cards start flush below the section rule.
  // CorrelationHeatmap still owns the default-init (All × All) and matrix
  // logic — these are just bound through via its bindable props.
  let rowBlock = ''
  let colBlock = ''

  $: correlation = $bundle?.correlation
  $: gating = $bundle?.gating
  $: blocks = correlation ? Array.from(new Set(correlation.block_labels)) : []
  const blockDisplay = (b: string) => (b === 'background' ? 'All' : (gating?.group_labels?.[b] ?? b))

  // Default the compare pair to the first two distinct topics as soon as the
  // correlation bundle is available, but only ever when nothing is selected
  // yet — never override a user's click.
  $: if (correlation && $comparePair === null && correlation.topic_order.length > 1) {
    comparePair.set({ a: correlation.topic_order[0], b: correlation.topic_order[1] })
  }
</script>

<section class="compare">
  <header class="section-head">
    <div class="title-block">
      <div class="title-row">
        <p class="kicker">{copy.correlation.kicker}</p>
      </div>
    </div>
    {#if correlation}
      <div class="controls">
        <div class="picker">
          <label>Rows
            <select bind:value={rowBlock}>
              {#each blocks as b}<option value={b}>{blockDisplay(b)}</option>{/each}
            </select>
          </label>
          <span class="times">×</span>
          <label>Columns
            <select bind:value={colBlock}>
              {#each blocks as b}<option value={b}>{blockDisplay(b)}</option>{/each}
            </select>
          </label>
        </div>
      </div>
    {/if}
  </header>

  <div class="grid">
    <div class="left-col">
      {#if correlation}
        <CorrelationHeatmap {correlation} pairSelect showBlockPickers={false} bind:rowBlock bind:colBlock />
      {:else}
        <p>Correlations are not available for this bundle.</p>
      {/if}
    </div>
    <DifferencePane />
  </div>
</section>

<style>
  .compare {
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
  .title-row {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    flex-wrap: wrap;
  }
  .kicker {
    margin: 0;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    max-width: 62ch;
    line-height: 1.55;
  }

  .controls {
    display: flex;
    align-items: end;
    gap: 1.25rem;
  }

  /* Block-picker chevron/appearance styling, matched to the CorrelationHeatmap's
     own (now-hidden-when-embedded) picker so the controls look identical
     whether the pickers render standalone or lifted into this header. */
  .picker {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-faint);
  }
  .picker label {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }
  .picker select {
    font-family: var(--font-mono);
    font-size: var(--fs-small);
    font-weight: 600;
    text-transform: none;
    letter-spacing: normal;
    padding: 0.35rem 1.75rem 0.35rem 0.6rem;
    border: 1px solid var(--rule-strong);
    background-color: var(--surface);
    /* explicit caret so it unmistakably reads as a dropdown */
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M1 1l4 4 4-4' fill='none' stroke='%2371717a' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 0.6rem center;
    appearance: none;
    -webkit-appearance: none;
    -moz-appearance: none;
    color: var(--ink);
    border-radius: var(--radius-sm);
    box-shadow: 0 1px 0 rgba(0, 0, 0, 0.04);
    cursor: pointer;
  }
  .picker select:hover {
    border-color: var(--ink-muted);
  }
  .picker select:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
  .picker .times {
    color: var(--ink-muted);
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
</style>
