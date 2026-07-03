<script lang="ts">
  import { bundle } from '../store'
  import { copy } from '../copy'
  import CorrelationHeatmap from './CorrelationHeatmap.svelte'
  import DifferencePane from './DifferencePane.svelte'
</script>

<div class="grid">
  <div class="left-col">
    {#if $bundle?.correlation}
      <p class="corr-kicker">{copy.correlation.kicker}</p>
      <CorrelationHeatmap correlation={$bundle.correlation} pairSelect />
    {:else}
      <p>Correlations are not available for this bundle.</p>
    {/if}
  </div>
  <DifferencePane />
</div>

<style>
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
  .corr-kicker {
    margin: 0;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    max-width: 52ch;
    line-height: 1.5;
  }
</style>
