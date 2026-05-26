<script lang="ts">
  import { bundle, tauThreshold } from './store'

  // Derive whether the histogram fields are present. When absent (HDP /
  // legacy bundles) we render nothing — no layout artefact, no DOM shell.
  $: hasHistogram = !!$bundle?.phenotypes.theta_histogram_bin_edges

  const TOOLTIP =
    'Patient-prevalence threshold. A phenotype is considered "present" in a patient when the topic\'s share of that patient\'s coded activity exceeds τ. Move the slider to see how prevalence numbers change at different cutoffs.'
</script>

{#if hasHistogram}
  <div class="tau-control">
    <span
      class="tau-kicker"
      title={TOOLTIP}
    >τ</span>
    <input
      type="range"
      min="0"
      max="0.5"
      step="0.02"
      bind:value={$tauThreshold}
      title={TOOLTIP}
    />
    <span class="tau-value" data-numeric>{$tauThreshold.toFixed(2)}</span>
  </div>
{/if}

<style>
  .tau-control {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    /* Compact container matching the seg-control height */
    padding: 0.2rem 0.65rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
  }

  .tau-kicker {
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-faint);
    font-weight: 500;
    cursor: help;
    flex-shrink: 0;
  }

  input[type="range"] {
    width: 140px;
    flex-shrink: 0;
  }

  .tau-value {
    font-family: var(--font-mono);
    font-size: var(--fs-small);
    font-weight: 500;
    color: var(--accent);
    min-width: 2.5ch;
    flex-shrink: 0;
  }
</style>
