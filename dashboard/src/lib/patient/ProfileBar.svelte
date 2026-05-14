<script lang="ts">
  import { phenotypesById } from '../store'
  export let theta: number[]
  export let height = 28
  export let labels = true
  export let onSelect: ((id: number) => void) | null = null
  export let otherThreshold = 0.05

  // Curated 8-color palette, indexed by topic-id mod 8 so the same
  // phenotype is always the same color across patients. Modern data-viz
  // hues, well-spaced for distinguishability.
  const PALETTE = [
    '#06b6d4', // cyan
    '#8b5cf6', // violet
    '#10b981', // emerald
    '#f59e0b', // amber
    '#ec4899', // pink
    '#3b82f6', // blue
    '#ef4444', // red
    '#64748b', // slate
  ]
  function hue(k: number): string { return PALETTE[k % PALETTE.length] }

  $: ordered = theta.map((v, k) => ({ k, v }))
    .filter((x) => x.v > 0)
    .sort((a, b) => b.v - a.v)
  $: mainBands = ordered.filter((x) => x.v >= otherThreshold)
  $: otherFrac = ordered.filter((x) => x.v < otherThreshold).reduce((a, x) => a + x.v, 0)
</script>

<div class="profile" style="--bar-h: {height}px">
  {#if labels}
    <div class="band-labels">
      {#each mainBands as b}
        <span class="band-label" style="width: {(b.v * 100).toFixed(2)}%; color: {hue(b.k)}">
          <span class="dot" style="background: {hue(b.k)}"></span>
          <span class="lab">{$phenotypesById.get(b.k)?.label || `Phenotype ${b.k}`}</span>
        </span>
      {/each}
      {#if otherFrac > 0}
        <span class="band-label other" style="width: {(otherFrac * 100).toFixed(2)}%">
          <span class="dot"></span>
          <span class="lab">Other</span>
        </span>
      {/if}
    </div>
  {/if}

  <div class="bar">
    {#each mainBands as b}
      <button class="band"
        style="width: {(b.v * 100).toFixed(2)}%; background: {hue(b.k)};"
        title={`${$phenotypesById.get(b.k)?.label || `Phenotype ${b.k}`}: ${(b.v * 100).toFixed(1)}%`}
        on:click={() => onSelect?.(b.k)}
        aria-label={`${$phenotypesById.get(b.k)?.label || `Phenotype ${b.k}`}, ${(b.v * 100).toFixed(1)} percent`}
      ></button>
    {/each}
    {#if otherFrac > 0}
      <span class="band other-band" style="width: {(otherFrac * 100).toFixed(2)}%"></span>
    {/if}
  </div>

  {#if labels}
    <div class="band-percents" data-numeric>
      {#each mainBands as b}
        <span class="pct" style="width: {(b.v * 100).toFixed(2)}%">{(b.v * 100).toFixed(0)}%</span>
      {/each}
      {#if otherFrac > 0}
        <span class="pct other" style="width: {(otherFrac * 100).toFixed(2)}%">{(otherFrac * 100).toFixed(0)}%</span>
      {/if}
    </div>
  {/if}
</div>

<style>
  .profile {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .band-labels {
    display: flex;
    width: 100%;
    align-items: baseline;
  }
  .band-label {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    padding-right: 0.4rem;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    overflow: hidden;
    white-space: nowrap;
    line-height: 1.2;
  }
  .band-label .lab {
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .band-label.other { color: var(--ink-faint); font-style: italic; }
  .band-label .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--ink-faint);
    flex-shrink: 0;
  }

  .bar {
    display: flex;
    width: 100%;
    height: var(--bar-h);
    border-radius: var(--radius-sm);
    overflow: hidden;
    background: var(--surface-recessed);
    gap: 1px;
  }
  .band {
    border: 0;
    padding: 0;
    cursor: pointer;
    height: 100%;
    transition: filter 0.15s ease;
  }
  .band:hover {
    filter: brightness(1.08);
  }
  .band:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: -2px;
  }
  .other-band {
    background: var(--surface-deep);
    background-image: repeating-linear-gradient(
      45deg,
      transparent,
      transparent 3px,
      rgba(82, 82, 91, 0.14) 3px,
      rgba(82, 82, 91, 0.14) 4px
    );
  }

  .band-percents {
    display: flex;
    width: 100%;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
  }
  .pct {
    overflow: hidden;
    white-space: nowrap;
    padding-right: 0.4rem;
    letter-spacing: 0.02em;
  }
</style>
