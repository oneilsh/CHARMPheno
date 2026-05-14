<script lang="ts">
  import { phenotypesById } from '../store'
  export let theta: number[]
  export let height = 24
  export let labels = true
  export let onSelect: ((id: number) => void) | null = null
  export let otherThreshold = 0.05

  $: ordered = theta.map((v, k) => ({ k, v }))
    .filter((x) => x.v > 0)
    .sort((a, b) => b.v - a.v)
  $: mainBands = ordered.filter((x) => x.v >= otherThreshold)
  $: otherFrac = ordered.filter((x) => x.v < otherThreshold).reduce((a, x) => a + x.v, 0)

  function hue(k: number): string { return `hsl(${(k * 47) % 360} 60% 55%)` }
</script>

<div class="bar" style="height: {height}px">
  {#each mainBands as b}
    <button class="band"
      style="width: {(b.v * 100).toFixed(2)}%; background: {hue(b.k)};"
      title={`${$phenotypesById.get(b.k)?.label || `Phenotype ${b.k}`}: ${(b.v * 100).toFixed(1)}%`}
      on:click={() => onSelect?.(b.k)}
    ></button>
  {/each}
  {#if otherFrac > 0}
    <span class="band other" style="width: {(otherFrac * 100).toFixed(2)}%">Other</span>
  {/if}
</div>

{#if labels}
  <ul class="legend">
    {#each mainBands as b}
      <li><span class="swatch" style="background: {hue(b.k)};"></span><span>{$phenotypesById.get(b.k)?.label || `Phenotype ${b.k}`}</span><span class="pct">{(b.v * 100).toFixed(0)}%</span></li>
    {/each}
    {#if otherFrac > 0}<li><span class="swatch" style="background: #999;"></span><span>Other</span><span class="pct">{(otherFrac * 100).toFixed(0)}%</span></li>{/if}
  </ul>
{/if}

<style>
  .bar { display: flex; width: 100%; border-radius: 4px; overflow: hidden; border: 1px solid #ccc; }
  .band { border: 0; padding: 0; cursor: pointer; height: 100%; color: #fff; font-size: 0.7rem; }
  .band.other { background: #aaa; cursor: default; display: flex; align-items: center; justify-content: center; }
  ul.legend { list-style: none; padding: 0; margin: 0.5rem 0 0; font-size: 0.85rem; }
  ul.legend li { display: grid; grid-template-columns: 1.25rem 1fr 3rem; gap: 0.5rem; align-items: center; padding: 0.15rem 0; }
  .swatch { width: 1rem; height: 1rem; border-radius: 2px; }
  .pct { text-align: right; font-variant-numeric: tabular-nums; color: #555; }
</style>
