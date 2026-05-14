<script lang="ts">
  import { route, go } from './router'
  const tabs: { id: 'atlas' | 'patient' | 'simulator'; label: string }[] = [
    { id: 'atlas', label: 'Phenotype Atlas' },
    { id: 'patient', label: 'Patient Explorer' },
    { id: 'simulator', label: 'Simulator' },
  ]
</script>

<nav class="tabs">
  {#each tabs as t}
    <button class:active={$route === t.id} on:click={() => go(t.id)}>{t.label}</button>
  {/each}
</nav>

<style>
  .tabs { display: flex; gap: 0.5rem; padding: 0.5rem 1rem; border-bottom: 1px solid #ddd; }
  button { padding: 0.5rem 1rem; border: 1px solid transparent; background: transparent; cursor: pointer; }
  button.active { border-color: #999; background: #f4f4f4; font-weight: 600; }
</style>
