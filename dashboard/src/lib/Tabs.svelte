<script lang="ts">
  import { route, go } from './router'
  const tabs: { id: 'atlas' | 'patient' | 'simulator'; label: string }[] = [
    { id: 'atlas',     label: 'Phenotype Atlas' },
    { id: 'patient',   label: 'Patient Explorer' },
    { id: 'simulator', label: 'Simulator' },
  ]
</script>

<nav class="tabs" aria-label="Sections">
  {#each tabs as t}
    <button
      class="tab"
      class:active={$route === t.id}
      on:click={() => go(t.id)}
      aria-current={$route === t.id ? 'page' : undefined}
    >
      <span class="label">{t.label}</span>
    </button>
  {/each}
</nav>

<style>
  .tabs {
    display: flex;
    gap: 0;
    padding: 0.85rem 0 0;
    margin-bottom: 1.75rem;
    border-bottom: 1px solid var(--rule);
  }
  .tab {
    position: relative;
    padding: 0.55rem 1rem 0.65rem;
    background: transparent;
    border: 0;
    cursor: pointer;
    color: var(--ink-muted);
    font-family: var(--font-body);
    font-size: var(--fs-small);
    font-weight: 500;
    letter-spacing: -0.005em;
    margin-bottom: -1px; /* overlap the parent border to make the active underline sit on it */
    border-bottom: 2px solid transparent;
    transition: color 0.15s ease, border-color 0.15s ease;
  }
  .tab:hover {
    color: var(--ink);
  }
  .tab.active {
    color: var(--ink);
    border-bottom-color: var(--accent);
  }
</style>
