<script lang="ts">
  import { route, go } from './router'
  const tabs: { id: 'atlas' | 'patient' | 'simulator'; label: string; num: string }[] = [
    { id: 'atlas',     label: 'Phenotype Atlas',  num: 'I' },
    { id: 'patient',   label: 'Patient Explorer', num: 'II' },
    { id: 'simulator', label: 'Simulator',        num: 'III' },
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
      <span class="num">{t.num}</span>
      <span class="label">{t.label}</span>
    </button>
  {/each}
</nav>

<style>
  .tabs {
    display: flex;
    gap: 2.25rem;
    padding: 1rem 0 0;
    margin-bottom: 1.5rem;
  }
  .tab {
    position: relative;
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    padding: 0.4rem 0;
    background: transparent;
    border: 0;
    border-bottom: 1.5px solid transparent;
    cursor: pointer;
    color: var(--ink-muted);
    font-family: var(--font-body);
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: -0.005em;
    transition: color 0.15s ease, border-color 0.15s ease;
  }
  .num {
    font-family: var(--font-display);
    font-style: italic;
    font-size: 0.78rem;
    color: var(--ink-faint);
    transition: color 0.15s ease;
  }
  .tab:hover {
    color: var(--ink);
  }
  .tab:hover .num {
    color: var(--terracotta);
  }
  .tab.active {
    color: var(--ink);
    border-bottom-color: var(--terracotta);
  }
  .tab.active .num {
    color: var(--terracotta);
  }
</style>
