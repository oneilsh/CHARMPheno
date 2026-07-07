<script lang="ts">
  import { subRoute, go, SUBTABS, type TopId } from './router'
  export let top: TopId
  $: subs = SUBTABS[top]
</script>

<div class="subtabs" role="tablist" aria-label="View">
  {#each subs as s}
    <button
      type="button"
      role="tab"
      data-tour="subtab-{top}-{s.id}"
      aria-selected={$subRoute === s.id}
      class:active={$subRoute === s.id}
      on:click={() => go(top, s.id)}
    >{s.label}</button>
  {/each}
</div>

<style>
  .subtabs {
    display: inline-flex;
    gap: 0.25rem;
    padding: 0.25rem;
    background: var(--surface-sunk, rgba(0, 0, 0, 0.03));
    border: 1px solid var(--rule);
    border-radius: 8px;
  }
  .subtabs button {
    padding: 0.35rem 0.9rem;
    border: 0;
    border-radius: 6px;
    background: transparent;
    color: var(--ink-muted);
    font-family: var(--font-body);
    font-size: var(--fs-small);
    cursor: pointer;
    transition: background 0.15s ease, color 0.15s ease;
  }
  .subtabs button:hover { color: var(--ink); }
  .subtabs button.active { background: var(--surface); color: var(--ink); box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08); }
</style>
