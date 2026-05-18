<script lang="ts">
  // Cohort selector: a button showing the currently loaded cohort's label,
  // clicking opens a popover listing every cohort in the manifest with its
  // full description. The popover doubles as the "what is this cohort?"
  // help affordance — putting the descriptions next to the choices keeps
  // the masthead uncluttered while still surfacing the metadata that
  // distinguishes cohorts (inclusion criteria, document window, etc).
  //
  // Changing the selection writes to the selectedCohort store; App.svelte
  // reacts to that and re-loads the bundle. We deliberately don't reset
  // bundle/cohort here — keeping side effects in the page-level reactive
  // statement avoids racing two re-fetches if the user clicks rapidly.
  import { manifest, selectedCohort } from './store'
  import { onMount } from 'svelte'

  let open = false
  let buttonEl: HTMLButtonElement
  let menuEl: HTMLDivElement

  $: current = $manifest?.cohorts.find((c) => c.id === $selectedCohort) ?? null

  function toggle() { open = !open }
  function close() { open = false }

  function pick(id: string) {
    if (id !== $selectedCohort) selectedCohort.set(id)
    close()
  }

  // Close on outside click / Escape. Bound only while open so we don't
  // pay event-listener cost when the menu is collapsed.
  function onWindowClick(e: MouseEvent) {
    const t = e.target as Node
    if (buttonEl?.contains(t) || menuEl?.contains(t)) return
    close()
  }
  function onKey(e: KeyboardEvent) {
    if (e.key === 'Escape') close()
  }

  onMount(() => {
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  $: if (open) {
    window.addEventListener('click', onWindowClick, true)
  } else {
    window.removeEventListener('click', onWindowClick, true)
  }
</script>

{#if $manifest}
  <div class="wrap">
    <button
      bind:this={buttonEl}
      class="selector"
      class:open
      type="button"
      aria-haspopup="listbox"
      aria-expanded={open}
      on:click={toggle}
    >
      <span class="kicker">cohort</span>
      <span class="label">{current?.label ?? 'Select cohort'}</span>
      <svg class="caret" viewBox="0 0 12 12" aria-hidden="true">
        <path d="M2 4.5 L6 8.5 L10 4.5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
    </button>

    {#if open}
      <div bind:this={menuEl} class="menu" role="listbox">
        {#each $manifest.cohorts as c (c.id)}
          <button
            class="item"
            class:active={c.id === $selectedCohort}
            role="option"
            aria-selected={c.id === $selectedCohort}
            on:click={() => pick(c.id)}
            type="button"
          >
            <span class="item-label">{c.label}</span>
            <span class="item-desc">{c.description}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>
{/if}

<style>
  .wrap {
    position: relative;
    display: inline-block;
  }

  .selector {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    background: var(--surface);
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius-sm);
    padding: 0.35rem 0.65rem 0.35rem 0.75rem;
    cursor: pointer;
    font-family: var(--font-body);
    color: var(--ink);
    transition: border-color 0.12s ease, background 0.12s ease;
    min-width: 14rem;
  }
  .selector:hover { border-color: var(--ink); }
  .selector.open {
    border-color: var(--accent);
  }
  .kicker {
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--ink-faint);
    padding-right: 0.55rem;
    border-right: 1px solid var(--rule);
  }
  .label {
    font-size: var(--fs-small);
    font-weight: 600;
    letter-spacing: -0.005em;
    flex: 1;
    text-align: left;
  }
  .caret {
    width: 11px;
    height: 11px;
    color: var(--ink-muted);
    transition: transform 0.15s ease;
    flex-shrink: 0;
  }
  .selector.open .caret { transform: rotate(180deg); }

  .menu {
    position: absolute;
    top: calc(100% + 6px);
    left: 0;
    min-width: 26rem;
    max-width: 32rem;
    background: var(--surface);
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius-sm);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.18);
    z-index: 50;
    padding: 0.3rem;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .item {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.25rem;
    text-align: left;
    background: transparent;
    border: 0;
    padding: 0.55rem 0.7rem;
    border-radius: 3px;
    cursor: pointer;
    font-family: var(--font-body);
    color: var(--ink);
    transition: background 0.1s ease;
  }
  .item:hover { background: var(--surface-alt, rgba(0,0,0,0.04)); }
  .item.active {
    background: var(--surface-alt, rgba(6, 182, 212, 0.08));
    box-shadow: inset 3px 0 0 var(--accent);
  }
  .item-label {
    font-size: var(--fs-small);
    font-weight: 600;
    letter-spacing: -0.005em;
  }
  .item-desc {
    font-size: var(--fs-micro);
    color: var(--ink-muted);
    line-height: 1.45;
  }
</style>
