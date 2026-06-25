<script lang="ts">
  import { onMount } from 'svelte'
  import { conditioning } from '../store'
  import type { CovariateSchema, GatingSpec } from '../types'
  import { initialValues, canInteract } from './covariate-panel'

  export let schema: CovariateSchema
  export let gating: GatingSpec | null | undefined = undefined

  $: interactive = canInteract(schema)

  // When the schema changes (or on mount) seed covariateValues with defaults
  // and force covariateMode off if the schema is unsupported.
  $: {
    if (!interactive) {
      conditioning.update((c) => ({ ...c, covariateActive: false }))
    }
  }

  onMount(() => {
    conditioning.update((c) => ({ ...c, values: initialValues(schema) }))
    if (!interactive) conditioning.update((c) => ({ ...c, covariateActive: false }))
  })

  // Keep a local mirror of the covariate values so we can bind individual controls.
  // When the local mirror changes we write the whole record back to the store.
  let local: Record<string, number | string> = initialValues(schema)

  $: conditioning.update((c) => ({ ...c, values: local }))

  function reset() {
    conditioning.update((c) => ({ ...c, covariateActive: false }))
    local = initialValues(schema)
  }
</script>

<aside class="covariate-panel">
  <header class="panel-head">
    <span class="eyebrow">Covariate controls</span>
    {#if interactive}
      <label class="toggle-label" title="When on, bubble sizes show model-predicted prevalence at the covariate values below rather than the corpus-average histogram estimate.">
        <input
          type="checkbox"
          class="toggle-input"
          checked={$conditioning.covariateActive}
          on:change={(e) => conditioning.update((c) => ({ ...c, covariateActive: e.currentTarget.checked }))}
          disabled={!interactive}
        />
        <span class="toggle-track">
          <span class="toggle-thumb"></span>
        </span>
        <span class="toggle-text">{$conditioning.covariateActive ? 'covariate prevalence' : 'corpus average'}</span>
      </label>
    {:else}
      <span class="unavailable-note">Covariate controls unavailable for this formula</span>
    {/if}
  </header>

  <div class="controls" class:disabled={!interactive}>
    {#each schema.controls as control (control.name)}
      <div class="control-row">
        <span class="control-label">{control.name}</span>
        {#if control.type === 'continuous'}
          {@const min = control.range?.[0] ?? 0}
          {@const max = control.range?.[1] ?? 100}
          <label class="slider">
            <span class="slider-head">
              <span class="slider-ends">
                <span>{min}</span>
                <span class="slider-val" data-numeric>{local[control.name]}</span>
                <span>{max}</span>
              </span>
            </span>
            <input
              type="range"
              {min}
              {max}
              step="1"
              bind:value={local[control.name]}
              disabled={!interactive}
            />
          </label>
        {:else if control.levels && control.levels.length === 2}
          <!-- 2-level categorical: toggle between the two levels -->
          <div class="cat-toggle">
            {#each control.levels as level}
              <button
                type="button"
                class="cat-btn"
                class:active={local[control.name] === level}
                disabled={!interactive}
                on:click={() => { local[control.name] = level }}
              >{level}</button>
            {/each}
          </div>
        {:else if control.levels}
          <!-- n-level categorical: select dropdown -->
          <select
            bind:value={local[control.name]}
            disabled={!interactive}
            class="cat-select"
          >
            {#each control.levels as level}
              <option value={level}>{level}</option>
            {/each}
          </select>
        {/if}
      </div>
    {/each}
  </div>

  {#if gating}
    <div class="control-row">
      <span class="control-label">{gating.group_var}</span>
      <select
        value={$conditioning.group ?? ''}
        on:change={(e) => conditioning.update((c) => ({ ...c, group: e.currentTarget.value === '' ? null : e.currentTarget.value }))}
        class="cat-select"
      >
        <option value="">Background only</option>
        {#each gating.groups as g}
          <option value={g}>{g}</option>
        {/each}
      </select>
    </div>
  {/if}

  {#if interactive}
    <div class="panel-foot">
      <button type="button" class="reset-btn" on:click={reset}>
        Reset to corpus average
      </button>
    </div>
  {/if}
</aside>

<style>
  .covariate-panel {
    padding: 0.9rem 1rem 0.75rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }

  .panel-head {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid var(--rule);
  }

  .eyebrow {
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-faint);
    font-weight: 500;
    flex-shrink: 0;
  }

  .unavailable-note {
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    font-style: italic;
  }

  /* Toggle switch */
  .toggle-label {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    cursor: pointer;
    user-select: none;
  }
  .toggle-input {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
  }
  .toggle-track {
    position: relative;
    display: inline-block;
    width: 28px;
    height: 15px;
    background: var(--rule-strong);
    border-radius: 8px;
    transition: background 0.15s ease;
    flex-shrink: 0;
  }
  .toggle-input:checked ~ .toggle-track {
    background: var(--accent);
  }
  .toggle-thumb {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 11px;
    height: 11px;
    background: #fff;
    border-radius: 50%;
    transition: transform 0.15s ease;
  }
  .toggle-input:checked ~ .toggle-track .toggle-thumb {
    transform: translateX(13px);
  }
  .toggle-text {
    font-size: var(--fs-micro);
    color: var(--ink-muted);
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  /* Controls list */
  .controls {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
  }
  .controls.disabled {
    opacity: 0.45;
    pointer-events: none;
  }

  .control-row {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }

  .control-label {
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--ink-faint);
    font-weight: 500;
  }

  /* Continuous slider */
  .slider {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }
  .slider-head {
    display: flex;
    flex-direction: column;
  }
  .slider-ends {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    color: var(--ink-faint);
  }
  .slider-val {
    color: var(--accent);
    font-weight: 500;
  }

  /* Categorical toggle (2-level) */
  .cat-toggle {
    display: flex;
    gap: 0.35rem;
  }
  .cat-btn {
    flex: 1;
    padding: 0.2rem 0.5rem;
    border: 1px solid var(--rule-strong);
    background: var(--surface);
    color: var(--ink-muted);
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    cursor: pointer;
    transition: color 0.12s ease, border-color 0.12s ease, background 0.12s ease;
  }
  .cat-btn.active {
    background: var(--accent-faint);
    border-color: var(--accent);
    color: var(--accent);
  }
  .cat-btn:disabled {
    cursor: default;
  }

  /* n-level select */
  .cat-select {
    font-size: var(--fs-small);
    padding: 0.2rem 0.4rem;
    border: 1px solid var(--rule-strong);
    background: var(--surface);
    color: var(--ink);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }
  .cat-select:disabled {
    cursor: default;
    opacity: 0.6;
  }

  /* Footer reset */
  .panel-foot {
    padding-top: 0.5rem;
    border-top: 1px solid var(--rule-faint);
  }
  .reset-btn {
    border: 1px solid var(--rule-strong);
    background: var(--surface);
    color: var(--ink-muted);
    padding: 0.25rem 0.6rem;
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    cursor: pointer;
    transition: color 0.12s ease, border-color 0.12s ease;
  }
  .reset-btn:hover {
    color: var(--ink);
    border-color: var(--ink-muted);
  }
</style>
