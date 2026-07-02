<script lang="ts">
  import { bundle } from '../store'
  import { populationLines } from './population'
  import { initialValues, canInteract } from '../atlas/covariate-panel'

  export let store: import('svelte/store').Writable<import('../store').Conditioning>

  $: schema = $bundle?.covariateSchema
  $: gating = $bundle?.gating
  $: hasCovariates = !!schema && canInteract(schema)
  $: hasGroup = !!gating
  $: visible = hasCovariates || hasGroup

  // Seed covariate values whenever the schema changes.
  let local: Record<string, number | string> = {}
  $: if (schema) { local = initialValues(schema) }
  $: store.update((c) => ({ ...c, values: local }))

  $: lines = populationLines(schema)

  function reset() {
    if (schema) local = initialValues(schema)
    store.update((c) => ({ ...c, covariateActive: false }))
  }
</script>

{#if visible}
  <div class="conditioning-bar">
    {#if hasGroup}
      <div class="bar-section group-section">
        <span class="bar-label">{gating.group_var}</span>
        <select
          class="cat-select"
          value={$store.group ?? ''}
          on:change={(e) => store.update((c) => ({ ...c, group: e.currentTarget.value === '' ? null : e.currentTarget.value }))}
        >
          <option value="">Background only</option>
          {#each gating.groups as g}<option value={g}>{g}</option>{/each}
        </select>
      </div>
    {/if}

    {#if hasCovariates}
      <div class="bar-section covariate-section">
        <label class="toggle-label" title="When on, bubble sizes show model-predicted prevalence at the covariate values below rather than the corpus-average histogram estimate.">
          <input
            type="checkbox"
            class="toggle-input"
            checked={$store.covariateActive}
            on:change={(e) => store.update((c) => ({ ...c, covariateActive: e.currentTarget.checked }))}
          />
          <span class="toggle-track">
            <span class="toggle-thumb"></span>
          </span>
          <span class="toggle-text">{$store.covariateActive ? 'covariate prevalence' : 'corpus average'}</span>
        </label>

        {#if $store.covariateActive}
          <div class="controls">
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
                        on:click={() => { local[control.name] = level }}
                      >{level}</button>
                    {/each}
                  </div>
                {:else if control.levels}
                  <!-- n-level categorical: select dropdown -->
                  <select
                    bind:value={local[control.name]}
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

          <div class="population-readout">
            {#each lines as l}
              <span class="pop-line"><span class="pop-name">{l.name}:</span> {l.summary}</span>
            {/each}
          </div>

          <button type="button" class="reset-btn" on:click={reset}>
            Reset
          </button>
        {/if}
      </div>
    {/if}
  </div>
{/if}

<style>
  .conditioning-bar {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    flex-wrap: wrap;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--rule);
    margin-bottom: 0;
  }

  .bar-section {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    flex-wrap: wrap;
  }

  .group-section {
    border-right: 1px solid var(--rule);
    padding-right: 1.5rem;
  }

  .bar-label {
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--ink-faint);
    font-weight: 500;
    flex-shrink: 0;
  }

  /* Toggle switch (lifted from the former covariate panel) */
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

  /* Controls list (lifted from the former covariate panel) */
  .controls {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
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

  /* Continuous slider (lifted from the former covariate panel) */
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
    min-width: 100px;
  }
  .slider-val {
    color: var(--accent);
    font-weight: 500;
  }

  /* Categorical toggle 2-level (lifted from the former covariate panel) */
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

  /* n-level select (lifted from the former covariate panel) */
  .cat-select {
    font-size: var(--fs-small);
    padding: 0.2rem 0.4rem;
    border: 1px solid var(--rule-strong);
    background: var(--surface);
    color: var(--ink);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  /* Population readout */
  .population-readout {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
  }
  .pop-line {
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    color: var(--ink-faint);
  }
  .pop-name {
    color: var(--ink-muted);
    font-weight: 500;
  }

  /* Footer reset (lifted from the former covariate panel) */
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
    flex-shrink: 0;
  }
  .reset-btn:hover {
    color: var(--ink);
    border-color: var(--ink-muted);
  }
</style>
