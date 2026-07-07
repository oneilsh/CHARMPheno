<script lang="ts">
  import { fly } from 'svelte/transition'
  import { bundle } from '../store'
  import type { CovariateSchema } from '../types'
  import { populationLines } from './population'
  import { initialValues, canInteract } from '../atlas/covariate-panel'
  import { ALL_SUBCOHORTS } from './marginalSampler'

  export let store: import('svelte/store').Writable<import('../store').Conditioning>
  // Layout variant for the non-inline (page) rendering:
  //   'bar'     — horizontal page-top strip (Patient tab). The default.
  //   'stacked' — a self-contained card with a full-width source-cohort select
  //               and vertically-stacked covariate controls, for embedding in a
  //               narrow column (the Simulator's left recipe column). Only the
  //               stacked variant offers the "all subcohorts" group option.
  export let layout: 'bar' | 'stacked' = 'bar'
  // Whether to show the gating-group (cohort) selector. The Phenotype Atlas
  // sets this false: it encodes cohort as node color and shows all cohorts, so
  // a filter dropdown there would be redundant. Simulator/Patient keep it (they
  // condition generation on a chosen group).
  export let showGroup = true
  // Inline mode (the Phenotype Atlas): render as a left pop-out drawer over the
  // map rather than a page-top strip. The drawer's OPEN state IS covariate mode:
  //   closed  -> neutral, bubbles show the empirical cohort average
  //   open    -> "configuring a patient", bubbles show the model-predicted
  //              prevalence for a single individual at the chosen features
  // so the two states are visually unambiguous (no controls are shown at all in
  // the average state). Simulator/Patient keep the default inline=false toggle
  // bar, where covariate mode gates whether generation conditions on covariates.
  export let inlineControls = false

  $: schema = $bundle?.covariateSchema
  $: gating = $bundle?.gating
  $: hasCovariates = !!schema && canInteract(schema)
  $: hasGroup = showGroup && !!gating
  $: visible = hasCovariates || hasGroup

  // Seed control values from the schema whenever it changes, pushing them to the
  // store WITHOUT activating covariate mode (neutral = corpus average).
  let local: Record<string, number | string> = {}
  $: if (schema) seed(schema)
  function seed(s: CovariateSchema) {
    local = initialValues(s)
    store.update((c) => ({ ...c, values: local }))
  }

  function onControl(name: string, value: number | string) {
    local = { ...local, [name]: value }
    store.update((c) => ({ ...c, values: local, covariateActive: true }))
  }

  // Per-control distribution summaries ("30-81 (med 60)", "F 63% / M 37%"),
  // shown as small text under each widget so the user can judge whether a chosen
  // value is typical or extreme.
  $: lines = populationLines(schema)
  $: summaryFor = (name: string) => lines.find((l) => l.name === name)?.summary ?? ''

  // Inline drawer: opening starts a patient at the reference values and enters
  // model-prediction mode; "Use Cohort Averages" collapses back to the average.
  function openDrawer() {
    if (schema) local = initialValues(schema)
    store.update((c) => ({ ...c, values: local, covariateActive: true }))
  }
  function reset() {
    if (schema) local = initialValues(schema)
    store.update((c) => ({ ...c, values: local, covariateActive: false }))
  }
</script>

{#if inlineControls}
  <!-- Phenotype Atlas: left pop-out drawer. Absolutely positioned by the host
       (TopicMap .map-canvas) so open/close never reflows the map. -->
  {#if hasCovariates && schema}
    <div class="cov-drawer">
      {#if !$store.covariateActive}
        <button type="button" class="drawer-open" on:click={openDrawer}>
          <span class="gear" aria-hidden="true">⚙</span> Configure Patient Features
        </button>
      {:else}
        <div class="drawer-panel" transition:fly={{ x: -10, duration: 140 }}>
          <header class="drawer-head">
            <span class="drawer-title">Patient Features</span>
            <button type="button" class="drawer-close" on:click={reset}>Use Cohort Averages</button>
          </header>
          <p class="drawer-hint">
            Bubbles show the model-predicted prevalence for a single patient with these features — not the cohort average.
          </p>
          <div class="drawer-controls">
            {#each schema.controls as control (control.name)}
              <div class="control-block">
                {#if control.type === 'continuous'}
                  {@const min = control.range?.[0] ?? 0}
                  {@const max = control.range?.[1] ?? 100}
                  <div class="control-top">
                    <span class="control-label">{control.name}</span>
                    <span class="control-value" data-numeric>{local[control.name]}</span>
                  </div>
                  <input
                    type="range"
                    {min}
                    {max}
                    step="1"
                    value={local[control.name]}
                    on:input={(e) => onControl(control.name, +e.currentTarget.value)}
                  />
                {:else}
                  <span class="control-label">{control.name}</span>
                  {#if control.levels && control.levels.length === 2}
                    <div class="cat-toggle">
                      {#each control.levels as level}
                        <button
                          type="button"
                          class="cat-btn"
                          class:active={local[control.name] === level}
                          on:click={() => onControl(control.name, level)}
                        >{level}</button>
                      {/each}
                    </div>
                  {:else if control.levels}
                    <select
                      value={local[control.name]}
                      on:change={(e) => onControl(control.name, e.currentTarget.value)}
                      class="cat-select"
                    >
                      {#each control.levels as level}
                        <option value={level}>{level}</option>
                      {/each}
                    </select>
                  {/if}
                {/if}
                <span class="control-dist">{summaryFor(control.name)}</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}
{:else if layout === 'stacked' && visible}
  <!-- Stacked card (Simulator left column): source-cohort select on top, then a
       covariate disclosure whose controls stack vertically like the Phenotype
       Atlas's Patient-Features drawer. -->
  <section class="cohort-panel">
    <header class="cp-head">
      <span class="eyebrow">Source cohort</span>
      <p class="cp-sub">Who to sample from — and, optionally, the covariate profile to condition on.</p>
    </header>

    {#if hasGroup && gating}
      <label class="cp-field">
        <span class="control-label">{gating.group_var_label ?? gating.group_var}</span>
        <select
          class="cat-select cp-select"
          value={$store.group ?? ''}
          on:change={(e) => store.update((c) => ({ ...c, group: e.currentTarget.value === '' ? null : e.currentTarget.value }))}
        >
          <option value={ALL_SUBCOHORTS}>All subcohorts</option>
          {#each gating.groups as g}
            <option value={g}>{gating.group_labels?.[g] ?? g}</option>
          {/each}
          <option value="">Background only</option>
        </select>
      </label>
    {/if}

    {#if hasCovariates && schema}
      <div class="cp-cov">
        <label class="toggle-label" title="When on, generation conditions each patient on the covariate values below instead of the corpus-average profile.">
          <input
            type="checkbox"
            class="toggle-input"
            checked={$store.covariateActive}
            on:change={(e) => store.update((c) => ({ ...c, covariateActive: e.currentTarget.checked }))}
          />
          <span class="toggle-track"><span class="toggle-thumb"></span></span>
          <span class="toggle-text">{$store.covariateActive ? 'custom covariates' : 'average covariates'}</span>
        </label>

        {#if $store.covariateActive}
          <div class="cp-controls">
            {#each schema.controls as control (control.name)}
              <div class="control-block">
                {#if control.type === 'continuous'}
                  {@const min = control.range?.[0] ?? 0}
                  {@const max = control.range?.[1] ?? 100}
                  <div class="control-top">
                    <span class="control-label">{control.name}</span>
                    <span class="control-value" data-numeric>{local[control.name]}</span>
                  </div>
                  <input
                    type="range"
                    {min}
                    {max}
                    step="1"
                    value={local[control.name]}
                    on:input={(e) => onControl(control.name, +e.currentTarget.value)}
                  />
                {:else}
                  <span class="control-label">{control.name}</span>
                  {#if control.levels && control.levels.length === 2}
                    <div class="cat-toggle">
                      {#each control.levels as level}
                        <button
                          type="button"
                          class="cat-btn"
                          class:active={local[control.name] === level}
                          on:click={() => onControl(control.name, level)}
                        >{level}</button>
                      {/each}
                    </div>
                  {:else if control.levels}
                    <select
                      value={local[control.name]}
                      on:change={(e) => onControl(control.name, e.currentTarget.value)}
                      class="cat-select"
                    >
                      {#each control.levels as level}
                        <option value={level}>{level}</option>
                      {/each}
                    </select>
                  {/if}
                {/if}
                <span class="control-dist">{summaryFor(control.name)}</span>
              </div>
            {/each}
          </div>
          <button type="button" class="reset-btn cp-reset" on:click={reset}>Reset to averages</button>
        {/if}
      </div>
    {/if}
  </section>
{:else if visible}
  <div class="conditioning-bar">
    {#if hasGroup && gating}
      <div class="bar-section group-section">
        <span class="bar-label">{gating.group_var_label ?? gating.group_var}</span>
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

    {#if hasCovariates && schema}
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
                      value={local[control.name]}
                      on:input={(e) => onControl(control.name, +e.currentTarget.value)}
                    />
                  </label>
                {:else if control.levels && control.levels.length === 2}
                  <div class="cat-toggle">
                    {#each control.levels as level}
                      <button
                        type="button"
                        class="cat-btn"
                        class:active={local[control.name] === level}
                        on:click={() => onControl(control.name, level)}
                      >{level}</button>
                    {/each}
                  </div>
                {:else if control.levels}
                  <select
                    value={local[control.name]}
                    on:change={(e) => onControl(control.name, e.currentTarget.value)}
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

          <button type="button" class="reset-btn" on:click={reset}>Reset</button>
        {/if}
      </div>
    {/if}
  </div>
{/if}

<style>
  /* ---- Inline drawer (Phenotype Atlas) ---------------------------------- */
  .cov-drawer {
    position: absolute;
    top: 0.5rem;
    left: 0.5rem;
    z-index: 6;
  }
  .drawer-open {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    border: 1px solid var(--rule-strong);
    background: color-mix(in srgb, var(--surface) 88%, transparent);
    color: var(--ink-muted);
    padding: 0.28rem 0.6rem;
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    cursor: pointer;
    transition: color 0.12s ease, border-color 0.12s ease;
  }
  .drawer-open:hover {
    color: var(--ink);
    border-color: var(--ink-muted);
  }
  .drawer-open .gear {
    font-size: var(--fs-small);
    line-height: 1;
  }
  .drawer-panel {
    width: 232px;
    max-height: 470px;
    overflow-y: auto;
    background: var(--surface);
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius-sm);
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.12);
    padding: 0.6rem 0.7rem 0.75rem;
  }
  .drawer-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
  }
  .drawer-title {
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--ink);
    font-weight: 600;
  }
  .drawer-close {
    border: 1px solid var(--rule-strong);
    background: var(--surface);
    color: var(--ink-muted);
    padding: 0.18rem 0.45rem;
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    cursor: pointer;
    transition: color 0.12s ease, border-color 0.12s ease;
    flex-shrink: 0;
  }
  .drawer-close:hover {
    color: var(--accent);
    border-color: var(--accent);
  }
  .drawer-hint {
    margin: 0 0 0.6rem;
    font-size: var(--fs-micro);
    line-height: 1.45;
    color: var(--ink-faint);
  }
  .drawer-controls {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
  }
  .control-block {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  .control-top {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }
  .control-value {
    font-family: var(--font-mono);
    font-size: var(--fs-small);
    color: var(--accent);
    font-weight: 500;
  }
  .control-block input[type='range'] {
    width: 100%;
  }
  .control-dist {
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    color: var(--ink-faint);
  }

  /* ---- Stacked card (Simulator left column) ----------------------------- */
  .cohort-panel {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    padding: 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
  }
  .cp-head {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--rule);
  }
  .cp-sub {
    margin: 0;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    font-style: italic;
    line-height: 1.5;
  }
  .cp-field {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .cp-select {
    width: 100%;
    font-size: var(--fs-small);
    padding: 0.35rem 0.4rem;
  }
  .cp-cov {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
    padding-top: 0.2rem;
    border-top: 1px solid var(--rule-faint);
  }
  .cp-controls {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
  }
  .cp-reset { align-self: flex-start; }

  /* ---- Page-top bar (Simulator / Patient) ------------------------------- */
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

  /* Continuous slider (page-top variant) */
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

  /* Categorical toggle 2-level */
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

  /* Population readout (page-top variant) */
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

  /* Footer reset */
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
