<script lang="ts">
  import {
    bundle, selectedPhenotypeId, advancedView,
    phenotypeFilter, phenotypeSortBy, searchedConditionIdx,
  } from '../store'
  import { phenotypesContainingCode } from '../inference'
  import type { Phenotype } from '../types'

  // Filter+sort phenotype list. Sort key state lives in the global store so
  // it's preserved across tab switches. Simple-mode hides dead+mixed (matches
  // the bubble atlas filter).
  $: phenotypes = ($bundle?.phenotypes.phenotypes ?? []) as Phenotype[]

  // Optional restriction: when the user has pinned a condition via the
  // search bar, restrict the table to phenotypes containing that condition
  // in their relevance-ranked top-N. Same rule as the bubble highlight.
  $: containingSet = $bundle && $searchedConditionIdx !== null
    ? phenotypesContainingCode({
        beta: $bundle.model.beta,
        corpusFreq: $bundle.vocab.codes.map((c) => c.corpus_freq),
        codeIdx: $searchedConditionIdx,
      })
    : null

  $: filtered = phenotypes
    .filter((p) => {
      if (!$advancedView && (p.quality === 'dead' || p.quality === 'mixed')) return false
      if (containingSet && !containingSet.has(p.id)) return false
      const q = $phenotypeFilter.trim().toLowerCase()
      if (!q) return true
      const label = (p.label || `Phenotype ${p.id}`).toLowerCase()
      const desc = (p.description || '').toLowerCase()
      return label.includes(q) || desc.includes(q)
    })
    .sort((a, b) => {
      switch ($phenotypeSortBy) {
        case 'label': {
          const la = a.label || `Phenotype ${a.id}`
          const lb = b.label || `Phenotype ${b.id}`
          return la.localeCompare(lb)
        }
        case 'coherence':
          return b.npmi - a.npmi
        case 'prevalence':
        default:
          return b.corpus_prevalence - a.corpus_prevalence
      }
    })

  // Color-code the quality chip the same way CodePanel does
  const qualityClass: Record<string, string> = {
    phenotype: 'q-phenotype',
    background: 'q-background',
    anchor: 'q-anchor',
    mixed: 'q-mixed',
    dead: 'q-dead',
  }

  // For the prevalence sparkline domain we want the max across the FULL set
  // so the bar widths stay stable as the filter narrows.
  $: maxPrev = phenotypes.length
    ? Math.max(...phenotypes.map((p) => p.corpus_prevalence))
    : 1
</script>

<details class="browser" open>
  <summary>
    <span class="summary-text">
      Browse all phenotypes ({filtered.length}/{phenotypes.length})
    </span>
    <span class="caret" aria-hidden="true">▾</span>
  </summary>

  <div class="controls">
    <div class="filter-wrap">
      <input
        type="search"
        placeholder="Filter by label or description…"
        bind:value={$phenotypeFilter}
        class="filter-input"
      />
      {#if $phenotypeFilter}
        <button
          class="filter-clear"
          type="button"
          on:click={() => phenotypeFilter.set('')}
          title="Clear the filter"
        >×</button>
      {/if}
    </div>
    <div class="sort">
      <span class="eyebrow">Sort by</span>
      <select bind:value={$phenotypeSortBy}>
        <option value="prevalence">Prevalence</option>
        <option value="label">Label (A–Z)</option>
        {#if $advancedView}
          <option value="coherence">Coherence</option>
        {/if}
      </select>
    </div>
    {#if containingSet}
      <span class="filter-chip filter-chip-search" title="Filtered to phenotypes containing the searched condition">
        contains searched condition · {filtered.length} match{filtered.length === 1 ? '' : 'es'}
      </span>
    {:else if $phenotypeFilter}
      <span class="filter-chip">{filtered.length} match{filtered.length === 1 ? '' : 'es'}</span>
    {/if}
  </div>

  <div class="table-wrap">
    <table class="phenos">
      <thead>
        <tr>
          <th class="col-id">#</th>
          <th class="col-label">Label</th>
          {#if $advancedView}
            <th class="col-quality">Quality</th>
            <th class="col-coh" data-numeric>Coherence</th>
          {/if}
          <th class="col-prev" data-numeric>Prevalence</th>
        </tr>
      </thead>
      <tbody>
        {#each filtered as p (p.id)}
          <tr
            class:selected={$selectedPhenotypeId === p.id}
            on:click={() => selectedPhenotypeId.set(p.id)}
          >
            <td class="col-id" data-numeric>{p.id}</td>
            <td class="col-label">{p.label || `Phenotype ${p.id}`}</td>
            {#if $advancedView}
              <td class="col-quality">
                {#if p.quality}
                  <span class="qchip {qualityClass[p.quality] ?? ''}">{p.quality}</span>
                {:else}
                  <span class="qchip qchip-empty">—</span>
                {/if}
              </td>
              <td class="col-coh" data-numeric>{p.npmi.toFixed(3)}</td>
            {/if}
            <td class="col-prev" data-numeric>
              <span class="prev-row">
                <span class="prev-bar">
                  <span
                    class="prev-fill"
                    style="width: {(p.corpus_prevalence / maxPrev) * 100}%"
                  ></span>
                </span>
                <span class="prev-num">{(p.corpus_prevalence * 100).toFixed(1)}%</span>
              </span>
            </td>
          </tr>
        {/each}
        {#if filtered.length === 0}
          <tr><td colspan={$advancedView ? 5 : 3} class="empty">No phenotypes match.</td></tr>
        {/if}
      </tbody>
    </table>
  </div>
</details>

<style>
  .browser {
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    background: var(--surface);
  }
  .browser summary {
    cursor: pointer;
    padding: 0.7rem 1rem;
    list-style: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: var(--fs-small);
    color: var(--ink);
    font-weight: 500;
    user-select: none;
  }
  .browser summary::-webkit-details-marker { display: none; }
  .browser summary::marker { display: none; }
  .browser summary:hover { color: var(--accent); }
  .caret {
    font-family: var(--font-mono);
    color: var(--ink-faint);
    transition: transform 0.15s ease;
  }
  .browser[open] .caret { transform: rotate(180deg); }

  .controls {
    display: flex;
    gap: 1rem;
    align-items: center;
    padding: 0.5rem 1rem 0.75rem;
    border-top: 1px solid var(--rule);
    flex-wrap: wrap;
  }
  .filter-wrap {
    position: relative;
    flex: 1;
    min-width: 220px;
    display: flex;
    align-items: center;
  }
  .filter-input {
    width: 100%;
    padding: 0.4rem 1.7rem 0.4rem 0.6rem;
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius-sm);
    background: var(--surface);
    color: var(--ink);
    font-family: var(--font-body);
    font-size: var(--fs-small);
  }
  .filter-input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .filter-clear {
    position: absolute;
    right: 0.4rem;
    top: 50%;
    transform: translateY(-50%);
    border: 0;
    background: transparent;
    color: var(--ink-faint);
    cursor: pointer;
    font-size: 1rem;
    line-height: 1;
    padding: 0 0.2rem;
  }
  .filter-clear:hover { color: var(--ink); }
  .sort {
    display: flex;
    align-items: center;
    gap: 0.45rem;
  }
  .filter-chip {
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-muted);
    padding: 0.2rem 0.55rem;
    background: var(--surface-recessed);
    border-radius: var(--radius-sm);
  }
  .filter-chip-search {
    color: var(--accent-search-ink);
    background: var(--accent-search-soft);
  }

  .table-wrap {
    max-height: 420px;
    overflow-y: auto;
    border-top: 1px solid var(--rule-faint);
  }
  .phenos {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--fs-small);
  }
  .phenos thead {
    position: sticky;
    top: 0;
    background: var(--surface);
    z-index: 1;
  }
  .phenos th {
    text-align: left;
    padding: 0.5rem 0.85rem;
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-faint);
    border-bottom: 1px solid var(--rule);
    font-weight: 500;
  }
  .phenos td {
    padding: 0.45rem 0.85rem;
    border-bottom: 1px solid var(--rule-faint);
  }
  .phenos tbody tr {
    cursor: pointer;
    transition: background 0.12s ease;
  }
  .phenos tbody tr:hover { background: var(--surface-recessed); }
  .phenos tbody tr.selected { background: var(--surface-recessed); }
  .phenos tbody tr.selected td.col-label { font-weight: 600; }

  .col-id { width: 3.5rem; color: var(--ink-faint); }
  .col-label { color: var(--ink); }
  .col-quality { width: 7.5rem; }
  .col-coh { width: 5.5rem; text-align: right; }
  .col-prev { width: 11rem; text-align: right; }

  .prev-row {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    width: 100%;
  }
  .prev-bar {
    flex: 1;
    display: block;
    height: 4px;
    background: var(--surface-recessed);
    border-radius: 2px;
    overflow: hidden;
  }
  .prev-fill {
    display: block;
    height: 100%;
    background: var(--accent);
  }
  .prev-num {
    color: var(--ink);
    min-width: 3rem;
    text-align: right;
  }

  .qchip {
    display: inline-block;
    padding: 0.1rem 0.45rem;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: capitalize;
    letter-spacing: 0.04em;
    background: var(--surface-recessed);
    color: var(--ink-muted);
  }
  .qchip-empty { opacity: 0.4; }
  .qchip.q-phenotype { color: var(--ink); }
  .qchip.q-background { color: var(--ink-muted); }
  .qchip.q-anchor { color: var(--accent); }
  .qchip.q-mixed { color: #b45309; }
  .qchip.q-dead { color: var(--danger); }

  .empty {
    text-align: center;
    color: var(--ink-faint);
    padding: 2rem 0;
    font-style: italic;
  }
</style>
