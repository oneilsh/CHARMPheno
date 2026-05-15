<script lang="ts">
  import {
    bundle, cohort, selectedPatientId, selectedPhenotypeId, advancedView,
    searchedConditionIdx, searchedPhenotypeForPatients,
  } from '../store'
  import { phenotypeHue } from '../palette'
  import { displayedDominant } from '../dominant'
  import type { SyntheticPatient } from '../types'

  let filter = ''
  let sortBy: 'id' | 'dominant' | 'concentration' = 'dominant'

  function concentration(theta: number[]): number {
    let H = 0
    const K = theta.length
    for (const t of theta) if (t > 0) H -= t * Math.log(t)
    const Hmax = Math.log(K)
    return Hmax > 0 ? 1 - H / Hmax : 1
  }

  $: phenotypes = $bundle?.phenotypes.phenotypes ?? []
  $: allPatients = ($cohort?.patients ?? []) as SyntheticPatient[]
  // Basic mode hides patients with dead/mixed dominants entirely from the
  // browser table; advanced shows them.
  $: patients = $advancedView ? allPatients : allPatients.filter((p) => p.isClean)

  // Decorate once per cohort/filter pass for sort stability. Dominant
  // label/id use displayedDominant so dead/mixed names never appear in
  // basic mode (a "clean" patient with secondary mass on a dead phenotype
  // shows their largest non-dead/mixed phenotype here).
  // Match the PatientMap's threshold so the row highlight and the ring
  // agree on what "patient has this phenotype" means.
  const PROFILE_THRESHOLD = 0.05

  $: decorated = patients.map((p) => {
    const d = displayedDominant(p.theta, phenotypes, $advancedView)
    const hasSearched = $searchedConditionIdx !== null
      && p.code_bag.indexOf($searchedConditionIdx) >= 0
    const hasPheno = $searchedPhenotypeForPatients !== null
      && (p.theta[$searchedPhenotypeForPatients] ?? 0) >= PROFILE_THRESHOLD
    return {
      id: p.id,
      patient: p,
      dominantId: d,
      dominantLabel: phenotypes[d]?.label || `Phenotype ${d}`,
      dominantWeight: p.theta[d] ?? 0,
      concentration: concentration(p.theta),
      hasSearched,
      hasPheno,
    }
  })

  $: filtered = decorated
    .filter((r) => {
      const q = filter.trim().toLowerCase()
      if (!q) return true
      return r.id.toLowerCase().includes(q)
        || r.dominantLabel.toLowerCase().includes(q)
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'id': return a.id.localeCompare(b.id)
        case 'dominant': return a.dominantLabel.localeCompare(b.dominantLabel)
        case 'concentration':
        default: return b.concentration - a.concentration
      }
    })

  function pick(row: typeof decorated[0]) {
    selectedPatientId.set(row.id)
    selectedPhenotypeId.set(row.dominantId)
  }
</script>

<details class="browser" open>
  <summary>
    <span class="summary-text">
      Browse all patients ({filtered.length}/{patients.length})
    </span>
    <span class="caret" aria-hidden="true">▾</span>
  </summary>

  <div class="controls">
    <div class="filter-wrap">
      <input
        type="search"
        placeholder="Filter by id or dominant phenotype…"
        bind:value={filter}
        class="filter-input"
      />
      {#if filter}
        <button class="filter-clear" type="button" on:click={() => (filter = '')} title="Clear the filter">×</button>
      {/if}
    </div>
    <div class="sort">
      <span class="eyebrow">Sort by</span>
      <select bind:value={sortBy}>
        <option value="concentration">Concentration</option>
        <option value="dominant">Dominant phenotype</option>
        <option value="id">ID</option>
      </select>
    </div>
  </div>

  <div class="table-wrap">
    <table class="patients">
      <thead>
        <tr>
          <th class="col-id">ID</th>
          <th class="col-dominant">Dominant phenotype</th>
          <th class="col-conc" data-numeric>Concentration</th>
        </tr>
      </thead>
      <tbody>
        {#each filtered as r (r.id)}
          <tr
            class:selected={$selectedPatientId === r.id}
            class:matched={r.hasSearched}
            class:phenomatched={r.hasPheno}
            on:click={() => pick(r)}
          >
            <td class="col-id">{r.id}</td>
            <td class="col-dominant">
              <!-- Colored dot mirrors the atlas dot color so the dominant
                   phenotype reads as the same color across atlas and table. -->
              <span class="dom-dot" style="background: {$phenotypeHue(r.dominantId)}" aria-hidden="true"></span>
              <span class="dom-label">{r.dominantLabel}</span>
              <span class="dom-w" data-numeric>{Math.round(r.dominantWeight * 100)}%</span>
            </td>
            <td class="col-conc" data-numeric>
              <span class="conc-row">
                <span class="conc-bar">
                  <span class="conc-fill" style="width: {r.concentration * 100}%"></span>
                </span>
                <span class="conc-num">{(r.concentration * 100).toFixed(0)}%</span>
              </span>
            </td>
          </tr>
        {/each}
        {#if filtered.length === 0}
          <tr><td colspan="3" class="empty">No patients match.</td></tr>
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
  .sort { display: flex; align-items: center; gap: 0.45rem; }

  .table-wrap {
    max-height: 420px;
    overflow-y: auto;
    border-top: 1px solid var(--rule-faint);
  }
  .patients {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--fs-small);
  }
  .patients thead {
    position: sticky;
    top: 0;
    background: var(--surface);
    z-index: 1;
  }
  .patients th {
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
  .patients td {
    padding: 0.45rem 0.85rem;
    border-bottom: 1px solid var(--rule-faint);
  }
  .patients tbody tr {
    cursor: pointer;
    transition: background 0.12s ease;
  }
  .patients tbody tr:hover { background: var(--surface-recessed); }
  .patients tbody tr.selected { background: var(--surface-recessed); }
  .patients tbody tr.selected td.col-id { font-weight: 600; }
  /* Row highlights for search activity. A patient that matches both a
     searched condition AND a searched phenotype gets BOTH bands (fuchsia
     left, amber right). Compatible with .selected (background channel). */
  .patients tbody tr.matched td:first-child {
    box-shadow: inset 3px 0 0 var(--accent-search);
  }
  .patients tbody tr.phenomatched td:last-child {
    box-shadow: inset -3px 0 0 var(--accent-find);
  }

  .col-id { width: 5rem; color: var(--ink-faint); font-family: var(--font-mono); }
  .col-dominant { color: var(--ink); }
  .col-conc { width: 13rem; text-align: right; }

  .dom-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 0.5rem;
    vertical-align: middle;
    flex-shrink: 0;
  }
  .dom-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: inline-block;
    max-width: calc(100% - 4.5rem);
    vertical-align: middle;
  }
  .dom-w {
    color: var(--ink-faint);
    margin-left: 0.5rem;
    font-size: var(--fs-micro);
  }

  .conc-row {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    width: 100%;
  }
  .conc-bar {
    flex: 1;
    display: block;
    height: 4px;
    background: var(--surface-recessed);
    border-radius: 2px;
    overflow: hidden;
  }
  .conc-fill {
    display: block;
    height: 100%;
    background: var(--accent);
  }
  .conc-num {
    color: var(--ink);
    min-width: 3rem;
    text-align: right;
  }

  .empty {
    text-align: center;
    color: var(--ink-faint);
    padding: 2rem 0;
    font-style: italic;
  }
</style>
