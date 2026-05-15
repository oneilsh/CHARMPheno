<script lang="ts">
  import {
    bundle, patientsById, selectedPatientId, simulatorPrefix,
    advancedView,
  } from '../store'
  import { phenotypeHue } from '../palette'

  let searchText = ''
  let phenotypeFilter = ''

  $: matches = $bundle && searchText.length >= 2
    ? $bundle.vocab.codes
        .filter((c) =>
          c.description.toLowerCase().includes(searchText.toLowerCase()) ||
          c.code.includes(searchText))
        .slice(0, 8)
    : []

  // Phenotype list mirrors the Patient atlas's "visible" rule: in basic
  // mode we hide dead/mixed phenotypes from this picker since drawing
  // from them is rarely the user's intent.
  $: phenotypes = (() => {
    const ps = $bundle?.phenotypes.phenotypes ?? []
    const filtered = $advancedView
      ? ps
      : ps.filter((p) => p.quality !== 'dead' && p.quality !== 'mixed')
    const q = phenotypeFilter.trim().toLowerCase()
    if (!q) return filtered
    return filtered.filter((p) => (p.label || '').toLowerCase().includes(q))
  })()

  function copyFromPatient() {
    const p = $selectedPatientId ? $patientsById.get($selectedPatientId) : null
    if (p) simulatorPrefix.set([...p.code_bag])
  }
  function clearAll() { simulatorPrefix.set([]) }
  function add(idx: number) {
    simulatorPrefix.update((prev) => [...prev, idx])
    searchText = ''
  }
  function removeAt(i: number) {
    simulatorPrefix.update((prev) => prev.filter((_, j) => j !== i))
  }

  // Sample ONE code from beta[k] via inverse-CDF on Math.random(). Each
  // click is a real generative draw - clicking a phenotype repeatedly
  // produces a varying set of codes weighted by that phenotype's beta
  // row, not the top-5 deterministic head. This matches the simulator's
  // generative metaphor (every action is a sample from the model).
  function drawFromPhenotype(k: number) {
    if (!$bundle) return
    const row = $bundle.model.beta[k]
    const r = Math.random()
    let acc = 0
    for (let i = 0; i < row.length; i++) {
      acc += row[i]
      if (r <= acc) { simulatorPrefix.update((prev) => [...prev, i]); return }
    }
    // Numerical safety net: if cumulative undershoots 1, append the
    // last entry rather than dropping the draw.
    simulatorPrefix.update((prev) => [...prev, row.length - 1])
  }

  // Show the most-recently-added condition near the top of the list -
  // the user just clicked, they want feedback that the click landed.
  $: prefixList = $simulatorPrefix.map((idx, i) => ({ idx, i })).reverse()
</script>

<section class="editor">
  <header class="head">
    <span class="eyebrow">Starting point</span>
    <h3>Conditions</h3>
    <p class="sub">Conditions this patient already has. The simulator fills in the rest of their year.</p>
  </header>

  <div class="actions">
    <button class="btn-link" on:click={copyFromPatient} disabled={!$selectedPatientId}>
      ← copy from current patient
    </button>
    <button class="btn-link clear" on:click={clearAll} disabled={$simulatorPrefix.length === 0}>
      clear
    </button>
  </div>

  <div class="search-wrap">
    <input
      type="search"
      placeholder="Search to add a condition…"
      bind:value={searchText}
    />
    {#if matches.length > 0}
      <ul class="matches">
        {#each matches as c}
          <li><button class="match-btn" on:click={() => add(c.id)}>
            <span class="plus">+</span>
            <span class="match-desc">{c.description || c.code}</span>
          </button></li>
        {/each}
      </ul>
    {/if}
  </div>

  <details class="from-phenotype">
    <summary>
      <span>Draw from a phenotype</span>
      <span class="caret" aria-hidden="true">▾</span>
    </summary>
    <p class="hint">Each click draws one random condition from that phenotype's profile.</p>
    <input
      type="search"
      class="pheno-filter"
      placeholder="Filter phenotypes…"
      bind:value={phenotypeFilter}
    />
    <ul class="phenotypes">
      {#each phenotypes as p}
        <li><button class="pheno-btn" on:click={() => drawFromPhenotype(p.id)}>
          <span class="dot" style="background: {$phenotypeHue(p.id)}" aria-hidden="true"></span>
          <span class="pheno-label">{p.label || `Phenotype ${p.id}`}</span>
          <span class="draw-icon" aria-hidden="true">+</span>
        </button></li>
      {/each}
    </ul>
  </details>

  <div class="prefix-head">
    <span class="eyebrow">Current set</span>
    <span class="count" data-numeric>{$simulatorPrefix.length} condition{$simulatorPrefix.length === 1 ? '' : 's'}</span>
  </div>
  <ul class="prefix">
    {#each prefixList as { idx, i } (i + ':' + idx)}
      {@const c = $bundle?.vocab.codes[idx]}
      <li>
        <span class="prefix-desc">{c?.description || c?.code || `#${idx}`}</span>
        <button class="remove" on:click={() => removeAt(i)} aria-label="remove">×</button>
      </li>
    {:else}
      <li class="empty">No conditions yet — search above or draw from a phenotype.</li>
    {/each}
  </ul>
</section>

<style>
  .editor {
    padding: 1.25rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
  }

  .head {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--rule);
  }
  .head h3 {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: var(--tracking-tight);
    margin: 0;
  }
  .sub {
    margin: 0.15rem 0 0;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    font-style: italic;
    line-height: 1.5;
  }

  .actions {
    display: flex;
    gap: 0.6rem;
    align-items: center;
  }
  .btn-link {
    background: transparent;
    border: 0;
    color: var(--accent);
    cursor: pointer;
    font-family: var(--font-body);
    font-size: var(--fs-small);
    padding: 0.15rem 0;
    border-bottom: 1px dotted var(--accent);
    text-underline-offset: 2px;
  }
  .btn-link:hover { color: var(--ink); border-bottom-color: var(--ink); }
  .btn-link:disabled {
    color: var(--ink-faint);
    border-bottom-color: transparent;
    cursor: not-allowed;
  }
  .clear { margin-left: auto; color: var(--ink-muted); border-bottom-color: var(--ink-muted); }

  .search-wrap {
    position: relative;
  }
  .search-wrap input[type="search"] { width: 100%; }

  .matches {
    list-style: none;
    padding: 0;
    margin: 0.25rem 0 0;
    border: 1px solid var(--rule-faint);
    border-radius: var(--radius-sm);
    max-height: 220px;
    overflow: auto;
    background: var(--surface);
  }
  .match-btn {
    display: grid;
    grid-template-columns: 1rem 1fr;
    gap: 0.5rem;
    align-items: baseline;
    width: 100%;
    text-align: left;
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--rule-faint);
    padding: 0.4rem 0.55rem;
    cursor: pointer;
    font-family: var(--font-body);
    font-size: var(--fs-small);
    color: var(--ink);
  }
  .match-btn:last-child { border-bottom: 0; }
  .match-btn:hover { background: var(--surface-recessed); }
  .plus { color: var(--accent); font-weight: 600; }
  .match-desc {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .from-phenotype {
    border-top: 1px solid var(--rule-faint);
    border-bottom: 1px solid var(--rule-faint);
    padding: 0.6rem 0 0.5rem;
  }
  .from-phenotype summary {
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: var(--fs-small);
    color: var(--ink);
    font-weight: 500;
    user-select: none;
  }
  .from-phenotype summary::-webkit-details-marker { display: none; }
  .from-phenotype summary::marker { display: none; }
  .from-phenotype summary:hover { color: var(--accent); }
  .caret {
    font-family: var(--font-mono);
    color: var(--ink-faint);
    transition: transform 0.15s ease;
  }
  .from-phenotype[open] .caret { transform: rotate(180deg); }
  .hint {
    margin: 0.5rem 0 0.45rem;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    font-style: italic;
  }
  .pheno-filter {
    width: 100%;
    margin-bottom: 0.5rem;
  }
  .phenotypes {
    list-style: none;
    padding: 0;
    margin: 0;
    max-height: 240px;
    overflow: auto;
  }
  .pheno-btn {
    display: grid;
    grid-template-columns: 10px 1fr 1rem;
    align-items: center;
    gap: 0.55rem;
    width: 100%;
    text-align: left;
    background: transparent;
    border: 0;
    padding: 0.35rem 0.5rem;
    cursor: pointer;
    font-family: var(--font-body);
    font-size: var(--fs-small);
    color: var(--ink);
    border-radius: 3px;
    transition: background 0.12s ease, color 0.12s ease;
  }
  .pheno-btn:hover { background: var(--surface-recessed); }
  .pheno-btn:hover .draw-icon { color: var(--accent); }
  .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .pheno-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .draw-icon {
    color: var(--ink-faint);
    font-weight: 600;
    text-align: center;
  }

  .prefix-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-top: 0.1rem;
  }
  .count { color: var(--ink-muted); font-size: var(--fs-small); }

  .prefix {
    list-style: none;
    padding: 0;
    margin: 0;
    max-height: 280px;
    overflow: auto;
  }
  .prefix li {
    display: grid;
    grid-template-columns: 1fr 1.5rem;
    gap: 0.5rem;
    align-items: center;
    padding: 0.32rem 0;
    font-size: var(--fs-small);
    border-bottom: 1px solid var(--rule-faint);
  }
  .prefix li:last-child { border-bottom: 0; }
  .prefix .empty {
    color: var(--ink-faint);
    font-style: italic;
    grid-template-columns: 1fr;
    padding: 0.85rem 0;
    border-bottom: 0;
  }
  .prefix-desc {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--ink);
  }
  .remove {
    border: 0;
    background: transparent;
    color: var(--ink-faint);
    cursor: pointer;
    font-size: 1rem;
    line-height: 1;
    padding: 0;
    width: 1.25rem;
    height: 1.25rem;
    border-radius: 50%;
    transition: color 0.12s ease, background 0.12s ease;
  }
  .remove:hover { color: var(--danger); background: var(--surface-recessed); }
</style>
