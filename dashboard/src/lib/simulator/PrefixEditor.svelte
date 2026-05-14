<script lang="ts">
  import {
    bundle, patientsById, selectedPatientId, simulatorPrefix,
  } from '../store'
  let searchText = ''

  $: matches = $bundle && searchText.length >= 2
    ? $bundle.vocab.codes
        .filter((c) =>
          c.description.toLowerCase().includes(searchText.toLowerCase()) ||
          c.code.includes(searchText))
        .slice(0, 8)
    : []

  function loadFromPatient() {
    const p = $selectedPatientId ? $patientsById.get($selectedPatientId) : null
    if (p) simulatorPrefix.set([...p.code_bag])
  }
  function clearAll() { simulatorPrefix.set([]) }
  function add(idx: number) {
    simulatorPrefix.update((prev) => [...prev, idx]); searchText = ''
  }
  function removeAt(i: number) {
    simulatorPrefix.update((prev) => prev.filter((_, j) => j !== i))
  }
  function forcePhenotype(k: number) {
    if (!$bundle) return
    const row = $bundle.model.beta[k]
    const topIdx = row.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, 5).map((x) => x.i)
    simulatorPrefix.update((prev) => [...prev, ...topIdx])
  }
</script>

<section class="editor">
  <header>
    <span class="eyebrow">Composition</span>
    <h3>Prefix</h3>
  </header>

  <div class="actions">
    <button class="btn" on:click={loadFromPatient} disabled={!$selectedPatientId}>
      ← from current patient
    </button>
    <button class="btn-ghost clear" on:click={clearAll}>clear</button>
  </div>

  <input type="text" placeholder="Search vocab to add a code…" bind:value={searchText} />

  {#if matches.length > 0}
    <ul class="matches">
      {#each matches as c}
        <li><button class="match-btn" on:click={() => add(c.id)}>
          <span class="plus">+</span>
          <span class="match-code" data-numeric>{c.code}</span>
          <span class="match-desc">{c.description}</span>
        </button></li>
      {/each}
    </ul>
  {/if}

  <details>
    <summary>Force a phenotype</summary>
    <p class="hint">Appends the top-5 codes for the chosen phenotype to the prefix.</p>
    <ul class="force">
      {#each $bundle?.phenotypes.phenotypes ?? [] as p}
        <li><button class="force-btn" on:click={() => forcePhenotype(p.id)}>
          {p.label || `Phenotype ${p.id}`}
        </button></li>
      {/each}
    </ul>
  </details>

  <div class="prefix-head">
    <span class="eyebrow">Current prefix</span>
    <span class="count" data-numeric>{$simulatorPrefix.length}</span>
  </div>
  <ul class="prefix">
    {#each $simulatorPrefix as idx, i}
      {@const c = $bundle?.vocab.codes[idx]}
      <li>
        <span class="prefix-desc">{c?.description || c?.code || `#${idx}`}</span>
        <button class="remove" on:click={() => removeAt(i)} aria-label="remove">×</button>
      </li>
    {:else}
      <li class="empty">No codes yet.</li>
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
    gap: 0.85rem;
  }

  header {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--rule);
  }
  header h3 {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: var(--tracking-tight);
  }

  .actions {
    display: flex;
    gap: 0.6rem;
    align-items: center;
  }
  .clear {
    font-size: var(--fs-small);
    margin-left: auto;
  }

  input[type="text"] { width: 100%; }

  .matches, .force, .prefix {
    list-style: none;
    padding: 0;
    margin: 0;
    max-height: 220px;
    overflow: auto;
  }

  .matches {
    border: 1px solid var(--rule-faint);
    border-radius: var(--radius-sm);
    margin-top: -0.4rem;
  }
  .match-btn {
    display: grid;
    grid-template-columns: 1rem 4rem 1fr;
    gap: 0.5rem;
    align-items: baseline;
    width: 100%;
    text-align: left;
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--rule-faint);
    padding: 0.35rem 0.5rem;
    cursor: pointer;
    font-family: var(--font-body);
    font-size: var(--fs-small);
    color: var(--ink);
    transition: background 0.12s ease;
  }
  .match-btn:last-child { border-bottom: 0; }
  .match-btn:hover { background: var(--surface-recessed); }
  .plus { color: var(--accent); font-weight: 600; }
  .match-code {
    color: var(--ink-faint);
    font-size: var(--fs-micro);
    letter-spacing: 0.02em;
  }
  .match-desc {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  details {
    border-top: 1px solid var(--rule-faint);
    border-bottom: 1px solid var(--rule-faint);
    padding: 0.5rem 0;
  }
  summary {
    cursor: pointer;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    list-style-position: inside;
  }
  summary:hover { color: var(--accent); }
  .hint {
    margin: 0.4rem 0 0.25rem;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
  }
  .force-btn {
    width: 100%;
    text-align: left;
    background: transparent;
    border: 0;
    padding: 0.3rem 0.5rem;
    cursor: pointer;
    font-family: var(--font-body);
    font-size: var(--fs-small);
    color: var(--ink-muted);
    transition: color 0.12s ease, background 0.12s ease;
  }
  .force-btn:hover { color: var(--accent); background: var(--surface-recessed); }

  .prefix-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-top: 0.25rem;
  }
  .count { color: var(--ink); font-size: var(--fs-small); }

  .prefix li {
    display: grid;
    grid-template-columns: 1fr 1.5rem;
    gap: 0.5rem;
    align-items: center;
    padding: 0.3rem 0;
    font-size: var(--fs-small);
    border-bottom: 1px solid var(--rule-faint);
  }
  .prefix li:last-child { border-bottom: 0; }
  .prefix .empty {
    color: var(--ink-faint);
    grid-template-columns: 1fr;
    padding: 0.6rem 0;
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
  .remove:hover {
    color: var(--danger);
    background: var(--surface-recessed);
  }
</style>
