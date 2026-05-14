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
  <h3>Prefix</h3>
  <div class="actions">
    <button on:click={loadFromPatient} disabled={!$selectedPatientId}>Load selected patient</button>
    <button on:click={clearAll}>Clear</button>
  </div>
  <input type="text" placeholder="Search vocab to add a code…" bind:value={searchText} />
  {#if matches.length > 0}
    <ul class="matches">
      {#each matches as c}<li><button on:click={() => add(c.id)}>+ {c.code} {c.description}</button></li>{/each}
    </ul>
  {/if}
  <details>
    <summary>Force a phenotype (adds top-5 codes for it)</summary>
    <ul class="force">
      {#each $bundle?.phenotypes.phenotypes ?? [] as p}
        <li><button on:click={() => forcePhenotype(p.id)}>{p.label || `Phenotype ${p.id}`}</button></li>
      {/each}
    </ul>
  </details>
  <h4>Current prefix ({$simulatorPrefix.length} codes)</h4>
  <ul class="prefix">
    {#each $simulatorPrefix as idx, i}
      {@const c = $bundle?.vocab.codes[idx]}
      <li>
        <span>{c?.description || c?.code || `#${idx}`}</span>
        <button on:click={() => removeAt(i)}>×</button>
      </li>
    {/each}
  </ul>
</section>

<style>
  .editor { padding: 1rem; border: 1px solid #ddd; }
  .actions { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; }
  input { width: 100%; padding: 0.4rem; margin-bottom: 0.5rem; }
  .matches, .force, .prefix { list-style: none; padding: 0; margin: 0; max-height: 220px; overflow: auto; }
  .matches li button, .force li button { width: 100%; text-align: left; background: transparent; border: 0; padding: 0.25rem; cursor: pointer; }
  .matches li button:hover, .force li button:hover { background: #f0f0f0; }
  .prefix li { display: grid; grid-template-columns: 1fr auto; gap: 0.5rem; padding: 0.2rem 0; font-size: 0.85rem; border-bottom: 1px solid #f4f4f4; }
  details { margin: 0.5rem 0; }
  h4 { margin: 0.5rem 0 0.25rem; font-size: 0.9rem; }
</style>
