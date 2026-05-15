<script lang="ts">
  import { bundle, searchedConditionIdx } from '../store'

  let query = ''
  let inputEl: HTMLInputElement
  let showResults = false

  // Top N matches by substring. Ordered by corpus_freq descending so common
  // conditions surface first — usually what a user actually wants when they
  // type "diabetes" rather than getting a rare specific subtype.
  const MAX_RESULTS = 8

  $: vocab = $bundle?.vocab.codes ?? []

  $: matches = (() => {
    const q = query.trim().toLowerCase()
    if (!q || q.length < 2) return []
    const out: { idx: number; description: string; corpus_freq: number }[] = []
    for (let i = 0; i < vocab.length; i++) {
      const desc = (vocab[i].description || vocab[i].code).toLowerCase()
      if (desc.includes(q)) {
        out.push({
          idx: i,
          description: vocab[i].description || vocab[i].code,
          corpus_freq: vocab[i].corpus_freq,
        })
      }
    }
    out.sort((a, b) => b.corpus_freq - a.corpus_freq)
    return out.slice(0, MAX_RESULTS)
  })()

  $: selectedCondition = $searchedConditionIdx !== null
    ? vocab[$searchedConditionIdx]
    : null

  function pick(idx: number) {
    searchedConditionIdx.set(idx)
    query = ''
    showResults = false
    inputEl?.blur()
  }

  function clear() {
    searchedConditionIdx.set(null)
    query = ''
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      showResults = false
      inputEl?.blur()
    } else if (e.key === 'Enter' && matches.length > 0) {
      e.preventDefault()
      pick(matches[0].idx)
    }
  }
</script>

<div class="condition-search">
  <label class="control">
    <span class="eyebrow">Find condition</span>
    <div class="input-wrap">
      <input
        bind:this={inputEl}
        type="search"
        placeholder="e.g. diabetes, psoriasis, valve…"
        bind:value={query}
        on:focus={() => (showResults = true)}
        on:keydown={onKeydown}
      />
      {#if selectedCondition}
        <button class="clear" on:click={clear} title="Clear searched condition" type="button">×</button>
      {/if}
    </div>
  </label>

  {#if showResults && matches.length > 0}
    <ul class="results" role="listbox">
      {#each matches as m}
        <li>
          <button type="button" on:mousedown|preventDefault={() => pick(m.idx)}>
            <span class="desc">{m.description}</span>
            <span class="freq" data-numeric>{(m.corpus_freq * 100).toFixed(2)}%</span>
          </button>
        </li>
      {/each}
    </ul>
  {/if}

  {#if selectedCondition}
    <div class="active-chip" title="Phenotypes containing this condition are highlighted on the atlas">
      <span class="chip-label">Highlighting phenotypes with</span>
      <span class="chip-val">{selectedCondition.description || selectedCondition.code}</span>
    </div>
  {/if}
</div>

<svelte:window
  on:click={(e) => {
    // Close the results dropdown when clicking outside.
    if (showResults && !(e.target as HTMLElement)?.closest('.condition-search')) {
      showResults = false
    }
  }}
/>

<style>
  .condition-search {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .control {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .input-wrap {
    position: relative;
    display: flex;
    align-items: center;
  }
  input[type="search"] {
    width: 240px;
    padding: 0.4rem 1.7rem 0.4rem 0.6rem;
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius-sm);
    background: var(--surface);
    color: var(--ink);
    font-family: var(--font-body);
    font-size: var(--fs-small);
  }
  input[type="search"]:focus {
    outline: none;
    border-color: var(--accent);
  }
  .clear {
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
  .clear:hover { color: var(--ink); }

  .results {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    margin: 0.15rem 0 0;
    padding: 0.25rem 0;
    list-style: none;
    background: var(--surface);
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius-sm);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    z-index: 10;
    max-height: 280px;
    overflow-y: auto;
  }
  .results li button {
    width: 100%;
    text-align: left;
    border: 0;
    background: transparent;
    padding: 0.35rem 0.7rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    justify-content: space-between;
    font-size: var(--fs-small);
    color: var(--ink);
    font-family: var(--font-body);
  }
  .results li button:hover {
    background: var(--surface-recessed);
    color: var(--accent);
  }
  .results .desc {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .results .freq {
    color: var(--ink-faint);
    font-size: var(--fs-micro);
    flex-shrink: 0;
  }

  .active-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.55rem;
    background: var(--surface-recessed);
    border-radius: var(--radius-sm);
    font-size: var(--fs-micro);
    color: var(--ink-muted);
    max-width: 280px;
  }
  .chip-label {
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .chip-val {
    color: var(--accent);
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
