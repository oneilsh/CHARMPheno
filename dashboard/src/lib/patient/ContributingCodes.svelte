<script lang="ts">
  import { bundle, selectedPhenotypeId, phenotypesById } from '../store'
  export let theta: number[]
  export let codeBag: number[]

  $: counts = (() => {
    const m = new Map<number, number>()
    for (const w of codeBag) m.set(w, (m.get(w) ?? 0) + 1)
    return m
  })()

  $: top = (() => {
    if (!$bundle || $selectedPhenotypeId === null) return []
    const k = $selectedPhenotypeId
    const beta = $bundle.model.beta
    const K = $bundle.model.K
    const scored: { w: number; c: number; score: number }[] = []
    for (const [w, c] of counts) {
      let z = 0
      for (let j = 0; j < K; j++) z += beta[j][w] * theta[j]
      const pzkw = z > 0 ? (beta[k][w] * theta[k]) / z : 0
      scored.push({ w, c, score: c * pzkw })
    }
    return scored.sort((a, b) => b.score - a.score).slice(0, 12)
  })()

  $: maxScore = top.length ? Math.max(...top.map((t) => t.score)) : 1
  $: selectedLabel = $selectedPhenotypeId !== null
    ? ($phenotypesById.get($selectedPhenotypeId)?.label || `Phenotype ${$selectedPhenotypeId}`)
    : null
</script>

<section class="contrib">
  <header class="head">
    <span class="eyebrow">Section</span>
    <h3>Top contributing codes{#if selectedLabel} <span class="for">for</span> <em>{selectedLabel}</em>{/if}</h3>
  </header>

  {#if $selectedPhenotypeId === null}
    <p class="hint">Click a phenotype band above to see which codes from this patient's bag drove the assignment.</p>
  {:else if top.length === 0}
    <p class="hint">No codes from this patient's bag contribute to {selectedLabel}.</p>
  {:else}
    <ol class="codes">
      {#each top as t}
        {@const c = $bundle!.vocab.codes[t.w]}
        <li>
          <span class="domain-mark dom-{c.domain}">{c.domain.slice(0, 3)}</span>
          <span class="desc">{c.description || c.code}</span>
          <span class="spark" aria-hidden="true">
            <span class="spark-bar" style="width: {(t.score / maxScore) * 100}%"></span>
          </span>
          <span class="count" data-numeric>×{t.c}</span>
        </li>
      {/each}
    </ol>
  {/if}
</section>

<style>
  .contrib {
    margin-top: 2.5rem;
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: 0.85rem;
    margin-bottom: 0.85rem;
  }
  .head h3 {
    font-size: 1.05rem;
    font-weight: 500;
  }
  .head .for {
    color: var(--ink-faint);
    font-weight: 400;
    font-family: var(--font-display);
    font-style: italic;
  }
  .head h3 em {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 500;
    color: var(--terracotta);
  }

  .hint {
    color: var(--ink-muted);
    font-style: italic;
    font-size: var(--fs-small);
    padding: 0.5rem 0;
    border-top: 1px solid var(--rule-faint);
  }

  .codes {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .codes li {
    display: grid;
    grid-template-columns: 5rem 1fr 4rem 2rem;
    align-items: center;
    gap: 0.85rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--rule-faint);
    font-size: var(--fs-small);
  }
  .codes li:last-child { border-bottom: 0; }
  .codes .desc {
    color: var(--ink);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .spark {
    display: block;
    height: 4px;
    background: var(--paper-recessed);
    border-radius: 1px;
    overflow: hidden;
  }
  .spark-bar {
    display: block;
    height: 100%;
    background: var(--terracotta);
    transition: width 0.2s ease;
  }
  .count {
    text-align: right;
    color: var(--ink-muted);
    font-size: var(--fs-small);
  }
</style>
