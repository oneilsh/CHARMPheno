<script lang="ts">
  import { bundle, selectedPhenotypeId } from '../store'
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
</script>

<section class="contrib">
  <h3>Top contributing codes</h3>
  {#if $selectedPhenotypeId === null}<p class="hint">Click a phenotype band above.</p>
  {:else if top.length === 0}<p>No codes from this patient's bag contribute to phenotype {$selectedPhenotypeId}.</p>
  {:else}
    <ol>
      {#each top as t}
        {@const c = $bundle!.vocab.codes[t.w]}
        <li>
          <span class="dom dom-{c.domain}">{c.domain.slice(0, 3)}</span>
          <span class="desc">{c.description || c.code}</span>
          <span class="count">×{t.c}</span>
        </li>
      {/each}
    </ol>
  {/if}
</section>

<style>
  .contrib { margin-top: 1rem; }
  .hint { color: #555; font-size: 0.85rem; }
  ol { list-style: none; padding: 0; }
  li { display: grid; grid-template-columns: 3rem 1fr 3rem; gap: 0.5rem; padding: 0.2rem 0; font-size: 0.85rem; border-bottom: 1px solid #f4f4f4; }
  .dom { font-size: 0.7rem; padding: 0.05rem 0.3rem; border-radius: 3px; text-align: center; }
  .dom-condition { background: #ffe4e1; }
  .dom-drug { background: #e0f2fe; }
  .dom-procedure { background: #ecfccb; }
  .dom-measurement { background: #fef3c7; }
  .dom-observation { background: #f5f5f5; }
  .count { text-align: right; font-variant-numeric: tabular-nums; color: #444; }
</style>
