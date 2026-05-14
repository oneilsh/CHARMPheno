<script lang="ts">
  import * as d3 from 'd3'
  import { bundle, phenotypesById } from '../store'
  import { quantiles } from './runSamples'
  export let thetaSamples: number[][]
  export let codeCountsSamples: Map<number, number>[]
  export let sortMode: 'median' | 'spread' | 'npmi' | 'id' = 'median'

  const W = 720, ROW_H = 14, X_MARGIN = 220
  let expandedK: number | null = null

  $: K = $bundle?.model.K ?? 0
  $: H = ROW_H * K + 20

  $: rows = (() => {
    if (thetaSamples.length === 0 || K === 0) return []
    const out = []
    for (let k = 0; k < K; k++) {
      const ks = thetaSamples.map((t) => t[k])
      const q = quantiles(ks, [0.1, 0.25, 0.5, 0.75, 0.9])
      out.push({ k, p10: q[0], p25: q[1], p50: q[2], p75: q[3], p90: q[4] })
    }
    if (sortMode === 'median') out.sort((a, b) => b.p50 - a.p50)
    else if (sortMode === 'spread') out.sort((a, b) => (b.p90 - b.p10) - (a.p90 - a.p10))
    else if (sortMode === 'npmi') {
      const npmi = (k: number) => $phenotypesById.get(k)?.npmi ?? 0
      out.sort((a, b) => npmi(b.k) - npmi(a.k))
    }
    return out
  })()

  $: xMax = rows.length > 0 ? Math.max(0.01, d3.max(rows, (r) => r.p90) ?? 0.01) : 0.01
  $: xScale = d3.scaleLinear().domain([0, xMax]).range([X_MARGIN, W - 20])

  $: drill = (() => {
    if (expandedK === null || !$bundle) return [] as { w: number; score: number }[]
    const k = expandedK
    const beta = $bundle.model.beta
    const K_ = $bundle.model.K
    const scores = new Map<number, number>()
    for (let s = 0; s < codeCountsSamples.length; s++) {
      const theta = thetaSamples[s]
      for (const [w, c] of codeCountsSamples[s]) {
        let z = 0
        for (let j = 0; j < K_; j++) z += beta[j][w] * theta[j]
        const pzkw = z > 0 ? (beta[k][w] * theta[k]) / z : 0
        scores.set(w, (scores.get(w) ?? 0) + c * pzkw)
      }
    }
    return Array.from(scores.entries())
      .map(([w, score]) => ({ w, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 8)
  })()
</script>

<svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img">
  {#each rows as r, i}
    {@const cy = 10 + i * ROW_H + ROW_H / 2}
    <g style="cursor: pointer;" on:click={() => expandedK = expandedK === r.k ? null : r.k}>
      <rect x="0" y={cy - ROW_H / 2} width={W} height={ROW_H} fill={expandedK === r.k ? '#f4f8ff' : 'transparent'} />
      <text x={X_MARGIN - 8} y={cy + 3} font-size="10" text-anchor="end">
        {$phenotypesById.get(r.k)?.label || `Phenotype ${r.k}`}
      </text>
      <line x1={xScale(r.p10)} y1={cy} x2={xScale(r.p90)} y2={cy} stroke="#999" stroke-width="1" />
      <rect x={xScale(r.p25)} y={cy - 4} width={Math.max(1, xScale(r.p75) - xScale(r.p25))} height="8" fill="#1e88e5" opacity="0.7" />
      <line x1={xScale(r.p50)} y1={cy - 5} x2={xScale(r.p50)} y2={cy + 5} stroke="#000" stroke-width="1.5" />
      <text x={W - 18} y={cy + 3} font-size="9" text-anchor="end" fill="#555">{(r.p50 * 100).toFixed(1)}%</text>
    </g>
  {/each}
</svg>

{#if expandedK !== null}
  <aside class="drill">
    <h4>Top codes driving {$phenotypesById.get(expandedK)?.label || `Phenotype ${expandedK}`}</h4>
    {#if drill.length === 0}<p>No completion codes scored above zero.</p>
    {:else}
      <ol>{#each drill as d}{@const c = $bundle!.vocab.codes[d.w]}<li><span>{c.description || c.code}</span><span class="score">{d.score.toFixed(2)}</span></li>{/each}</ol>
    {/if}
  </aside>
{/if}

<style>
  .drill { margin-top: 0.5rem; padding: 0.75rem; background: #f4f8ff; border: 1px solid #cfe2ff; }
  .drill h4 { margin: 0 0 0.5rem; font-size: 0.9rem; }
  ol { list-style: none; padding: 0; margin: 0; font-size: 0.85rem; }
  ol li { display: grid; grid-template-columns: 1fr 4rem; gap: 0.5rem; padding: 0.15rem 0; border-bottom: 1px solid #e0e0e0; }
  .score { text-align: right; font-variant-numeric: tabular-nums; }
</style>
