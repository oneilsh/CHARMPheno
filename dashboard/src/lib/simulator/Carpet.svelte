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
  $: H = ROW_H * K + 36

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

  // Adaptive x-scale: clip to the 95th-percentile of p90s with 15% headroom.
  // Rows whose p50 exceeds the scale overflow visually and get a chevron.
  // Floor at 0.1 (10%) so the scale never shrinks past readability, ceiling
  // at 1.0 since the simplex never goes higher. This fixes the "one dominant
  // phenotype pins the scale and everything else looks like zero" pathology.
  $: xMax = (() => {
    if (rows.length === 0) return 0.1
    const sorted = rows.map((r) => r.p90).slice().sort((a, b) => a - b)
    const p95 = sorted[Math.floor(sorted.length * 0.95)] ?? 0.1
    return Math.max(0.1, Math.min(1.0, p95 * 1.15))
  })()
  $: xScale = d3.scaleLinear().domain([0, xMax]).range([X_MARGIN, W - 28])
  $: ticks = xScale.ticks(5)

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

<div class="carpet-wrap">
  <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img" aria-label="Phenotype posterior carpet">
    <!-- gridlines + tick labels -->
    {#each ticks as t}
      <line
        x1={xScale(t)} y1={26} x2={xScale(t)} y2={H - 8}
        stroke="#e4e4e7" stroke-width="0.5" stroke-dasharray="2,3"
      />
      <text x={xScale(t)} y={18} font-size="9" text-anchor="middle" fill="#a1a1aa" font-family="JetBrains Mono, monospace">
        {(t * 100).toFixed(0)}%
      </text>
    {/each}

    <!-- axis labels -->
    <text x={X_MARGIN - 8} y={18} font-size="9" text-anchor="end" fill="#a1a1aa"
          font-family="JetBrains Mono, monospace"
          letter-spacing="0.08em" text-transform="uppercase">
      PHENOTYPE
    </text>
    <text x={W - 22} y={18} font-size="9" text-anchor="end" fill="#a1a1aa"
          font-family="JetBrains Mono, monospace"
          letter-spacing="0.08em" text-transform="uppercase">
      POSTERIOR θ
    </text>

    {#each rows as r, i}
      {@const cy = 30 + i * ROW_H + ROW_H / 2}
      {@const overflow = r.p50 > xMax}
      {@const selected = expandedK === r.k}
      <g style="cursor: pointer;"
         role="button" tabindex="0"
         aria-label={`Toggle drill-down for ${$phenotypesById.get(r.k)?.label || `Phenotype ${r.k}`}`}
         on:click={() => expandedK = expandedK === r.k ? null : r.k}
         on:keydown={(e) => (e.key === 'Enter' || e.key === ' ') && (expandedK = expandedK === r.k ? null : r.k)}>
        <!-- row highlight -->
        <rect x="0" y={cy - ROW_H / 2} width={W} height={ROW_H}
              fill={selected ? '#ecfeff' : 'transparent'} />

        <!-- label -->
        <text x={X_MARGIN - 8} y={cy + 3} font-size="10" text-anchor="end"
              fill={selected ? '#0a0a0a' : '#52525b'}
              font-family="Geist, sans-serif"
              font-weight={selected ? 600 : 400}>
          {$phenotypesById.get(r.k)?.label || `Phenotype ${r.k}`}
        </text>

        {#if overflow}
          <!-- Overflow chevron at the right edge of the plot area -->
          <text x={W - 22} y={cy + 3} font-size="11" fill="#06b6d4" text-anchor="end"
                font-family="Geist, sans-serif" font-weight="600">›</text>
          <line x1={xScale(Math.min(r.p10, xMax))} y1={cy} x2={W - 26} y2={cy}
                stroke="#06b6d4" stroke-width="1" />
          <rect x={xScale(Math.min(r.p25, xMax))} y={cy - 4}
                width={Math.max(1, (W - 26) - xScale(Math.min(r.p25, xMax)))} height="8"
                fill="#06b6d4" opacity="0.5" />
        {:else}
          <!-- P10–P90 line -->
          <line x1={xScale(r.p10)} y1={cy} x2={xScale(r.p90)} y2={cy}
                stroke="#d4d4d8" stroke-width="1" />
          <!-- P25–P75 box -->
          <rect x={xScale(r.p25)} y={cy - 4}
                width={Math.max(1, xScale(r.p75) - xScale(r.p25))} height="8"
                fill="#06b6d4" opacity="0.75" />
          <!-- median tick -->
          <line x1={xScale(r.p50)} y1={cy - 5} x2={xScale(r.p50)} y2={cy + 5}
                stroke="#0a0a0a" stroke-width="1.5" />
        {/if}

        <!-- numeric % -->
        <text x={W - 4} y={cy + 3} font-size="9" text-anchor="end"
              fill={overflow ? '#06b6d4' : '#52525b'}
              font-family="JetBrains Mono, monospace">
          {(r.p50 * 100).toFixed(1)}%
        </text>
      </g>
    {/each}
  </svg>

  {#if expandedK !== null}
    <aside class="drill">
      <header>
        <span class="eyebrow">Drill-down</span>
        <h4>{$phenotypesById.get(expandedK)?.label || `Phenotype ${expandedK}`}</h4>
      </header>
      {#if drill.length === 0}
        <p class="hint">No completion codes scored above zero for this phenotype.</p>
      {:else}
        {@const dMax = Math.max(...drill.map((x) => x.score))}
        <ol>
          {#each drill as d}
            {@const c = $bundle!.vocab.codes[d.w]}
            <li>
              <span class="domain-mark dom-{c.domain}">{c.domain.slice(0, 3)}</span>
              <span class="desc">{c.description || c.code}</span>
              <span class="spark" aria-hidden="true">
                <span class="spark-bar" style="width: {(d.score / dMax) * 100}%"></span>
              </span>
              <span class="score" data-numeric>{d.score.toFixed(2)}</span>
            </li>
          {/each}
        </ol>
      {/if}
    </aside>
  {/if}
</div>

<style>
  .carpet-wrap {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    padding: 0.5rem 0.5rem 0.25rem;
  }

  svg { display: block; }

  .drill {
    margin: 0.5rem 0.25rem 0.25rem;
    padding: 0.85rem 1rem;
    background: var(--accent-faint);
    border: 1px solid var(--accent-soft);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-sm);
  }
  .drill header {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    margin-bottom: 0.65rem;
  }
  .drill h4 {
    margin: 0;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--ink);
  }
  .hint {
    color: var(--ink-muted);
    font-size: var(--fs-small);
    margin: 0;
  }
  ol {
    list-style: none;
    padding: 0;
    margin: 0;
    font-size: var(--fs-small);
  }
  ol li {
    display: grid;
    grid-template-columns: 4.5rem 1fr 56px 3.5rem;
    align-items: center;
    gap: 0.7rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(6, 182, 212, 0.15);
  }
  ol li:last-child { border-bottom: 0; }
  .desc {
    color: var(--ink);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .spark {
    display: block;
    height: 3px;
    background: rgba(255, 255, 255, 0.6);
    border-radius: 1.5px;
    overflow: hidden;
  }
  .spark-bar {
    display: block;
    height: 100%;
    background: var(--accent);
  }
  .score {
    text-align: right;
    color: var(--ink-muted);
  }
</style>
