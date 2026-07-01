<script lang="ts">
  import * as d3 from 'd3'
  import type { Correlation } from '../types'
  import { copy } from '../copy'

  export let correlation: Correlation

  // Diverging R ramp: red (R = -1) -> neutral gray (R = 0) -> cyan (R = +1).
  // Mirrors the NPMI ramp used on the phenotype atlas (TopicMap.svelte) so
  // "diverging metric" reads consistently across the dashboard.
  const rRamp = d3.scaleLinear<string>()
    .domain([-1, 0, 1])
    .range(['#ef4444', '#d4d4d8', '#06b6d4'])
    .clamp(true)

  const CELL = 22
  const MARGIN_LEFT = 90
  const MARGIN_TOP = 90

  $: order = correlation.topic_order
  $: n = order.length
  $: labels = correlation.block_labels
  $: W = MARGIN_LEFT + n * CELL
  $: H = MARGIN_TOP + n * CELL

  // Block-boundary positions (in grid-index space) where a separator line
  // should be drawn, derived from block_labels running in lockstep with
  // topic_order. A boundary sits just before the first index whose label
  // differs from its predecessor's.
  $: boundaries = (() => {
    const out: number[] = []
    for (let i = 1; i < n; i++) {
      if (labels[i] !== labels[i - 1]) out.push(i)
    }
    return out
  })()

  function naTitle(support: number): string {
    return `no joint support: ${support} < min_pair_support`
  }

  function cellTitle(i: number, j: number): string {
    const r = correlation.R[i][j]
    const identified = correlation.identified[i][j]
    const support = correlation.support[i][j]
    if (r === null || !identified) return naTitle(support)
    return `R = ${r.toFixed(3)} · N = ${support}`
  }
</script>

<figure class="heatmap" data-tour="correlation-heatmap">
  <svg
    viewBox="0 0 {W} {H}"
    width={W}
    height={H}
    role="img"
    aria-label={copy.correlation.ariaLabel}
  >
    <!-- Row labels (block labels, one per topic row) -->
    {#each order as _topic, i}
      <text
        x={MARGIN_LEFT - 6}
        y={MARGIN_TOP + i * CELL + CELL / 2}
        font-family="var(--font-mono)"
        font-size="8"
        fill="var(--ink-faint)"
        text-anchor="end"
        dominant-baseline="middle"
      >{labels[i]}</text>
    {/each}

    <!-- Column labels (rotated) -->
    {#each order as _topic, j}
      <text
        x={MARGIN_LEFT + j * CELL + CELL / 2}
        y={MARGIN_TOP - 6}
        font-family="var(--font-mono)"
        font-size="8"
        fill="var(--ink-faint)"
        text-anchor="start"
        dominant-baseline="middle"
        transform="rotate(-90 {MARGIN_LEFT + j * CELL + CELL / 2} {MARGIN_TOP - 6})"
      >{labels[j]}</text>
    {/each}

    <!-- Grid cells -->
    {#each order as _rowTopic, i}
      {#each order as _colTopic, j}
        {@const r = correlation.R[i][j]}
        {@const identified = correlation.identified[i][j]}
        {@const isNa = r === null || !identified}
        <rect
          class="cell"
          class:na={isNa}
          data-row={i}
          data-col={j}
          x={MARGIN_LEFT + j * CELL}
          y={MARGIN_TOP + i * CELL}
          width={CELL}
          height={CELL}
          fill={isNa ? 'var(--rule-strong)' : rRamp(r as number)}
          data-tip={cellTitle(i, j)}
        ><title>{cellTitle(i, j)}</title></rect>
      {/each}
    {/each}

    <!-- Block separators: heavier rule at each boundary, both row and column. -->
    {#each boundaries as b}
      <line
        x1={MARGIN_LEFT} y1={MARGIN_TOP + b * CELL}
        x2={MARGIN_LEFT + n * CELL} y2={MARGIN_TOP + b * CELL}
        stroke="var(--ink-muted)"
        stroke-width="1.5"
      />
      <line
        x1={MARGIN_LEFT + b * CELL} y1={MARGIN_TOP}
        x2={MARGIN_LEFT + b * CELL} y2={MARGIN_TOP + n * CELL}
        stroke="var(--ink-muted)"
        stroke-width="1.5"
      />
    {/each}

    <!-- Outer frame -->
    <rect
      x={MARGIN_LEFT} y={MARGIN_TOP}
      width={n * CELL} height={n * CELL}
      fill="none"
      stroke="var(--rule)"
      stroke-width="1"
    />
  </svg>

  <figcaption class="legend">
    <div class="legend-group">
      <span class="eyebrow">R</span>
      <span class="grad grad-r" aria-hidden="true"></span>
      <span class="ticks" data-numeric><span>-1</span><span>0</span><span>+1</span></span>
    </div>
    <div class="legend-group">
      <span class="swatch na" aria-hidden="true"></span>
      <span class="eyebrow">no joint support</span>
    </div>
  </figcaption>
</figure>

<style>
  .heatmap {
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  svg {
    display: block;
    overflow: visible;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
  }
  .cell {
    cursor: help;
  }
  .legend {
    display: flex;
    gap: 1.5rem;
    align-items: center;
    padding: 0.25rem 0.25rem;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    flex-wrap: wrap;
  }
  .legend-group {
    display: flex;
    align-items: center;
    gap: 0.55rem;
  }
  .eyebrow {
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .grad {
    display: inline-block;
    width: 96px;
    height: 6px;
    border-radius: 3px;
  }
  .grad-r {
    background: linear-gradient(to right, #ef4444, #d4d4d8, #06b6d4);
  }
  .ticks {
    display: inline-flex;
    gap: 0.4rem;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
  }
  .swatch {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 2px;
    background: var(--rule-strong);
  }
</style>
