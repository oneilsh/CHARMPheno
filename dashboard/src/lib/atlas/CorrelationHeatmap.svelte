<script lang="ts">
  import * as d3 from 'd3'
  import type { Correlation } from '../types'
  import { bundle, selectedPhenotypeId } from '../store'
  import { copy } from '../copy'

  export let correlation: Correlation

  // Diverging R ramp: red (R = -1) -> WHITE (R = 0) -> cyan (R = +1). A white
  // midpoint (not gray) keeps near-zero correlations from muddying the field, so
  // the genuinely strong pairs pop. Mirrors the atlas's red→neutral→cyan sense.
  const rRamp = d3.scaleLinear<string>()
    .domain([-1, 0, 1])
    .range(['#ef4444', '#ffffff', '#06b6d4'])
    .clamp(true)

  // viewBox units (the SVG scales to fit via CSS). MARGIN pads the grid inside
  // the card so the plot floats within its frame (like the bubble map) rather
  // than bleeding to the border. GAP is a small break drawn between blocks in
  // place of a separator line.
  const CELL = 22
  const MARGIN = 10
  const GAP = 18

  $: order = correlation.topic_order
  $: n = order.length
  $: labels = correlation.block_labels

  // Block boundaries (grid-index space): first index of each new block.
  $: boundaries = (() => {
    const out: number[] = []
    for (let i = 1; i < n; i++) if (labels[i] !== labels[i - 1]) out.push(i)
    return out
  })()

  // Cumulative gap count applied before each index (one GAP per block boundary
  // at-or-before the index), so blocks are visually separated by whitespace.
  $: cumGap = (() => {
    const bset = new Set(boundaries)
    const arr = new Array<number>(n)
    let g = 0
    for (let i = 0; i < n; i++) {
      if (bset.has(i)) g++
      arr[i] = g
    }
    return arr
  })()
  $: pos = (i: number) => MARGIN + i * CELL + (cumGap[i] ?? 0) * GAP
  $: gridSpan = n * CELL + boundaries.length * GAP
  $: S = MARGIN * 2 + gridSpan

  // Phenotype id -> label, for the hover tooltip (row × column topic names).
  $: labelById = new Map(
    ($bundle?.phenotypes.phenotypes ?? []).map((p) => [p.id, p.label || `Phenotype ${p.id}`]),
  )

  // Selected phenotype's position in the matrix ordering (-1 if absent, e.g. the
  // reference topic, which has no Sigma row/column).
  $: selIndex = $selectedPhenotypeId == null ? -1 : order.indexOf($selectedPhenotypeId)

  function cellTitle(i: number, j: number): string {
    const r = correlation.R[i][j]
    const identified = correlation.identified[i][j]
    const support = correlation.support[i][j]
    const pair = `${labelById.get(order[i]) ?? order[i]} × ${labelById.get(order[j]) ?? order[j]}`
    if (r === null || !identified) return `${pair}\nno joint support: ${support} < min_pair_support`
    return `${pair}\nR = ${r.toFixed(3)} · N = ${support}`
  }

  // Clicking a cell selects its COLUMN topic (order[j]); the row topic is named
  // in the tooltip, so the pairing stays legible while the click has one target.
  function selectCol(j: number) {
    selectedPhenotypeId.set(order[j])
  }
</script>

<figure class="heatmap" data-tour="correlation-heatmap">
  <div class="card">
    <svg
      viewBox="0 0 {S} {S}"
      role="img"
      aria-label={copy.correlation.ariaLabel}
      preserveAspectRatio="xMidYMid meet"
    >
    <!-- Grid cells -->
    {#each order as _rowTopic, i}
      {#each order as _colTopic, j}
        {@const r = correlation.R[i][j]}
        {@const identified = correlation.identified[i][j]}
        {@const isNa = r === null || !identified}
        <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
        <!-- The grid is thousands of cells; per-cell keyboard focus would be a
             tab-trap. The same selection is keyboard-reachable via the phenotype
             table, so the cell click is a mouse-only enhancement. -->
        <rect
          class="cell"
          class:na={isNa}
          data-row={i}
          data-col={j}
          x={pos(j)}
          y={pos(i)}
          width={CELL}
          height={CELL}
          fill={isNa ? 'var(--rule)' : rRamp(r as number)}
          data-tip={cellTitle(i, j)}
          on:click={() => selectCol(j)}
        ><title>{cellTitle(i, j)}</title></rect>
      {/each}
    {/each}

    <!-- Selection: gently OUTLINE the selected topic's row + column (no fill, so
         the underlying R values aren't visually altered), and ring its diagonal.
         pointer-events:none so clicks still reach the cells underneath. -->
    {#if selIndex >= 0}
      <rect class="sel-line" x={MARGIN} y={pos(selIndex)} width={gridSpan} height={CELL} />
      <rect class="sel-line" x={pos(selIndex)} y={MARGIN} width={CELL} height={gridSpan} />
      <rect class="sel-diag" x={pos(selIndex)} y={pos(selIndex)} width={CELL} height={CELL} />
    {/if}
    </svg>
  </div>

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
    <span class="hint">hover a cell for the topic pair · click to select its column topic</span>
  </figcaption>
</figure>

<style>
  .heatmap {
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  /* Full-column-width framed card; the grid itself stays a fixed square, centered
     inside with left/right + top/bottom breathing room. */
  .card {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    padding: 1rem 0;
    display: flex;
    justify-content: center;
  }
  svg {
    width: 100%;
    max-width: 560px;
    aspect-ratio: 1 / 1;
    height: auto;
    display: block;
  }
  .cell {
    cursor: pointer;
  }
  .cell:hover {
    stroke: var(--ink);
    stroke-width: 1.5;
    paint-order: stroke;
  }
  /* Selection outline (accent cyan), no fill so R values read true. */
  .sel-line {
    fill: none;
    stroke: var(--accent);
    stroke-width: 1.5;
    pointer-events: none;
  }
  .sel-diag {
    fill: none;
    stroke: var(--accent);
    stroke-width: 2.5;
    pointer-events: none;
  }
  .legend {
    display: flex;
    gap: 1.5rem;
    align-items: center;
    padding: 0.25rem 0.25rem;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    flex-wrap: wrap;
    max-width: 600px;
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
    border: 1px solid var(--rule);
  }
  .grad-r {
    background: linear-gradient(to right, #ef4444, #ffffff, #06b6d4);
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
    background: var(--rule);
  }
  .hint {
    color: var(--ink-faint);
    font-style: italic;
  }
</style>
