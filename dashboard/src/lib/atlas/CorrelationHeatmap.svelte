<script lang="ts">
  import * as d3 from 'd3'
  import type { Correlation } from '../types'
  import { bundle, selectedPhenotypeId, comparePair } from '../store'
  import { seriateTSPCorr, seriateRect } from './seriation'
  import { copy } from '../copy'

  export let correlation: Correlation
  // When true, clicking a cell sets `comparePair` to its (row, col) phenotype
  // pair (diagonal clicks select the phenotype against itself, a "self-view")
  // instead of the default single-select (`selectedPhenotypeId`) behavior
  // used by the Explore heatmap.
  export let pairSelect = false

  // Diverging R ramp: red (−1) → white (0) → cyan (+1). White midpoint keeps
  // near-zero correlations from muddying the field.
  const rRamp = d3.scaleLinear<string>()
    .domain([-1, 0, 1])
    .range(['#ef4444', '#ffffff', '#06b6d4'])
    .clamp(true)

  const CELL = 20
  const AXIS = 26 // gutter for the single axis title (left rotated, bottom flat)
  const PAD = 10 // top / right breathing room
  const CELL_PX_MAX = 30 // cap the RENDERED cell size so small blocks don't balloon

  $: order = correlation.topic_order // matrix position -> compacted phenotype id
  $: labels = correlation.block_labels
  $: blocks = Array.from(new Set(labels)) // block names, first-appearance order
  $: gating = $bundle?.gating
  $: blockDisplay = (b: string) => (b === 'background' ? 'All' : (gating?.group_labels?.[b] ?? b))
  $: labelById = new Map(
    ($bundle?.phenotypes.phenotypes ?? []).map((p) => [p.id, p.label || `Phenotype ${p.id}`]),
  )
  const nameOf = (mi: number) => labelById.get(order[mi]) ?? `Phenotype ${order[mi]}`

  // Row / column block selection. Default to All × All (the background block,
  // first in first-appearance order) — the broadest cross-phenotype view.
  // Bindable so a host (Compare.svelte) can lift the pickers into its own
  // header while this component still drives the default-init + matrix logic.
  export let rowBlock = ''
  export let colBlock = ''
  // When false, the host renders its own row/col <select>s (bound to the
  // props above) and this component omits its internal pickers.
  export let showBlockPickers = true
  $: if (blocks.length && !blocks.includes(rowBlock)) rowBlock = blocks[0]
  $: if (blocks.length && !blocks.includes(colBlock)) colBlock = blocks[0]

  // Matrix positions belonging to the chosen row / column blocks.
  $: rowIdx = labels.reduce<number[]>((a, l, i) => (l === rowBlock ? (a.push(i), a) : a), [])
  $: colIdx = labels.reduce<number[]>((a, l, i) => (l === colBlock ? (a.push(i), a) : a), [])
  $: nr = rowIdx.length
  $: nc = colIdx.length
  $: symmetric = rowBlock === colBlock

  // Sub-matrix over the selection, then seriate: symmetric (TSP path-length) on
  // the diagonal, rectangular (co-oriented singular vectors) off it.
  $: subR = rowIdx.map((ri) => colIdx.map((ci) => correlation.R[ri][ci]))
  $: seriation = symmetric
    ? (() => {
        const o = seriateTSPCorr(subR)
        return { rowOrder: o, colOrder: o }
      })()
    : seriateRect(subR)
  $: rowOrder = seriation.rowOrder
  $: colOrder = seriation.colOrder

  $: gridW = nc * CELL
  $: gridH = nr * CELL
  $: W = AXIS + gridW + PAD
  $: H = PAD + gridH + AXIS

  // Uniform on-screen cell size across every block: the size the WIDEST block
  // (the most columns — the "All" background) renders at when it fills the card,
  // capped at CELL_PX_MAX. Smaller blocks then use that same px/cell and center
  // (a narrower svg) instead of stretching to full width. cardW is the measured
  // card content-box width; CARD_PAD accounts for its 1rem padding either side.
  let cardW = 0
  const CARD_PAD = 32
  $: maxCols = blocks.length
    ? Math.max(...blocks.map((b) => labels.filter((l) => l === b).length))
    : 1
  $: widestW = AXIS + maxCols * CELL + PAD
  $: cellPx =
    cardW > 0
      ? Math.min(CELL_PX_MAX, ((cardW - CARD_PAD) * CELL) / widestW)
      : CELL_PX_MAX
  $: svgPxW = (W * cellPx) / CELL

  function cellTitle(mr: number, mc: number): string {
    const r = correlation.R[mr][mc]
    const identified = correlation.identified[mr][mc]
    const head = `Row: ${nameOf(mr)}\nCol: ${nameOf(mc)}`
    if (r === null || !identified) return `${head}\nnot measured (no joint support)`
    return `${head}\nR = ${r.toFixed(3)}`
  }

  // Flat cell list keyed by (matrixRow:matrixCol) so the SAME DOM rect persists
  // across reorders and CSS-transitions to its new position (the "slide").
  $: cells = (() => {
    const out: {
      key: string; mr: number; mc: number; x: number; y: number; fill: string; na: boolean; tip: string
    }[] = []
    for (let dr = 0; dr < nr; dr++) {
      const mr = rowIdx[rowOrder[dr]]
      for (let dc = 0; dc < nc; dc++) {
        const mc = colIdx[colOrder[dc]]
        const r = correlation.R[mr][mc]
        const na = r === null || !correlation.identified[mr][mc]
        out.push({
          key: `${mr}:${mc}`,
          mr,
          mc,
          x: AXIS + dc * CELL,
          y: PAD + dr * CELL,
          fill: na ? 'var(--rule)' : rRamp(r as number),
          na,
          tip: cellTitle(mr, mc),
        })
      }
    }
    return out
  })()
  $: allNa = cells.length > 0 && cells.every((c) => c.na)

  // Selected phenotype's display row / column within the current selection.
  $: selMatrix = $selectedPhenotypeId == null ? -1 : order.indexOf($selectedPhenotypeId)
  $: selRowDisp = selMatrix < 0 ? -1 : Array.from({ length: nr }).findIndex((_, dr) => rowIdx[rowOrder[dr]] === selMatrix)
  $: selColDisp = selMatrix < 0 ? -1 : Array.from({ length: nc }).findIndex((_, dc) => colIdx[colOrder[dc]] === selMatrix)

  function selectCol(mc: number) {
    selectedPhenotypeId.set(order[mc])
  }

  function onCellClick(mr: number, mc: number) {
    if (pairSelect) {
      const a = order[mr], b = order[mc]
      // Diagonal (a === b) is a real selection too — DifferencePane renders
      // a self-view (that phenotype's own top conditions) instead of an
      // empty state.
      comparePair.set({ a, b })
    } else {
      selectCol(mc)
    }
  }
</script>

<figure class="heatmap" data-tour="correlation-heatmap">
  {#if showBlockPickers}
    <div class="picker">
      <label>Rows
        <select bind:value={rowBlock}>
          {#each blocks as b}<option value={b}>{blockDisplay(b)}</option>{/each}
        </select>
      </label>
      <span class="times">×</span>
      <label>Columns
        <select bind:value={colBlock}>
          {#each blocks as b}<option value={b}>{blockDisplay(b)}</option>{/each}
        </select>
      </label>
    </div>
  {/if}

  <div class="card" bind:clientWidth={cardW}>
    <svg viewBox="0 0 {W} {H}" role="img" aria-label={copy.correlation.ariaLabel} preserveAspectRatio="xMidYMid meet" style="width: {svgPxW}px">
      <!-- cells -->
      {#each cells as c (c.key)}
        <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
        <rect
          class="cell"
          class:na={c.na}
          class:selected={pairSelect && $comparePair !== null && order[c.mr] === $comparePair.a && order[c.mc] === $comparePair.b}
          data-mr={c.mr}
          data-mc={c.mc}
          width={CELL}
          height={CELL}
          fill={c.fill}
          style="transform: translate({c.x}px, {c.y}px)"
          data-tip={c.tip}
          on:click={() => onCellClick(c.mr, c.mc)}
        />
      {/each}

      <!-- selection outline (row + column of the selected topic) -->
      {#if selRowDisp >= 0}
        <rect class="sel-line" x={AXIS} y={PAD + selRowDisp * CELL} width={gridW} height={CELL} />
      {/if}
      {#if selColDisp >= 0}
        <rect class="sel-line" x={AXIS + selColDisp * CELL} y={PAD} width={CELL} height={gridH} />
      {/if}

      <!-- axis titles (per-cell labels dropped; details live in the hover) -->
      <text
        class="axis-title"
        x={13}
        y={PAD + gridH / 2}
        text-anchor="middle"
        transform="rotate(-90 13 {PAD + gridH / 2})"
      >Phenotypes: {blockDisplay(rowBlock)}</text>
      <text class="axis-title" x={AXIS + gridW / 2} y={H - 7} text-anchor="middle">
        Phenotypes: {blockDisplay(colBlock)}
      </text>

      {#if allNa}
        <text class="na-note" x={AXIS + gridW / 2} y={PAD + gridH / 2} text-anchor="middle">
          not measured under single-label gating
        </text>
      {/if}
    </svg>
  </div>

  <figcaption class="legend">
    <div class="legend-group">
      <span class="eyebrow">Correlation Coefficient (R)</span>
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
    gap: 0.6rem;
  }
  .picker {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-faint);
  }
  .picker label {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }
  .picker select {
    font-family: var(--font-mono);
    font-size: var(--fs-small);
    font-weight: 600;
    text-transform: none;
    letter-spacing: normal;
    padding: 0.35rem 1.75rem 0.35rem 0.6rem;
    border: 1px solid var(--rule-strong);
    background-color: var(--surface);
    /* explicit caret so it unmistakably reads as a dropdown */
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M1 1l4 4 4-4' fill='none' stroke='%2371717a' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 0.6rem center;
    appearance: none;
    -webkit-appearance: none;
    -moz-appearance: none;
    color: var(--ink);
    border-radius: var(--radius-sm);
    box-shadow: 0 1px 0 rgba(0, 0, 0, 0.04);
    cursor: pointer;
  }
  .picker select:hover {
    border-color: var(--ink-muted);
  }
  .picker select:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
  .picker .times {
    color: var(--ink-muted);
  }

  .card {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    padding: 1rem;
    display: flex;
    justify-content: center;
  }
  svg {
    max-width: 100%;
    height: auto;
    display: block;
  }
  .cell {
    cursor: pointer;
    /* slide to the new position on reseriation */
    transition: transform 0.4s ease;
  }
  .cell:hover {
    stroke: var(--ink);
    stroke-width: 1.5;
    paint-order: stroke;
  }
  .cell.selected {
    stroke: var(--accent);
    stroke-width: 2;
    paint-order: stroke;
  }
  .sel-line {
    fill: none;
    stroke: var(--accent);
    stroke-width: 1.5;
    pointer-events: none;
    transition: x 0.4s ease, y 0.4s ease;
  }
  .axis-title {
    font-family: var(--font-mono);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    fill: var(--ink-muted);
  }
  .na-note {
    font-family: var(--font-mono);
    font-size: 11px;
    fill: var(--ink-faint);
    font-style: italic;
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
</style>
