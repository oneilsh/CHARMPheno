<script lang="ts">
  import {
    bundle, advancedView, phenotypesById, phenotypeOrder,
  } from '../store'
  import { phenotypeHue } from '../palette'
  import { displayedDominant } from '../dominant'

  // N theta vectors from the simulator. Each becomes one vertical column
  // in a STRUCTURE-style plot (population genetics): each column shows
  // its phenotype mix as a stack of colored segments summing to 1. With
  // columns sorted by dominant phenotype, similar-looking samples clump
  // together visually - the same kind of read you get from a STRUCTURE
  // plot, where a tightly-inferred prior produces a near-uniform color
  // block and an ambiguous prior produces a rainbow.
  export let thetaSamples: number[][] = []

  // Threshold below which a phenotype is folded into "Other" in basic
  // mode. Matches ProfileBar's default so the structure plot and the
  // profile bar above it agree on what counts as a labeled band.
  const OTHER_THRESHOLD = 0.01

  // SVG geometry. Width is responsive (viewBox-based); columns auto-size
  // to fill the width based on sample count.
  const W = 720
  const H = 120
  const TOP_PAD = 4
  const BOT_PAD = 4

  // Phenotypes that should be visually "hidden" in basic mode. Their
  // mass folds into a striped Other segment drawn at the bottom of each
  // column so it never names a dead/mixed phenotype.
  $: badSet = (() => {
    if ($advancedView) return new Set<number>()
    const s = new Set<number>()
    for (const p of $bundle?.phenotypes.phenotypes ?? []) {
      if (p.quality === 'dead' || p.quality === 'mixed') s.add(p.id)
    }
    return s
  })()

  // Order phenotypes by the JSD-similarity walk so adjacent topics in
  // the stack are similar - colors transition smoothly within a column
  // (and stripes of similar colors across columns read as related).
  $: stackOrder = (() => {
    const ord = $phenotypeOrder
    if (ord.length === 0) return [] as number[]
    return ord.filter((k) => !badSet.has(k))
  })()

  // Seriate by similarity-order-of-dominant. phenotypeOrder is the
  // JSD-similarity walk through all K phenotypes used by the palette
  // (similar phenotypes adjacent), so sorting samples primarily by
  // their displayedDominant's position in that order means adjacent
  // sample blocks have related dominants and the strip reads as a
  // smooth color gradient rather than blocks-by-id. Secondary sort by
  // dominant weight desc puts the most-confident samples at the head
  // of each block. Tertiary by second-largest phenotype's position so
  // within a block, secondary mixtures also progress smoothly. Cost
  // is O(N log N) plus O(N*K) for the secondary key - trivial.

  // Position of each phenotype in the JSD-similarity walk. Used as the
  // primary sort key so adjacent sample-blocks share similar dominants.
  $: phenoRank = (() => {
    const r = new Map<number, number>()
    $phenotypeOrder.forEach((k, i) => r.set(k, i))
    return r
  })()

  // Decorate each sample (for hover tooltip + per-column rendering),
  // then sort by (rank of dominant, dominant weight desc, rank of 2nd).
  // Pre-computing per-segment heights here keeps the render path to
  // plain rect-drawing.
  $: columns = (() => {
    if (thetaSamples.length === 0 || !$bundle) return []
    const phenotypes = $bundle.phenotypes.phenotypes
    const advanced = $advancedView
    const decorated = thetaSamples.map((theta, i) => {
      const dom = displayedDominant(theta, phenotypes, advanced)
      const domW = theta[dom] ?? 0
      // Find second-largest displayed component for tertiary sort.
      let second = -1, secondV = -1
      for (let k = 0; k < theta.length; k++) {
        if (k === dom) continue
        if (!advanced && badSet.has(k)) continue
        const v = theta[k]
        if (v > secondV) { secondV = v; second = k }
      }
      // Sum mass that lands in the Other band (dead/mixed in basic,
      // anything below threshold in either mode).
      let otherSum = 0
      const segs: { k: number; h: number }[] = []
      for (const k of stackOrder) {
        const v = theta[k] ?? 0
        if (v < OTHER_THRESHOLD) { otherSum += v; continue }
        segs.push({ k, h: v })
      }
      // Bad-quality phenotypes (basic only) also roll into otherSum.
      if (!advanced) {
        for (const k of badSet) otherSum += theta[k] ?? 0
      }
      const domRank = phenoRank.get(dom) ?? 0
      const secondRank = second >= 0 ? (phenoRank.get(second) ?? 0) : 0
      return { i, theta, dom, domW, segs, otherSum, domRank, secondRank }
    })
    decorated.sort((a, b) =>
      (a.domRank - b.domRank)
        || (b.domW - a.domW)
        || (a.secondRank - b.secondRank)
    )
    return decorated
  })()

  $: colW = columns.length > 0 ? W / columns.length : 0

  // Hover tooltip - which column is the user pointing at and what
  // does that sample look like.
  let hoverIdx: number | null = null
  $: hoverCol = hoverIdx !== null ? columns[hoverIdx] : null

  function onMove(e: MouseEvent) {
    const svg = e.currentTarget as SVGSVGElement
    const rect = svg.getBoundingClientRect()
    const xPx = e.clientX - rect.left
    const xSvg = (xPx / rect.width) * W
    const idx = Math.floor(xSvg / colW)
    hoverIdx = (idx >= 0 && idx < columns.length) ? idx : null
  }
  function onLeave() { hoverIdx = null }
</script>

<section class="structure">
  <header>
    <span class="eyebrow">Per-sample mix</span>
    <h4>How confident is the model?</h4>
    <p class="sub">
      Each column is one simulated patient. A solid color block means the
      model agrees with itself across draws; a rainbow means the
      starting conditions are consistent with several phenotype mixes.
    </p>
  </header>

  {#if columns.length === 0}
    <p class="hint">Run the simulator to see the per-sample distribution.</p>
  {:else}
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      role="img"
      aria-label="Per-sample phenotype structure"
      on:mousemove={onMove}
      on:mouseleave={onLeave}
    >
      <defs>
        <!-- Striped pattern for the Other segment: same vocabulary as
             ProfileBar's other-band so the link reads. -->
        <pattern id="sim-structure-other" patternUnits="userSpaceOnUse" width="4" height="4" patternTransform="rotate(45)">
          <rect width="4" height="4" fill="var(--surface-deep)" />
          <rect width="1" height="4" fill="rgba(82, 82, 91, 0.28)" />
        </pattern>
      </defs>

      {#each columns as col, ci}
        {@const x = ci * colW}
        {@const yScale = H - TOP_PAD - BOT_PAD}
        {@const yBase = H - BOT_PAD}
        <!-- Bad-quality / below-threshold mass at bottom -->
        {#if col.otherSum > 0}
          <rect
            x={x}
            y={yBase - col.otherSum * yScale}
            width={Math.max(0.5, colW + 0.5)}
            height={col.otherSum * yScale}
            fill="url(#sim-structure-other)"
          />
        {/if}
        {#each col.segs as seg, si}
          {@const yOffset = (col.otherSum + col.segs.slice(0, si).reduce((s, x) => s + x.h, 0)) * yScale}
          <rect
            x={x}
            y={yBase - yOffset - seg.h * yScale}
            width={Math.max(0.5, colW + 0.5)}
            height={seg.h * yScale}
            fill={$phenotypeHue(seg.k)}
          />
        {/each}
      {/each}

      <!-- Hover indicator: vertical line + faint outline on the column -->
      {#if hoverIdx !== null}
        <rect
          x={hoverIdx * colW}
          y={TOP_PAD}
          width={Math.max(1, colW)}
          height={H - TOP_PAD - BOT_PAD}
          fill="none"
          stroke="var(--ink)"
          stroke-width="1"
          pointer-events="none"
        />
      {/if}
    </svg>

    {#if hoverCol}
      <div class="tooltip">
        <span class="tip-dot" style="background: {$phenotypeHue(hoverCol.dom)}" aria-hidden="true"></span>
        Sample {hoverCol.i + 1} of {columns.length}: dominant
        <strong>{$phenotypesById.get(hoverCol.dom)?.label || `Phenotype ${hoverCol.dom}`}</strong>
        at {(hoverCol.domW * 100).toFixed(0)}%
      </div>
    {:else}
      <div class="tooltip placeholder">Hover a column to inspect one simulated patient.</div>
    {/if}
  {/if}
</section>

<style>
  .structure {
    padding: 1.25rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
  }
  header {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    margin-bottom: 0.85rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid var(--rule);
  }
  header h4 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: var(--tracking-tight);
  }
  .sub {
    margin: 0.2rem 0 0;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    font-style: italic;
    line-height: 1.5;
  }
  svg {
    width: 100%;
    height: 120px;
    display: block;
    background: var(--surface-recessed);
    border-radius: var(--radius-sm);
    cursor: crosshair;
  }
  .tooltip {
    margin-top: 0.55rem;
    font-size: var(--fs-small);
    color: var(--ink);
    display: flex;
    align-items: center;
    gap: 0.45rem;
    min-height: 1.2rem;
  }
  .tooltip.placeholder {
    color: var(--ink-faint);
    font-style: italic;
  }
  .tip-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .tooltip strong { color: var(--ink); font-weight: 600; }
  .hint {
    color: var(--ink-faint);
    font-size: var(--fs-small);
    margin: 0;
    font-style: italic;
  }
</style>
