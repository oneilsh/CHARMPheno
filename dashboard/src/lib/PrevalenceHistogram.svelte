<script lang="ts">
  import type { ThetaPercentiles } from './types'
  import { copy } from './copy'

  export let histogram: (number | null)[]
  export let binEdges: number[]
  export let percentiles: ThetaPercentiles
  export let tau: number
  export let width: number = 360
  export let height: number = 120

  const PAD_TOP    = 8    // small breathing room above the tallest bar
  const PAD_BOTTOM = 30   // x-axis tick labels + axis title
  const PAD_LEFT   = 34   // y-axis tick labels + rotated axis title
  $: chartW = width - PAD_LEFT
  $: chartH = height - PAD_TOP - PAD_BOTTOM

  // Hover state — track which bar is hovered for fill-opacity toggle
  let hoveredBin: number | null = null

  // ── Adaptive x-range ──────────────────────────────────────────────────────
  // Start the axis at τ: the [0, τ) bin holds the "patients without the
  // phenotype" mass, which is summarised separately by the "< τ" label rather
  // than drawn as a giant spike that flattens everything else.  The upper edge
  // tracks p95 (+ breathing room), floored at 0.20 so concentrated topics
  // still show enough bins to read their shape.
  $: xMin = tau
  $: xMax = Math.min(1.0, Math.max(0.20, percentiles.p95 + Math.max(percentiles.p95 * 0.15, 0.02)))

  // Map a value in [xMin, xMax] to SVG x coordinate (offset by PAD_LEFT)
  $: xScale = (v: number) => PAD_LEFT + ((v - xMin) / (xMax - xMin)) * chartW

  // ── Visible bins ──────────────────────────────────────────────────────────
  // A bin is "visible" when its midpoint falls within [xMin, xMax].
  $: visibleBins = histogram.map((val, i) => {
    const lo = binEdges[i]
    const hi = binEdges[i + 1]
    const mid = (lo + hi) / 2
    const inRange = mid >= xMin && mid <= xMax
    return { i, val, lo, hi, inRange }
  })

  // ── Y-scale ─────────────────────────────────────────────────────────────
  // With the sub-τ spike excluded from the x-range, the visible bins are on a
  // comparable scale, so the y-axis is just driven by the tallest visible bar.
  $: visibleValues = visibleBins
    .filter((b) => b.inRange)
    .map((b) => b.val)
    .filter((v): v is number => v != null && v > 0)
  $: yMax = visibleValues.length ? Math.max(...visibleValues) : 1e-4

  // ── Bar geometry ──────────────────────────────────────────────────────────
  // Bar pixel width = width of one bin in the display domain.
  // Use the first bin width as representative (bins are uniform in local-test).
  $: binPixW = binEdges.length >= 2
    ? xScale(binEdges[1]) - xScale(binEdges[0])
    : 4

  function barHeight(val: number | null): number {
    if (val === null || val <= 0) return 0
    return Math.min(val / yMax, 1) * chartH
  }

  // ── X-axis ticks ──────────────────────────────────────────────────────────
  // 5 evenly-spaced ticks from xMin to xMax.
  $: xTickValues = [xMin, xMin + (xMax - xMin) / 4, xMin + (xMax - xMin) / 2, xMin + ((xMax - xMin) * 3) / 4, xMax]
  $: xTicks = (() => {
    const seen = new Set<string>()
    return xTickValues.map((v) => {
      const label = Math.round(v * 100) + '%'
      const show = !seen.has(label)
      seen.add(label)
      return { x: xScale(v), label, show }
    })
  })()

  // ── Y-axis ticks ──────────────────────────────────────────────────────────
  // 3 ticks at 0, yMax/2, yMax (rendered as percent of patients).
  $: yTickValues = [0, yMax / 2, yMax]
  $: yTicks = (() => {
    const seen = new Set<string>()
    return yTickValues.map((v) => {
      const label = Math.round(v * 100) + '%'
      const show = !seen.has(label)
      seen.add(label)
      // y=0 fraction → bottom of chart; y=yMax → top of chart
      const svgY = PAD_TOP + chartH - (v / yMax) * chartH
      return { y: svgY, label, show }
    })
  })()
</script>

<svg
  viewBox="0 0 {width} {height}"
  width={width}
  height={height}
  aria-label={copy.histogram.ariaLabel}
  role="img"
>
  <!-- Y-axis title (rotated, far left) -->
  <text
    x={8} y={PAD_TOP + chartH / 2}
    font-family="var(--font-mono)"
    font-size="8"
    fill="var(--ink-faint)"
    text-anchor="middle"
    transform="rotate(-90 8 {PAD_TOP + chartH / 2})"
  >{copy.histogram.axisY}</text>

  <!-- Y-axis ticks and labels (left margin) -->
  {#each yTicks as { y, label, show }}
    <!-- Short horizontal tick projecting left from the chart area -->
    <line
      x1={PAD_LEFT - 3} y1={y}
      x2={PAD_LEFT}     y2={y}
      stroke="var(--ink-faint)"
      stroke-width="0.75"
    />
    {#if show}
      <text
        x={PAD_LEFT - 5} y={y}
        font-family="var(--font-mono)"
        font-size="8"
        fill="var(--ink-faint)"
        text-anchor="end"
        dominant-baseline="middle"
      >{label}</text>
    {/if}
  {/each}

  <!-- X-axis baseline -->
  <line
    x1={PAD_LEFT} y1={PAD_TOP + chartH}
    x2={PAD_LEFT + chartW} y2={PAD_TOP + chartH}
    stroke="var(--ink-faint)"
    stroke-width="0.5"
  />

  <!-- X-axis ticks and labels (below baseline) -->
  {#each xTicks as { x, label, show }}
    <!-- Short vertical tick dropping below the baseline -->
    <line
      x1={x} y1={PAD_TOP + chartH}
      x2={x} y2={PAD_TOP + chartH + 3}
      stroke="var(--ink-faint)"
      stroke-width="0.75"
    />
    {#if show}
      <text
        x={x} y={PAD_TOP + chartH + 5}
        font-family="var(--font-mono)"
        font-size="8"
        fill="var(--ink-faint)"
        text-anchor="middle"
        dominant-baseline="hanging"
      >{label}</text>
    {/if}
  {/each}

  <!-- X-axis title -->
  <text
    x={PAD_LEFT + chartW / 2} y={height - 2}
    font-family="var(--font-mono)"
    font-size="8"
    fill="var(--ink-faint)"
    text-anchor="middle"
    dominant-baseline="auto"
  >{copy.histogram.axisX}</text>

  <!-- Bars -->
  {#each visibleBins as { i, val, lo, hi, inRange }}
    {#if inRange}
      {@const bx = xScale(lo)}
      {@const bh = barHeight(val)}
      {@const by = PAD_TOP + chartH - bh}
      {@const isHovered = hoveredBin === i}
      {@const barW = Math.max(binPixW - 0.5, 1)}

      <!-- Null bins still get a full-height transparent hit-target so the
           tooltip is reachable; non-null bins get the visible colored bar. -->
      {#if val === null}
        <!-- Transparent hit area — hover-invisible, tooltip fires on hover -->
        <rect
          x={bx} y={PAD_TOP}
          width={barW} height={chartH}
          fill="var(--accent)"
          fill-opacity="0"
          on:mouseenter={() => (hoveredBin = i)}
          on:mouseleave={() => (hoveredBin = null)}
        >
          <title>{copy.histogram.suppressedTip}</title>
        </rect>
      {:else}
        <rect
          x={bx} y={by}
          width={barW} height={Math.max(bh, 0)}
          fill="var(--accent)"
          fill-opacity={isHovered ? 1 : 0.7}
          on:mouseenter={() => (hoveredBin = i)}
          on:mouseleave={() => (hoveredBin = null)}
        >
          <title>{'Patients with phenotype prominence [' + lo.toFixed(3) + ', ' + hi.toFixed(3) + '): ' + (val * 100).toFixed(1) + '%'}</title>
        </rect>
      {/if}
    {/if}
  {/each}
</svg>

<style>
  svg {
    display: block;
    overflow: visible;
  }
</style>
