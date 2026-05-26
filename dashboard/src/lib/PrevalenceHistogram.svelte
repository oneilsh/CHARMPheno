<script lang="ts">
  import type { ThetaPercentiles } from './types'

  export let histogram: (number | null)[]
  export let binEdges: number[]
  export let percentiles: ThetaPercentiles
  export let tau: number
  export let width: number = 360
  export let height: number = 80

  const PAD_TOP = 8          // room for τ label and percentile tick marks
  const PAD_BOTTOM = 2
  const chartH = height - PAD_TOP - PAD_BOTTOM

  // Hover state — track which bar is hovered for fill-opacity toggle
  let hoveredBin: number | null = null

  // ── Adaptive x-range ──────────────────────────────────────────────────────
  // Clip to [0, p95 + some breathing room] so rare phenotypes don't squish
  // all mass against the left edge.  When p95 ≈ 0 (truly empty topic) use a
  // fallback minimum so the chart isn't degenerate.
  $: xMax = Math.min(1, Math.max(0.05, percentiles.p95 + Math.max(percentiles.p95 * 0.1, 0.02)))
  $: xMin = 0

  // Map a value in [xMin, xMax] to SVG x coordinate
  $: xScale = (v: number) => ((v - xMin) / (xMax - xMin)) * width

  // ── Visible bins ──────────────────────────────────────────────────────────
  // A bin is "visible" when its midpoint falls within [xMin, xMax].
  $: visibleBins = histogram.map((val, i) => {
    const lo = binEdges[i]
    const hi = binEdges[i + 1]
    const mid = (lo + hi) / 2
    const inRange = mid >= xMin && mid <= xMax
    return { i, val, lo, hi, inRange }
  })

  $: maxVisible = Math.max(
    ...visibleBins.filter((b) => b.inRange && b.val !== null).map((b) => b.val as number),
    1e-9,  // prevent divide-by-zero when all visible bins are null
  )

  // ── Bar geometry ──────────────────────────────────────────────────────────
  // Bar pixel width = width of one bin in the display domain.
  // Use the first bin width as representative (bins are uniform in local-test).
  $: binPixW = binEdges.length >= 2
    ? xScale(binEdges[1]) - xScale(binEdges[0])
    : 4

  function barHeight(val: number | null): number {
    if (val === null || val <= 0) return 0
    return (val / maxVisible) * chartH
  }

  // ── τ in range? ───────────────────────────────────────────────────────────
  $: tauInRange = tau >= xMin && tau <= xMax
  $: tauX = xScale(tau)

  // ── Percentile lines in range ─────────────────────────────────────────────
  $: p50InRange = percentiles.p50 >= xMin && percentiles.p50 <= xMax
  $: p50X = xScale(percentiles.p50)

  // Tick marks: p5, p25, p75, p95 — small downward marks at the top
  $: pTicks = (['p5', 'p25', 'p75', 'p95'] as const)
    .map((k) => ({ k, v: percentiles[k] }))
    .filter(({ v }) => v >= xMin && v <= xMax)
    .map(({ k, v }) => ({ k, x: xScale(v) }))
</script>

<svg
  viewBox="0 0 {width} {height}"
  width={width}
  height={height}
  aria-label="Prevalence distribution histogram"
  role="img"
>
  <!-- Bars -->
  {#each visibleBins as { i, val, lo, hi, inRange }}
    {#if inRange}
      {@const bx = xScale(lo)}
      {@const bh = barHeight(val)}
      {@const by = PAD_TOP + chartH - bh}
      {@const isHovered = hoveredBin === i}

      <!-- Null bins still get a full-height transparent hit-target so the
           tooltip is reachable; non-null bins get the visible colored bar. -->
      {#if val === null}
        <!-- Transparent hit area — hover-invisible, tooltip fires on hover -->
        <rect
          x={bx} y={PAD_TOP}
          width={Math.max(binPixW - 0.5, 1)} height={chartH}
          fill="var(--accent)"
          fill-opacity="0"
          on:mouseenter={() => (hoveredBin = i)}
          on:mouseleave={() => (hoveredBin = null)}
        >
          <title>{'< 20 patients (suppressed for privacy)'}</title>
        </rect>
      {:else}
        <rect
          x={bx} y={by}
          width={Math.max(binPixW - 0.5, 1)} height={Math.max(bh, 0)}
          fill="var(--accent)"
          fill-opacity={isHovered ? 1 : 0.7}
          on:mouseenter={() => (hoveredBin = i)}
          on:mouseleave={() => (hoveredBin = null)}
        >
          <title>{'Patients with θ in [' + lo.toFixed(3) + ', ' + hi.toFixed(3) + '): ' + (val * 100).toFixed(1) + '%'}</title>
        </rect>
      {/if}
    {/if}
  {/each}

  <!-- p50 — solid faint vertical line through the full chart area -->
  {#if p50InRange}
    <line
      x1={p50X} y1={PAD_TOP}
      x2={p50X} y2={PAD_TOP + chartH}
      stroke="var(--ink-faint)"
      stroke-width="1"
      stroke-opacity="0.6"
    />
  {/if}

  <!-- p5/p25/p75/p95 — small downward tick marks at the top edge -->
  {#each pTicks as { x }}
    <line
      x1={x} y1={PAD_TOP}
      x2={x} y2={PAD_TOP + 5}
      stroke="var(--ink-faint)"
      stroke-width="1"
      stroke-opacity="0.6"
    />
  {/each}

  <!-- τ marker — dashed fuchsia line + small label -->
  {#if tauInRange}
    <line
      x1={tauX} y1={PAD_TOP}
      x2={tauX} y2={PAD_TOP + chartH}
      stroke="#d946ef"
      stroke-width="1.5"
      stroke-dasharray="3,2"
    />
    <!-- τ label sits just above the top padding line -->
    <text
      x={tauX + 3} y={PAD_TOP - 1}
      font-family="var(--font-mono)"
      font-size="var(--fs-micro)"
      fill="#d946ef"
      dominant-baseline="auto"
    >τ</text>
  {/if}
</svg>

<style>
  svg {
    display: block;
    overflow: visible;
  }
</style>
