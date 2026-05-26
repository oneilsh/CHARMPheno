<script lang="ts">
  import type { ThetaPercentiles } from './types'

  export let histogram: (number | null)[]
  export let binEdges: number[]
  export let percentiles: ThetaPercentiles
  export let tau: number
  export let width: number = 360
  export let height: number = 100

  const PAD_TOP    = 18   // room for overflow labels above bars + τ label
  const PAD_BOTTOM = 14   // room for x-axis tick labels
  const PAD_LEFT   = 22   // room for y-axis tick labels
  $: chartW = width - PAD_LEFT
  $: chartH = height - PAD_TOP - PAD_BOTTOM

  // Hover state — track which bar is hovered for fill-opacity toggle
  let hoveredBin: number | null = null

  // ── Adaptive x-range ──────────────────────────────────────────────────────
  // Clip to [0, p95 + some breathing room] so rare phenotypes don't squish
  // all mass against the left edge.  Floor at 0.20 so we always show at least
  // ~10 of the 50 bins — enough shape detail for concentrated topics.
  $: xMax = Math.min(1.0, Math.max(0.20, percentiles.p95 + Math.max(percentiles.p95 * 0.15, 0.02)))
  $: xMin = 0

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

  // ── Tail-focused y-scale ──────────────────────────────────────────────────
  // The leading (leftmost visible) bin is the "patients without the phenotype"
  // spike and should not drive the y-axis scale.  Compute yMax from the tail
  // bins only, so the clinically interesting shape is visible.
  $: visibleIndices = visibleBins.filter((b) => b.inRange).map((b) => b.i)
  $: tailIndices = visibleIndices.slice(1)   // drop the leading bin
  $: tailValues = tailIndices
    .map((i) => histogram[i])
    .filter((v): v is number => v != null && v > 0)
  $: tailMax = tailValues.length ? Math.max(...tailValues) : 0
  // Guard: if the tail is truly empty, fall back to 5% of the leading bin's
  // value (so the spike renders as 20× the next-bin height, not literally
  // infinite), with a small absolute floor to prevent divide-by-zero.
  $: leadingVal = (visibleIndices.length > 0 ? histogram[visibleIndices[0]] : null) ?? 0
  $: yMax = Math.max(tailMax, leadingVal * 0.05, 1e-4)

  // ── Bar geometry ──────────────────────────────────────────────────────────
  // Bar pixel width = width of one bin in the display domain.
  // Use the first bin width as representative (bins are uniform in local-test).
  $: binPixW = binEdges.length >= 2
    ? xScale(binEdges[1]) - xScale(binEdges[0])
    : 4

  function barHeight(val: number | null): number {
    if (val === null || val <= 0) return 0
    // Clamp to chart height — leading bin may overflow the tail-focused scale
    return Math.min(val / yMax, 1) * chartH
  }

  function isClipped(val: number | null): boolean {
    if (val === null || val <= 0) return false
    return val / yMax > 1
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

  // ── X-axis ticks ──────────────────────────────────────────────────────────
  // 5 evenly-spaced ticks at 0, 1/4, 1/2, 3/4, 1 of xMax.
  $: xTickValues = [0, xMax / 4, xMax / 2, (xMax * 3) / 4, xMax]
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
  // 3 ticks at 0, yMax/2, yMax.
  $: yTickValues = [0, yMax / 2, yMax]
  $: yTicks = (() => {
    const seen = new Set<string>()
    // yMax is a fraction of patients; render as percent
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
  aria-label="Prevalence distribution histogram"
  role="img"
>
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
          <title>{'< 20 patients (suppressed for privacy)'}</title>
        </rect>
      {:else}
        {@const clipped = isClipped(val)}
        <rect
          x={bx} y={by}
          width={barW} height={Math.max(bh, 0)}
          fill="var(--accent)"
          fill-opacity={isHovered ? 1 : 0.7}
          on:mouseenter={() => (hoveredBin = i)}
          on:mouseleave={() => (hoveredBin = null)}
        >
          <title>{'Patients with phenotype prominence [' + lo.toFixed(3) + ', ' + hi.toFixed(3) + '): ' + (val * 100).toFixed(1) + '%' + (clipped ? ' (bar clipped — y-scale set by the tail so other bins are visible)' : '')}</title>
        </rect>

        <!-- Broken-axis zigzag + overflow label for clipped bars -->
        {#if clipped}
          {@const zx0 = bx}
          {@const zx1 = bx + barW}
          {@const zy  = PAD_TOP}           // top of chart area
          {@const amp = 2}                  // zigzag amplitude
          {@const seg = Math.max(barW / 4, 2)}
          <!-- 4-point zigzag across the top of the clipped bar -->
          <polyline
            points="{zx0},{zy}  {zx0 + seg},{zy - amp}  {zx0 + seg * 2},{zy + amp}  {zx0 + seg * 3},{zy - amp}  {zx1},{zy}"
            fill="none"
            stroke="var(--accent)"
            stroke-width="1.5"
            stroke-opacity="0.9"
          />
          <!-- Numeric overflow label above the zigzag -->
          <text
            x={bx + barW / 2} y={zy - 7}
            font-family="var(--font-mono)"
            font-size="10"
            fill="var(--ink-muted)"
            text-anchor="middle"
            dominant-baseline="auto"
          >{Math.round(val * 100) + '%'}</text>
        {/if}
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
