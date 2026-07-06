<script lang="ts">
  // Cloned from PrevalenceHistogram.svelte (read that file's history for the
  // rationale behind the bar/suppressed-bin rendering approach). This variant
  // plots a phenotype's predictive-gain "prominence" distribution: the
  // per-patient held-out predictive gain (nats), binned per
  // `predictive_gain.prominence_bin_edges` — NOT the [0,1] theta-mixture
  // scale PrevalenceHistogram uses. Differences from PrevalenceHistogram:
  //   - x-axis domain is data-driven from the FULL bin-edge range (no
  //     tau-anchored start, no percentile-based crop: nats can be negative
  //     and there is no tau concept in this space).
  //   - no `percentiles`/`tau` props; every bin is visible.
  import { copy } from './copy'

  export let histogram: (number | null)[]
  export let binEdges: number[]
  export let width: number = 360
  export let height: number = 120

  const PAD_TOP    = 8    // small breathing room above the tallest bar
  const PAD_BOTTOM = 30   // x-axis tick labels + axis title
  const PAD_LEFT   = 34   // y-axis tick labels + rotated axis title
  $: chartW = width - PAD_LEFT
  $: chartH = height - PAD_TOP - PAD_BOTTOM

  // Hover state — track which bar is hovered for fill-opacity toggle
  let hoveredBin: number | null = null

  // ── Data-driven x-range ─────────────────────────────────────────────────
  // The full span of the shared bin edges, whatever it is (nats can be
  // negative) — never a hardcoded range.
  $: xMin = binEdges.length ? binEdges[0] : 0
  $: xMax = binEdges.length ? binEdges[binEdges.length - 1] : 1

  // Map a value in [xMin, xMax] to SVG x coordinate (offset by PAD_LEFT)
  $: xScale = (v: number) => PAD_LEFT + ((v - xMin) / (xMax - xMin || 1)) * chartW

  // Every bin is visible (no percentile-based crop for the nats scale).
  $: bins = histogram.map((val, i) => {
    const lo = binEdges[i]
    const hi = binEdges[i + 1]
    return { i, val, lo, hi }
  })

  // ── Y-scale ─────────────────────────────────────────────────────────────
  $: visibleValues = bins
    .map((b) => b.val)
    .filter((v): v is number => v != null && v > 0)
  $: yMax = visibleValues.length ? Math.max(...visibleValues) : 1e-4

  // ── Bar geometry ──────────────────────────────────────────────────────────
  $: binPixW = binEdges.length >= 2
    ? xScale(binEdges[1]) - xScale(binEdges[0])
    : 4

  function barHeight(val: number | null): number {
    if (val === null || val <= 0) return 0
    return Math.min(val / yMax, 1) * chartH
  }

  // ── X-axis ticks ──────────────────────────────────────────────────────────
  $: xTickValues = [xMin, xMin + (xMax - xMin) / 4, xMin + (xMax - xMin) / 2, xMin + ((xMax - xMin) * 3) / 4, xMax]
  $: xTicks = (() => {
    const seen = new Set<string>()
    return xTickValues.map((v) => {
      const label = v.toFixed(2)
      const show = !seen.has(label)
      seen.add(label)
      return { x: xScale(v), label, show }
    })
  })()

  // ── Y-axis ticks ──────────────────────────────────────────────────────────
  $: yTickValues = [0, yMax / 2, yMax]
  $: yTicks = (() => {
    const seen = new Set<string>()
    return yTickValues.map((v) => {
      const label = Math.round(v * 100) + '%'
      const show = !seen.has(label)
      seen.add(label)
      const svgY = PAD_TOP + chartH - (v / yMax) * chartH
      return { y: svgY, label, show }
    })
  })()
</script>

<svg
  viewBox="0 0 {width} {height}"
  width={width}
  height={height}
  aria-label={copy.prominenceHistogram.ariaLabel}
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
  >{copy.prominenceHistogram.axisY}</text>

  <!-- Y-axis ticks and labels (left margin) -->
  {#each yTicks as { y, label, show }}
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
  >{copy.prominenceHistogram.axisX}</text>

  <!-- Bars -->
  {#each bins as { i, val, lo, hi }}
    {@const bx = xScale(lo)}
    {@const bh = barHeight(val)}
    {@const by = PAD_TOP + chartH - bh}
    {@const isHovered = hoveredBin === i}
    {@const barW = Math.max(binPixW - 0.5, 1)}

    <!-- Null bins still get a full-height transparent hit-target so the
         tooltip is reachable; non-null bins get the visible colored bar. -->
    {#if val === null}
      <rect
        x={bx} y={PAD_TOP}
        width={barW} height={chartH}
        fill="var(--accent)"
        fill-opacity="0"
        on:mouseenter={() => (hoveredBin = i)}
        on:mouseleave={() => (hoveredBin = null)}
      >
        <title>{copy.prominenceHistogram.suppressedTip}</title>
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
        <title>{'Patients with predictive gain [' + lo.toFixed(3) + ', ' + hi.toFixed(3) + ') nats: ' + (val * 100).toFixed(1) + '%'}</title>
      </rect>
    {/if}
  {/each}
</svg>

<style>
  svg {
    display: block;
    overflow: visible;
  }
</style>
