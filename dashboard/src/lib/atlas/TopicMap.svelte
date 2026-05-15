<script lang="ts">
  import { onMount } from 'svelte'
  import * as d3 from 'd3'
  import { bundle, selectedPhenotypeId, colorMode, hoveredCodeIdx, advancedView } from '../store'
  import { computeJsdMds } from '../mds'

  // Simple mode hides `dead` and `mixed` topics. `null` (unlabeled bundle)
  // and the other three categories all show. Advanced mode shows everything.
  function isHiddenInSimpleMode(p: { quality: string | null }): boolean {
    return p.quality === 'dead' || p.quality === 'mixed'
  }

  function phenotypesWithCode(idx: number | null, n = 20): Set<number> {
    if (!$bundle || idx === null) return new Set()
    const out = new Set<number>()
    const K = $bundle.model.K
    for (let k = 0; k < K; k++) {
      const row = $bundle.model.beta[k]
      const top = row.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, n)
      if (top.some((s) => s.i === idx)) out.add(k)
    }
    return out
  }

  $: highlighted = phenotypesWithCode($hoveredCodeIdx)

  let svgEl: SVGSVGElement
  const W = 720, H = 560, MARGIN = 24

  let coords: number[][] = []
  $: if ($bundle && coords.length !== $bundle.model.K) {
    coords = computeJsdMds($bundle.model.beta)
  }

  // Diverging NPMI ramp: red (low) → neutral gray → cyan (high).
  // Aligns with the global accent and avoids the rainbow look.
  const npmiRamp = d3.scaleLinear<string>()
    .domain([-0.2, 0, 0.2, 0.4])
    .range(['#ef4444', '#d4d4d8', '#67e8f9', '#06b6d4'])
    .clamp(true)

  // Sequential prevalence ramp: pale sky → deep cyan.
  const prevRamp = d3.scaleSequential<string>(
    (t) => d3.interpolateRgb('#f0fdfa', '#0e7490')(t),
  )

  function render() {
    if (!$bundle || !svgEl || coords.length === 0) return
    // Mode-aware filter: simple hides dead+mixed; advanced shows all.
    // x/y/r scales are computed against the FULL set so the layout is
    // stable across mode toggles (we don't want bubbles jumping around).
    const allPhenotypes = $bundle.phenotypes.phenotypes
    const phenotypes = $advancedView
      ? allPhenotypes
      : allPhenotypes.filter((p) => !isHiddenInSimpleMode(p))
    const xExt = d3.extent(coords, (c) => c[0]) as [number, number]
    const yExt = d3.extent(coords, (c) => c[1]) as [number, number]
    const x = d3.scaleLinear().domain(xExt).range([MARGIN, W - MARGIN])
    const y = d3.scaleLinear().domain(yExt).range([H - MARGIN, MARGIN])
    // Use the FULL phenotype set for the prevalence scale domain so bubble
    // size doesn't rescale between simple and advanced modes.
    const r = d3.scaleSqrt()
      .domain(d3.extent(allPhenotypes, (p) => p.corpus_prevalence) as [number, number])
      .range([5, 26])

    const prevExt = d3.extent(allPhenotypes, (p) => p.corpus_prevalence) as [number, number]
    prevRamp.domain(prevExt)
    const colorFn = (p: typeof phenotypes[0]) =>
      $colorMode === 'prevalence' ? prevRamp(p.corpus_prevalence) : npmiRamp(p.npmi)

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()
    svg.attr('viewBox', `0 0 ${W} ${H}`)

    const g = svg.append('g')

    // Solid-fill bubbles with thin border. Cleaner than the previous
    // ring-style; the encoding is in the fill, not the ring.
    const nodes = g.selectAll('g.node')
      .data(phenotypes)
      .join('g')
      .attr('class', 'node')
      .attr('transform', (p) => `translate(${x(coords[p.id][0])}, ${y(coords[p.id][1])})`)
      .style('cursor', 'pointer')
      .on('click', (_, p) => selectedPhenotypeId.set(p.id))

    // Main bubble — filled with the encoded color, thin ink-tinted border
    nodes.append('circle')
      .attr('r', (p) => r(p.corpus_prevalence))
      .attr('fill', (p) => colorFn(p))
      .attr('fill-opacity', 0.85)
      .attr('stroke', '#18181b')
      .attr('stroke-opacity', 0.25)
      .attr('stroke-width', 0.75)

    // Selection ring — cyan accent
    nodes.append('circle')
      .attr('r', (p) => r(p.corpus_prevalence) + 3)
      .attr('fill', 'none')
      .attr('stroke', '#06b6d4')
      .attr('stroke-width', (p) => ($selectedPhenotypeId === p.id ? 2 : 0))

    // Hover-highlight ring — same color, dashed, when a code is hovered
    nodes.append('circle')
      .attr('r', (p) => r(p.corpus_prevalence) + 5)
      .attr('fill', 'none')
      .attr('stroke', '#06b6d4')
      .attr('stroke-dasharray', '3,2')
      .attr('stroke-width', (p) => (highlighted.has(p.id) ? 1.25 : 0))

    // Quality indicator (advanced mode only) — small colored dot on the
    // bubble's edge for dead/mixed topics. Simple mode hides these
    // topics entirely so no marker is needed there.
    const qualityMarkColor: Record<string, string> = {
      dead: '#ef4444',   // red
      mixed: '#f59e0b',  // amber
    }
    if ($advancedView) {
      g.selectAll('circle.quality-mark')
        .data(phenotypes.filter((p) => p.quality && qualityMarkColor[p.quality]))
        .join('circle')
        .attr('class', 'quality-mark')
        .attr('cx', (p) => x(coords[p.id][0]) + r(p.corpus_prevalence) * 0.7)
        .attr('cy', (p) => y(coords[p.id][1]) - r(p.corpus_prevalence) * 0.7)
        .attr('r', 2.5)
        .attr('fill', (p) => qualityMarkColor[p.quality!])
        .attr('stroke', '#fff')
        .attr('stroke-width', 1)
    }

    // Label for the selected phenotype above its bubble
    if ($selectedPhenotypeId !== null) {
      const sel = phenotypes[$selectedPhenotypeId]
      if (sel) {
        const cx = x(coords[sel.id][0])
        const cy = y(coords[sel.id][1])
        const rr = r(sel.corpus_prevalence)
        const labelText = sel.label || `Phenotype ${sel.id}`
        const labelW = Math.max(8 * labelText.length, 60)
        // Pill background
        g.append('rect')
          .attr('x', cx - labelW / 2)
          .attr('y', cy - rr - 26)
          .attr('rx', 3).attr('ry', 3)
          .attr('width', labelW)
          .attr('height', 18)
          .attr('fill', '#0a0a0a')
        g.append('text')
          .attr('x', cx).attr('y', cy - rr - 13)
          .attr('text-anchor', 'middle')
          .attr('font-family', 'Geist, sans-serif')
          .attr('font-size', 11)
          .attr('font-weight', 500)
          .attr('fill', '#fff')
          .text(labelText)
      }
    }

    nodes.append('title')
      .text((p) => `${p.label || `Phenotype ${p.id}`}\nNPMI ${p.npmi.toFixed(3)} · prev ${(p.corpus_prevalence * 100).toFixed(1)}%`)
  }

  $: $colorMode, $selectedPhenotypeId, $hoveredCodeIdx, $advancedView, $bundle && svgEl && coords.length && render()
  onMount(render)
</script>

<figure class="map">
  <svg bind:this={svgEl} role="img" aria-label="Phenotype atlas" preserveAspectRatio="xMidYMid meet"></svg>
  <figcaption class="legend">
    {#if $bundle}
      <div class="legend-group">
        {#if $colorMode === 'prevalence'}
          <span class="eyebrow">Color · prevalence</span>
          <span class="grad grad-prev" aria-hidden="true"></span>
        {:else}
          <span class="eyebrow">Color · NPMI</span>
          <span class="grad grad-npmi" aria-hidden="true"></span>
          <span class="ticks" data-numeric><span>−0.2</span><span>0</span><span>+0.4</span></span>
        {/if}
      </div>
      <div class="legend-group">
        <span class="eyebrow">Size · prevalence</span>
        <span class="size-marks" aria-hidden="true">
          <span class="dot s1"></span><span class="dot s2"></span><span class="dot s3"></span>
        </span>
      </div>
    {/if}
  </figcaption>
</figure>

<style>
  .map {
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  svg {
    width: 100%;
    height: auto;
    display: block;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
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
  .grad {
    display: inline-block;
    width: 96px;
    height: 6px;
    border-radius: 3px;
  }
  .grad-prev {
    background: linear-gradient(to right, #f0fdfa, #0e7490);
  }
  .grad-npmi {
    background: linear-gradient(to right, #ef4444, #d4d4d8, #06b6d4);
  }
  .ticks {
    display: inline-flex;
    gap: 0.4rem;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
  }
  .size-marks {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .size-marks .dot {
    border-radius: 50%;
    background: var(--rule-strong);
  }
  .size-marks .s1 { width: 5px; height: 5px; }
  .size-marks .s2 { width: 9px; height: 9px; }
  .size-marks .s3 { width: 14px; height: 14px; }
</style>
