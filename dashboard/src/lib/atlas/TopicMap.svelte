<script lang="ts">
  import { onMount } from 'svelte'
  import * as d3 from 'd3'
  import { bundle, selectedPhenotypeId, colorMode, hoveredCodeIdx, phenotypesById } from '../store'
  import { computeJsdMds } from '../mds'

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
  const W = 720, H = 560, MARGIN = 40

  let coords: number[][] = []
  $: if ($bundle && coords.length !== $bundle.model.K) {
    coords = computeJsdMds($bundle.model.beta)
  }

  // Custom diverging NPMI ramp (brick → bone → moss). Hand-tuned so the
  // mid-range reads as neutral paper, not the default RdYlGn jaundice.
  const npmiRamp = d3.scaleLinear<string>()
    .domain([-0.2, 0, 0.2, 0.4])
    .range(['#8c3b2e', '#cfb88a', '#9ba87d', '#5b6e3d'])
    .clamp(true)

  // Sequential indigo ramp for prevalence (warm-paper-compatible).
  const prevRamp = d3.scaleSequential<string>(
    (t) => d3.interpolateRgb('#e8e0d0', '#3d4f6e')(t),
  )

  function render() {
    if (!$bundle || !svgEl || coords.length === 0) return
    const phenotypes = $bundle.phenotypes.phenotypes
    const xExt = d3.extent(coords, (c) => c[0]) as [number, number]
    const yExt = d3.extent(coords, (c) => c[1]) as [number, number]
    const x = d3.scaleLinear().domain(xExt).range([MARGIN, W - MARGIN])
    const y = d3.scaleLinear().domain(yExt).range([H - MARGIN, MARGIN])
    const r = d3.scaleSqrt()
      .domain(d3.extent(phenotypes, (p) => p.corpus_prevalence) as [number, number])
      .range([6, 28])

    const prevExt = d3.extent(phenotypes, (p) => p.corpus_prevalence) as [number, number]
    prevRamp.domain(prevExt)
    const colorFn = (p: typeof phenotypes[0]) =>
      $colorMode === 'prevalence' ? prevRamp(p.corpus_prevalence) : npmiRamp(p.npmi)

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()
    svg.attr('viewBox', `0 0 ${W} ${H}`)

    // Defs: dotted-grid pattern for the map background
    const defs = svg.append('defs')
    const pat = defs.append('pattern')
      .attr('id', 'atlas-grid')
      .attr('width', 18)
      .attr('height', 18)
      .attr('patternUnits', 'userSpaceOnUse')
    pat.append('circle').attr('cx', 9).attr('cy', 9).attr('r', 0.6).attr('fill', '#d8ccb8')

    // Frame + grid
    svg.append('rect')
      .attr('width', W).attr('height', H)
      .attr('fill', 'url(#atlas-grid)')
      .attr('stroke', '#d8ccb8')
      .attr('stroke-width', 1)

    // Corner crosshair ornaments — small atlas-style ticks
    const corner = (cx: number, cy: number, dx: number, dy: number) => {
      svg.append('line').attr('x1', cx).attr('y1', cy).attr('x2', cx + dx).attr('y2', cy)
        .attr('stroke', '#1f1b16').attr('stroke-width', 1)
      svg.append('line').attr('x1', cx).attr('y1', cy).attr('x2', cx).attr('y2', cy + dy)
        .attr('stroke', '#1f1b16').attr('stroke-width', 1)
    }
    corner(0.5, 0.5, 12, 12); corner(W - 0.5, 0.5, -12, 12)
    corner(0.5, H - 0.5, 12, -12); corner(W - 0.5, H - 0.5, -12, -12)

    // Axis labels (subtle, italic — feels cartographic)
    svg.append('text').attr('x', W / 2).attr('y', H - 8)
      .attr('text-anchor', 'middle')
      .attr('font-family', 'Newsreader, serif')
      .attr('font-style', 'italic').attr('font-size', 10)
      .attr('fill', '#9c8e7a')
      .text('JSD–MDS  ·  dimension 1')
    svg.append('text').attr('x', 12).attr('y', H / 2)
      .attr('text-anchor', 'middle')
      .attr('font-family', 'Newsreader, serif')
      .attr('font-style', 'italic').attr('font-size', 10)
      .attr('fill', '#9c8e7a')
      .attr('transform', `rotate(-90 12 ${H / 2})`)
      .text('JSD–MDS  ·  dimension 2')

    const g = svg.append('g')

    // Bubbles — concentric ring + small fill core (atlas-like)
    const nodes = g.selectAll('g.node')
      .data(phenotypes)
      .join('g')
      .attr('class', 'node')
      .attr('transform', (p) => `translate(${x(coords[p.id][0])}, ${y(coords[p.id][1])})`)
      .style('cursor', 'pointer')
      .on('click', (_, p) => selectedPhenotypeId.set(p.id))

    nodes.append('circle')
      .attr('r', (p) => r(p.corpus_prevalence))
      .attr('fill', (p) => colorFn(p))
      .attr('fill-opacity', 0.18)
      .attr('stroke', (p) => colorFn(p))
      .attr('stroke-width', (p) => (highlighted.has(p.id) ? 2.5 : 1.25))

    // Inner solid core — opacity scales with corpus_prevalence so dense
    // phenotypes feel weighty, rare ones feel thready
    nodes.append('circle')
      .attr('r', (p) => Math.max(1.5, r(p.corpus_prevalence) * 0.35))
      .attr('fill', (p) => colorFn(p))

    // Selection ring
    nodes.append('circle')
      .attr('r', (p) => r(p.corpus_prevalence) + 4)
      .attr('fill', 'none')
      .attr('stroke', '#1f1b16')
      .attr('stroke-width', (p) => ($selectedPhenotypeId === p.id ? 1 : 0))
      .attr('stroke-dasharray', '2,2')

    // Hover-highlight ring (when a code is hovered in the right panel)
    nodes.append('circle')
      .attr('r', (p) => r(p.corpus_prevalence) + 7)
      .attr('fill', 'none')
      .attr('stroke', '#b25b2c')
      .attr('stroke-width', (p) => (highlighted.has(p.id) ? 1.5 : 0))

    // Junk indicator — small italic mark in brick
    g.selectAll('text.junk')
      .data(phenotypes.filter((p) => p.junk_flag))
      .join('text')
      .attr('class', 'junk')
      .attr('x', (p) => x(coords[p.id][0]) + r(p.corpus_prevalence) + 3)
      .attr('y', (p) => y(coords[p.id][1]) - r(p.corpus_prevalence) + 4)
      .attr('font-family', 'Newsreader, serif')
      .attr('font-style', 'italic')
      .attr('font-size', 11)
      .attr('fill', '#8c3b2e')
      .text('†')

    // Label for the selected phenotype, ink color, anchored above its bubble
    if ($selectedPhenotypeId !== null) {
      const sel = phenotypes[$selectedPhenotypeId]
      if (sel) {
        g.append('text')
          .attr('x', x(coords[sel.id][0]))
          .attr('y', y(coords[sel.id][1]) - r(sel.corpus_prevalence) - 10)
          .attr('text-anchor', 'middle')
          .attr('font-family', 'Newsreader, serif')
          .attr('font-style', 'italic')
          .attr('font-size', 12)
          .attr('fill', '#1f1b16')
          .text(sel.label || `Phenotype ${sel.id}`)
      }
    }

    nodes.append('title')
      .text((p) => `${p.label || `Phenotype ${p.id}`}\nNPMI ${p.npmi.toFixed(3)} · prev ${(p.corpus_prevalence * 100).toFixed(1)}%`)
  }

  $: $colorMode, $selectedPhenotypeId, $hoveredCodeIdx, $bundle && svgEl && coords.length && render()
  onMount(render)
</script>

<figure class="map">
  <svg bind:this={svgEl} role="img" aria-label="Phenotype topic map" preserveAspectRatio="xMidYMid meet"></svg>
  <figcaption class="legend">
    {#if $bundle}
      {#if $colorMode === 'prevalence'}
        <span class="eyebrow">color</span>
        <span class="grad grad-prev" aria-hidden="true"></span>
        <span class="lend" data-numeric>rare</span>
        <span class="rend" data-numeric>common</span>
      {:else}
        <span class="eyebrow">npmi</span>
        <span class="grad grad-npmi" aria-hidden="true"></span>
        <span class="lend" data-numeric>−0.2</span>
        <span class="rend" data-numeric>+0.4</span>
      {/if}
      <span class="sep"></span>
      <span class="eyebrow">size</span>
      <span class="size-marks" aria-hidden="true">
        <span class="dot s1"></span><span class="dot s2"></span><span class="dot s3"></span>
      </span>
      <span class="lend">prevalence</span>
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
    background: var(--paper-elevated);
  }
  .legend {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.4rem 0.55rem;
    font-size: var(--fs-micro);
    color: var(--ink-muted);
    flex-wrap: wrap;
  }
  .legend .grad {
    display: inline-block;
    width: 100px;
    height: 6px;
    border-radius: 3px;
  }
  .grad-prev {
    background: linear-gradient(to right, #e8e0d0, #3d4f6e);
  }
  .grad-npmi {
    background: linear-gradient(to right, #8c3b2e, #cfb88a, #5b6e3d);
  }
  .legend .lend,
  .legend .rend {
    font-size: var(--fs-micro);
    color: var(--ink-faint);
  }
  .sep {
    width: 1px;
    height: 14px;
    background: var(--rule);
    margin: 0 0.3rem;
  }
  .size-marks {
    display: inline-flex;
    align-items: baseline;
    gap: 4px;
  }
  .size-marks .dot {
    border-radius: 50%;
    border: 1px solid var(--ink-muted);
    background: transparent;
  }
  .size-marks .s1 { width: 5px; height: 5px; }
  .size-marks .s2 { width: 9px; height: 9px; }
  .size-marks .s3 { width: 14px; height: 14px; }
</style>
