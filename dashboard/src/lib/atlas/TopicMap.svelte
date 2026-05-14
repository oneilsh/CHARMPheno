<script lang="ts">
  import { onMount } from 'svelte'
  import * as d3 from 'd3'
  import { bundle, selectedPhenotypeId, colorMode } from '../store'
  import { computeJsdMds } from '../mds'

  let svgEl: SVGSVGElement
  const W = 560, H = 480, MARGIN = 24

  let coords: number[][] = []
  $: if ($bundle && coords.length !== $bundle.model.K) {
    coords = computeJsdMds($bundle.model.beta)
  }

  function render() {
    if (!$bundle || !svgEl || coords.length === 0) return
    const phenotypes = $bundle.phenotypes.phenotypes
    const xExt = d3.extent(coords, (c) => c[0]) as [number, number]
    const yExt = d3.extent(coords, (c) => c[1]) as [number, number]
    const x = d3.scaleLinear().domain(xExt).range([MARGIN, W - MARGIN])
    const y = d3.scaleLinear().domain(yExt).range([H - MARGIN, MARGIN])
    const r = d3.scaleSqrt()
      .domain(d3.extent(phenotypes, (p) => p.corpus_prevalence) as [number, number])
      .range([4, 24])
    const colorFn = $colorMode === 'prevalence'
      ? d3.scaleSequential(d3.interpolateBlues)
          .domain(d3.extent(phenotypes, (p) => p.corpus_prevalence) as [number, number])
      : d3.scaleSequential(d3.interpolateRdYlGn).domain([-0.2, 0.4])

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()
    svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').attr('height', H)
    const g = svg.append('g')

    g.selectAll('circle')
      .data(phenotypes)
      .join('circle')
      .attr('cx', (p) => x(coords[p.id][0]))
      .attr('cy', (p) => y(coords[p.id][1]))
      .attr('r', (p) => r(p.corpus_prevalence))
      .attr('fill', (p) => ($colorMode === 'prevalence' ? colorFn(p.corpus_prevalence) : colorFn(p.npmi)) as string)
      .attr('stroke', (p) => ($selectedPhenotypeId === p.id ? '#000' : '#444'))
      .attr('stroke-width', (p) => ($selectedPhenotypeId === p.id ? 2.5 : 0.5))
      .style('cursor', 'pointer')
      .on('click', (_, p) => selectedPhenotypeId.set(p.id))
      .append('title')
      .text((p) => `${p.label || `Phenotype ${p.id}`}\nNPMI ${p.npmi.toFixed(3)} · prev ${(p.corpus_prevalence * 100).toFixed(1)}%`)

    g.selectAll('text.junk')
      .data(phenotypes.filter((p) => p.junk_flag))
      .join('text')
      .attr('class', 'junk')
      .attr('x', (p) => x(coords[p.id][0]) + 8)
      .attr('y', (p) => y(coords[p.id][1]) - 8)
      .attr('font-size', 9)
      .attr('fill', '#b00020')
      .text('!')
  }

  $: $colorMode, $selectedPhenotypeId, $bundle && svgEl && coords.length && render()
  onMount(render)
</script>

<svg bind:this={svgEl} role="img" aria-label="Phenotype topic map"></svg>
