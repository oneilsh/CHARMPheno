<script lang="ts">
  import * as d3 from 'd3'
  import {
    bundle, cohort, patientProjection, patientProjectionFitting, advancedView,
  } from '../store'
  import { phenotypeHue } from '../palette'
  import { displayedDominant } from '../dominant'
  import { ensurePatientProjection } from '../patient/projection'
  import type { Phenotype, SyntheticPatient } from '../types'

  // N theta vectors from the latest simulator run. Each is transformed
  // into the same 2D UMAP space as the cohort and drawn as a bright dot
  // on top of the dimmed cohort cloud. A tight sim cluster means the
  // prefix nails one kind of patient; a smeared cloud means the prefix
  // is consistent with several phenotype mixes.
  export let thetaSamples: number[][] = []

  let svgEl: SVGSVGElement
  const W = 520, H = 380, MARGIN = 24
  const COHORT_R = 2.4
  const SIM_R = 3.4
  const COHORT_OPACITY = 0.22

  // Trigger the cohort projection on mount in case the user lands on
  // the Simulator tab before they ever visit the Patient atlas.
  $: $cohort, ensurePatientProjection()

  // Project the sim samples through the SAME fitted UMAP. Recomputed
  // only when the sim samples or the cohort projection changes.
  $: simCoords = (() => {
    if (!$patientProjection || thetaSamples.length === 0) return [] as number[][]
    try {
      return $patientProjection.umap.transform(thetaSamples)
    } catch {
      return [] as number[][]
    }
  })()

  function render() {
    if (!$bundle || !$cohort || !svgEl || !$patientProjection) return
    const phenotypes: Phenotype[] = $bundle.phenotypes.phenotypes
    const allPatients: SyntheticPatient[] = $cohort.patients
    const { patientCoords } = $patientProjection
    const hue = $phenotypeHue
    const advanced = $advancedView

    const allX = patientCoords.map((c) => c[0])
    const allY = patientCoords.map((c) => c[1])
    const xExt = d3.extent(allX) as [number, number]
    const yExt = d3.extent(allY) as [number, number]
    const x = d3.scaleLinear().domain(xExt).range([MARGIN, W - MARGIN])
    const y = d3.scaleLinear().domain(yExt).range([H - MARGIN, MARGIN])

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()
    svg.attr('viewBox', `0 0 ${W} ${H}`)

    // Cohort layer: same color vocabulary as the Patient atlas (dominant
    // phenotype hue) but heavily dimmed so the sim cloud reads as the
    // figure and the cohort reads as the ground.
    const cohortG = svg.append('g')
    const visible: SyntheticPatient[] = []
    const visibleCoords: number[][] = []
    for (let i = 0; i < allPatients.length; i++) {
      if (advanced || allPatients[i].isClean) {
        visible.push(allPatients[i])
        visibleCoords.push(patientCoords[i])
      }
    }
    cohortG.selectAll('circle')
      .data(visible)
      .join('circle')
      .attr('cx', (_, i) => x(visibleCoords[i][0]))
      .attr('cy', (_, i) => y(visibleCoords[i][1]))
      .attr('r', COHORT_R)
      .attr('fill', (d) => hue(displayedDominant(d.theta, phenotypes, advanced)))
      .attr('fill-opacity', COHORT_OPACITY)
      .attr('stroke', 'none')

    // Sim layer: full-opacity dots colored by each sample's own dominant.
    // Sits on top of the cohort so a glance reads "these N points are
    // the simulated patients."
    const simG = svg.append('g')
    simG.selectAll('circle')
      .data(thetaSamples)
      .join('circle')
      .attr('cx', (_, i) => simCoords[i] ? x(simCoords[i][0]) : -10)
      .attr('cy', (_, i) => simCoords[i] ? y(simCoords[i][1]) : -10)
      .attr('r', SIM_R)
      .attr('fill', (t) => hue(displayedDominant(t, phenotypes, advanced)))
      .attr('fill-opacity', 0.9)
      .attr('stroke', '#18181b')
      .attr('stroke-opacity', 0.4)
      .attr('stroke-width', 0.6)
  }

  $: $patientProjection, simCoords, $phenotypeHue, $advancedView, svgEl && render()
</script>

<figure class="map">
  <figcaption class="caption-top">
    <span class="eyebrow">Where this patient lands</span>
    <span class="sub">Each bright dot is one simulated patient on the same atlas as the patient cohort.</span>
  </figcaption>
  <div class="canvas">
    <svg bind:this={svgEl} role="img" aria-label="Simulator placement on patient atlas" preserveAspectRatio="xMidYMid meet"></svg>
    {#if $patientProjectionFitting || !$patientProjection}
      <div class="overlay">
        <span class="loading-dot"></span>
        <span>computing layout</span>
      </div>
    {/if}
  </div>
</figure>

<style>
  .map {
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    padding: 1rem 1rem 0.75rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
  }
  .caption-top {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--rule-faint);
  }
  .sub {
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    font-style: italic;
  }
  .canvas { position: relative; }
  svg {
    width: 100%;
    height: auto;
    display: block;
    background: var(--surface-recessed);
    border-radius: var(--radius-sm);
  }
  .overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.55rem;
    background: rgba(243, 243, 243, 0.7);
    color: var(--ink-muted);
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-radius: var(--radius-sm);
    pointer-events: none;
  }
  .loading-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 0.3; transform: scale(0.85); }
    50% { opacity: 1; transform: scale(1.15); }
  }
</style>
