<script lang="ts">
  import { onMount } from 'svelte'
  import * as d3 from 'd3'
  import { UMAP, cosine } from 'umap-js'
  import {
    bundle, cohort, selectedPatientId, selectedPhenotypeId,
    searchedConditionIdx, searchedPhenotypeForPatients, advancedView,
    patientProjection,
  } from '../store'
  import { createRng } from '../sampling'
  import { phenotypeHue } from '../palette'
  import { displayedDominant } from '../dominant'
  import type { Phenotype, SyntheticPatient } from '../types'

  // Patient atlas is a de-novo 2D UMAP of the patient theta vectors using
  // cosine distance (the right metric for topic mixtures). UMAP surfaces
  // the cluster structure that PCA flattened onto rays from the origin.
  // Color (dominant phenotype hue) carries the "what's in each cluster"
  // signal, so no on-canvas labels or phenotype anchors are needed.

  let svgEl: SVGSVGElement
  const W = 720, H = 560, MARGIN = 60
  // Dots are intentionally small. with up to ~1000 patients on the canvas
  // the cluster shape is the read, not individual dots.
  const DOT_R = 3.2

  // Projection is cached in a store keyed by cohort.seed. Storing it
  // outside this component means navigating away from the Patient tab
  // (which unmounts PatientMap) and back doesn't retrigger UMAP fitting
  // on an unchanged cohort. `projecting` is local because it's transient
  // UI state - the overlay only matters while this component is mounted.
  let projecting = false

  // We tried exposing n_neighbors as a user slider and it didn't surface
  // useful tunability for our cohort - patients come out near-mono-
  // phenotype (the trained Dirichlet alpha is concentrated), so there
  // isn't much higher-order similarity for larger n_neighbors to surface.
  // 50 sits a notch above umap-js's default (15) for a marginally less
  // shattered layout without paying a fit-time penalty.
  const UMAP_N_NEIGHBORS = 50

  function computeProjection(patients: SyntheticPatient[], seed: number) {
    const thetas = patients.map((p) => p.theta)
    const umap = new UMAP({
      nComponents: 2,
      nNeighbors: Math.min(UMAP_N_NEIGHBORS, Math.max(2, patients.length - 1)),
      minDist: 0.15,
      distanceFn: cosine,
      random: createRng(seed),
    })
    const patientCoords = umap.fit(thetas)
    return { patientCoords, seed }
  }


  // Per-patient deterministic jitter prevents perfect overlap on the rare
  // collisions PCA produces (patients with near-identical theta).
  function hashJitter(id: string): [number, number] {
    let h = 2166136261 >>> 0
    for (let i = 0; i < id.length; i++) {
      h ^= id.charCodeAt(i)
      h = Math.imul(h, 16777619) >>> 0
    }
    const a = ((h & 0xffff) / 65535) * 2 - 1
    const b = ((h >>> 16) / 65535) * 2 - 1
    return [a, b]
  }

  function render() {
    if (!$bundle || !$cohort || !svgEl || !$patientProjection) return
    const phenotypes: Phenotype[] = $bundle.phenotypes.phenotypes
    const allPatients: SyntheticPatient[] = $cohort.patients
    if (allPatients.length === 0) return
    const { patientCoords } = $patientProjection
    const hue = $phenotypeHue
    const advanced = $advancedView

    // Basic mode hides patients whose dominant is dead/mixed. The UMAP
    // layout is still fit on the full cohort so coords are stable across
    // mode toggles; we just don't draw the messy ones in basic.
    const visibleIdx: number[] = []
    for (let i = 0; i < allPatients.length; i++) {
      if (advanced || allPatients[i].isClean) visibleIdx.push(i)
    }
    const data = visibleIdx.map((i) => ({
      patient: allPatients[i],
      x: patientCoords[i][0],
      y: patientCoords[i][1],
    }))

    const allX = patientCoords.map((c) => c[0])
    const allY = patientCoords.map((c) => c[1])
    const xExt = d3.extent(allX) as [number, number]
    const yExt = d3.extent(allY) as [number, number]
    const x = d3.scaleLinear().domain(xExt).range([MARGIN, W - MARGIN])
    const y = d3.scaleLinear().domain(yExt).range([H - MARGIN, MARGIN])

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()
    svg.attr('viewBox', `0 0 ${W} ${H}`)

    const g = svg.append('g')

    // Search highlight: which patients have the searched condition somewhere
    // in their record. Same fuchsia ring vocabulary as the phenotype atlas's
    // pinned-condition highlight so the visual language is consistent.
    const searchedIdx = $searchedConditionIdx
    const searchedMatch = (p: SyntheticPatient): boolean => {
      if (searchedIdx === null) return false
      // Linear scan over the bag is fine. typical bag length is in the dozens.
      const bag = p.code_bag
      for (let i = 0; i < bag.length; i++) if (bag[i] === searchedIdx) return true
      return false
    }

    // Phenotype-find highlight: which patients carry the pinned phenotype
    // with at least the same theta threshold the profile-bar uses for
    // labeled bands (so "the phenotype shows up in their profile"). Amber
    // ring, distinct from the fuchsia condition ring - a patient with both
    // matches gets two rings, one inside the other.
    const PROFILE_THRESHOLD = 0.05
    const phenoIdx = $searchedPhenotypeForPatients
    const phenotypeMatch = (p: SyntheticPatient): boolean => {
      if (phenoIdx === null) return false
      return (p.theta[phenoIdx] ?? 0) >= PROFILE_THRESHOLD
    }

    const JIT = 6
    const nodes = g.selectAll('g.patient')
      .data(data)
      .join('g')
      .attr('class', 'patient')
      .attr('transform', (d) => {
        const [jx, jy] = hashJitter(d.patient.id)
        return `translate(${x(d.x) + jx * JIT}, ${y(d.y) + jy * JIT})`
      })
      .style('cursor', 'pointer')
      .on('click', (_, d) => {
        selectedPatientId.set(d.patient.id)
        selectedPhenotypeId.set(displayedDominant(d.patient.theta, phenotypes, advanced))
      })

    nodes.append('circle')
      .attr('r', DOT_R)
      .attr('fill', (d) => hue(displayedDominant(d.patient.theta, phenotypes, advanced)))
      .attr('fill-opacity', 0.78)
      .attr('stroke', '#18181b')
      .attr('stroke-opacity', 0.25)
      .attr('stroke-width', 0.6)

    // Selection: double cyan ring, same vocabulary as TopicMap.
    nodes.append('circle')
      .attr('r', DOT_R + 5)
      .attr('fill', 'none')
      .attr('stroke', '#06b6d4')
      .attr('stroke-opacity', 0.25)
      .attr('stroke-width', (d) => ($selectedPatientId === d.patient.id ? 5 : 0))
    nodes.append('circle')
      .attr('r', DOT_R + 2.5)
      .attr('fill', 'none')
      .attr('stroke', '#06b6d4')
      .attr('stroke-width', (d) => ($selectedPatientId === d.patient.id ? 2 : 0))

    // Search highlight ring: fuchsia, solid, slightly larger than the dot.
    // Visible only on patients whose record contains the searched condition.
    nodes.append('circle')
      .attr('r', DOT_R + 3)
      .attr('fill', 'none')
      .attr('stroke', '#d946ef')
      .attr('stroke-width', (d) => (searchedMatch(d.patient) ? 2 : 0))

    // Phenotype-find ring: amber, sits at a slightly larger radius so it
    // can co-exist with the fuchsia condition ring on patients that
    // match both.
    nodes.append('circle')
      .attr('r', DOT_R + 5.5)
      .attr('fill', 'none')
      .attr('stroke', '#d97706')
      .attr('stroke-width', (d) => (phenotypeMatch(d.patient) ? 2 : 0))

    nodes.attr('data-tip', (d) => {
      const p = d.patient
      const top = displayedDominant(p.theta, phenotypes, advanced)
      const tlabel = phenotypes[top]?.label || `Phenotype ${top}`
      const pct = (theta: number) => `${(theta * 100).toFixed(0)}%`
      return `${p.id}\nDominant: ${tlabel} (${pct(p.theta[top])})`
    })
  }

  // Refit UMAP only when the cohort identity (seed) changes. Selection
  // changes re-render but reuse the cached projection. We defer the fit
  // with setTimeout(0) so the "Computing layout..." overlay paints before
  // UMAP blocks the main thread. The projection lives in a store keyed
  // by seed, so unmount/remount (tab navigation) doesn't trigger a refit.
  $: if ($bundle && $cohort && !projecting
        && (!$patientProjection || $patientProjection.seed !== $cohort.seed)) {
    projecting = true
    const c = $cohort
    setTimeout(() => {
      try {
        patientProjection.set(computeProjection(c.patients, c.seed))
      } finally {
        projecting = false
        render()
      }
    }, 0)
  }
  $: $selectedPatientId, $phenotypeHue, $searchedConditionIdx, $searchedPhenotypeForPatients, $advancedView, $patientProjection && svgEl && render()

  // For the legend caption: count of patients currently visible on the
  // atlas. Basic = clean only; advanced = full cohort.
  $: visibleCount = $cohort
    ? ($advancedView ? $cohort.patients.length : $cohort.patients.filter((p) => p.isClean).length)
    : 0
  onMount(render)
</script>

<figure class="map">
  <div class="canvas">
    <svg bind:this={svgEl} role="img" aria-label="Patient atlas" preserveAspectRatio="xMidYMid meet"></svg>
    {#if projecting || !$patientProjection}
      <div class="overlay">
        <span class="loading-dot"></span>
        <span>computing layout</span>
      </div>
    {/if}
  </div>
  <figcaption class="legend">
    {#if $cohort}
      <div class="legend-row">
        <span class="eyebrow" title="Each dot is one synthetic patient. Position comes from a 2D UMAP of patient phenotype mixes using cosine distance. Patients near each other have similar phenotype profiles. Dot color matches the dominant phenotype's color in this patient's profile bar.">{visibleCount} synthetic patients · color = dominant phenotype</span>
      </div>
      <div class="legend-note">
        Patients with mixed or unclear phenotypes are excluded in basic view, toggle advanced mode to see all patients.
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
  .canvas {
    position: relative;
  }
  svg {
    width: 100%;
    height: auto;
    display: block;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
  }
  .overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    background: rgba(250, 250, 250, 0.7);
    color: var(--ink-muted);
    font-family: var(--font-mono);
    font-size: var(--fs-small);
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
  .legend {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 0.25rem 0.25rem;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
  }
  .legend-row {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    flex-wrap: wrap;
  }
  .legend-note {
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    font-style: italic;
  }
</style>
