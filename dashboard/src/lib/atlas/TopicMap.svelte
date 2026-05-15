<script lang="ts">
  import { onMount } from 'svelte'
  import * as d3 from 'd3'
  import {
    bundle, selectedPhenotypeId, hoveredCodeIdx, advancedView,
    searchedConditionIdx, phenotypeCoords,
  } from '../store'
  import { phenotypesContainingCode } from '../inference'

  // Simple mode hides `dead` and `mixed` topics. `null` (unlabeled bundle)
  // and the other three categories all show. Advanced mode shows everything.
  function isHiddenInSimpleMode(p: { quality: string | null }): boolean {
    return p.quality === 'dead' || p.quality === 'mixed'
  }

  // Phenotype-containment for highlight: switched from raw-β top-N to
  // relevance-ranked top-N (λ=0.6), matching the CodePanel's displayed
  // ordering. The two views now agree on "which phenotypes feature this
  // condition prominently."
  function containingSet(idx: number | null): Set<number> {
    if (!$bundle || idx === null) return new Set()
    return phenotypesContainingCode({
      beta: $bundle.model.beta,
      corpusFreq: $bundle.vocab.codes.map((c) => c.corpus_freq),
      codeIdx: idx,
    })
  }

  // Two sources of highlight:
  //   - hoveredCodeIdx: transient, set by CodePanel mouseover
  //   - searchedConditionIdx: persistent, set by ConditionSearch
  // The searched condition takes precedence when both are present (a user
  // pinned that condition; the mouseover shouldn't override the pin).
  $: highlighted = $searchedConditionIdx !== null
    ? containingSet($searchedConditionIdx)
    : containingSet($hoveredCodeIdx)

  // When the searched condition is active we also draw a stronger,
  // solid-line ring rather than the dashed hover ring.
  $: highlightStyle = $searchedConditionIdx !== null ? 'pinned' : 'hover'

  let svgEl: SVGSVGElement
  // Wider margin gives the largest bubbles + their selection/highlight rings
  // room to breathe; the previous 24 sat right at the SVG edge for prevalent
  // phenotypes near the layout boundary.
  const W = 720, H = 560, MARGIN = 60
  // How many of the most prevalent bubbles get always-on labels.
  const ALWAYS_LABEL_N = 8

  $: coords = $phenotypeCoords

  // Diverging NPMI ramp: red (low) → neutral gray → cyan (high).
  // Aligns with the global accent and avoids the rainbow look.
  const npmiRamp = d3.scaleLinear<string>()
    .domain([-0.2, 0, 0.2, 0.4])
    .range(['#ef4444', '#d4d4d8', '#67e8f9', '#06b6d4'])
    .clamp(true)

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

    const colorFn = (p: typeof phenotypes[0]) => npmiRamp(p.npmi)

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

    // Main bubble . filled with the encoded color, thin ink-tinted border
    nodes.append('circle')
      .attr('r', (p) => r(p.corpus_prevalence))
      .attr('fill', (p) => colorFn(p))
      .attr('fill-opacity', 0.85)
      .attr('stroke', '#18181b')
      .attr('stroke-opacity', 0.25)
      .attr('stroke-width', 0.75)

    // Selection: thicker double-ring in the cyan accent . a faint outer halo
    // plus a crisp inner band so the picked phenotype reads at a glance even
    // when it sits inside a crowded cluster. The cyan matches the colored
    // bullet in the CodePanel header so "this bubble = this detail" is
    // unambiguous.
    nodes.append('circle')
      .attr('r', (p) => r(p.corpus_prevalence) + 6)
      .attr('fill', 'none')
      .attr('stroke', '#06b6d4')
      .attr('stroke-opacity', 0.25)
      .attr('stroke-width', (p) => ($selectedPhenotypeId === p.id ? 6 : 0))
    nodes.append('circle')
      .attr('r', (p) => r(p.corpus_prevalence) + 3)
      .attr('fill', 'none')
      .attr('stroke', '#06b6d4')
      .attr('stroke-width', (p) => ($selectedPhenotypeId === p.id ? 2.25 : 0))

    // Condition-highlight ring . fuchsia, distinct from the cyan selection
    // accent so the eye can separate "selected" from "matched the searched
    // condition". Dashed for transient hover (from CodePanel mouseover);
    // solid + thicker for a pinned search.
    nodes.append('circle')
      .attr('r', (p) => r(p.corpus_prevalence) + 5)
      .attr('fill', 'none')
      .attr('stroke', '#d946ef')
      .attr('stroke-dasharray', highlightStyle === 'pinned' ? '0' : '3,2')
      .attr('stroke-width', (p) =>
        highlighted.has(p.id) ? (highlightStyle === 'pinned' ? 2.25 : 1.5) : 0
      )

    // Quality indicator (advanced mode only) . small colored dot on the
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

    // Always-on labels for the N most prevalent bubbles (including the
    // currently selected one, if it's in the top N), so the map has some
    // textual anchors a user can scan without clicking. Truncate long
    // labels.
    const truncate = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + '…' : s)
    const topPrevalent = phenotypes
      .slice()
      .sort((a, b) => b.corpus_prevalence - a.corpus_prevalence)
      .slice(0, ALWAYS_LABEL_N)

    g.selectAll('text.minor-label')
      .data(topPrevalent)
      .join('text')
      .attr('class', 'minor-label')
      .attr('x', (p) => x(coords[p.id][0]))
      .attr('y', (p) => y(coords[p.id][1]) - r(p.corpus_prevalence) - 5)
      .attr('text-anchor', 'middle')
      .attr('font-family', 'Geist, sans-serif')
      .attr('font-size', 10)
      .attr('font-weight', 400)
      .attr('fill', '#52525b')
      .attr('paint-order', 'stroke')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 3)
      .attr('stroke-linejoin', 'round')
      .text((p) => truncate(p.label || `Phenotype ${p.id}`, 22))

    // Custom tooltip . `data-tip` is picked up by the global tooltip
    // overlay (lib/tooltip.ts) so it appears with no hover delay. Avoiding
    // SVG `<title>` here means the browser-native delayed tooltip doesn't
    // also fire.
    nodes.attr('data-tip', (p) =>
      `${p.label || `Phenotype ${p.id}`}\nCoherence ${p.npmi.toFixed(3)} · prev ${(p.corpus_prevalence * 100).toFixed(1)}%`,
    )
  }

  $: $selectedPhenotypeId, $hoveredCodeIdx, $advancedView, $searchedConditionIdx, $bundle && svgEl && coords.length && render()
  onMount(render)
</script>

<figure class="map">
  <svg bind:this={svgEl} role="img" aria-label="Phenotype atlas" preserveAspectRatio="xMidYMid meet"></svg>
  <figcaption class="legend">
    {#if $bundle}
      <div class="legend-group">
        <span class="eyebrow" title="Coherence: how reliably the phenotype's leading conditions actually co-occur in the same patients. Higher means the conditions really do show up together; lower means the pattern is weaker or more diffuse. (Bubble color encodes this.)">Coherence</span>
        <span class="grad grad-npmi" aria-hidden="true"></span>
        <span class="ticks" data-numeric><span>low</span><span>high</span></span>
      </div>
      <div class="legend-group">
        <span class="eyebrow" title="Prevalence: roughly how many patients show this phenotype in their records. Bigger bubble means more patients show this pattern.">Prevalence</span>
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
