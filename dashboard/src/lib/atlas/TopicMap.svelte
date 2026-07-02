<script lang="ts">
  import { onMount } from 'svelte'
  import * as d3 from 'd3'
  import {
    bundle, selectedPhenotypeId, hoveredCodeIdx, advancedView,
    searchedConditionIdx, phenotypeCoords,
    prevalenceReader, tauThreshold, isVisibleInCurrentMode, conditioning,
  } from '../store'
  import { groupHue } from '../palette'
  import { phenotypesContainingCode } from '../inference'
  import { copy } from '../copy'

  // Cohort color + label helpers (gated bundles). Cohort now owns the node
  // hue; groupHue gives foreground groups distinct colors and 'background'
  // (not a group) a neutral gray.
  $: grpColor = $groupHue
  $: groupLabel = (g: string) => $bundle?.gating?.group_labels?.[g] ?? g

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
  $: reader = $prevalenceReader

  // Diverging NPMI ramp: red (low) → neutral gray → cyan (high). Used as the
  // node HUE only for non-gated bundles (where cohort color doesn't apply).
  const npmiRamp = d3.scaleLinear<string>()
    .domain([-0.2, 0, 0.2, 0.4])
    .range(['#ef4444', '#d4d4d8', '#67e8f9', '#06b6d4'])
    .clamp(true)

  // For gated bundles cohort owns the hue, so coherence (NPMI) is encoded as
  // fill OPACITY instead: faded = low coherence, solid = high.
  const cohOpacity = d3.scaleLinear().domain([0.05, 0.4]).range([0.45, 0.95]).clamp(true)

  function render() {
    if (!$bundle || !svgEl || coords.length === 0) return
    // Mode-aware filter: simple hides dead+mixed; advanced shows all.
    // x/y/r scales are computed against the FULL set so the layout is
    // stable across mode toggles (we don't want bubbles jumping around).
    const allPhenotypes = $bundle.phenotypes.phenotypes
    const phenotypes = allPhenotypes.filter($isVisibleInCurrentMode)
    const xExt = d3.extent(coords, (c) => c[0]) as [number, number]
    const yExt = d3.extent(coords, (c) => c[1]) as [number, number]
    const x = d3.scaleLinear().domain(xExt).range([MARGIN, W - MARGIN])
    const y = d3.scaleLinear().domain(yExt).range([H - MARGIN, MARGIN])
    // Use the FULL phenotype set for the prevalence scale domain so bubble
    // size doesn't rescale between simple and advanced modes.
    const r_of = reader
    // Bubble area encodes prevalence. In covariate mode or when gating is active
    // the user expects ABSOLUTE size changes; anchor the scale's domain to a
    // stable per-bundle reference (the corpus-average prevalence max) rather than
    // the live reader's max. Otherwise the most-prevalent bubble is re-pinned to
    // the range top every frame and never appears to change as the covariates move.
    // Outside conditioning, keep the self-scaling domain.
    const conditioningActive = $conditioning.covariateActive || !!$bundle?.gating
    const domainMax = conditioningActive
      ? Math.max(...allPhenotypes.map((p) => p.corpus_prevalence), 1e-9)
      : Math.max(...allPhenotypes.map(r_of), 1e-9)
    const r = d3.scaleSqrt()
      .domain([0, domainMax])
      .range([5, 26])

    // Node hue: cohort (topic block) for gated bundles, else the NPMI ramp.
    // Coherence rides on opacity in the gated case (see opacityFn).
    const gated = !!$bundle.gating
    const blocks = $bundle.gating?.topic_blocks
    const colorFn = (p: typeof phenotypes[0]) =>
      gated ? grpColor(blocks![p.id]) : npmiRamp(p.npmi ?? 0)
    const opacityFn = (p: typeof phenotypes[0]) =>
      gated ? cohOpacity(p.npmi ?? 0.05) : 0.85

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
    // Zero-prevalence (out-of-group foreground) topics get radius 0 so they
    // vanish entirely; positive values keep the existing [5, 26] scaleSqrt mapping.
    const rad = (p: typeof phenotypes[0]) => {
      const v = r_of(p)
      return v === 0 ? 0 : r(v)
    }
    nodes.append('circle')
      .attr('r', (p) => rad(p))
      .attr('fill', (p) => colorFn(p))
      .attr('fill-opacity', (p) => opacityFn(p))
      .attr('stroke', '#18181b')
      .attr('stroke-opacity', 0.25)
      .attr('stroke-width', 0.75)

    // Selection: thicker double-ring in the cyan accent . a faint outer halo
    // plus a crisp inner band so the picked phenotype reads at a glance even
    // when it sits inside a crowded cluster. The cyan matches the colored
    // bullet in the CodePanel header so "this bubble = this detail" is
    // unambiguous.
    nodes.append('circle')
      .attr('r', (p) => rad(p) + 6)
      .attr('fill', 'none')
      .attr('stroke', '#06b6d4')
      .attr('stroke-opacity', 0.25)
      .attr('stroke-width', (p) => ($selectedPhenotypeId === p.id ? 6 : 0))
    nodes.append('circle')
      .attr('r', (p) => rad(p) + 3)
      .attr('fill', 'none')
      .attr('stroke', '#06b6d4')
      .attr('stroke-width', (p) => ($selectedPhenotypeId === p.id ? 2.25 : 0))

    // Condition-highlight ring . fuchsia, distinct from the cyan selection
    // accent so the eye can separate "selected" from "matched the searched
    // condition". Dashed for transient hover (from CodePanel mouseover);
    // solid + thicker for a pinned search.
    nodes.append('circle')
      .attr('r', (p) => rad(p) + 5)
      .attr('fill', 'none')
      .attr('stroke', '#d946ef')
      .attr('stroke-dasharray', highlightStyle === 'pinned' ? '0' : '3,2')
      .attr('stroke-width', (p) =>
        highlighted.has(p.id) ? (highlightStyle === 'pinned' ? 2.25 : 1.5) : 0
      )

    // Quality glyph (advanced mode only): ⊘ dead / ◑ mixed, drawn as a text
    // element APPENDED TO EACH NODE GROUP so it inherits the bubble's transform
    // and tracks it exactly (the previous separate-selection version recomputed
    // absolute coords and drifted off bubbles whose prevalence was masked to 0).
    // Good-quality topics get an empty string, which renders nothing.
    const qualityGlyph: Record<string, string> = { dead: '⊘', mixed: '◑' }
    const qualityColor: Record<string, string> = { dead: '#dc2626', mixed: '#d97706' }
    if ($advancedView) {
      nodes.append('text')
        .attr('class', 'quality-glyph')
        .attr('x', (p) => rad(p) * 0.72 + 4)
        .attr('y', (p) => -rad(p) * 0.72 - 4)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .attr('font-size', 13)
        .attr('paint-order', 'stroke')
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 2.5)
        .attr('stroke-linejoin', 'round')
        .attr('fill', (p) => qualityColor[p.quality ?? ''] ?? 'transparent')
        .text((p) => qualityGlyph[p.quality ?? ''] ?? '')
    }

    // Always-on labels for the N most prevalent bubbles (including the
    // currently selected one, if it's in the top N), so the map has some
    // textual anchors a user can scan without clicking. Truncate long
    // labels.
    const truncate = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + '…' : s)
    const topPrevalent = phenotypes
      .slice()
      .sort((a, b) => r_of(b) - r_of(a))
      .slice(0, ALWAYS_LABEL_N)

    g.selectAll('text.minor-label')
      .data(topPrevalent)
      .join('text')
      .attr('class', 'minor-label')
      .attr('x', (p) => x(coords[p.id][0]))
      .attr('y', (p) => y(coords[p.id][1]) - rad(p) - 5)
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
    nodes.attr('data-tip', (p) => {
      const pat = (r_of(p) * 100).toFixed(1)
      const npmi = p.npmi == null ? '—' : p.npmi.toFixed(3)
      const tauStr = $tauThreshold.toFixed(2)
      const label = p.label || `Phenotype ${p.id}`
      if ($advancedView) {
        const mass = (p.corpus_prevalence * 100).toFixed(1)
        return `${label}\nCoherence ${npmi} · prev ${pat}% (patients, θ > ${tauStr}) · topic mass ${mass}%`
      }
      return `${label}\nCoherence ${npmi} · prev ${pat}% (θ > ${tauStr})`
    })
  }

  // `reader` is listed so the atlas re-renders whenever the prevalence reader
  // changes for ANY reason - covariate active toggling, covariate-value edits,
  // or the gating group selector - not only on the tau/selection/mode stores.
  $: reader, $conditioning, $tauThreshold, $selectedPhenotypeId, $hoveredCodeIdx, $advancedView, $searchedConditionIdx, $bundle && svgEl && coords.length && render()
  onMount(render)
</script>

<figure class="map" data-tour="atlas-map">
  <svg bind:this={svgEl} role="img" aria-label="Phenotype atlas" preserveAspectRatio="xMidYMid meet"></svg>
  <figcaption class="legend">
    {#if $bundle}
      <!-- Top row: encodings shared by all bundles (coherence, prevalence,
           and — in advanced mode — the quality glyphs). -->
      <div class="legend-row">
        {#if $bundle.gating}
          <div class="legend-group">
            <span class="eyebrow" title="Bubble opacity encodes topic coherence (NPMI): faded = low, solid = high.">Coherence<span class="help-mark" aria-hidden="true">?</span></span>
            <span class="grad grad-coh" aria-hidden="true"></span>
            <span class="ticks" data-numeric><span>low</span><span>high</span></span>
          </div>
        {:else}
          <div class="legend-group">
            <span class="eyebrow" title={copy.atlas.legend.coherence}>Coherence<span class="help-mark" aria-hidden="true">?</span></span>
            <span class="grad grad-npmi" aria-hidden="true"></span>
            <span class="ticks" data-numeric><span>low</span><span>high</span></span>
          </div>
        {/if}
        <div class="legend-group">
          <span class="eyebrow" title={copy.atlas.legend.prevalence($tauThreshold)}>Prevalence<span class="help-mark" aria-hidden="true">?</span></span>
          <span class="size-marks" aria-hidden="true">
            <span class="dot s1"></span><span class="dot s2"></span><span class="dot s3"></span>
          </span>
        </div>
        {#if $advancedView}
          <div class="legend-group">
            <span class="eyebrow" title="Low-quality topics: ⊘ dead (no usable signal), ◑ mixed (blended themes).">Quality<span class="help-mark" aria-hidden="true">?</span></span>
            <span class="quality-legend" aria-hidden="true">
              <span class="q-glyph q-dead">⊘</span>dead
              <span class="q-glyph q-mixed">◑</span>mixed
            </span>
          </div>
        {/if}
      </div>
      <!-- Bottom row: cohort color key (gated bundles only). -->
      {#if $bundle.gating}
        <div class="legend-row">
          <div class="legend-group">
            <span class="eyebrow" title="Node color = the source cohort each phenotype belongs to.">Cohort<span class="help-mark" aria-hidden="true">?</span></span>
            <span class="cohort-marks">
              <span class="cohort-item"><span class="sw" style="background:{grpColor('background')}"></span>Background</span>
              {#each $bundle.gating.groups as g}
                <span class="cohort-item"><span class="sw" style="background:{grpColor(g)}"></span>{groupLabel(g)}</span>
              {/each}
            </span>
          </div>
        </div>
      {/if}
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
    flex-direction: column;
    gap: 0.45rem;
    align-items: flex-start;
    padding: 0.25rem 0.25rem;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
  }
  .legend-row {
    display: flex;
    gap: 1.5rem;
    align-items: center;
    flex-wrap: wrap;
  }
  .legend-group {
    display: flex;
    align-items: center;
    gap: 0.55rem;
  }
  /* Small circled "?" cueing a hover explanation on the label. Matches the
     phenotype-detail panel's .help-mark convention (CodePanel.svelte); the
     tooltip itself lives on the parent .eyebrow title. */
  .eyebrow { cursor: help; }
  .help-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 11px;
    height: 11px;
    margin-left: 0.3rem;
    border: 1px solid var(--ink-faint);
    border-radius: 50%;
    font-family: var(--font-body);
    font-size: 8px;
    line-height: 1;
    font-weight: 600;
    color: var(--ink-faint);
    vertical-align: middle;
    transition: color 0.12s ease, border-color 0.12s ease;
  }
  .eyebrow:hover .help-mark {
    color: var(--accent);
    border-color: var(--accent);
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
  /* Coherence-as-opacity ramp: a neutral swatch fading from low to high. */
  .grad-coh {
    background: linear-gradient(to right, rgba(100, 116, 139, 0.3), rgba(100, 116, 139, 0.95));
  }
  /* Cohort swatches. */
  .cohort-marks {
    display: inline-flex;
    align-items: center;
    gap: 0.7rem;
    flex-wrap: wrap;
  }
  .cohort-item {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
  }
  .cohort-item .sw {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    border: 0.5px solid rgba(24, 24, 27, 0.25);
  }
  /* Quality-glyph legend. */
  .quality-legend {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
  }
  .q-glyph {
    font-size: var(--fs-small);
    line-height: 1;
  }
  .q-dead { color: #dc2626; }
  .q-mixed { color: #d97706; }
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
