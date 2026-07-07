<script lang="ts">
  import { onMount } from 'svelte'
  import * as d3 from 'd3'
  import {
    bundle, selectedPhenotypeId, hoveredCodeIdx, advancedView,
    searchedConditionIdx, phenotypeCoords,
    prevalenceReader, meanGainReader, tauThreshold, isVisibleInCurrentMode, conditioning,
  } from '../store'
  import { groupHue } from '../palette'
  import { phenotypesContainingCode } from '../inference'
  import { labelRanks } from './labels'
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

  // ---- Semantic zoom + progressive labels --------------------------------
  // The zoom transform scales node POSITIONS (they spread apart) but each node's
  // markers live in a counter-scaled inner group so they keep a CONSTANT on-screen
  // size — zooming in therefore decongests overlapping bubbles instead of just
  // magnifying the same overlap. Labels are revealed progressively: a node is
  // labeled when its within-block prevalence rank is below a zoom-dependent
  // cutoff, so zooming in only ADDS lower-ranked labels (stable, non-jumpy) and
  // fixes the "no labels when zoomed in" discovery gap.
  //
  // How many labels per block at the overview (k=1). Ungated bundles form one
  // implicit block so this is effectively the global overview count.
  const BASE_LABELS_GATED = 3
  const BASE_LABELS_UNGATED = 8
  function labelsCut(k: number): number {
    // grows ~linearly with zoom; never drops below the overview count
    return Math.max(labelBaseCut, Math.round(labelBaseCut * k))
  }
  function labelVisible(id: number, k: number): boolean {
    const rank = labelRankMap.get(id)
    return rank !== undefined && rank < labelsCut(k)
  }

  // d3-zoom state. render() rebuilds the <g> every frame (selectAll('*').remove),
  // so we keep the zoom behavior + a handle to the current <g> at module scope
  // and re-apply the persisted transform (stored by d3 on the svg node) to each
  // freshly-created group. gSel is read inside the zoom handler so panning/
  // zooming always drives the live group.
  let zoomBehavior: d3.ZoomBehavior<SVGSVGElement, unknown> | null = null
  let gSel: d3.Selection<SVGGElement, unknown, null, undefined> | null = null
  // Set each render() so the zoom handler can recompute label visibility.
  let labelRankMap: Map<number, number> = new Map()
  let labelBaseCut = BASE_LABELS_UNGATED
  let currentK = 1

  // Applied on every zoom/pan tick: move positions (g transform), keep markers a
  // constant size (counter-scale the inner groups), and re-reveal labels for the
  // new zoom level.
  function applyZoomVisuals(t: d3.ZoomTransform) {
    currentK = t.k
    if (!gSel) return
    gSel.attr('transform', t.toString())
    gSel.selectAll<SVGGElement, unknown>('g.inner').attr('transform', `scale(${1 / t.k})`)
    gSel.selectAll<SVGGElement, { id: number }>('g.label')
      .attr('display', (d) => (labelVisible(d.id, t.k) ? null : 'none'))
  }

  function resetView() {
    if (svgEl && zoomBehavior) {
      d3.select(svgEl).transition().duration(250).call(zoomBehavior.transform, d3.zoomIdentity)
    }
  }

  $: coords = $phenotypeCoords
  $: reader = $prevalenceReader

  // Task 6b: bubble SIZE. Predictive-gain bundles (gated STM with the
  // held-out predictive-gain metric computed) encode size as `mean_gain`
  // (unique predictive contribution, nats) — the most discriminating of the
  // new fields. Bundles without predictive_gain (HDP/LDA/legacy) fall back
  // to the original prevalence-based sizing, UNCHANGED. This is the single
  // swap point: to try a different size source later (e.g. presence), just
  // change what `sizeReader` reads when `hasPredictiveGain` is true.
  $: hasPredictiveGain = !!$bundle?.phenotypes.predictive_gain
  $: sizeReader = hasPredictiveGain ? $meanGainReader : reader

  // Diverging NPMI ramp: red (low) → neutral gray → cyan (high). Used as the
  // node HUE only for non-gated bundles (where cohort color doesn't apply).
  const npmiRamp = d3.scaleLinear<string>()
    .domain([-0.2, 0, 0.2, 0.4])
    .range(['#ef4444', '#d4d4d8', '#67e8f9', '#06b6d4'])
    .clamp(true)

  // For gated bundles cohort owns the hue, so coherence (NPMI) is encoded as
  // fill OPACITY instead: faded = low coherence, solid = high. The DOMAIN is
  // fitted to the bundle's actual NPMI spread each render (see render()), so the
  // full opacity range is used across the observed coherence range rather than a
  // fixed [0.05, 0.4] that may under-use it. Range floor 0.45 keeps a wide span.
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
    // Use the FULL phenotype set for the size scale domain so bubble size
    // doesn't rescale between simple and advanced modes. `r_of` is the size
    // source: mean_gain when predictive_gain is present, else prevalence
    // (see `sizeReader` above).
    const r_of = sizeReader
    // Bubble area encodes prevalence (fallback case only). In covariate mode
    // or when gating is active the user expects ABSOLUTE size changes; anchor
    // the scale's domain to a stable per-bundle reference (the corpus-average
    // prevalence max) rather than the live reader's max. Otherwise the most-
    // prevalent bubble is re-pinned to the range top every frame and never
    // appears to change as the covariates move. Outside conditioning, keep
    // the self-scaling domain. mean_gain is a static per-topic value (it
    // doesn't move with covariate sliders), so the predictive-gain case
    // always uses the plain self-scaling domain regardless of conditioning.
    const conditioningActive = $conditioning.covariateActive || !!$bundle?.gating
    const domainMax = !hasPredictiveGain && conditioningActive
      ? Math.max(...allPhenotypes.map((p) => p.corpus_prevalence), 1e-9)
      : Math.max(...allPhenotypes.map(r_of), 1e-9)
    // scaleSqrt = area-proportional. A low floor (2, not 5) lets a 0.2%-
    // prevalence bubble actually read as tiny next to a 4%+ one instead of being
    // pinned near the mid-size floor; zero stays 0 (handled in rad()).
    const r = d3.scaleSqrt()
      .domain([0, domainMax])
      .range([2, 26])

    // Node hue: cohort (topic block) for gated bundles, else the NPMI ramp.
    // Coherence rides on opacity in the gated case (see opacityFn).
    const gated = !!$bundle.gating
    const blocks = $bundle.gating?.topic_blocks
    // Fit the coherence-opacity domain to THIS bundle's NPMI spread so the full
    // opacity range spans the observed coherence range. Guard degenerate spreads
    // (all-equal / missing) by falling back to the fixed [0.05, 0.4].
    const npmiVals = allPhenotypes.map((p) => p.npmi).filter((v): v is number => v != null)
    const npmiExt = d3.extent(npmiVals) as [number, number] | [undefined, undefined]
    if (npmiExt[0] != null && npmiExt[1] != null && npmiExt[1] - npmiExt[0] > 1e-6) {
      cohOpacity.domain(npmiExt)
    } else {
      cohOpacity.domain([0.05, 0.4])
    }
    const npmiFloor = npmiExt[0] ?? 0.05
    const colorFn = (p: typeof phenotypes[0]) =>
      gated ? grpColor(blocks![p.id]) : npmiRamp(p.npmi ?? 0)
    const opacityFn = (p: typeof phenotypes[0]) =>
      gated ? cohOpacity(p.npmi ?? npmiFloor) : 0.85

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()
    svg.attr('viewBox', `0 0 ${W} ${H}`)

    const g = svg.append('g')
    gSel = g as unknown as d3.Selection<SVGGElement, unknown, null, undefined>

    // Progressive-label state for this render (read by the zoom handler).
    labelRankMap = labelRanks(allPhenotypes, {
      blocks: gated ? blocks : undefined,
      isVisible: $isVisibleInCurrentMode,
    })
    labelBaseCut = gated ? BASE_LABELS_GATED : BASE_LABELS_UNGATED

    // Zoom / pan. d3 stashes the current transform on the svg node (__zoom), so
    // it survives render()'s teardown; re-apply it to the new <g>. Attach the
    // behavior once and reuse it (re-calling .call would re-register listeners).
    currentK = d3.zoomTransform(svgEl).k
    g.attr('transform', d3.zoomTransform(svgEl).toString())
    if (!zoomBehavior) {
      zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.6, 8])
        .on('zoom', (event) => applyZoomVisuals(event.transform))
      svg.call(zoomBehavior)
      // Disable dblclick-to-zoom so double-clicking a bubble doesn't also zoom;
      // wheel + drag-pan remain. The "Reset view" button restores the default.
      svg.on('dblclick.zoom', null)
    }

    const rad = (p: typeof phenotypes[0]) => {
      const v = r_of(p)
      return v === 0 ? 0 : r(v)
    }

    // Node groups carry only POSITION (translate). Under the zoom transform g is
    // scaled by k, so these positions spread apart as you zoom in. The visual
    // markers live in an INNER group counter-scaled by 1/k, so they keep a
    // constant on-screen size — the net effect is decongestion, not magnification.
    const nodes = g.selectAll('g.node')
      .data(phenotypes)
      .join('g')
      .attr('class', 'node')
      .attr('transform', (p) => `translate(${x(coords[p.id][0])}, ${y(coords[p.id][1])})`)
      .style('cursor', 'pointer')
      .on('click', (_, p) => selectedPhenotypeId.set(p.id))

    const inner = nodes.append('g')
      .attr('class', 'inner')
      .attr('transform', `scale(${1 / currentK})`)

    // Main bubble . filled with the encoded color, thin ink-tinted border
    // Zero-prevalence (out-of-group foreground) topics get radius 0 so they
    // vanish entirely; positive values keep the existing [5, 26] scaleSqrt mapping.
    inner.append('circle')
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
    inner.append('circle')
      .attr('r', (p) => rad(p) + 6)
      .attr('fill', 'none')
      .attr('stroke', '#06b6d4')
      .attr('stroke-opacity', 0.25)
      .attr('stroke-width', (p) => ($selectedPhenotypeId === p.id ? 6 : 0))
    inner.append('circle')
      .attr('r', (p) => rad(p) + 3)
      .attr('fill', 'none')
      .attr('stroke', '#06b6d4')
      .attr('stroke-width', (p) => ($selectedPhenotypeId === p.id ? 2.25 : 0))

    // Condition-highlight ring . fuchsia, distinct from the cyan selection
    // accent so the eye can separate "selected" from "matched the searched
    // condition". Dashed for transient hover (from CodePanel mouseover);
    // solid + thicker for a pinned search.
    inner.append('circle')
      .attr('r', (p) => rad(p) + 5)
      .attr('fill', 'none')
      .attr('stroke', '#d946ef')
      .attr('stroke-dasharray', highlightStyle === 'pinned' ? '0' : '3,2')
      .attr('stroke-width', (p) =>
        highlighted.has(p.id) ? (highlightStyle === 'pinned' ? 2.25 : 1.5) : 0
      )

    // Quality glyph (advanced mode only): ⊘ dead / ◑ mixed, drawn inside the
    // counter-scaled inner group so it tracks the bubble at any zoom.
    const qualityGlyph: Record<string, string> = { dead: '⊘', mixed: '◑' }
    const qualityColor: Record<string, string> = { dead: '#dc2626', mixed: '#d97706' }
    if ($advancedView) {
      inner.append('text')
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

    // Labels live in a SEPARATE layer appended after all nodes so they always sit
    // on top of neighboring bubbles. Each label mirrors a node: translate(pos) for
    // spread + an inner scale(1/k) so the text stays a constant size. Which labels
    // are shown is set via the `display` attr from the progressive-reveal cutoff
    // (see labelVisible) — the SET is keyed off stable corpus_prevalence rank, so
    // it never reshuffles as the covariate controls move, only reveals more on zoom.
    const truncate = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + '…' : s)
    const labelCandidates = phenotypes.filter((p) => labelRankMap.has(p.id))
    const labelLayer = g.append('g').attr('class', 'label-layer')
    const labelSel = labelLayer.selectAll('g.label')
      .data(labelCandidates)
      .join('g')
      .attr('class', 'label')
      .attr('transform', (p) => `translate(${x(coords[p.id][0])}, ${y(coords[p.id][1])})`)
      .attr('display', (p) => (labelVisible(p.id, currentK) ? null : 'none'))
    labelSel.append('g')
      .attr('class', 'inner')
      .attr('transform', `scale(${1 / currentK})`)
      .append('text')
      .attr('class', 'minor-label')
      .attr('x', 0)
      .attr('y', (p) => -(rad(p) + 6))
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
      // Prevalence text always reads off the prevalence reader (`reader`),
      // NOT the size source (`r_of`) — the two diverge once predictive_gain
      // is present and size encodes mean_gain (nats) instead. A gain suffix
      // is appended only in that case so the fallback tooltip is byte-for-
      // byte unchanged.
      const pat = (reader(p) * 100).toFixed(1)
      const npmi = p.npmi == null ? '—' : p.npmi.toFixed(3)
      const tauStr = $tauThreshold.toFixed(2)
      const label = p.label || `Phenotype ${p.id}`
      const gainSuffix = hasPredictiveGain ? ` · gain ${r_of(p).toFixed(2)} nats` : ''
      if ($advancedView) {
        const mass = (p.corpus_prevalence * 100).toFixed(1)
        return `${label}\nCoherence ${npmi} · prev ${pat}% (patients, θ > ${tauStr}) · topic mass ${mass}%${gainSuffix}`
      }
      return `${label}\nCoherence ${npmi} · prev ${pat}% (θ > ${tauStr})${gainSuffix}`
    })
  }

  // `reader`/`sizeReader` are listed so the atlas re-renders whenever the
  // prevalence reader or the size source changes for ANY reason - covariate
  // active toggling, covariate-value edits, or the gating group selector -
  // not only on the tau/selection/mode stores.
  $: reader, sizeReader, $conditioning, $tauThreshold, $selectedPhenotypeId, $hoveredCodeIdx, $advancedView, $searchedConditionIdx, $bundle && svgEl && coords.length && render()
  onMount(render)
</script>

<figure class="map" data-tour="atlas-map">
  <div class="map-canvas">
    <svg bind:this={svgEl} role="img" aria-label="Phenotype atlas" preserveAspectRatio="xMidYMid meet"></svg>
    <!-- Slotted covariate drawer (ConditioningBar inline mode): absolutely
         positioned within this canvas so opening/closing never reflows the map. -->
    <slot />
    <button class="reset-view" type="button" on:click={resetView} title="Reset zoom and pan" aria-label="Reset zoom and pan">↺</button>
    <!-- Cohort color key, floated into the plot (lower-right) so it reads against
         the bubbles it explains. Semi-opaque so underlying nodes stay visible. -->
    {#if $bundle?.gating}
      <div class="cohort-key">
        <span class="cohort-item"><span class="sw" style="background:{grpColor('background')}"></span>All</span>
        {#each $bundle.gating.groups as g}
          <span class="cohort-item"><span class="sw" style="background:{grpColor(g)}"></span>{groupLabel(g)}</span>
        {/each}
      </div>
    {/if}
  </div>
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
        {#if hasPredictiveGain}
          <div class="legend-group">
            <span class="eyebrow" title={copy.atlas.legend.meanGain}>Predictive contribution (nats)<span class="help-mark" aria-hidden="true">?</span></span>
            <span class="size-marks" aria-hidden="true">
              <span class="dot s1"></span><span class="dot s2"></span><span class="dot s3"></span>
            </span>
          </div>
        {:else}
          <div class="legend-group">
            <span class="eyebrow" title={copy.atlas.legend.prevalence($tauThreshold)}>Prevalence (% Over Threshold)<span class="help-mark" aria-hidden="true">?</span></span>
            <span class="size-marks" aria-hidden="true">
              <span class="dot s1"></span><span class="dot s2"></span><span class="dot s3"></span>
            </span>
          </div>
        {/if}
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
  /* Positioned wrapper so the slotted covariate drawer + the reset-view button
     can float over the canvas without reflowing the map. */
  .map-canvas {
    position: relative;
  }
  svg {
    width: 100%;
    height: auto;
    display: block;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    cursor: grab;
    touch-action: none;
  }
  svg:active { cursor: grabbing; }
  /* Compact icon button (↺). Square, so it stays out of the way of the drawer
     button top-left and the cohort key bottom-right. */
  .reset-view {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border: 1px solid var(--rule-strong);
    background: color-mix(in srgb, var(--surface) 88%, transparent);
    color: var(--ink-muted);
    border-radius: var(--radius-sm);
    font-size: 15px;
    line-height: 1;
    cursor: pointer;
    transition: color 0.12s ease, border-color 0.12s ease;
  }
  .reset-view:hover {
    color: var(--ink);
    border-color: var(--ink-muted);
  }
  /* In-plot cohort color key, lower-right, semi-opaque so bubbles show through. */
  .cohort-key {
    position: absolute;
    right: 0.5rem;
    bottom: 0.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 0.4rem 0.55rem;
    background: color-mix(in srgb, var(--surface) 82%, transparent);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    font-size: var(--fs-micro);
    color: var(--ink-muted);
    backdrop-filter: blur(2px);
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
    background: linear-gradient(to right, rgba(100, 116, 139, 0.45), rgba(100, 116, 139, 0.95));
  }
  /* Cohort swatches (in the in-plot .cohort-key). */
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
