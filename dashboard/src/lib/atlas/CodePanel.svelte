<script lang="ts">
  import {
    bundle, selectedPhenotypeId, advancedView, hoveredCodeIdx,
    searchedConditionIdx, searchedPhenotypeForPatients,
    prevalenceReader, tauThreshold,
  } from '../store'
  import { topRelevantCodes } from '../inference'
  import { go } from '../router'
  import PrevalenceHistogram from '../PrevalenceHistogram.svelte'
  import { copy } from '../copy'

  function findInPatients() {
    if ($selectedPhenotypeId === null) return
    searchedPhenotypeForPatients.set($selectedPhenotypeId)
    go('patient')
  }

  let lambda = 0.6
  const topN = 20

  $: pheno = $bundle && $selectedPhenotypeId !== null
    ? $bundle.phenotypes.phenotypes[$selectedPhenotypeId] : null

  $: top = $bundle && pheno
    ? topRelevantCodes({
        pwk: $bundle.model.beta[pheno.id],
        pw: $bundle.vocab.codes.map((c) => c.corpus_freq),
        lambda,
        n: topN,
      })
    : []

  $: maxPwk = top.length ? Math.max(...top.map((r) => r.pwk)) : 1

  $: reader = $prevalenceReader
  $: hasHistogram = !!(pheno?.theta_histogram && pheno?.theta_percentiles && $bundle?.phenotypes.theta_histogram_bin_edges)

  // Share of patients below τ — the mass the histogram omits because its
  // x-axis starts at τ. Summed from bins whose upper edge is at or below τ;
  // suppressed bins (null) contribute 0, matching the privacy round-to-zero
  // rule used by fractionAboveTau in the store.
  function fractionBelowTau(hist: (number | null)[], edges: number[], tau: number): number {
    let s = 0
    for (let i = 0; i < hist.length; i++) {
      if (edges[i + 1] <= tau + 1e-9) {
        const v = hist[i]
        if (v != null) s += v
      }
    }
    return s
  }

  // Lay-readable rubric explanations for the quality chip; copy lives in copy.ts.
  function qualityTooltip(q: string): string {
    return (copy.phenotypeDetail.quality as Record<string, string>)[q] ?? ''
  }
</script>

<aside class="code-panel" data-tour="phenotype-detail">
  {#if !pheno}
    <div class="empty">
      <span class="eyebrow">Detail</span>
      <p class="empty-msg">{copy.phenotypeDetail.empty}</p>
    </div>
  {:else}
    <header class="head">
      <div class="eyebrow-row">
        <!-- Cyan dot mirrors the cyan selection ring on the bubble map,
             so the eye links "this detail" with "that selected bubble". -->
        <span class="sel-dot" aria-hidden="true"></span>
        <span class="eyebrow">Phenotype detail</span>
        <button
          class="find-in-patients"
          type="button"
          on:click={findInPatients}
          title={copy.phenotypeDetail.findInPatientsTip}
          data-tour="find-in-patients"
        >
          find in patients →
        </button>
      </div>
      <h2 class="title">{pheno.label || `Phenotype ${pheno.id}`}</h2>
      {#if pheno.description}
        <p class="desc-text">{pheno.description}</p>
      {/if}
      <div class="stats" data-numeric data-tour="detail-stats">
        <span class="stat" title={hasHistogram
          ? ($advancedView
            ? copy.phenotypeDetail.prevalence.tipAdvanced($tauThreshold)
            : copy.phenotypeDetail.prevalence.tipBasic($tauThreshold))
          : copy.phenotypeDetail.prevalence.tipNoHistogram
        }>
          <span class="stat-k">{hasHistogram && $advancedView ? copy.phenotypeDetail.prevalence.labelAdvanced : copy.phenotypeDetail.prevalence.labelBasic}<span class="help-mark" aria-hidden="true">?</span></span>
          <span class="stat-v">{(reader(pheno) * 100).toFixed(1)}%</span>
        </span>
        {#if $advancedView}
          {#if hasHistogram}
            <span class="stat" title={copy.phenotypeDetail.topicMassTip}>
              <span class="stat-k">Topic mass</span>
              <span class="stat-v">{(pheno.corpus_prevalence * 100).toFixed(1)}%</span>
            </span>
          {/if}
          <span class="stat" title={copy.phenotypeDetail.coherenceTip}>
            <span class="stat-k">Coherence</span>
            <span class="stat-v">{pheno.npmi == null ? '—' : pheno.npmi.toFixed(3)}</span>
          </span>
          <span class="stat" title={copy.phenotypeDetail.pairCoverageTip}>
            <span class="stat-k">Pair cov</span>
            <span class="stat-v">{pheno.pair_coverage == null ? '—' : (pheno.pair_coverage * 100).toFixed(0) + '%'}</span>
          </span>
          {#if pheno.original_topic_id !== pheno.id}
            <span class="stat" title={copy.phenotypeDetail.sourceTip}>
              <span class="stat-k">Source #</span>
              <span class="stat-v">{pheno.original_topic_id}</span>
            </span>
          {/if}
          {#if pheno.quality}
            <span class="stat quality q-{pheno.quality}" title={qualityTooltip(pheno.quality)}>
              <span class="stat-k">Quality</span>
              <span class="stat-v">{pheno.quality}</span>
            </span>
          {/if}
        {/if}
      </div>
    </header>

    {#if hasHistogram && $advancedView}
      {@const edges = $bundle!.phenotypes.theta_histogram_bin_edges!}
      {@const hist = pheno.theta_histogram!}
      {@const pcts = pheno.theta_percentiles!}
      {@const belowTau = fractionBelowTau(hist, edges, $tauThreshold)}
      <div class="hist-wrap" data-tour="histogram">
        <span class="hist-head">
          <span class="eyebrow" title={copy.phenotypeDetail.histogram.tip}>{copy.phenotypeDetail.histogram.title}</span>
          <span class="hist-below" data-numeric title={copy.phenotypeDetail.histogram.belowTauTip}>
            {(belowTau * 100).toFixed(0)}% &lt; {($tauThreshold * 100).toFixed(0)}%
          </span>
        </span>
        <PrevalenceHistogram
          histogram={hist}
          binEdges={edges}
          percentiles={pcts}
          tau={$tauThreshold}
          width={360}
          height={120}
        />
      </div>
    {:else}
      <!-- Basic view, and HDP / legacy bundles: skip the histogram. -->
    {/if}

    <!--
      The leading-conditions table.

      Basic mode shows a single column header "Relevance" with a lay-readable
      tooltip. Advanced mode exposes the underlying lift/frequency weighting
      via a slider so the user can re-rank by pure lift (rare-but-concentrated
      conditions) versus pure frequency (raw probability under the topic).
      λ=0.6 is the default (Sievert & Shirley's recommended compromise).
    -->
    {#if $advancedView}
      <div class="slider-row" data-tour="relevance">
        <label class="slider">
          <span class="slider-head">
            <span class="slider-k" title={copy.phenotypeDetail.relevance.weightingTip}>
              <span class="eyebrow">Relevance term weighting</span>
            </span>
            <span class="slider-v" data-numeric>λ {lambda.toFixed(2)}</span>
          </span>
          <input type="range" min="0" max="1" step="0.05" bind:value={lambda} />
          <span class="slider-ends">
            <span title={copy.phenotypeDetail.relevance.liftEndTip}>lift</span>
            <span title={copy.phenotypeDetail.relevance.freqEndTip}>frequency</span>
          </span>
        </label>
      </div>
    {/if}

    <div class="col-head">
      <span class="ch-rank">#</span>
      <span class="ch-cond">Condition</span>
      <span
        class="ch-rel"
        title={
          $advancedView
            ? copy.phenotypeDetail.relevance.colTipAdvanced
            : copy.phenotypeDetail.relevance.colTipBasic
        }
      >Relevance<span class="help-mark" aria-hidden="true">?</span> ▾</span>
    </div>

    <ol class="codes">
      {#each top as r, i}
        {@const c = $bundle!.vocab.codes[r.index]}
        {@const isSearched = $searchedConditionIdx === r.index}
        <li
          class:searched={isSearched}
          on:mouseenter={() => hoveredCodeIdx.set(r.index)}
          on:mouseleave={() => hoveredCodeIdx.set(null)}
        >
          <span class="rank" data-numeric>{i + 1}</span>
          <span class="desc" title={c.code}>{c.description || c.code}</span>
          <span class="spark" aria-hidden="true">
            <span class="spark-bar" style="width: {(r.pwk / maxPwk) * 100}%"></span>
          </span>
          <span class="num" data-numeric>{(r.pwk * 100).toFixed(2)}<span class="unit">%</span></span>
        </li>
      {/each}
    </ol>
  {/if}
</aside>

<style>
  .code-panel {
    padding: 1.25rem 1.25rem 1rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    min-height: 560px;
    display: flex;
    flex-direction: column;
  }

  .empty {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    padding: 2rem 0;
  }
  .empty-msg {
    margin: 0;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    max-width: 32ch;
    line-height: 1.6;
  }

  .head {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    padding-bottom: 1rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--rule);
  }
  .eyebrow-row {
    display: flex;
    align-items: center;
    gap: 0.45rem;
  }
  .find-in-patients {
    margin-left: auto;
    border: 1px solid var(--rule-strong);
    background: var(--surface);
    color: var(--ink-muted);
    padding: 0.25rem 0.6rem;
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    cursor: pointer;
    flex-shrink: 0;
    transition: color 0.12s ease, border-color 0.12s ease;
  }
  .find-in-patients:hover {
    color: var(--accent-find);
    border-color: var(--accent-find);
  }
  .sel-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 0 2px var(--accent-faint);
  }
  .title {
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: var(--tracking-display);
    line-height: 1.15;
    color: var(--ink);
  }
  .stats {
    display: flex;
    flex-wrap: wrap;
    gap: 1.1rem;
    margin-top: 0.35rem;
  }
  .stat {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    cursor: help;
  }
  .stat-k {
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-faint);
    font-weight: 500;
  }
  .stat-v {
    font-size: var(--fs-small);
    color: var(--ink);
    font-weight: 500;
  }
  /* Small circled "?" cueing that the prevalence label carries a hover
     explanation. Inherits the .stat's `cursor: help`; the tooltip itself
     lives on the parent .stat title. */
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
    letter-spacing: 0;
    color: var(--ink-faint);
    vertical-align: middle;
    transition: color 0.12s ease, border-color 0.12s ease;
  }
  .stat:hover .help-mark {
    color: var(--accent);
    border-color: var(--accent);
  }
  .desc-text {
    margin: 0.1rem 0 0.15rem;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    line-height: 1.5;
    max-width: 60ch;
  }
  .stat.quality .stat-v {
    text-transform: capitalize;
    font-weight: 600;
  }
  .stat.q-dead .stat-v { color: var(--danger); }
  .stat.q-mixed .stat-v { color: #b45309; }
  .stat.q-anchor .stat-v { color: var(--accent); }
  .stat.q-background .stat-v { color: var(--ink-muted); }

  /* Slider row */
  .slider-row {
    padding: 0.55rem 0 0.85rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid var(--rule-faint);
  }
  .slider {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  .slider-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }
  .slider-v {
    font-size: var(--fs-small);
    color: var(--accent);
    font-weight: 500;
  }
  .slider-ends {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .slider-ends span {
    cursor: help;
  }

  /* Column header sits above the list, mirrors the table grid so the labels
     line up with the corresponding columns. */
  .col-head {
    display: grid;
    grid-template-columns: 1.8rem 1fr 56px 4rem;
    align-items: baseline;
    gap: 0.7rem;
    padding: 0.2rem 0 0.45rem;
    border-bottom: 1px solid var(--rule);
    margin-bottom: 0.1rem;
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-faint);
    font-weight: 500;
  }
  .ch-rank { text-align: right; }
  .ch-rel { text-align: right; cursor: help; grid-column: 3 / span 2; }
  .ch-rel:hover .help-mark { color: var(--accent); border-color: var(--accent); }

  /* Code list */
  .codes {
    list-style: none;
    padding: 0;
    margin: 0;
    overflow-y: auto;
    flex: 1;
  }
  .codes li {
    display: grid;
    grid-template-columns: 1.8rem 1fr 56px 4rem;
    align-items: center;
    gap: 0.7rem;
    padding: 0.42rem 0.45rem;
    border-bottom: 1px solid var(--rule-faint);
    font-size: var(--fs-small);
    transition: background 0.12s ease;
    border-radius: 3px;
  }
  .codes li:hover {
    background: var(--surface-recessed);
  }
  .codes li:last-child { border-bottom: 0; }
  .codes li.searched {
    background: var(--accent-search-soft);
    box-shadow: inset 3px 0 0 var(--accent-search);
  }
  .codes li.searched:hover {
    background: var(--accent-search-soft);
  }
  .codes .rank {
    color: var(--ink-faint);
    text-align: right;
    font-size: var(--fs-micro);
    font-family: var(--font-mono);
  }
  .codes li.searched .rank {
    color: var(--accent-search-ink);
    font-weight: 600;
  }
  .codes .desc {
    color: var(--ink);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .spark {
    display: block;
    height: 3px;
    background: var(--surface-recessed);
    border-radius: 1.5px;
    overflow: hidden;
  }
  .spark-bar {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 0.25s ease;
  }
  .codes li.searched .spark-bar {
    background: var(--accent-search);
  }
  .num {
    text-align: right;
    color: var(--ink);
    font-size: var(--fs-small);
  }
  .num .unit {
    color: var(--ink-faint);
    font-weight: 400;
    margin-left: 1px;
  }

  .hist-wrap {
    padding: 0.55rem 0 0.85rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid var(--rule-faint);
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .hist-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }
  .hist-below {
    font-size: var(--fs-micro);
    font-family: var(--font-mono);
    color: var(--ink-faint);
    cursor: help;
  }
</style>
