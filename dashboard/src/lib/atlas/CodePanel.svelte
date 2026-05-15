<script lang="ts">
  import {
    bundle, selectedPhenotypeId, advancedView, hoveredCodeIdx,
    searchedConditionIdx,
  } from '../store'
  import { topRelevantCodes } from '../inference'

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

  // Lay-readable rubric explanations for the quality chip.
  function qualityTooltip(q: string): string {
    switch (q) {
      case 'phenotype': return 'Quality: phenotype — a clinically coherent pattern of conditions; the cluster names a recognisable disease or related family.'
      case 'background': return 'Quality: background — broad, non-specific health-care activity (general checkups, common comorbidities). Real signal but not disease-specific.'
      case 'anchor': return 'Quality: anchor — dominated by one or two very common conditions; useful as a reference point but lower information content than a fuller phenotype.'
      case 'mixed': return 'Quality: mixed — the leading conditions span multiple unrelated clinical areas; the topic merged what should probably be separate phenotypes.'
      case 'dead': return 'Quality: dead — minimal usage by the model and very small divergence from the corpus-average distribution; this topic slot was effectively unused. (Hidden in simple mode.)'
      default: return ''
    }
  }
</script>

<aside class="code-panel">
  {#if !pheno}
    <div class="empty">
      <span class="eyebrow">Detail</span>
      <p class="empty-msg">Select a phenotype on the map to read its top conditions.</p>
    </div>
  {:else}
    <header class="head">
      <div class="eyebrow-row">
        <!-- Cyan dot mirrors the cyan selection ring on the bubble map,
             so the eye links "this detail" with "that selected bubble". -->
        <span class="sel-dot" aria-hidden="true"></span>
        <span class="eyebrow">Phenotype detail</span>
      </div>
      <h2 class="title">{pheno.label || `Phenotype ${pheno.id}`}</h2>
      {#if pheno.description}
        <p class="desc-text">{pheno.description}</p>
      {/if}
      <div class="stats" data-numeric>
        <span class="stat" title="Prevalence — fraction of patient-year documents that draw meaningfully from this phenotype.">
          <span class="stat-k">Prevalence</span>
          <span class="stat-v">{(pheno.corpus_prevalence * 100).toFixed(1)}%</span>
        </span>
        {#if $advancedView}
          <span class="stat" title="Coherence — how reliably this phenotype's leading conditions co-occur in real patients (NPMI: normalized pointwise mutual information). Higher means the conditions really do show up together.">
            <span class="stat-k">Coherence</span>
            <span class="stat-v">{pheno.npmi.toFixed(3)}</span>
          </span>
          <span class="stat" title="Pair coverage — fraction of the leading-condition pairs that had enough joint observations to actually contribute to the coherence number. Low coverage means the coherence value was computed on only a few pairs and is less trustworthy.">
            <span class="stat-k">Pair cov</span>
            <span class="stat-v">{(pheno.pair_coverage * 100).toFixed(0)}%</span>
          </span>
          {#if pheno.original_topic_id !== pheno.id}
            <span class="stat" title="Source # — the raw topic index from the LDA fit before sorting. Useful for cross-referencing the underlying model.">
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

    <!-- The leading-conditions table.

         Simple mode shows a single sortable column header "Relevance" with a
         lay-readable tooltip. Advanced mode exposes the underlying
         lift/frequency weighting via a slider so the user can re-rank by
         pure lift (rare-but-concentrated conditions) versus pure frequency
         (raw probability under the topic). λ=0.6 is the default — Sievert &
         Shirley's recommended compromise. -->
    {#if $advancedView}
      <div class="slider-row">
        <label class="slider">
          <span class="slider-head">
            <span class="slider-k" title="Relevance term weighting — the slider blends two views of 'top conditions': raw frequency (how much of the phenotype's mass falls on this condition) and lift (how much more this condition shows up here than in the overall corpus). Slide left for surprise/lift, right for sheer frequency.">
              <span class="eyebrow">Relevance term weighting</span>
            </span>
            <span class="slider-v" data-numeric>λ {lambda.toFixed(2)}</span>
          </span>
          <input type="range" min="0" max="1" step="0.05" bind:value={lambda} />
          <span class="slider-ends">
            <span title="Lift — how much more this condition appears in this phenotype than in the corpus overall. Surface rare-but-concentrated conditions.">lift</span>
            <span title="Frequency — the condition's raw probability under this phenotype.">frequency</span>
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
            ? 'Relevance — λ·log p(w|k) + (1−λ)·log lift. The slider tunes how much frequency vs lift the ranking favors. Bar shows raw frequency p(w|k); number is its percentage.'
            : "Relevance — the leading conditions for this phenotype, ranked by a balance of how often they appear here AND how distinctive they are to this phenotype. The bar shows the condition's share of the phenotype's probability mass."
        }
      >Relevance ▾</span>
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
</style>
