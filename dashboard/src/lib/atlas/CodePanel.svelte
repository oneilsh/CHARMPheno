<script lang="ts">
  import { bundle, selectedPhenotypeId, advancedView, hoveredCodeIdx } from '../store'
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
</script>

<aside class="code-panel">
  {#if !pheno}
    <div class="empty">
      <span class="eyebrow">Detail</span>
      <p class="empty-msg">Select a phenotype on the map to read its top codes.</p>
    </div>
  {:else}
    <header class="head">
      <span class="eyebrow">Phenotype detail</span>
      <h2 class="title">{pheno.label || `Phenotype ${pheno.id}`}</h2>
      {#if pheno.description}
        <p class="desc-text">{pheno.description}</p>
      {/if}
      <div class="stats" data-numeric>
        <span class="stat"><span class="stat-k">Prev</span> <span class="stat-v">{(pheno.corpus_prevalence * 100).toFixed(1)}%</span></span>
        {#if $advancedView}
          <span class="stat"><span class="stat-k">NPMI</span> <span class="stat-v">{pheno.npmi.toFixed(3)}</span></span>
          <span class="stat"><span class="stat-k">Pair cov</span> <span class="stat-v">{(pheno.pair_coverage * 100).toFixed(0)}%</span></span>
          {#if pheno.original_topic_id !== pheno.id}
            <span class="stat"><span class="stat-k">Topic</span> <span class="stat-v">#{pheno.original_topic_id}</span></span>
          {/if}
          {#if pheno.quality}
            <span class="stat quality q-{pheno.quality}"><span class="stat-k">Quality</span> <span class="stat-v">{pheno.quality}</span></span>
          {/if}
        {/if}
      </div>
    </header>

    {#if $advancedView}
      <div class="slider-row">
        <label class="slider">
          <span class="slider-head">
            <span class="slider-k"><span class="eyebrow">Relevance λ</span></span>
            <span class="slider-v" data-numeric>{lambda.toFixed(2)}</span>
          </span>
          <input type="range" min="0" max="1" step="0.05" bind:value={lambda} />
          <span class="slider-ends">
            <span>lift</span><span>frequency</span>
          </span>
        </label>
      </div>
    {/if}

    <ol class="codes">
      {#each top as r}
        {@const c = $bundle!.vocab.codes[r.index]}
        <li
          on:mouseenter={() => hoveredCodeIdx.set(r.index)}
          on:mouseleave={() => hoveredCodeIdx.set(null)}
        >
          <span class="domain-mark dom-{c.domain}">{c.domain.slice(0, 3)}</span>
          <span class="desc">{c.description || c.code}</span>
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
    grid-template-columns: 4.5rem 1fr 56px 4rem;
    align-items: center;
    gap: 0.7rem;
    padding: 0.42rem 0;
    border-bottom: 1px solid var(--rule-faint);
    font-size: var(--fs-small);
    transition: background 0.12s ease;
  }
  .codes li:hover {
    background: var(--surface-recessed);
  }
  .codes li:last-child { border-bottom: 0; }
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
