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
</script>

<aside class="code-panel">
  {#if !pheno}<p>Click a phenotype on the map.</p>
  {:else}
    <h3>{pheno.label || `Phenotype ${pheno.id}`}</h3>
    <p class="meta">
      prevalence {(pheno.corpus_prevalence * 100).toFixed(1)}%
      {#if $advancedView} · NPMI {pheno.npmi.toFixed(3)}{/if}
      {#if $advancedView && pheno.original_topic_id !== pheno.id} · model topic #{pheno.original_topic_id}{/if}
      {#if pheno.junk_flag}<span class="junk">low-coherence</span>{/if}
    </p>

    {#if $advancedView}
      <label class="slider">
        Relevance λ = {lambda.toFixed(2)}
        <input type="range" min="0" max="1" step="0.05" bind:value={lambda} />
      </label>
    {/if}

    <ol class="codes">
      {#each top as r}
        {@const c = $bundle!.vocab.codes[r.index]}
        <li
          on:mouseenter={() => hoveredCodeIdx.set(r.index)}
          on:mouseleave={() => hoveredCodeIdx.set(null)}
        >
          <span class="dom dom-{c.domain}">{c.domain.slice(0, 3)}</span>
          <span class="desc">{c.description || c.code}</span>
          <span class="num">{(r.pwk * 100).toFixed(2)}%</span>
        </li>
      {/each}
    </ol>
  {/if}
</aside>

<style>
  .code-panel { padding: 1rem; border: 1px solid #ddd; min-height: 480px; }
  h3 { margin: 0 0 0.25rem; }
  .meta { font-size: 0.85rem; color: #555; margin: 0 0 1rem; }
  .junk { color: #b00020; margin-left: 0.5rem; font-size: 0.75rem; }
  .slider { display: block; margin-bottom: 0.75rem; font-size: 0.85rem; }
  .slider input { width: 100%; }
  ol.codes { list-style: none; padding: 0; margin: 0; }
  ol.codes li { display: grid; grid-template-columns: 3rem 1fr auto; gap: 0.5rem; padding: 0.2rem 0; font-size: 0.85rem; border-bottom: 1px solid #f4f4f4; }
  .dom { font-size: 0.7rem; padding: 0.05rem 0.3rem; border-radius: 3px; text-align: center; }
  .dom-condition { background: #ffe4e1; }
  .dom-drug { background: #e0f2fe; }
  .dom-procedure { background: #ecfccb; }
  .dom-measurement { background: #fef3c7; }
  .dom-observation { background: #f5f5f5; }
  .num { font-variant-numeric: tabular-nums; color: #444; }
</style>
