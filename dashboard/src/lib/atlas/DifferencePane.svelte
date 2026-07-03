<script lang="ts">
  import { bundle, comparePair } from '../store'
  import { topDifferentialCodes } from './difference'

  // Relevance-term weighting for the delta ranking — same λ semantics as
  // CodePanel's relevance slider (Sievert & Shirley 2014), just applied to
  // the A-vs-B contrast instead of a single phenotype's own ranking.
  let lambda = 0.6

  $: labelById = new Map(
    ($bundle?.phenotypes.phenotypes ?? []).map((p) => [p.id, p.label || `Phenotype ${p.id}`]),
  )
  const nameOf = (id: number) => labelById.get(id) ?? `Phenotype ${id}`

  $: corpusFreq = $bundle ? $bundle.vocab.codes.map((c) => c.corpus_freq) : []

  $: pair = $comparePair
  $: valid = !!(pair && pair.a !== pair.b && $bundle)

  $: result = valid && pair && $bundle
    ? topDifferentialCodes({
        betaA: $bundle.model.beta[pair.a],
        betaB: $bundle.model.beta[pair.b],
        pw: corpusFreq,
        lambda,
        n: 15,
      })
    : null

  function describe(index: number): string {
    if (!$bundle) return ''
    const c = $bundle.vocab.codes[index]
    return c.description || c.code
  }
</script>

<aside class="difference-pane" data-tour="phenotype-difference">
  {#if !valid || !result || !pair}
    <div class="empty">
      <span class="eyebrow">Difference</span>
      <p class="empty-msg">Click a cell in the heatmap to compare two phenotypes.</p>
    </div>
  {:else}
    <header class="head">
      <span class="eyebrow">Phenotype difference</span>
      <h2 class="title">{nameOf(pair.a)} <span class="vs">vs</span> {nameOf(pair.b)}</h2>
    </header>

    <div class="slider-row">
      <label class="slider">
        <span class="slider-head">
          <span class="slider-k">
            <span class="eyebrow">Relevance term weighting</span>
          </span>
          <span class="slider-v" data-numeric>λ {lambda.toFixed(2)}</span>
        </span>
        <input type="range" min="0" max="1" step="0.05" bind:value={lambda} />
        <span class="slider-ends">
          <span>Lift</span>
          <span>Frequency</span>
        </span>
      </label>
    </div>

    <div class="sides">
      <div class="side">
        <h3 class="side-head">More in {nameOf(pair.a)}</h3>
        <ol class="rows">
          {#each result.aSide as r (r.index)}
            <li>
              <span class="desc" title={describe(r.index)}>{describe(r.index)}</span>
              <span class="num" data-numeric>{r.delta.toFixed(2)}</span>
            </li>
          {/each}
        </ol>
      </div>
      <div class="side">
        <h3 class="side-head">More in {nameOf(pair.b)}</h3>
        <ol class="rows">
          {#each result.bSide as r (r.index)}
            <li>
              <span class="desc" title={describe(r.index)}>{describe(r.index)}</span>
              <span class="num" data-numeric>{(-r.delta).toFixed(2)}</span>
            </li>
          {/each}
        </ol>
      </div>
    </div>
  {/if}
</aside>

<style>
  .difference-pane {
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

  .eyebrow {
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-faint);
    font-weight: 500;
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
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: var(--tracking-display);
    line-height: 1.3;
    color: var(--ink);
  }
  .vs {
    color: var(--ink-faint);
    font-weight: 400;
    font-size: 0.85em;
  }

  .slider-row {
    padding: 0.55rem 0 0.85rem;
    margin-bottom: 0.85rem;
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

  .sides {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.25rem;
    flex: 1;
    min-height: 0;
  }
  .side {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .side-head {
    margin: 0 0 0.5rem;
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-faint);
    font-weight: 500;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--rule);
  }
  .rows {
    list-style: none;
    padding: 0;
    margin: 0;
    overflow-y: auto;
  }
  .rows li {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.5rem;
    padding: 0.42rem 0.2rem;
    border-bottom: 1px solid var(--rule-faint);
    font-size: var(--fs-small);
  }
  .rows li:last-child { border-bottom: 0; }
  .rows .desc {
    color: var(--ink);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .rows .num {
    color: var(--ink-muted);
    flex-shrink: 0;
  }
</style>
