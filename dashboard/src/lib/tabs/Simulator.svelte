<script lang="ts">
  import { bundle, simulatorPrefix } from '../store'
  import { runSimulator } from '../simulator/runSamples'
  import PrefixEditor from '../simulator/PrefixEditor.svelte'
  import Carpet from '../simulator/Carpet.svelte'
  import ExpectedCodes from '../simulator/ExpectedCodes.svelte'

  let nSamples = 1000
  let sortMode: 'median' | 'spread' | 'npmi' | 'id' = 'median'
  let seed = 0
  let result: ReturnType<typeof runSimulator> | null = null
  let running = false

  async function runSim() {
    if (!$bundle) return
    running = true
    await new Promise((r) => setTimeout(r, 0))
    result = runSimulator({
      alpha: $bundle.model.alpha,
      beta: $bundle.model.beta,
      meanCodesPerDoc: $bundle.corpusStats.mean_codes_per_doc,
      prefix: $simulatorPrefix,
      nSamples, seed,
    })
    running = false
  }
</script>

<section class="sim">
  <header class="section-head">
    <div class="title-block">
      <h1>Simulator</h1>
      <p class="kicker">Given a partial code bag, draw <span data-numeric>N</span> samples from the model's posterior predictive. Each sample is one complete year-of-life bag.</p>
    </div>
    <div class="controls">
      <label class="control n-control">
        <span class="ctl-head"><span class="eyebrow">N samples</span> <span class="ctl-v" data-numeric>{nSamples}</span></span>
        <input type="range" min="10" max="2000" step="10" bind:value={nSamples} />
      </label>
      <label class="control">
        <span class="eyebrow">Sort</span>
        <select bind:value={sortMode}>
          <option value="median">Median θ</option>
          <option value="spread">Spread (P90–P10)</option>
          <option value="npmi" title="How reliably the leading conditions co-occur in the corpus (NPMI).">Coherence</option>
          <option value="id">Phenotype id</option>
        </select>
      </label>
      <label class="control seed-control">
        <span class="eyebrow">Seed</span>
        <input type="number" bind:value={seed} />
      </label>
      <button class="btn btn-primary run-btn" on:click={runSim} disabled={running}>
        {running ? 'sampling…' : 'run sampler →'}
      </button>
    </div>
  </header>

  <div class="grid">
    <PrefixEditor />
    <div class="main">
      {#if result}
        <Carpet thetaSamples={result.thetaSamples} codeCountsSamples={result.codeCountsSamples} {sortMode} />
        <ExpectedCodes codeCountsSamples={result.codeCountsSamples} />
      {:else}
        <div class="empty">
          <span class="eyebrow">Awaiting input</span>
          <p class="empty-msg">Compose a prefix on the left, then run the sampler to see the posterior over phenotypes and expected codes.</p>
        </div>
      {/if}
    </div>
  </div>

  <p class="footnote">
    <strong>Scope.</strong> Year-of-life; code ordering and timing are not modeled. Each sample is one complete bag.
    <span data-numeric>N = {nSamples}</span>,
    <span data-numeric>K = {$bundle?.model.K ?? '?'}</span>,
    <span data-numeric>prefix = {$simulatorPrefix.length}</span>.
  </p>
</section>

<style>
  .sim { padding: 0.25rem 0 3rem; }

  .section-head {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    gap: 2rem;
    padding-bottom: 1.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--rule);
  }
  .title-block { display: flex; flex-direction: column; gap: 0.45rem; }
  .title-block h1 { margin: 0.1rem 0 0; }
  .kicker {
    margin: 0.25rem 0 0;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    max-width: 62ch;
    line-height: 1.55;
  }

  .controls {
    display: flex;
    align-items: end;
    gap: 1rem;
  }
  .control {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .ctl-head {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
  }
  .ctl-v {
    font-size: var(--fs-small);
    color: var(--accent);
    font-weight: 500;
  }
  .n-control { min-width: 180px; }
  .n-control input[type="range"] { width: 180px; }
  .seed-control input { width: 5rem; }
  .run-btn {
    font-size: var(--fs-small);
    padding: 0.5rem 1rem;
  }

  .grid {
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 1.25rem;
  }
  .main {
    display: grid;
    gap: 1rem;
    align-content: start;
  }

  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.65rem;
    padding: 3rem 1rem;
    background: var(--surface);
    border: 1px dashed var(--rule-strong);
    border-radius: var(--radius-sm);
    text-align: center;
  }
  .empty-msg {
    margin: 0 auto;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    max-width: 42ch;
    line-height: 1.6;
  }

  .footnote {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--rule-faint);
    font-size: var(--fs-micro);
    color: var(--ink-faint);
  }
  .footnote strong {
    color: var(--ink-muted);
    font-weight: 600;
  }
</style>
