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
      <span class="eyebrow">Section III</span>
      <h1>Simulator</h1>
      <p class="kicker">Given a partial code bag, draw <em>N</em> samples from the model's posterior predictive. Each sample is one complete year-of-life bag.</p>
    </div>
    <div class="controls">
      <label class="control n-control">
        <span class="eyebrow">N samples <span class="ctl-v" data-numeric>{nSamples}</span></span>
        <input type="range" min="10" max="2000" step="10" bind:value={nSamples} />
      </label>
      <label class="control">
        <span class="eyebrow">Sort</span>
        <select bind:value={sortMode}>
          <option value="median">Median θ</option>
          <option value="spread">Spread (P90–P10)</option>
          <option value="npmi">NPMI</option>
          <option value="id">Phenotype id</option>
        </select>
      </label>
      <label class="control seed-control">
        <span class="eyebrow">Seed</span>
        <input type="number" bind:value={seed} />
      </label>
      <button class="btn run-btn" on:click={runSim} disabled={running}>
        {running ? '… sampling' : '→ run sampler'}
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
          <p class="empty-msg">Compose a prefix on the left, then run the sampler to see the posterior.</p>
        </div>
      {/if}
    </div>
  </div>

  <p class="footnote">
    <em>Note.</em> Year-of-life scope; code ordering and timing are not modeled. Each sample is one complete bag.
    N = <span data-numeric>{nSamples}</span>, K = <span data-numeric>{$bundle?.model.K ?? '?'}</span>,
    prefix length = <span data-numeric>{$simulatorPrefix.length}</span>.
  </p>
</section>

<style>
  .sim {
    padding: 0.5rem 0 3rem;
  }

  .section-head {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    gap: 2rem;
    padding-bottom: 1.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--rule);
  }
  .title-block { display: flex; flex-direction: column; gap: 0.4rem; }
  .title-block h1 { margin: 0; }
  .title-block h1 em {
    font-style: italic;
    color: var(--terracotta);
  }
  .kicker {
    margin: 0.15rem 0 0;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    max-width: 62ch;
  }
  .kicker em {
    font-family: var(--font-display);
    font-style: italic;
    color: var(--ink);
  }

  .controls {
    display: flex;
    align-items: end;
    gap: 1.5rem;
  }
  .control {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    font-family: var(--font-mono);
    font-size: var(--fs-small);
  }
  .n-control {
    min-width: 180px;
  }
  .n-control input[type="range"] {
    width: 180px;
  }
  .ctl-v {
    color: var(--terracotta);
    margin-left: 0.25rem;
  }
  .seed-control input {
    width: 5rem;
  }
  .run-btn {
    font-family: var(--font-body);
    font-size: var(--fs-small);
    padding: 0.5rem 0.9rem;
    background: var(--ink);
    color: var(--paper);
    border-color: var(--ink);
  }
  .run-btn:hover:not(:disabled) {
    background: var(--terracotta);
    border-color: var(--terracotta);
    color: var(--paper);
  }

  .grid {
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 1.5rem;
  }
  .main {
    display: grid;
    gap: 1.25rem;
    align-content: start;
  }

  .empty {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 3rem 1rem;
    background: var(--paper-elevated);
    border: 1px solid var(--rule);
    border-style: dashed;
    border-radius: var(--radius-sm);
    text-align: center;
  }
  .empty-msg {
    margin: 0 auto;
    font-family: var(--font-display);
    font-style: italic;
    font-size: 1.05rem;
    color: var(--ink-muted);
    max-width: 36ch;
  }

  .footnote {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--rule-faint);
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    font-style: italic;
  }
  .footnote em {
    font-family: var(--font-display);
    color: var(--ink-muted);
  }
</style>
