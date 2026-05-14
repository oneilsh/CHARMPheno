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
  <header>
    <h2>Simulator</h2>
    <label>N <input type="range" min="10" max="2000" step="10" bind:value={nSamples} /> <span class="num">{nSamples}</span></label>
    <label>Sort <select bind:value={sortMode}>
      <option value="median">Median θ</option>
      <option value="spread">Spread (P90-P10)</option>
      <option value="npmi">NPMI</option>
      <option value="id">Phenotype id</option>
    </select></label>
    <label>Seed <input type="number" bind:value={seed} style="width: 5rem" /></label>
    <button on:click={runSim} disabled={running}>{running ? 'Sampling…' : 'Re-sample'}</button>
  </header>

  <div class="grid">
    <PrefixEditor />
    <div class="main">
      {#if result}
        <Carpet thetaSamples={result.thetaSamples} codeCountsSamples={result.codeCountsSamples} {sortMode} />
        <ExpectedCodes codeCountsSamples={result.codeCountsSamples} />
      {:else}
        <p class="hint">Compose a prefix on the left, then Re-sample.</p>
      {/if}
    </div>
  </div>
  <p class="footnote">
    Year-of-life scope; code ordering and timing are not modeled. Each sample is one complete bag.
    N = {nSamples}, K = {$bundle?.model.K ?? '?'}, prefix length = {$simulatorPrefix.length}.
  </p>
</section>

<style>
  .sim { padding: 1rem; }
  header { display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
  header label { display: flex; align-items: baseline; gap: 0.25rem; font-size: 0.85rem; }
  .grid { display: grid; grid-template-columns: 320px 1fr; gap: 1rem; }
  .main { display: grid; gap: 1rem; }
  .footnote { margin-top: 1rem; font-size: 0.75rem; color: #777; }
  .hint { color: #555; }
  .num { font-variant-numeric: tabular-nums; }
</style>
