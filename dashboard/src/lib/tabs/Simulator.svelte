<script lang="ts">
  import {
    bundle, simulatorPrefix, advancedView,
  } from '../store'
  import { runSimulator } from '../simulator/runSamples'
  import ConditionsEditor from '../simulator/ConditionsEditor.svelte'
  import PredictedRecord from '../simulator/PredictedRecord.svelte'
  import SimMiniMap from '../simulator/SimMiniMap.svelte'
  import StructurePlot from '../simulator/StructurePlot.svelte'
  import ProfileBar from '../patient/ProfileBar.svelte'

  // Default N: enough samples for a stable median and a smooth atlas
  // cloud, low enough that even autoregressive mode (which re-fits theta
  // per token) feels responsive.
  const DEFAULT_N = 200
  // Seed sequence starts at 42 (matches the Patient atlas's seed default)
  // and auto-bumps on each Simulate click so the user sees variation
  // without managing a seed input. The first run is therefore always
  // reproducible across reloads, which keeps walkthrough demos stable.
  let seedCounter = 42

  let nSamples = DEFAULT_N
  let autoregressive = false
  let result: ReturnType<typeof runSimulator> | null = null
  let running = false

  let whatIsEl: HTMLDetailsElement
  let whatIsOpen = false

  async function simulate() {
    if (!$bundle || running) return
    running = true
    const seed = seedCounter++
    // Yield to the browser so the running spinner paints before the
    // simulator (variational E-step in a loop) blocks the main thread.
    await new Promise((r) => setTimeout(r, 0))
    try {
      result = runSimulator({
        alpha: $bundle.model.alpha,
        beta: $bundle.model.beta,
        meanCodesPerDoc: $bundle.corpusStats.mean_codes_per_doc,
        prefix: $simulatorPrefix,
        nSamples,
        seed,
        autoregressive,
      })
    } finally {
      running = false
    }
  }

  // Clear the result whenever the prefix changes so the output never
  // reflects a stale set of starting conditions. Subscribing to the
  // prefix store rather than reading it directly so this fires on every
  // edit (add, remove, draw-from-phenotype).
  simulatorPrefix.subscribe(() => { result = null })

  // Mean theta across samples for the profile bar. Aggregating to mean
  // is the right move here - taking per-component medians would not sum
  // to 1 across phenotypes. The atlas cloud below carries the
  // uncertainty story; the bar shows the typical mix.
  $: meanTheta = (() => {
    if (!result || result.thetaSamples.length === 0) return null
    const K = result.thetaSamples[0].length
    const m = new Array(K).fill(0)
    for (const t of result.thetaSamples) for (let k = 0; k < K; k++) m[k] += t[k]
    for (let k = 0; k < K; k++) m[k] /= result.thetaSamples.length
    return m as number[]
  })()
</script>

<svelte:window on:click={(e) => {
  if (whatIsOpen && whatIsEl && !whatIsEl.contains(e.target as Node)) {
    whatIsOpen = false
  }
}} />

<section class="sim">
  <header class="section-head">
    <div class="title-block">
      <div class="title-row">
        <h1>Simulator</h1>
        <details class="what-is" bind:this={whatIsEl} bind:open={whatIsOpen}>
          <summary>What is this?</summary>
          <div class="what-is-body popover">
            <p>
              The simulator asks the model: <em>given these conditions,
              what kind of patient could this be?</em> It answers by
              drawing many possible complete year-of-life records from
              the model's distribution.
            </p>
            <p>
              Each draw is one plausible patient. The <strong>profile
              bar</strong> shows the average phenotype mix across those
              draws. The <strong>expected codes</strong> table shows
              what the model thinks fills in the rest of the year. The
              <strong>atlas</strong> shows where these patients land
              relative to the synthetic cohort - tight cluster means
              the conditions you gave nail one kind of patient, smeared
              cloud means they're consistent with several.
            </p>
            <p>
              Start by clicking conditions on the left, or just hit
              Simulate to draw new patients from scratch.
            </p>
          </div>
        </details>
      </div>
      <p class="kicker">
        Pick some starting conditions and the model will tell you what kind of patient this looks like and what else would round out their year.
      </p>
    </div>
    <div class="controls">
      {#if $advancedView}
        <label class="control n-control">
          <span class="ctl-head"><span class="eyebrow">Samples</span> <span class="ctl-v" data-numeric>{nSamples}</span></span>
          <input type="range" min="20" max="1000" step="20" bind:value={nSamples} />
        </label>
        <label class="control toggle" title="When on, the model re-evaluates the phenotype mix after every drawn code so each token shifts the next one's distribution.">
          <input type="checkbox" bind:checked={autoregressive} />
          <span class="eyebrow">Autoregressive</span>
        </label>
      {/if}
      <button class="btn btn-primary run-btn" on:click={simulate} disabled={running || !$bundle}>
        {running ? 'sampling…' : 'simulate →'}
      </button>
    </div>
  </header>

  <div class="grid">
    <div class="left-col">
      <ConditionsEditor />
    </div>

    <div class="right-col">
      {#if result && meanTheta}
        <div class="profile-block">
          <header class="profile-head">
            <span class="eyebrow">Phenotype mix</span>
            <h3>This patient is a mix of…</h3>
            <p class="sub">Average across {result.thetaSamples.length} simulated draws.</p>
          </header>
          <ProfileBar theta={meanTheta} height={44} />
        </div>
        <StructurePlot thetaSamples={result.thetaSamples} />
        <PredictedRecord codeCountsSamples={result.codeCountsSamples} />
        <SimMiniMap thetaSamples={result.thetaSamples} />
      {:else}
        <div class="empty-card">
          <span class="eyebrow">Awaiting input</span>
          <p class="empty-msg">
            {#if $simulatorPrefix.length === 0}
              Add some starting conditions on the left (or just hit Simulate to draw patients from scratch), then click <strong>simulate →</strong> to see what kind of patient this looks like.
            {:else}
              {$simulatorPrefix.length} starting condition{$simulatorPrefix.length === 1 ? '' : 's'} ready. Click <strong>simulate →</strong> to see what kind of patient this looks like.
            {/if}
          </p>
        </div>
      {/if}
    </div>
  </div>
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
  .title-row {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    flex-wrap: wrap;
    position: relative;
  }
  .kicker {
    margin: 0.25rem 0 0;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    max-width: 62ch;
    line-height: 1.55;
  }

  .what-is { position: relative; }
  .what-is summary {
    cursor: pointer;
    color: var(--accent);
    font-size: var(--fs-small);
    list-style: none;
    display: inline-block;
    border-bottom: 1px dotted var(--accent);
    text-underline-offset: 2px;
  }
  .what-is summary::-webkit-details-marker { display: none; }
  .what-is summary::marker { display: none; }
  .what-is summary:hover { color: var(--ink); border-bottom-color: var(--ink); }
  .what-is[open] summary { color: var(--ink); border-bottom-color: transparent; }
  .what-is-body {
    margin-top: 0.6rem;
    max-width: 62ch;
    padding: 0.85rem 1rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-sm);
  }
  .what-is-body.popover {
    position: absolute;
    top: 1.6rem;
    left: 0;
    z-index: 5;
    width: 62ch;
    max-width: min(62ch, calc(100vw - 4rem));
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
  }
  .what-is-body p {
    margin: 0 0 0.55rem;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    line-height: 1.6;
  }
  .what-is-body p:last-child { margin-bottom: 0; }
  .what-is-body em { font-style: italic; color: var(--ink); }
  .what-is-body strong { color: var(--ink); font-weight: 600; }

  .controls {
    display: flex;
    align-items: end;
    gap: 1.25rem;
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
  .toggle {
    flex-direction: row;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
  }
  .toggle input { margin: 0; accent-color: var(--accent); }
  .run-btn {
    font-size: var(--fs-small);
    padding: 0.5rem 1rem;
  }

  .grid {
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 1.5rem;
    align-items: start;
  }
  .left-col, .right-col {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    min-width: 0;
  }

  .profile-block {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    padding: 1.25rem;
  }
  .profile-head {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    margin-bottom: 1rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid var(--rule);
  }
  .profile-head h3 {
    margin: 0;
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: var(--tracking-display);
    line-height: 1.15;
    color: var(--ink);
    font-family: var(--font-mono);
  }
  .profile-head .sub {
    margin: 0.15rem 0 0;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    font-style: italic;
  }
  .empty-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.85rem;
    padding: 4rem 1.5rem;
    background: var(--surface);
    border: 1px dashed var(--rule-strong);
    border-radius: var(--radius-sm);
    text-align: center;
  }
  .empty-msg {
    margin: 0;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    max-width: 46ch;
    line-height: 1.6;
  }
  .empty-msg strong { color: var(--ink); font-weight: 600; }
</style>
