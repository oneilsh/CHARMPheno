<script lang="ts">
  import {
    bundle, cohort, simulatorPrefix, advancedView, simulatorConditioning,
  } from '../store'
  import { runSimulator } from '../simulator/runSamples'
  import { buildDesignVector } from '../covariate'
  import { sampleRecordPosterior } from '../conditioning/recordPosterior'
  import { createRng } from '../sampling'
  import { generateCohort } from '../cohort'
  import { dominantVote } from '../dominant'
  import { ensurePatientProjection } from '../patient/projection'
  import ConditionsEditor from '../simulator/ConditionsEditor.svelte'
  import PredictedRecord from '../simulator/PredictedRecord.svelte'
  import StructurePlot from '../simulator/StructurePlot.svelte'
  import ConditioningBar from '../conditioning/ConditioningBar.svelte'
  import { phenotypeHue } from '../palette'
  import { copy } from '../copy'

  // Default N: enough samples for stable occurrence-rate estimates in the
  // posterior-predictive panel and a smooth per-sample strip. The fast
  // (non-autoregressive) path is ~2 E-steps per sample, so 500 stays snappy;
  // autoregressive mode (re-fits theta per token) is the slow opt-in.
  const DEFAULT_N = 500

  // Explore-Cohort ("$cohort") sizing for the cohort this Simulate click
  // generates - matches App.svelte's initial-load defaults (same N/
  // neighbors) so the shared atlas has consistent density regardless of
  // whether it was populated on load or via Simulate Cohort.
  const COHORT_N = 1500
  const COHORT_NEIGHBORS = 8
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
      const b = $bundle
      // STM bundles carry per-topic covariate effects and a topic-correlation
      // block; when both are present, condition the generative theta on the
      // panel's covariate values/group AND the starting-condition prefix via
      // the logistic-normal posterior sampler (see conditioning/
      // recordPosterior.ts) instead of drawing from the Dirichlet prior. With
      // an empty prefix this reduces to the covariate/group prior draw.
      // Non-STM bundles take the unchanged Dirichlet path.
      const isStm = !!b.covariateEffects && !!b.correlation
      const prefixCounts = new Map<number, number>()
      for (const w of $simulatorPrefix) prefixCounts.set(w, (prefixCounts.get(w) ?? 0) + 1)
      let conditionedTheta: (() => number[]) | undefined
      if (isStm) {
        const cond = $simulatorConditioning
        const schema = b.covariateSchema!
        const x = buildDesignVector(schema.design_columns, cond.values)
        const tRng = createRng(seed ^ 0x9e3779b9)
        conditionedTheta = () => sampleRecordPosterior({
          effects: b.covariateEffects!,
          x,
          correlation: b.correlation!,
          topicBlocks: b.gating?.topic_blocks ?? null,
          group: cond.group,
          prefixCounts,
          beta: b.model.beta,
          rng: tRng,
        })
      }
      result = runSimulator({
        alpha: b.model.alpha,
        beta: b.model.beta,
        meanCodesPerDoc: b.corpusStats.mean_codes_per_doc,
        prefix: $simulatorPrefix,
        nSamples,
        seed,
        autoregressive,
        conditionedTheta,
      })

      // Simulate Cohort also (re)generates the shared cohort that Explore
      // Cohort displays: a 'set'-mode cohort at the same covariate
      // values/group, conditioned on the same starting-condition prefix
      // (empty prefix -> sampleConditionedTheta's ordinary prior draw, via
      // cohort.ts's delegation). Reusing this Simulate click as the single
      // trigger for both panels keeps "what you configured" and "what you
      // see in Explore Cohort" in sync without a second button.
      const newCohort = generateCohort({
        model: b.model,
        meanCodesPerDoc: b.corpusStats.mean_codes_per_doc,
        n: COHORT_N,
        seed,
        nNeighbors: COHORT_NEIGHBORS,
        qualityByPhenotype: b.phenotypes.phenotypes.map((p) => p.quality),
        conditioning: {
          mode: 'set',
          values: $simulatorConditioning.values,
          group: $simulatorConditioning.group,
          bundle: b,
          prefixCounts,
          beta: b.model.beta,
        },
      })
      cohort.set(newCohort)
      ensurePatientProjection()
    } finally {
      running = false
    }
  }

  // Clear the result whenever the prefix changes so the output never
  // reflects a stale set of starting conditions. Using Svelte's reactive
  // syntax (rather than a bare .subscribe()) so the dependency is auto-
  // unsubscribed when this component unmounts - a raw .subscribe() leaks
  // a handler every time the Simulator tab is left and re-entered.
  $: $simulatorPrefix, (result = null)

  // Dominant-phenotype vote across the draws: the fraction of simulated patients
  // whose LEADING phenotype is each one. Used only for the one-line confidence
  // readout above the per-sample strip (the mean-θ profile bar was dropped — it
  // flattened toward an even mix regardless of the model; the per-sample strip is
  // the real overview). See dominantVote.
  $: voteTheta = (result && result.thetaSamples.length > 0 && $bundle)
    ? dominantVote(result.thetaSamples, $bundle.phenotypes.phenotypes, $advancedView)
    : null

  // One-line "how confident is the model" verdict from the vote: a clear leader
  // (concentrated conditions) vs a split across profiles (ambiguous conditions).
  $: confidence = (() => {
    if (!voteTheta || !$bundle) return null
    const ordered = voteTheta
      .map((v, k) => ({ k, v }))
      .filter((x) => x.v > 0)
      .sort((a, b) => b.v - a.v)
    if (ordered.length === 0) return null
    const top = ordered[0]
    const name = (k: number) => $bundle!.phenotypes.phenotypes[k]?.label || `Phenotype ${k}`
    const pct = (v: number) => Math.round(v * 100)
    if (top.v >= 0.5) return { text: `Mostly ${name(top.k)} — leads ${pct(top.v)}% of draws`, hue: $phenotypeHue(top.k) }
    if (top.v >= 0.3) return { text: `Leans ${name(top.k)} (${pct(top.v)}% of draws), but mixed`, hue: $phenotypeHue(top.k) }
    const names = ordered.slice(0, 3).map((x) => name(x.k)).join(', ')
    return { text: `Split across several profiles — ${names}…`, hue: null as string | null }
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
        <p class="kicker">{copy.simulator.kicker}</p>
        <details class="what-is" bind:this={whatIsEl} bind:open={whatIsOpen}>
          <summary>{copy.simulator.whatIsSummary}</summary>
          <div class="what-is-body popover">
            {#each copy.simulator.whatIs as para}
              <p>{@html para}</p>
            {/each}
          </div>
        </details>
      </div>
    </div>
  </header>

  <ConditioningBar store={simulatorConditioning} />

  <div class="grid">
    <div class="left-col" data-tour="simulator-input">
      <ConditionsEditor />

      <!-- Run panel: the advanced sampling knobs (if any) and the Simulate
           button, grouped under the conditions so the left column reads as a
           top-to-bottom recipe — set conditions, tune the run, simulate. -->
      <div class="run-panel" data-tour="sim-controls">
        <div class="run-head">
          <span class="eyebrow">Run the model</span>
          <span class="run-sub">{copy.simulator.runSub}</span>
        </div>
        <div class="run-opts">
          {#if $advancedView}
            <label class="control n-control">
              <span class="ctl-head"><span class="eyebrow">Samples</span> <span class="ctl-v" data-numeric>{nSamples}</span></span>
              <input type="range" min="20" max="1000" step="20" bind:value={nSamples} />
            </label>
            <label class="control toggle" title={copy.simulator.autoregressiveTip}>
              <input type="checkbox" bind:checked={autoregressive} />
              <span class="eyebrow">Autoregressive</span>
            </label>
          {/if}
        </div>
        <button class="btn btn-primary run-btn" on:click={simulate} disabled={running || !$bundle}>
          {running ? 'sampling…' : 'simulate →'}
        </button>
      </div>
    </div>

    <div class="right-col">
      {#if result}
        <StructurePlot thetaSamples={result.thetaSamples} summary={confidence} />
        <PredictedRecord
          codeCountsSamples={result.codeCountsSamples}
          codeTopicCounts={result.codeTopicCounts}
        />
      {:else}
        <div class="empty-card">
          <span class="eyebrow">Awaiting input</span>
          <p class="empty-msg">
            {#if $simulatorPrefix.length === 0}
              {@html copy.simulator.emptyFromScratch}
            {:else}
              {@html copy.simulator.emptyReady($simulatorPrefix.length)}
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
    grid-template-columns: 1fr;
    align-items: end;
    gap: 2rem;
    padding-bottom: 1.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--rule);
  }
  .title-block { display: flex; flex-direction: column; gap: 0.45rem; }
  .title-row {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    flex-wrap: wrap;
    position: relative;
  }
  .kicker {
    margin: 0;
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
  /* :global because the popover paragraphs are injected via {@html} from
     copy.ts, so their <em>/<strong> don't receive Svelte's scoping hash. */
  .what-is-body :global(em) { font-style: italic; color: var(--ink); }
  .what-is-body :global(strong) { color: var(--ink); font-weight: 600; }

  /* Run panel: a card under the conditions editor holding the sampling
     controls (advanced) and the Simulate button. */
  .run-panel {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    padding: 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
  }
  .run-head {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--rule);
  }
  .run-sub {
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    font-style: italic;
    line-height: 1.5;
  }
  .run-opts {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
  }
  .control {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .ctl-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.4rem;
  }
  .ctl-v {
    font-size: var(--fs-small);
    color: var(--accent);
    font-weight: 500;
  }
  .n-control { width: 100%; }
  .n-control input[type="range"] { width: 100%; }
  .toggle {
    flex-direction: row;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
  }
  .toggle input { margin: 0; accent-color: var(--accent); }
  .run-btn {
    width: 100%;
    font-size: var(--fs-small);
    padding: 0.6rem 1rem;
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
  /* :global: the empty-state copy is injected via {@html} from copy.ts. */
  .empty-msg :global(strong) { color: var(--ink); font-weight: 600; }
</style>
