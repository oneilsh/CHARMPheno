<script lang="ts">
  import { bundle } from '../store'
  import TopicMap from '../atlas/TopicMap.svelte'
  import CodePanel from '../atlas/CodePanel.svelte'
  import ConditionSearch from '../atlas/ConditionSearch.svelte'
  import PhenotypeBrowser from '../atlas/PhenotypeBrowser.svelte'

  // Background-click closes the disclosure popover so the user doesn't
  // have to find and click the same link again to dismiss it.
  let whatIsEl: HTMLDetailsElement
  let whatIsOpen = false
</script>

<svelte:window on:click={(e) => {
  if (whatIsOpen && whatIsEl && !whatIsEl.contains(e.target as Node)) {
    whatIsOpen = false
  }
}} />

<section class="atlas">
  <header class="section-head">
    <div class="title-block">
      <div class="title-row">
        <h1>Phenotype Atlas</h1>
        <details class="what-is" bind:this={whatIsEl} bind:open={whatIsOpen}>
          <summary>What's a phenotype?</summary>
          <div class="what-is-body popover">
            <p>
              A <em>phenotype</em> here is a recurring pattern of clinical
              conditions that tends to appear together across patients.
              For example, "Type 2 diabetes care" concentrates on diabetes,
              retinopathy, neuropathy, and related conditions.
            </p>
            <p>
              These phenotypes were learned automatically from de-identified
              patient records using a topic model (Latent Dirichlet Allocation).
              The model didn't know about diseases ahead of time; it just looked
              for groups of conditions that tend to co-occur, and produced
              {$bundle?.model.K ?? '~80'} phenotypes.
            </p>
            <p>
              A patient is a mix of phenotypes, not a single one. A phenotype
              is not a diagnosis; it's a pattern. Some patterns name a single
              disease, others name a family of related conditions, and some
              describe broad health backgrounds (e.g. chronic comorbidity
              follow-up).
            </p>
          </div>
        </details>
      </div>
      <p class="kicker">
        Each marker is a learned phenotype. Bubbles that sit closer together
        share more of their leading conditions; bubble size shows how widely
        the phenotype shows up across patients.
      </p>
    </div>
    <div class="controls">
      <ConditionSearch />
    </div>
  </header>

  <div class="grid">
    <div class="left-col">
      <TopicMap />
      <PhenotypeBrowser />
    </div>
    <CodePanel />
  </div>
</section>

<style>
  .atlas {
    padding: 0.25rem 0 3rem;
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
  .title-block {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
  }
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

  /* "What's a phenotype?" disclosure. Body floats as a popover so it
     doesn't shove the kicker and the rest of the page down when opened. */
  .what-is {
    position: relative;
  }
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
  .what-is summary:hover {
    color: var(--ink);
    border-bottom-color: var(--ink);
  }
  .what-is[open] summary {
    color: var(--ink);
    border-bottom-color: transparent;
  }
  .what-is-body {
    margin-top: 0.6rem;
    max-width: 62ch;
    padding: 0.85rem 1rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-sm);
  }
  /* Popover variant: anchored under the inline summary, floats over the
     content below so the masthead doesn't grow when expanded. */
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
  .what-is-body em {
    font-style: italic;
    color: var(--ink);
  }

  .controls {
    display: flex;
    align-items: end;
    gap: 1.25rem;
  }

  .grid {
    display: grid;
    grid-template-columns: 1.1fr 1fr;
    gap: 1.5rem;
    align-items: start;
  }
  .left-col {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    min-width: 0;
  }
</style>
