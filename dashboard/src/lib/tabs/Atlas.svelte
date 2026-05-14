<script lang="ts">
  import { colorMode } from '../store'
  import TopicMap from '../atlas/TopicMap.svelte'
  import CodePanel from '../atlas/CodePanel.svelte'
</script>

<section class="atlas">
  <header class="section-head">
    <div class="title-block">
      <span class="eyebrow">01 · Atlas</span>
      <h1>Phenotype Atlas</h1>
      <p class="kicker">Each marker is a learned phenotype. Distance is Jensen–Shannon divergence between code distributions; size encodes corpus prevalence.</p>
    </div>
    <div class="controls">
      <label class="control">
        <span class="eyebrow">Color by</span>
        <select bind:value={$colorMode}>
          <option value="npmi">NPMI</option>
          <option value="prevalence">Prevalence</option>
        </select>
      </label>
    </div>
  </header>

  <div class="grid">
    <TopicMap />
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
    gap: 1.25rem;
  }
  .control {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .grid {
    display: grid;
    grid-template-columns: 1.1fr 1fr;
    gap: 1.5rem;
  }
</style>
