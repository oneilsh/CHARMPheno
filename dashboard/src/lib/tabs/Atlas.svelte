<script lang="ts">
  import { colorMode } from '../store'
  import TopicMap from '../atlas/TopicMap.svelte'
  import CodePanel from '../atlas/CodePanel.svelte'
</script>

<section class="atlas">
  <header>
    <h2>Phenotype Atlas</h2>
    <label>Color
      <select bind:value={$colorMode}>
        <option value="npmi">NPMI</option>
        <option value="prevalence">Prevalence</option>
      </select>
    </label>
  </header>
  <div class="grid">
    <TopicMap />
    <CodePanel />
  </div>
</section>

<style>
  .atlas { padding: 1rem; }
  header { display: flex; align-items: baseline; gap: 1rem; margin-bottom: 1rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
</style>
