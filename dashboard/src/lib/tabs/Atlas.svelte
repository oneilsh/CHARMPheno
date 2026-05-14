<script lang="ts">
  import { colorMode } from '../store'
  import TopicMap from '../atlas/TopicMap.svelte'
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
    <aside class="code-panel"><p>Select a phenotype to see its top codes.</p></aside>
  </div>
</section>

<style>
  .atlas { padding: 1rem; }
  header { display: flex; align-items: baseline; gap: 1rem; margin-bottom: 1rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  aside { padding: 1rem; border: 1px solid #ddd; min-height: 480px; }
</style>
