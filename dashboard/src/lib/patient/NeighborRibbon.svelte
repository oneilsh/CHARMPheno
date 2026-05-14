<script lang="ts">
  import { selectedPatientId, patientsById } from '../store'
  import ProfileBar from './ProfileBar.svelte'
  export let neighbors: string[]
</script>

<section class="ribbon">
  <h3>Patients with similar profiles</h3>
  <div class="strip">
    {#each neighbors as nid}
      {@const n = $patientsById.get(nid)}
      {#if n}
        <button class="card" on:click={() => selectedPatientId.set(nid)}>
          <span class="id">{n.id}</span>
          <ProfileBar theta={n.theta} height={14} labels={false} />
        </button>
      {/if}
    {/each}
  </div>
</section>

<style>
  .ribbon { margin-top: 1.5rem; }
  .strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.5rem; }
  .card { display: grid; gap: 0.25rem; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; background: #fff; cursor: pointer; text-align: left; }
  .card:hover { background: #f8f8f8; }
  .id { font-size: 0.75rem; color: #555; }
</style>
