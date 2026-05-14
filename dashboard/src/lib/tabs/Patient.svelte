<script lang="ts">
  import { cohort, patientsById, selectedPatientId, selectedPhenotypeId } from '../store'
  import ProfileBar from '../patient/ProfileBar.svelte'
  import ContributingCodes from '../patient/ContributingCodes.svelte'
  import NeighborRibbon from '../patient/NeighborRibbon.svelte'

  $: patients = $cohort?.patients ?? []
  $: current = $selectedPatientId ? $patientsById.get($selectedPatientId) : (patients[0] ?? null)
  $: if (current && $selectedPatientId !== current.id) selectedPatientId.set(current.id)

  function shuffle() {
    if (patients.length === 0) return
    selectedPatientId.set(patients[Math.floor(Math.random() * patients.length)].id)
  }
</script>

<section class="patient">
  <header>
    <h2>Patient Explorer</h2>
    <label>Patient
      <select bind:value={$selectedPatientId}>
        {#each patients as p}<option value={p.id}>{p.id}</option>{/each}
      </select>
    </label>
    <button on:click={shuffle}>Shuffle</button>
  </header>

  {#if current}
    <div class="profile">
      <h3>Profile</h3>
      <ProfileBar theta={current.theta} height={40} onSelect={(k) => selectedPhenotypeId.set(k)} />
    </div>
    <ContributingCodes theta={current.theta} codeBag={current.code_bag} />
    <NeighborRibbon neighbors={current.neighbors} />
  {/if}
</section>

<style>
  .patient { padding: 1rem; }
  header { display: flex; align-items: baseline; gap: 1rem; margin-bottom: 1rem; }
  .profile h3 { margin: 0 0 0.5rem; }
</style>
