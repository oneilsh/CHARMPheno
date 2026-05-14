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
  <header class="section-head">
    <div class="title-block">
      <span class="eyebrow">02 · Patient</span>
      <h1>Patient Explorer</h1>
      <p class="kicker">Synthetic patient drawn from the model. Click a phenotype band to inspect which codes from this patient's bag drove the assignment.</p>
    </div>
    <div class="picker">
      <label class="picker-label">
        <span class="eyebrow">Patient</span>
        <select bind:value={$selectedPatientId}>
          {#each patients as p}<option value={p.id}>{p.id}</option>{/each}
        </select>
      </label>
      <button class="btn-ghost shuffle" on:click={shuffle}>↻ shuffle</button>
    </div>
  </header>

  {#if current}
    <div class="profile-block">
      <div class="profile-head">
        <span class="eyebrow">Phenotype profile</span>
        <span class="profile-id" data-numeric>{current.id}</span>
      </div>
      <ProfileBar theta={current.theta} height={44} onSelect={(k) => selectedPhenotypeId.set(k)} />
    </div>
    <ContributingCodes theta={current.theta} codeBag={current.code_bag} />
    <NeighborRibbon neighbors={current.neighbors} />
  {/if}
</section>

<style>
  .patient { padding: 0.25rem 0 3rem; }

  .section-head {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    gap: 2rem;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--rule);
  }
  .title-block { display: flex; flex-direction: column; gap: 0.45rem; }
  .title-block h1 { margin: 0.1rem 0 0; }
  .kicker {
    margin: 0.25rem 0 0;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    max-width: 60ch;
    line-height: 1.55;
  }

  .picker {
    display: flex;
    align-items: end;
    gap: 1rem;
  }
  .picker-label {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .picker-label select { min-width: 9rem; }
  .shuffle { font-family: var(--font-body); font-size: var(--fs-small); }
  .shuffle:hover { color: var(--accent); }

  .profile-block {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    padding: 1.25rem 1.25rem 1rem;
  }
  .profile-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.85rem;
  }
  .profile-id {
    font-size: var(--fs-small);
    color: var(--ink-faint);
  }
</style>
