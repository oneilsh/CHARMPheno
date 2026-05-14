<script lang="ts">
  import { selectedPatientId, patientsById } from '../store'
  import ProfileBar from './ProfileBar.svelte'
  export let neighbors: string[]
</script>

<section class="ribbon">
  <header class="head">
    <span class="eyebrow">Cohort</span>
    <h3>Patients with similar profiles</h3>
  </header>

  <div class="strip">
    {#each neighbors as nid}
      {@const n = $patientsById.get(nid)}
      {#if n}
        <button class="card" on:click={() => selectedPatientId.set(nid)} aria-label={`Open ${n.id}`}>
          <div class="card-head">
            <span class="dot"></span>
            <span class="id" data-numeric>{n.id}</span>
          </div>
          <div class="card-bar">
            <ProfileBar theta={n.theta} height={10} labels={false} />
          </div>
        </button>
      {/if}
    {/each}
  </div>
</section>

<style>
  .ribbon {
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--rule);
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: 0.85rem;
    margin-bottom: 0.85rem;
  }
  .head h3 {
    font-size: 1.05rem;
    font-weight: 500;
  }

  .strip {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 0.6rem;
  }

  .card {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    padding: 0.65rem 0.75rem;
    background: var(--paper-elevated);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    cursor: pointer;
    text-align: left;
    transition: border-color 0.15s ease, transform 0.15s ease, background 0.15s ease;
  }
  .card:hover {
    border-color: var(--terracotta);
    background: var(--paper);
    transform: translateY(-1px);
  }
  .card:focus-visible {
    outline: 2px solid var(--terracotta);
    outline-offset: 2px;
  }

  .card-head {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .dot {
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: var(--ink-faint);
    flex-shrink: 0;
  }
  .card:hover .dot {
    background: var(--terracotta);
  }
  .id {
    font-size: var(--fs-small);
    color: var(--ink-muted);
    letter-spacing: 0.02em;
  }

  /* Compact inline bar; no labels/percents inside neighbor cards. */
  .card-bar :global(.profile) {
    gap: 0;
  }
</style>
