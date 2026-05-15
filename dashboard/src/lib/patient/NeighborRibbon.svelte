<script lang="ts">
  import { selectedPatientId, patientsById, advancedView } from '../store'
  import ProfileBar from './ProfileBar.svelte'
  export let neighbors: string[]

  // In basic mode, drop neighbors whose dominant is dead/mixed so the
  // "similar patients" cards never surface messy patients by name. The
  // ranking itself is still cosine over the full cohort; we just hide
  // the ones the rest of the basic-mode UI hides too.
  $: visibleNeighbors = neighbors
    .map((nid) => $patientsById.get(nid))
    .filter((n) => n && ($advancedView || n.isClean))
</script>

<section class="ribbon">
  <header class="head">
    <span class="eyebrow">Cohort</span>
    <h3>Patients with similar profiles</h3>
  </header>

  <div class="strip">
    {#each visibleNeighbors as n}
      {#if n}
        <button class="card" on:click={() => selectedPatientId.set(n.id)} aria-label={`Open ${n.id}`}>
          <div class="card-head">
            <span class="dot"></span>
            <span class="id" data-numeric>{n.id}</span>
          </div>
          <div class="card-bar">
            <ProfileBar theta={n.theta} codeBag={n.code_bag} height={10} labels={false} />
          </div>
        </button>
      {/if}
    {/each}
  </div>
</section>

<style>
  .ribbon {
    margin-top: 2rem;
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: 0.85rem;
    margin-bottom: 0.85rem;
  }
  .head h3 {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: var(--tracking-tight);
  }

  .strip {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 0.65rem;
  }

  .card {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    padding: 0.7rem 0.85rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    cursor: pointer;
    text-align: left;
    transition: border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
  }
  .card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.12);
    transform: translateY(-1px);
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
    background: var(--accent);
  }
  .id {
    font-family: var(--font-mono);
    font-size: var(--fs-small);
    color: var(--ink-muted);
    letter-spacing: -0.01em;
  }

  /* Compact inline bar; no labels/percents inside neighbor cards. */
  .card-bar :global(.profile) {
    gap: 0;
  }
</style>
