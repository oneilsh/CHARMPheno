<script lang="ts">
  import {
    bundle, cohort, patientsById, selectedPatientId, selectedPhenotypeId,
    advancedView, searchedPhenotypeForPatients, phenotypesById,
    colorByGroup, patientConditioning,
  } from '../store'
  import { phenotypeHue } from '../palette'
  import { displayedDominant } from '../dominant'
  import { generateCohort } from '../cohort'
  import { ensurePatientProjection } from '../patient/projection'
  import PatientMap from '../patient/PatientMap.svelte'
  import PatientBrowser from '../patient/PatientBrowser.svelte'
  import ProfileBar from '../patient/ProfileBar.svelte'
  import ContributingCodes from '../patient/ContributingCodes.svelte'
  import NeighborRibbon from '../patient/NeighborRibbon.svelte'
  import ConditionSearch from '../atlas/ConditionSearch.svelte'
  import ConditioningBar from '../conditioning/ConditioningBar.svelte'
  import { copy } from '../copy'

  // Explore-Cohort ("$cohort") sizing for a Regenerate click — matches
  // App.svelte's initial-load defaults (same N/neighbors) so the shared
  // atlas has consistent density regardless of whether it was populated on
  // load or via this panel's Regenerate button. Duplicated here rather than
  // imported because App.svelte doesn't export them (see Simulator.svelte,
  // which follows the same pattern for its own Simulate Cohort trigger).
  const COHORT_N = 1500
  const COHORT_NEIGHBORS = 8
  // Bumps on every Regenerate click so each run differs from the last; starts
  // one past App.svelte's DEFAULT_COHORT_SEED (42) so the first Regenerate
  // is never a no-op re-draw of the load-time cohort.
  let seedCounter = 43

  // Sample-vs-set conditioning mode for Regenerate. 'sample' (default): each
  // patient draws its own covariates/group from the bundle's marginals — a
  // mixed, representative cohort (matches the faithful load-time default in
  // App.svelte). 'set': every patient shares $patientConditioning's
  // values/group — "what does this subpopulation look like". See
  // cohort.ts's CohortConditioning for the underlying modes.
  let conditioningMode: 'sample' | 'set' = 'sample'

  $: isStm = !!$bundle?.covariateEffects && !!$bundle?.correlation

  function regenerate() {
    const b = $bundle
    if (!b) return
    const seed = seedCounter++
    const newCohort = generateCohort({
      model: b.model,
      meanCodesPerDoc: b.corpusStats.mean_codes_per_doc,
      n: COHORT_N,
      seed,
      nNeighbors: COHORT_NEIGHBORS,
      qualityByPhenotype: b.phenotypes.phenotypes.map((p) => p.quality),
      ...(isStm
        ? { conditioning: {
            mode: conditioningMode,
            values: $patientConditioning.values,
            group: $patientConditioning.group,
            bundle: b,
          } }
        : {}),
    })
    cohort.set(newCohort)
    ensurePatientProjection()
  }

  // This view displays the current cohort ($cohort, populated on app load /
  // on cohort switch — see App.svelte) and can regenerate it via the
  // conditioning cluster above. A gated STM bundle carries a group per
  // patient, which the color-by-group toggle below can display.
  $: hasGroup = !!$bundle?.gating

  // Visible-in-current-mode patients (basic = clean only, advanced = all).
  // We default the detail panel selection to one of these so basic mode
  // never opens onto a hidden messy patient.
  $: allPatients = $cohort?.patients ?? []
  $: visiblePatients = $advancedView
    ? allPatients
    : allPatients.filter((p) => p.isClean)
  $: current = (() => {
    if ($selectedPatientId) {
      const sel = $patientsById.get($selectedPatientId)
      if (sel && ($advancedView || sel.isClean)) return sel
    }
    return visiblePatients[0] ?? null
  })()
  $: if (current && $selectedPatientId !== current.id) selectedPatientId.set(current.id)

  $: phenotypes = $bundle?.phenotypes.phenotypes ?? []
  // Color the patient-panel selection dot to match the patient's atlas dot
  // (which is colored by displayedDominant). Keeps the "this detail is
  // about that dot" link consistent across the two panels and avoids
  // naming a dead/mixed phenotype in basic mode.
  $: dotColor = current
    ? $phenotypeHue(displayedDominant(current.theta, phenotypes, $advancedView))
    : 'var(--accent)'

  // Background-click closes the disclosure popover so it dismisses with a
  // click anywhere outside, not only by re-clicking the link.
  let whatIsEl: HTMLDetailsElement
  let whatIsOpen = false
</script>

<svelte:window on:click={(e) => {
  if (whatIsOpen && whatIsEl && !whatIsEl.contains(e.target as Node)) {
    whatIsOpen = false
  }
}} />

<section class="patient">
  <header class="section-head">
    <div class="title-block">
      <div class="title-row">
        <p class="kicker">{copy.patient.kicker}</p>
        <details class="what-is" bind:this={whatIsEl} bind:open={whatIsOpen}>
          <summary>{copy.patient.whatIsSummary}</summary>
          <div class="what-is-body popover">
            {#each copy.patient.whatIs as para}
              <p>{@html para}</p>
            {/each}
          </div>
        </details>
      </div>
    </div>
  </header>

  <div class="grid">
    <div class="left-col">
      <PatientMap />

      <ConditioningBar store={patientConditioning} showGroup={true} />

      <div class="regen-panel">
        {#if isStm}
          <div class="mode-toggle" role="group" aria-label="Cohort conditioning mode">
            <button
              type="button"
              class="mode-btn"
              class:active={conditioningMode === 'sample'}
              title="Each patient draws its own covariates/group from the corpus's natural mix — a representative cohort."
              on:click={() => conditioningMode = 'sample'}
            >sample from distribution</button>
            <button
              type="button"
              class="mode-btn"
              class:active={conditioningMode === 'set'}
              title="Every patient shares the covariate values/group selected above — this subpopulation's cohort."
              on:click={() => conditioningMode = 'set'}
            >use set covariates/group</button>
          </div>
        {/if}
        <button class="btn btn-primary regen-btn" on:click={regenerate} disabled={!$bundle}>
          Regenerate cohort
        </button>
      </div>

      <div class="map-actions">
        <div class="control-stack">
          <ConditionSearch entityLabel="patients" />
          {#if $searchedPhenotypeForPatients !== null}
            <div class="phenotype-chip" title={copy.patient.findPhenotypeChipTip}>
              <span class="chip-label">Highlighting patients with</span>
              <span class="chip-val">{$phenotypesById.get($searchedPhenotypeForPatients)?.label || `Phenotype ${$searchedPhenotypeForPatients}`}</span>
              <button class="chip-clear" type="button" on:click={() => searchedPhenotypeForPatients.set(null)} title="Clear">×</button>
            </div>
          {/if}
        </div>
        {#if hasGroup}
          <label class="color-toggle" title="Color patient-atlas points by each patient's gating group instead of dominant phenotype.">
            <input type="checkbox" bind:checked={$colorByGroup} />
            <span>color by group</span>
          </label>
        {/if}
      </div>
      <PatientBrowser />
    </div>

    <div class="right-col">
      {#if current}
        <div class="profile-block" data-tour="patient-profile">
          <header class="profile-head">
            <div class="eyebrow-row">
              <!-- Selection dot uses the patient's dominant-phenotype hue
                   so it visually matches the patient's atlas dot - "this
                   detail is about that dot". -->
              <span class="sel-dot" style="background: {dotColor}; box-shadow: 0 0 0 2px color-mix(in srgb, {dotColor} 28%, transparent);" aria-hidden="true"></span>
              <span class="eyebrow">Patient</span>
            </div>
            <h2 class="title">Patient {current.id}</h2>
            <p class="profile-sub">{copy.patient.profileSub}</p>
          </header>
          <ProfileBar
            theta={current.theta}
            codeBag={current.code_bag}
            height={44}
            onSelect={(k) => selectedPhenotypeId.set(k)}
          />
        </div>
        <ContributingCodes theta={current.theta} codeBag={current.code_bag} />
        <NeighborRibbon neighbors={current.neighbors} />
      {:else}
        <p class="empty">{copy.patient.empty}</p>
      {/if}
    </div>
  </div>
</section>

<style>
  .patient { padding: 0.25rem 0 3rem; }

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
  .what-is-body :global(strong) {
    color: var(--ink);
    font-weight: 600;
  }

  /* Regenerate cluster: sample-vs-set mode toggle + Regenerate button,
     sitting between the ConditioningBar and the map-actions row. Mirrors
     Simulator.svelte's .run-panel card so the two generative panels feel
     like the same family of control. */
  .regen-panel {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.85rem;
    padding: 0.6rem 0;
  }
  .mode-toggle {
    display: inline-flex;
    background: var(--surface);
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius-sm);
    padding: 2px;
    gap: 1px;
  }
  .mode-btn {
    border: 0;
    background: transparent;
    padding: 0.28rem 0.7rem;
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--ink-muted);
    cursor: pointer;
    border-radius: 3px;
    transition: all 0.12s ease;
  }
  .mode-btn:hover { color: var(--ink); }
  .mode-btn.active {
    background: var(--ink);
    color: var(--surface);
  }
  .regen-btn {
    font-size: var(--fs-small);
    padding: 0.4rem 0.9rem;
    flex-shrink: 0;
  }

  .control-stack {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    align-items: flex-start;
  }
  /* Sits just below the patient map: the condition-search control-stack on
     the left, the color-by-group toggle on the right, pulled up tight
     against the map (overrides the left-col's 1.25rem gap). When there's
     no gating group the toggle is absent and the stack just sits left. */
  .map-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.85rem;
    margin-top: -0.75rem;
  }

  .color-toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    color: var(--ink-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    cursor: pointer;
    user-select: none;
  }

  .phenotype-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.55rem;
    background: var(--accent-find-soft);
    border-radius: var(--radius-sm);
    font-size: var(--fs-micro);
    color: var(--accent-find-ink);
    max-width: 280px;
    box-shadow: inset 3px 0 0 var(--accent-find);
  }
  .phenotype-chip .chip-label {
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.85;
  }
  .phenotype-chip .chip-val {
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .phenotype-chip .chip-clear {
    border: 0;
    background: transparent;
    color: var(--accent-find-ink);
    cursor: pointer;
    font-size: 1rem;
    line-height: 1;
    padding: 0 0.1rem;
    margin-left: 0.1rem;
  }
  .phenotype-chip .chip-clear:hover { color: var(--ink); }

  .grid {
    display: grid;
    grid-template-columns: 1.1fr 1fr;
    gap: 1.5rem;
    align-items: start;
  }
  .left-col, .right-col {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    min-width: 0;
  }

  .profile-block {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    padding: 1.25rem 1.25rem 1rem;
  }
  .profile-head {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    padding-bottom: 1rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--rule);
  }
  .eyebrow-row {
    display: flex;
    align-items: center;
    gap: 0.45rem;
  }
  .sel-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 0 2px var(--accent-faint);
  }
  .profile-head .title {
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: var(--tracking-display);
    line-height: 1.15;
    color: var(--ink);
    font-family: var(--font-mono);
    margin: 0;
  }
  .profile-sub {
    margin: 0.15rem 0 0;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    font-style: italic;
  }
  .empty {
    color: var(--ink-faint);
    font-style: italic;
    padding: 2rem 0;
  }
</style>
