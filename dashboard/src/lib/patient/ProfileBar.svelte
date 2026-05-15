<script lang="ts">
  import {
    phenotypesById, advancedView, searchedConditionIdx, searchedPhenotypeSet,
  } from '../store'
  import { phenotypeHue } from '../palette'
  export let theta: number[]
  export let height = 28
  export let labels = true
  export let onSelect: ((id: number) => void) | null = null
  export let otherThreshold = 0.05
  // The patient's code bag. When provided AND the user has pinned a
  // condition via the search box AND this patient actually has that
  // condition, magenta dots mark every band whose phenotype contains
  // the searched condition in its top relevance-ranked codes.
  export let codeBag: number[] | null = null

  // In basic mode, dead/mixed phenotype bands ALSO fold into "Other"
  // regardless of weight, so bad-phenotype names never appear in the
  // basic UI. Their mass still shows up in the Other band where the
  // user can drill in without seeing the name.
  function bandHidden(k: number, v: number, advanced: boolean): boolean {
    if (v < otherThreshold) return true
    if (advanced) return false
    const q = $phenotypesById.get(k)?.quality
    return q === 'dead' || q === 'mixed'
  }

  $: ordered = theta.map((v, k) => ({ k, v }))
    .filter((x) => x.v > 0)
    .sort((a, b) => b.v - a.v)
  $: mainBands = ordered.filter((x) => !bandHidden(x.k, x.v, $advancedView))
  $: hiddenBands = ordered.filter((x) => bandHidden(x.k, x.v, $advancedView))
  $: otherFrac = hiddenBands.reduce((a, x) => a + x.v, 0)

  // Search-match logic. patientHasSearchedCode gates everything; without
  // it the patient doesn't actually carry the condition so we don't mark
  // anything (even if some of their phenotypes "contain" it).
  $: patientHasSearchedCode = (() => {
    if ($searchedConditionIdx === null || !codeBag) return false
    const idx = $searchedConditionIdx
    for (let i = 0; i < codeBag.length; i++) if (codeBag[i] === idx) return true
    return false
  })()
  $: matchSet = ($searchedPhenotypeSet && patientHasSearchedCode) ? $searchedPhenotypeSet : null
  // Did any of the phenotypes that folded into "Other" contain the
  // searched condition? If so the Other slot gets a match dot too.
  $: otherHasMatch = !!matchSet && hiddenBands.some((x) => matchSet!.has(x.k))
</script>

<div class="profile" style="--bar-h: {height}px">
  {#if labels}
    <div class="band-labels">
      {#each mainBands as b}
        <span class="band-label" style="width: {(b.v * 100).toFixed(2)}%; color: {$phenotypeHue(b.k)}">
          <span class="dot" style="background: {$phenotypeHue(b.k)}"></span>
          <span class="lab">{$phenotypesById.get(b.k)?.label || `Phenotype ${b.k}`}</span>
        </span>
      {/each}
      {#if otherFrac > 0}
        <span class="band-label other" style="width: {(otherFrac * 100).toFixed(2)}%">
          <span class="dot"></span>
          <span class="lab">Other</span>
        </span>
      {/if}
    </div>
  {/if}

  <div class="bar">
    {#each mainBands as b}
      <button class="band"
        style="width: {(b.v * 100).toFixed(2)}%; background: {$phenotypeHue(b.k)};"
        title={`${$phenotypesById.get(b.k)?.label || `Phenotype ${b.k}`}: ${(b.v * 100).toFixed(1)}%`}
        on:click={() => onSelect?.(b.k)}
        aria-label={`${$phenotypesById.get(b.k)?.label || `Phenotype ${b.k}`}, ${(b.v * 100).toFixed(1)} percent`}
      ></button>
    {/each}
    {#if otherFrac > 0}
      <!-- onSelect(-1) is the sentinel "Other / tail phenotypes" view.
           ContributingCodes special-cases id < 0 to surface codes whose
           responsibility lies in the long tail rather than the dominants. -->
      <button class="band other-band"
        style="width: {(otherFrac * 100).toFixed(2)}%"
        title={`Other (long-tail phenotypes): ${(otherFrac * 100).toFixed(1)}%`}
        on:click={() => onSelect?.(-1)}
        aria-label={`Other tail phenotypes, ${(otherFrac * 100).toFixed(1)} percent`}
      ></button>
    {/if}
  </div>

  {#if matchSet}
    <!-- One match-slot per band, same widths as the bar above. A fuchsia
         dot appears under any band whose phenotype contains the searched
         condition. Patient must actually have the condition (gated by
         the caller passing codeBag). -->
    <div class="band-marks" aria-hidden="true">
      {#each mainBands as b}
        <span class="mark-slot" style="width: {(b.v * 100).toFixed(2)}%">
          {#if matchSet.has(b.k)}<span class="match-dot"></span>{/if}
        </span>
      {/each}
      {#if otherFrac > 0}
        <span class="mark-slot" style="width: {(otherFrac * 100).toFixed(2)}%">
          {#if otherHasMatch}<span class="match-dot"></span>{/if}
        </span>
      {/if}
    </div>
  {/if}

  {#if labels}
    <div class="band-percents" data-numeric>
      {#each mainBands as b}
        <span class="pct" style="width: {(b.v * 100).toFixed(2)}%">{(b.v * 100).toFixed(0)}%</span>
      {/each}
      {#if otherFrac > 0}
        <span class="pct other" style="width: {(otherFrac * 100).toFixed(2)}%">{(otherFrac * 100).toFixed(0)}%</span>
      {/if}
    </div>
  {/if}
</div>

<style>
  .profile {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .band-labels {
    display: flex;
    width: 100%;
    align-items: baseline;
  }
  .band-label {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    padding-right: 0.4rem;
    font-size: var(--fs-small);
    color: var(--ink-muted);
    overflow: hidden;
    white-space: nowrap;
    line-height: 1.2;
  }
  .band-label .lab {
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .band-label.other { color: var(--ink-faint); font-style: italic; }
  .band-label .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--ink-faint);
    flex-shrink: 0;
  }

  .bar {
    display: flex;
    width: 100%;
    height: var(--bar-h);
    border-radius: var(--radius-sm);
    overflow: hidden;
    background: var(--surface-recessed);
    gap: 1px;
  }
  .band {
    border: 0;
    padding: 0;
    cursor: pointer;
    height: 100%;
    transition: filter 0.15s ease;
  }
  .band:hover {
    filter: brightness(1.08);
  }
  .band:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: -2px;
  }
  .other-band {
    background: var(--surface-deep);
    background-image: repeating-linear-gradient(
      45deg,
      transparent,
      transparent 3px,
      rgba(82, 82, 91, 0.14) 3px,
      rgba(82, 82, 91, 0.14) 4px
    );
  }

  .band-marks {
    display: flex;
    width: 100%;
    height: 8px;
  }
  .mark-slot {
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .match-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-search);
  }

  .band-percents {
    display: flex;
    width: 100%;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
  }
  .pct {
    overflow: hidden;
    white-space: nowrap;
    padding-right: 0.4rem;
    letter-spacing: 0.02em;
  }
</style>
