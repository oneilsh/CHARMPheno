<script lang="ts">
  import { bundle, selectedPhenotypeId, phenotypesById, searchedConditionIdx, advancedView } from '../store'
  import { phenotypeHue } from '../palette'
  import { go } from '../router'
  import { copy } from '../copy'
  import { codeComposition, sortRowsForSelection, OTHER_ID } from './codeComposition'

  export let theta: number[]
  export let codeBag: number[]

  const MAX_ROWS = 12

  function openInAtlas() {
    // selectedPhenotypeId is already set to the clicked phenotype; just
    // switch tabs and the atlas's TopicMap + CodePanel pick it up.
    go('atlas')
  }

  $: isOther = $selectedPhenotypeId === OTHER_ID

  // Mirrors ProfileBar's bandHidden() basic-mode branch: dead/mixed
  // phenotypes fold into Other regardless of theta so the code bar's
  // Other bucket matches what the profile bar above it shows.
  $: hiddenPhenotypes = $bundle && !$advancedView
    ? new Set(
        Array.from({ length: $bundle.model.K }, (_, k) => k).filter((k) => {
          const q = $phenotypesById.get(k)?.quality
          return q === 'dead' || q === 'mixed'
        }),
      )
    : undefined

  $: rows = $bundle
    ? codeComposition(theta, codeBag, $bundle.model.beta, $bundle.model.K, 0.05, hiddenPhenotypes)
    : []
  $: sorted = sortRowsForSelection(rows, $selectedPhenotypeId).slice(0, MAX_ROWS)
  $: hasSelection = $selectedPhenotypeId !== null
  // Gates the header's selection-claiming bits (h3, open-in-atlas, subMatch/
  // subOther). With zero rows the body falls back to emptyRecord, so the
  // header must not claim a selection either — that mismatch (Fix 2) is what
  // this guards against.
  $: hasRows = rows.length > 0

  // Focus: when a band is selected, dim every segment that is not it.
  function segActive(k: number): boolean {
    return !hasSelection || k === $selectedPhenotypeId
  }
  function segColor(k: number): string {
    return k === OTHER_ID ? 'var(--surface-deep)' : $phenotypeHue(k)
  }
  // Hover label for each composition segment, mirroring ProfileBar's band
  // title ("<phenotype>: <pct>%") so a code bar reads the same way as the
  // profile bars elsewhere on the page.
  function segLabel(k: number): string {
    return k === OTHER_ID
      ? copy.contributingCodes.otherLabel
      : $phenotypesById.get(k)?.label || `Phenotype ${k}`
  }

  $: selectedLabel = $selectedPhenotypeId === null
    ? null
    : isOther
      ? copy.contributingCodes.otherLabel
      : ($phenotypesById.get($selectedPhenotypeId)?.label || `Phenotype ${$selectedPhenotypeId}`)
</script>

<section class="contrib" data-tour="contributing-codes">
  <header class="head">
    <div class="top-row">
      <span class="eyebrow">{copy.contributingCodes.heading}</span>
      {#if selectedLabel && $selectedPhenotypeId !== null && !isOther && hasRows}
        <button
          class="open-in-atlas"
          type="button"
          on:click={openInAtlas}
          title={copy.contributingCodes.openInAtlasTip}
          data-tour="open-in-atlas"
        >
          open in atlas →
        </button>
      {/if}
    </div>
    {#if selectedLabel && $selectedPhenotypeId !== null && hasRows}
      <h3>
        <!-- Bullet matches the clicked band in the profile bar above so the
             link "I clicked that band → these are its codes" reads at a
             glance. The Other band uses a hatched grey to match. -->
        {#if isOther}
          <span class="link-dot link-dot-other" aria-hidden="true"></span>
        {:else}
          <span class="link-dot" style="background: {$phenotypeHue($selectedPhenotypeId)}" aria-hidden="true"></span>
        {/if}
        {selectedLabel}
      </h3>
    {/if}
    {#if !hasRows || $selectedPhenotypeId === null}
      <p class="sub">{copy.contributingCodes.composition}</p>
    {:else if isOther}
      <p class="sub">{copy.contributingCodes.subOther}</p>
    {:else}
      <p class="sub">{copy.contributingCodes.subMatch}</p>
    {/if}
  </header>

  {#if !$bundle || rows.length === 0}
    <p class="hint">{copy.contributingCodes.emptyRecord}</p>
  {:else}
    <ol class="codes">
      {#each sorted as row (row.w)}
        {@const c = $bundle.vocab.codes[row.w]}
        {@const matched = $searchedConditionIdx === row.w}
        <li class="code" class:matched>
          <span class="desc">
            {#if matched}<span class="match-dot" aria-hidden="true"></span>{/if}{c.description || c.code}
          </span>
          <span class="bar" aria-hidden="true">
            {#each row.segments as s}
              <span
                class="seg"
                class:dim={!segActive(s.k)}
                style="width: {(s.weight * 100).toFixed(2)}%; background: {segColor(s.k)}"
                title={`${segLabel(s.k)}: ${(s.weight * 100).toFixed(1)}%`}
              ></span>
            {/each}
          </span>
          <span class="count" data-numeric>×{row.count}</span>
        </li>
      {/each}
    </ol>
  {/if}
</section>

<style>
  .contrib {
    margin-top: 2rem;
    padding: 1.25rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
  }
  .head {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    margin-bottom: 0.85rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid var(--rule);
  }
  .top-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.85rem;
  }
  .head h3 {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: var(--tracking-tight);
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin: 0;
  }
  .open-in-atlas {
    border: 1px solid var(--rule-strong);
    background: var(--surface);
    color: var(--ink-muted);
    padding: 0.25rem 0.6rem;
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    cursor: pointer;
    flex-shrink: 0;
    transition: color 0.12s ease, border-color 0.12s ease;
  }
  .open-in-atlas:hover { color: var(--accent); border-color: var(--accent); }
  .link-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  /* Matches the Other band's striped fill in ProfileBar so the link from
     the band to this dot is visually obvious. */
  .link-dot-other {
    background: var(--surface-deep);
    background-image: repeating-linear-gradient(
      45deg,
      transparent,
      transparent 2px,
      rgba(82, 82, 91, 0.35) 2px,
      rgba(82, 82, 91, 0.35) 3px
    );
    border: 1px solid var(--rule-strong);
  }

  .hint {
    color: var(--ink-muted);
    font-size: var(--fs-small);
    margin: 0;
    padding: 0.5rem 0;
  }

  .codes {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .codes li.code {
    display: grid;
    grid-template-columns: 1fr 8rem 2.5rem;
    align-items: center;
    gap: 0.85rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--rule-faint);
    font-size: var(--fs-small);
  }
  .sub {
    margin: 0.2rem 0 0;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    font-style: italic;
  }
  .codes li:last-child { border-bottom: 0; }
  .codes li.matched .desc { color: var(--accent-search-ink); font-weight: 500; }
  .codes .desc {
    color: var(--ink);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  /* Fuchsia bullet matching the search highlight vocabulary elsewhere. */
  .match-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-search);
    margin-right: 0.45rem;
    vertical-align: middle;
  }
  .bar {
    display: flex;
    height: 8px;
    border-radius: 2px;
    overflow: hidden;
    background: var(--surface-recessed);
  }
  .seg {
    height: 100%;
    /* 2px surface gap between fills, per dataviz mark spec */
    box-shadow: inset -2px 0 0 var(--surface);
    transition: opacity 0.15s ease;
  }
  .seg.dim { opacity: 0.2; }
  .seg:last-child { box-shadow: none; }
  .count {
    text-align: right;
    color: var(--ink-muted);
  }
</style>
