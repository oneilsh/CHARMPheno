<script lang="ts">
  import { bundle, selectedPhenotypeId, phenotypesById, searchedConditionIdx } from '../store'
  import { phenotypeHue } from '../palette'
  import { go } from '../router'

  export let theta: number[]
  export let codeBag: number[]

  function openInAtlas() {
    // selectedPhenotypeId is already set to the clicked phenotype; just
    // switch tabs and the atlas's TopicMap + CodePanel pick it up.
    go('atlas')
  }

  // Mirrors ProfileBar's default otherThreshold. Phenotypes with theta
  // below this make up the patient's long-tail "Other" band. Clicking
  // that band sets selectedPhenotypeId = -1 (sentinel) and we aggregate
  // responsibility across every phenotype in the tail.
  const OTHER_THRESHOLD = 0.05

  $: isOther = $selectedPhenotypeId === -1

  $: counts = (() => {
    const m = new Map<number, number>()
    for (const w of codeBag) m.set(w, (m.get(w) ?? 0) + 1)
    return m
  })()

  $: top = (() => {
    if (!$bundle || $selectedPhenotypeId === null) return []
    const beta = $bundle.model.beta
    const K = $bundle.model.K
    // For the "Other" view, build the set of tail phenotype ids once so
    // the inner loop just sums their responsibilities for each code.
    const otherIds: number[] = []
    if (isOther) {
      for (let j = 0; j < K; j++) if (theta[j] < OTHER_THRESHOLD) otherIds.push(j)
    }
    const scored: { w: number; c: number; score: number }[] = []
    for (const [w, c] of counts) {
      let z = 0
      for (let j = 0; j < K; j++) z += beta[j][w] * theta[j]
      let pzkw = 0
      if (isOther) {
        let num = 0
        for (const j of otherIds) num += beta[j][w] * theta[j]
        pzkw = z > 0 ? num / z : 0
      } else {
        const k = $selectedPhenotypeId!
        pzkw = z > 0 ? (beta[k][w] * theta[k]) / z : 0
      }
      scored.push({ w, c, score: c * pzkw })
    }
    return scored.sort((a, b) => b.score - a.score).slice(0, 12)
  })()

  $: maxScore = top.length ? Math.max(...top.map((t) => t.score)) : 1
  $: selectedLabel = $selectedPhenotypeId === null
    ? null
    : isOther
      ? 'Other / tail phenotypes'
      : ($phenotypesById.get($selectedPhenotypeId)?.label || `Phenotype ${$selectedPhenotypeId}`)
</script>

<section class="contrib">
  <header class="head">
    <div class="top-row">
      <span class="eyebrow">Top contributing codes</span>
      {#if selectedLabel && $selectedPhenotypeId !== null && !isOther}
        <button
          class="open-in-atlas"
          type="button"
          on:click={openInAtlas}
          title="Switch to the Phenotype Atlas with this phenotype selected"
        >
          open in atlas →
        </button>
      {/if}
    </div>
    {#if selectedLabel && $selectedPhenotypeId !== null}
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
    {#if isOther}
      <p class="sub">Codes from this patient's record that the model attributes to phenotypes outside this patient's dominant mix, ordered by tail-responsibility.</p>
    {:else}
      <p class="sub">Codes from this patient's record that match this phenotype, ordered by occurrence count.</p>
    {/if}
  </header>

  {#if $selectedPhenotypeId === null}
    <p class="hint">Click a phenotype band above to see which codes from this patient's record drove the assignment.</p>
  {:else if top.length === 0}
    <p class="hint">No codes from this patient's record contribute to {selectedLabel}.</p>
  {:else}
    <ol class="codes">
      {#each top as t}
        {@const c = $bundle!.vocab.codes[t.w]}
        {@const matched = $searchedConditionIdx === t.w}
        <li class:matched>
          <span class="desc">
            {#if matched}<span class="match-dot" aria-hidden="true"></span>{/if}{c.description || c.code}
          </span>
          <span class="spark" aria-hidden="true">
            <span class="spark-bar" style="width: {(t.score / maxScore) * 100}%"></span>
          </span>
          <span class="count" data-numeric>×{t.c}</span>
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
  .codes li {
    display: grid;
    grid-template-columns: 1fr 6rem 2.5rem;
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
  .spark {
    display: block;
    height: 3px;
    background: var(--surface-recessed);
    border-radius: 1.5px;
    overflow: hidden;
  }
  .spark-bar {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 0.2s ease;
  }
  .count {
    text-align: right;
    color: var(--ink-muted);
  }
</style>
