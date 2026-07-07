<script lang="ts">
  import { bundle, simulatorPrefix, phenotypesById, advancedView } from '../store'
  import { phenotypeHue } from '../palette'
  import { copy } from '../copy'
  export let codeCountsSamples: Map<number, number>[] = []
  // code id -> (generating topic id -> count). Drives the per-phenotype grouping.
  export let codeTopicCounts: Map<number, Map<number, number>> = new Map()
  // Display limits. Because codes are GROUPED by generating phenotype, a global
  // top-N would starve phenotypes with several moderate codes; instead we keep any
  // code above an occurrence-rate floor and cap PER phenotype (and cap the number
  // of phenotype groups) so the table stays full but bounded.
  export let minRate = 0.03      // show codes appearing in ≥3% of simulated years
  export let perPhenotype = 8    // max codes shown under each phenotype
  export let maxGroups = 12      // max phenotype groups shown

  // Build a set of prefix codes so we can exclude them from the
  // "expects also" list - we only want to show NEW codes the model added.
  // The prefix itself is already visible in the conditions editor on the
  // left, so we don't duplicate it here.
  $: prefixSet = new Set($simulatorPrefix)

  // Predicted rollup: codes the simulator drew across N samples that AREN'T in the
  // prefix. A simulated record is ~one year, so a given code appears 0 or 1 times
  // in most draws — per-sample count quantiles are degenerate (median 0, range
  // 0→1) and carry no signal. We instead report the OCCURRENCE RATE: the share of
  // simulated years in which the code shows up at least once (its posterior
  // predictive probability), which is interpretable across the whole long tail.
  $: predictedRows = (() => {
    if (codeCountsSamples.length === 0 || !$bundle) return [] as { w: number; rate: number }[]
    const N = codeCountsSamples.length
    const codes = new Set<number>()
    for (const m of codeCountsSamples) for (const w of m.keys()) {
      if (!prefixSet.has(w)) codes.add(w)
    }
    const rows: { w: number; rate: number }[] = []
    for (const w of codes) {
      let occ = 0
      for (const m of codeCountsSamples) if ((m.get(w) ?? 0) > 0) occ++
      const rate = occ / N
      if (rate < minRate) continue
      rows.push({ w, rate })
    }
    rows.sort((a, b) => b.rate - a.rate)
    return rows
  })()

  // Dominant generating phenotype for a code: the topic that emitted it most
  // often across all draws (argmax of codeTopicCounts). -1 when unattributed.
  function domTopic(w: number): number {
    const tm = codeTopicCounts.get(w)
    if (!tm) return -1
    let best = -1, bestC = -1
    for (const [z, c] of tm) if (c > bestC) { bestC = c; best = z }
    return best
  }

  // Group the predicted codes under the phenotype that generated them, keeping the
  // FULL sorted rows per group (the per-phenotype / group caps are applied at
  // render time so "show more" can reveal the rest). In basic mode a dead/mixed
  // generator folds into the "Other phenotypes" group (id -1) so a bad-quality
  // phenotype name never surfaces — matching the rest of the UI.
  $: allGroups = (() => {
    const OTHER = -1
    const byPheno = new Map<number, typeof predictedRows>()
    for (const r of predictedRows) {
      let k = domTopic(r.w)
      if (k >= 0 && !$advancedView) {
        const q = $phenotypesById.get(k)?.quality
        if (q === 'dead' || q === 'mixed') k = OTHER
      }
      const arr = byPheno.get(k) ?? []
      arr.push(r)
      byPheno.set(k, arr)
    }
    return [...byPheno.entries()]
      .map(([k, rows]) => ({
        k,
        rows: rows.slice().sort((a, b) => b.rate - a.rate),
        total: rows.reduce((s, r) => s + r.rate, 0),
      }))
      // Real phenotypes first (by total predicted mass), the Other bucket last.
      .sort((a, b) => (a.k < 0 ? 1 : 0) - (b.k < 0 ? 1 : 0) || (b.total - a.total))
  })()

  // Progressive disclosure: cap groups + codes-per-group by default; the buttons
  // below reveal the rest. Expanded state is keyed by phenotype id.
  let expandedGroups = new Set<number>()
  let showAllGroups = false
  // A fresh run resets the disclosure back to the capped default.
  $: codeCountsSamples, (expandedGroups = new Set(), showAllGroups = false)
  function toggleGroup(k: number) {
    const s = new Set(expandedGroups)
    s.has(k) ? s.delete(k) : s.add(k)
    expandedGroups = s
  }

  $: visibleGroups = showAllGroups ? allGroups : allGroups.slice(0, maxGroups)
  $: hiddenGroupCount = Math.max(0, allGroups.length - maxGroups)

  const groupLabel = (k: number) =>
    k < 0 ? 'Other phenotypes' : ($phenotypesById.get(k)?.label || `Phenotype ${k}`)

  // Phenotype-level share: the fraction of ALL generated new-code tokens attributed
  // to each (display) phenotype — how much of the model's added record this
  // phenotype drives. Summed over every generated code (not just the shown ones)
  // from codeTopicCounts, excluding prefix codes; dead/mixed fold into -1 in basic
  // mode, matching the groups. Additive, so it differentiates phenotypes rather
  // than saturating near 100% the way a per-year occurrence would.
  $: phenoTokens = (() => {
    const tok = new Map<number, number>()
    let grand = 0
    for (const [w, tm] of codeTopicCounts) {
      if (prefixSet.has(w)) continue
      for (const [z, c] of tm) {
        let k = z
        if (!$advancedView) {
          const q = $phenotypesById.get(z)?.quality
          if (q === 'dead' || q === 'mixed') k = -1
        }
        tok.set(k, (tok.get(k) ?? 0) + c)
        grand += c
      }
    }
    return { tok, grand }
  })()
  $: groupShare = (k: number) =>
    phenoTokens.grand > 0 ? (phenoTokens.tok.get(k) ?? 0) / phenoTokens.grand : 0
</script>

<section class="predicted">
  <div class="block">
    <header>
      <span class="eyebrow">Posterior predictive</span>
      <h4>{copy.predictedRecord.heading}</h4>
      <p class="sub">{copy.predictedRecord.sub(codeCountsSamples.length)}</p>
    </header>
    {#if predictedRows.length === 0}
      <p class="hint">{codeCountsSamples.length === 0 ? copy.predictedRecord.hintEmpty : copy.predictedRecord.hintNone}</p>
    {:else}
      {#each visibleGroups as g}
        {@const expanded = expandedGroups.has(g.k)}
        {@const shownRows = expanded ? g.rows : g.rows.slice(0, perPhenotype)}
        <div class="pheno-group">
          <div class="pheno-head" title="Share of the model's added codes generated by this phenotype">
            <span class="pheno-dot" style="background: {g.k < 0 ? 'var(--ink-faint)' : $phenotypeHue(g.k)}"></span>
            <span class="pheno-name">{groupLabel(g.k)}</span>
            <span class="pheno-share">
              <span class="pheno-bar">
                <span class="pheno-fill" style="width: {Math.max(1, groupShare(g.k) * 100)}%; background: {g.k < 0 ? 'var(--ink-faint)' : $phenotypeHue(g.k)}"></span>
              </span>
              <span class="pheno-pct" data-numeric>{(groupShare(g.k) * 100).toFixed(0)}%</span>
            </span>
          </div>
          <table>
            <tbody>
              {#each shownRows as r}
                {@const c = $bundle!.vocab.codes[r.w]}
                <tr>
                  <td class="desc">{c.description || c.code}</td>
                  <td class="bar">
                    <span class="fill" style="width: {Math.max(1, r.rate * 100)}%"></span>
                  </td>
                  <td class="num" data-numeric>{(r.rate * 100).toFixed(0)}%</td>
                </tr>
              {/each}
            </tbody>
          </table>
          {#if g.rows.length > perPhenotype}
            <button class="show-more" on:click={() => toggleGroup(g.k)}>
              {expanded ? 'Show fewer' : `Show ${g.rows.length - perPhenotype} more`}
            </button>
          {/if}
        </div>
      {/each}
      {#if hiddenGroupCount > 0 || showAllGroups}
        <button class="show-more show-more-groups" on:click={() => (showAllGroups = !showAllGroups)}>
          {showAllGroups ? 'Show fewer phenotypes' : `Show ${hiddenGroupCount} more phenotype${hiddenGroupCount === 1 ? '' : 's'}`}
        </button>
      {/if}
    {/if}
  </div>
</section>

<style>
  .predicted {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }
  .block {
    padding: 1.25rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
  }
  header {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    margin-bottom: 0.85rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid var(--rule);
  }
  header h4 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: var(--tracking-tight);
  }
  .sub {
    margin: 0.2rem 0 0;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    font-style: italic;
    line-height: 1.5;
  }
  /* Per-generating-phenotype group: a colored header over that phenotype's
     predicted codes. */
  .pheno-group + .pheno-group { margin-top: 0.9rem; }
  .pheno-head {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.1rem 0 0.35rem;
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-muted);
  }
  .pheno-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .pheno-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
    flex: 1;
  }
  /* Phenotype-level share bar, right-aligned in the group header. */
  .pheno-share {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    flex-shrink: 0;
  }
  .pheno-bar {
    position: relative;
    width: 64px;
    height: 5px;
    border-radius: 2.5px;
    background: var(--surface-recessed);
    overflow: hidden;
  }
  .pheno-fill {
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    border-radius: 2.5px;
    opacity: 0.8;
  }
  .pheno-pct {
    width: 2.5rem;
    text-align: right;
    color: var(--ink-muted);
  }
  /* Progressive-disclosure toggles: a subtle text button under each group and a
     wider one at the bottom for revealing more phenotype groups. */
  .show-more {
    margin-top: 0.3rem;
    padding: 0.15rem 0;
    border: 0;
    background: none;
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: var(--fs-micro);
    letter-spacing: 0.04em;
    cursor: pointer;
  }
  .show-more:hover { text-decoration: underline; }
  .show-more-groups {
    display: block;
    width: 100%;
    text-align: left;
    margin-top: 0.9rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--rule);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--fs-small);
  }
  td {
    padding: 0.38rem 0.25rem;
    border-bottom: 1px solid var(--rule-faint);
    vertical-align: middle;
  }
  tr:last-child td { border-bottom: 0; }
  td.desc {
    color: var(--ink);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 0;
  }
  td.bar {
    width: 38%;
    position: relative;
    height: 1.4rem;
  }
  /* Occurrence-rate fill: a bar from the left proportional to the share of
     simulated years the condition appears in. */
  td.bar .fill {
    position: absolute;
    top: 0.5rem;
    left: 0;
    height: 5px;
    background: var(--accent);
    opacity: 0.55;
    border-radius: 2.5px;
  }
  td.num {
    width: 3.5rem;
    text-align: right;
    color: var(--ink-muted);
  }
  .hint {
    color: var(--ink-faint);
    font-size: var(--fs-small);
    margin: 0;
    padding: 0.5rem 0 0;
    font-style: italic;
  }
</style>
