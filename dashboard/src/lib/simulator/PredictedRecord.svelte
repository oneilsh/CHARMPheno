<script lang="ts">
  import { bundle, simulatorPrefix } from '../store'
  import { quantiles } from './runSamples'
  export let codeCountsSamples: Map<number, number>[] = []
  export let topN = 12

  // Build a set of prefix codes so we can exclude them from the
  // "expects also" list - we only want to show NEW codes the model added.
  // The prefix itself is already visible in the conditions editor on the
  // left, so we don't duplicate it here.
  $: prefixSet = new Set($simulatorPrefix)

  // Predicted rollup: codes the simulator drew across N samples that
  // AREN'T in the prefix. For each code, compute the P10 / median / P90
  // of its per-sample completion count, then take the top by median.
  // A code with median 0 but a non-zero P90 is interesting too (rarely
  // appears but when it does it's plausible), so we keep entries with
  // P90 > 0 even when median is 0.
  $: predictedRows = (() => {
    if (codeCountsSamples.length === 0 || !$bundle) return [] as { w: number; p10: number; p50: number; p90: number }[]
    const codes = new Set<number>()
    for (const m of codeCountsSamples) for (const w of m.keys()) {
      if (!prefixSet.has(w)) codes.add(w)
    }
    const rows: { w: number; p10: number; p50: number; p90: number }[] = []
    for (const w of codes) {
      const counts = codeCountsSamples.map((m) => m.get(w) ?? 0)
      const q = quantiles(counts, [0.1, 0.5, 0.9])
      if (q[2] === 0) continue
      rows.push({ w, p10: q[0], p50: q[1], p90: q[2] })
    }
    rows.sort((a, b) => (b.p50 - a.p50) || (b.p90 - a.p90))
    return rows.slice(0, topN)
  })()

  $: maxP90 = predictedRows.length > 0 ? Math.max(...predictedRows.map((r) => r.p90)) : 1
</script>

<section class="predicted">
  <div class="block">
    <header>
      <span class="eyebrow">Posterior predictive</span>
      <h4>Model expects to also see</h4>
      <p class="sub">Across {codeCountsSamples.length} simulated years, the model most often fills these in. The bar shows how often this condition shows up (P10 → P90, tick at median).</p>
    </header>
    {#if predictedRows.length === 0}
      <p class="hint">{codeCountsSamples.length === 0 ? 'Run the simulator to see what the model predicts.' : 'The model did not predict any additional conditions in this run.'}</p>
    {:else}
      <table>
        <tbody>
          {#each predictedRows as r}
            {@const c = $bundle!.vocab.codes[r.w]}
            <tr>
              <td class="dom-cell"><span class="domain-mark dom-{c.domain}">{c.domain.slice(0, 3)}</span></td>
              <td class="desc">{c.description || c.code}</td>
              <td class="bar">
                <span class="rng" style="left: {(r.p10 / maxP90) * 100}%; width: {Math.max(1, ((r.p90 - r.p10) / maxP90) * 100)}%"></span>
                <span class="med" style="left: {(r.p50 / maxP90) * 100}%"></span>
              </td>
              <td class="num" data-numeric>{r.p50.toFixed(1)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
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
  td.dom-cell { width: 4rem; }
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
  td.bar .rng {
    position: absolute;
    top: 0.6rem;
    height: 3px;
    background: var(--accent);
    opacity: 0.35;
    border-radius: 1.5px;
  }
  td.bar .med {
    position: absolute;
    top: 0.35rem;
    width: 2px;
    height: 11px;
    background: var(--accent);
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
