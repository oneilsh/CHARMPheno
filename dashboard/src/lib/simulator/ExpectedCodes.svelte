<script lang="ts">
  import { bundle } from '../store'
  import { quantiles } from './runSamples'
  export let codeCountsSamples: Map<number, number>[]
  export let topN = 20

  $: top = (() => {
    if (codeCountsSamples.length === 0 || !$bundle) return []
    const all = new Set<number>()
    for (const m of codeCountsSamples) for (const w of m.keys()) all.add(w)
    const rows: { w: number; p10: number; p50: number; p90: number }[] = []
    for (const w of all) {
      const counts = codeCountsSamples.map((m) => m.get(w) ?? 0)
      const q = quantiles(counts, [0.1, 0.5, 0.9])
      if (q[1] === 0 && q[2] === 0) continue
      rows.push({ w, p10: q[0], p50: q[1], p90: q[2] })
    }
    rows.sort((a, b) => b.p50 - a.p50)
    return rows.slice(0, topN)
  })()

  $: maxP90 = top.length > 0 ? Math.max(...top.map((r) => r.p90)) : 1
</script>

<section class="expected">
  <header>
    <span class="eyebrow">Posterior predictive</span>
    <h3>Top expected codes</h3>
    <p class="kicker">Per-sample completion counts at the 10th / 50th / 90th percentile.</p>
  </header>

  <table>
    <tbody>
    {#each top as r}
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

  {#if top.length === 0}
    <p class="hint">No expected codes yet. Run a sample first.</p>
  {/if}
</section>

<style>
  .expected {
    padding: 1.25rem;
    background: var(--paper-elevated);
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
  }

  header {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    margin-bottom: 0.85rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid var(--rule);
  }
  header h3 {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 500;
    font-size: 1.4rem;
    letter-spacing: -0.005em;
  }
  .kicker {
    margin: 0.05rem 0 0;
    font-size: var(--fs-micro);
    color: var(--ink-faint);
    font-style: italic;
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
    width: 45%;
    color: var(--ink);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 0;
  }
  td.bar {
    width: 40%;
    position: relative;
    height: 1.4rem;
  }
  td.bar .rng {
    position: absolute;
    top: 0.6rem;
    height: 3px;
    background: var(--terracotta-soft);
    opacity: 0.55;
    border-radius: 1.5px;
  }
  td.bar .med {
    position: absolute;
    top: 0.35rem;
    width: 2px;
    height: 11px;
    background: var(--terracotta);
  }
  td.num {
    width: 3.5rem;
    text-align: right;
    color: var(--ink-muted);
  }

  .hint {
    color: var(--ink-faint);
    font-style: italic;
    font-size: var(--fs-small);
    padding: 0.5rem 0 0;
  }
</style>
