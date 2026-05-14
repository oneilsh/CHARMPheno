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
  <h3>Top expected codes</h3>
  <table>
    <tbody>
    {#each top as r}
      {@const c = $bundle!.vocab.codes[r.w]}
      <tr>
        <td class="desc">{c.description || c.code}</td>
        <td class="bar">
          <span class="rng" style="left: {(r.p10 / maxP90) * 100}%; width: {((r.p90 - r.p10) / maxP90) * 100}%"></span>
          <span class="med" style="left: {(r.p50 / maxP90) * 100}%"></span>
        </td>
        <td class="num">{r.p50.toFixed(1)}</td>
      </tr>
    {/each}
    </tbody>
  </table>
</section>

<style>
  .expected { padding: 1rem; border: 1px solid #ddd; }
  h3 { margin: 0 0 0.5rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  td { padding: 0.15rem 0.25rem; border-bottom: 1px solid #f4f4f4; }
  td.desc { width: 50%; }
  td.bar { width: 40%; position: relative; height: 1.2rem; }
  td.bar .rng { position: absolute; top: 0.45rem; height: 0.25rem; background: #cfe2ff; }
  td.bar .med { position: absolute; top: 0.2rem; width: 2px; height: 0.7rem; background: #1e88e5; }
  td.num { width: 4rem; text-align: right; font-variant-numeric: tabular-nums; }
</style>
