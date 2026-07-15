// Bubble size follows generative patient coverage for STM bundles, and the
// empirical fractionAboveTau fallback for non-STM bundles. Reads the main
// bubble's `r` off the rendered DOM so it catches real wiring regressions.
import { it, expect, afterEach } from 'vitest'
import { render, cleanup } from '@testing-library/svelte'
import TopicMap from './TopicMap.svelte'
import { bundle, atlasConditioning } from '../store'
import { makeStmBundleFixture } from '../test-fixtures'
import type { PredictiveGain } from '../types'

afterEach(() => { cleanup(); atlasConditioning.set({ covariateActive: false, values: {}, group: null }) })

function mainBubbleRadii(container: HTMLElement): number[] {
  return Array.from(container.querySelectorAll('svg g.node')).map((n) => {
    const c = n.querySelector('g.inner circle')
    return c ? Number(c.getAttribute('r')) : NaN
  })
}

it('STM bundle: bubble size follows generative coverage (age-loaded topic 2 > topic 1 at marginal)', () => {
  bundle.set(makeStmBundleFixture())
  const { container } = render(TopicMap)
  const [, r1, r2] = mainBubbleRadii(container)
  // Marginal cohort favors topic 2 (strong positive age effect) over topic 1.
  expect(r2).toBeGreaterThan(r1)
})

it('non-STM bundle: bubble size falls back to fractionAboveTau (corpus_prevalence order)', () => {
  const b = makeStmBundleFixture()
  // Strip the STM signals so the reader uses the histogram/corpus_prevalence fallback.
  delete (b as any).covariateEffects
  delete (b as any).correlation
  bundle.set(b)
  const { container } = render(TopicMap)
  const [, r1, r2] = mainBubbleRadii(container)
  // corpus_prevalence: id1=0.3 > id2=0.2 -> id1 bigger.
  expect(r1).toBeGreaterThan(r2)
})

it('predictive_gain present does NOT change the size source (still coverage)', () => {
  const b = makeStmBundleFixture()
  const PG: PredictiveGain = {
    presence: [0.4, 0.4, 0.4], mean_gain: [1, 20, 1], depth: [0.1, 0.1, 0.1],
    prominence_hist: [[1], [1], [1]], length_corr: [0, 0, 0], dedup_gain: [0, 0, 0],
    prominence_bin_edges: [0, 1], null_band: { mean: 0, std: 1, n: 100, p95: 1, hist: [1] },
    observed_delta_range: [-1, 1], downdate_audit: { max_abs_overall: 0.01, n_docs_audited: 100 },
    scale: 1.0, n_docs: 100,
  }
  b.phenotypes.predictive_gain = PG
  b.phenotypes.phenotypes.forEach((p, i) => { (p as any).mean_gain = PG.mean_gain[i] })
  bundle.set(b)
  const { container } = render(TopicMap)
  const [, r1, r2] = mainBubbleRadii(container)
  // mean_gain now orders topic1 (20) > topic2 (1) — the OPPOSITE of the
  // coverage order (topic 2 > topic 1 at the marginal profile, per the
  // fixture's strong positive age effect on topic 2). This makes the
  // assertion discriminating: if size were still driven by mean_gain
  // (reverted behavior), topic 1 would be bigger and r2 > r1 would FAIL.
  // It only passes because size follows coverage and ignores mean_gain.
  expect(r2).toBeGreaterThan(r1)
  expect(r1).toBeGreaterThan(0)
})
