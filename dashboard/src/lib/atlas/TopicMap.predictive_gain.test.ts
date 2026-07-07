// Task 6b: bubble-size encoding smoke tests. TopicMap sizes its bubbles from
// the prevalence reader by default; when the bundle carries predictive_gain
// it should switch the size SOURCE to mean_gain instead — while everything
// else (color, layout) stays put. These tests render the real component and
// read the main bubble's `r` attribute off the DOM rather than re-deriving
// the d3 scale, so they catch a regression in the actual wiring, not just in
// a helper function.
import { it, expect, afterEach } from 'vitest'
import { render, cleanup } from '@testing-library/svelte'
import TopicMap from './TopicMap.svelte'
import { bundle } from '../store'
import { makeStmBundleFixture } from '../test-fixtures'
import type { PredictiveGain } from '../types'

afterEach(() => cleanup())

// Fixture has 3 phenotypes (ids 0,1,2) with corpus_prevalence 0.5/0.3/0.2 —
// so on the prevalence-based (fallback) encoding, id1's bubble is bigger
// than id2's. mean_gain below is chosen to INVERT that relationship, so a
// test that observes id2 > id1 in the predictive_gain case proves the size
// source actually switched (not just that numbers use a different scale).
const PG: PredictiveGain = {
  presence: [0.4, 0.4, 0.4],
  mean_gain: [1, 1, 20],
  depth: [0.1, 0.1, 0.1],
  prominence_hist: [[1], [1], [1]],
  length_corr: [0, 0, 0],
  dedup_gain: [0, 0, 0],
  prominence_bin_edges: [0, 1],
  null_band: { mean: 0, std: 1, n: 100, p95: 1, hist: [1] },
  observed_delta_range: [-1, 1],
  downdate_audit: { max_abs_overall: 0.01, n_docs_audited: 100 },
  scale: 1.0,
  n_docs: 100,
}

// Main bubble is the first <circle> inside each node's counter-scaled
// `.inner` group (appended before the selection/highlight rings).
function mainBubbleRadii(container: HTMLElement): number[] {
  const nodes = Array.from(container.querySelectorAll('svg g.node'))
  return nodes.map((n) => {
    const c = n.querySelector('g.inner circle')
    return c ? Number(c.getAttribute('r')) : NaN
  })
}

it('without predictive_gain, bubble size follows prevalence (fallback, unchanged)', () => {
  const b = makeStmBundleFixture()
  bundle.set(b)
  const { container } = render(TopicMap)
  const [, r1, r2] = mainBubbleRadii(container)
  // corpus_prevalence: id1=0.3 > id2=0.2 -> id1's bubble is bigger.
  expect(r1).toBeGreaterThan(r2)
})

it('with predictive_gain present, bubble size follows mean_gain instead of prevalence', () => {
  const b = makeStmBundleFixture()
  b.phenotypes.predictive_gain = PG
  b.phenotypes.phenotypes.forEach((p, i) => {
    p.presence = PG.presence[i]
    p.mean_gain = PG.mean_gain[i]
    p.depth = PG.depth[i]
  })
  bundle.set(b)
  const { container } = render(TopicMap)
  const [, r1, r2] = mainBubbleRadii(container)
  // mean_gain: id1=1 < id2=20 -> id2's bubble is now bigger, the OPPOSITE
  // of the prevalence-based ordering above — proves the size source swapped.
  expect(r2).toBeGreaterThan(r1)
})
