import { it, expect, afterEach, beforeEach } from 'vitest'
import { render, cleanup } from '@testing-library/svelte'
import CodePanel from './CodePanel.svelte'
import { bundle, selectedPhenotypeId, advancedView } from '../store'
import { makeStmBundleFixture } from '../test-fixtures'
import type { PredictiveGain } from '../types'

afterEach(() => cleanup())

beforeEach(() => {
  advancedView.set(true)
  selectedPhenotypeId.set(1)
})

const PG: PredictiveGain = {
  presence: [0.5, 0.6, 0.7],
  mean_gain: [0.01, 0.02, 0.03],
  depth: [0.1, 0.2, 0.3],
  prominence_hist: [[0.4, 0.6], [0.3, 0.7], [0.5, 0.5]],
  length_corr: [0, 0.1, -0.1],
  dedup_gain: [0.01, 0.01, 0.01],
  prominence_bin_edges: [-1, 0, 1],
  null_band: { mean: 0, std: 1, n: 100, p95: 1, hist: [1, 2] },
  observed_delta_range: [-1, 1],
  downdate_audit: { max_abs_overall: 0.01, n_docs_audited: 100 },
  scale: 1.0,
  n_docs: 100,
}

it('backward-compat: without predictive_gain, the theta prominence histogram (PrevalenceHistogram) still renders in advanced view', () => {
  const b = makeStmBundleFixture()
  // Give the selected phenotype a theta_histogram so hasHistogram is true,
  // matching the pre-existing behavior this task must not regress.
  b.phenotypes.phenotypes[1] = {
    ...b.phenotypes.phenotypes[1],
    theta_histogram: [0.3, 0.7],
    theta_percentiles: { p5: 0.01, p25: 0.05, p50: 0.1, p75: 0.2, p95: 0.3 },
  }
  b.phenotypes.theta_histogram_bin_edges = [0, 0.5, 1]
  bundle.set(b)
  const { container, getByText } = render(CodePanel)
  expect(getByText('Phenotype Prominence')).toBeTruthy()
  // The old below-tau summary line only exists on the theta-histogram path.
  expect(container.querySelector('.hist-below')).toBeTruthy()
})

it('with predictive_gain present and hydrated, ProminenceHistogram (nats scale) renders instead, with no below-tau summary', () => {
  const b = makeStmBundleFixture()
  b.phenotypes.predictive_gain = PG
  b.phenotypes.phenotypes.forEach((p, i) => {
    p.presence = PG.presence[i]
    p.depth = PG.depth[i]
    p.prominence_hist = PG.prominence_hist[i]
  })
  bundle.set(b)
  const { container, getByText } = render(CodePanel)
  expect(getByText('Phenotype Prominence')).toBeTruthy()
  // Prominence path has no tau concept -> no below-tau summary chip.
  expect(container.querySelector('.hist-below')).toBeFalsy()
  // The chart renders bars for the 2-bin prominence_hist fixture.
  const rects = container.querySelectorAll('.hist-wrap svg rect')
  expect(rects.length).toBe(2)
})
