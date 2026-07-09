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

it('with predictive_gain present, the theta-coverage histogram still renders — the prominence nats plot is no longer preferred', () => {
  const b = makeStmBundleFixture()
  b.phenotypes.predictive_gain = PG
  b.phenotypes.phenotypes.forEach((p, i) => {
    p.presence = PG.presence[i]
    p.depth = PG.depth[i]
    p.prominence_hist = PG.prominence_hist[i]
  })
  // The selected phenotype also carries a theta_histogram: the interpretable
  // theta-coverage path must win even though predictive_gain (and its
  // prominence_hist) are present — the nats prominence plot was retired.
  b.phenotypes.phenotypes[1] = {
    ...b.phenotypes.phenotypes[1],
    theta_histogram: [0.3, 0.7],
    theta_percentiles: { p5: 0.01, p25: 0.05, p50: 0.1, p75: 0.2, p95: 0.3 },
  }
  b.phenotypes.theta_histogram_bin_edges = [0, 0.5, 1]
  bundle.set(b)
  const { container, getByText } = render(CodePanel)
  expect(getByText('Phenotype Prominence')).toBeTruthy()
  // The below-tau summary chip exists ONLY on the theta-coverage path, so its
  // presence proves the PrevalenceHistogram (not the prominence plot) rendered.
  expect(container.querySelector('.hist-below')).toBeTruthy()
  const rects = container.querySelectorAll('.hist-wrap svg rect')
  expect(rects.length).toBeGreaterThan(0)
})

it('advanced view shows a Distinctiveness stat (mean_gain) and a Presence chip; depth/dedup/length are bundled in the Presence hover (no standalone chips)', () => {
  const b = makeStmBundleFixture()
  b.phenotypes.predictive_gain = PG
  b.phenotypes.phenotypes.forEach((p, i) => {
    (p as any).mean_gain = PG.mean_gain[i]
    ;(p as any).presence = PG.presence[i]
    ;(p as any).depth = PG.depth[i]
    ;(p as any).dedup_gain = PG.dedup_gain[i]
    ;(p as any).length_corr = PG.length_corr[i]
  })
  bundle.set(b)
  const { queryByText, container } = render(CodePanel)   // advancedView is true (beforeEach)
  expect(queryByText('Distinctiveness')).toBeTruthy()
  // Presence is now a first-class chip (value = presence %); phenotype 1 -> 60%.
  expect(queryByText('Presence')).toBeTruthy()
  expect(queryByText('60%')).toBeTruthy()
  // Depth/dedup/length are NOT standalone visible chips — they live in the
  // Presence chip's hover (title attribute), so no visible "Depth" text.
  expect(queryByText('Depth')).toBeNull()
  const presenceChip = container.querySelector('span.stat[title*="Depth:"]')
  expect(presenceChip).toBeTruthy()
  expect(presenceChip!.getAttribute('title')).toContain('Dedup gain:')
  expect(presenceChip!.getAttribute('title')).toContain('Length corr:')
})

it('basic view hides the Distinctiveness stat', () => {
  advancedView.set(false)
  const b = makeStmBundleFixture()
  b.phenotypes.predictive_gain = PG
  b.phenotypes.phenotypes.forEach((p, i) => { (p as any).mean_gain = PG.mean_gain[i] })
  bundle.set(b)
  const { queryByText } = render(CodePanel)
  expect(queryByText('Distinctiveness')).toBeNull()
})
