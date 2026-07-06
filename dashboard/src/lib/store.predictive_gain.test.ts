import { it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { bundle, presenceReader, depthReader, predictiveGain, tauThreshold } from './store'
import type { Phenotype, PredictiveGain } from './types'

const PG: PredictiveGain = {
  presence: [0.1, 0.5],
  mean_gain: [0.01, 0.05],
  depth: [0.2, 0.8],
  prominence_hist: [[1, 2], [3, 4]],
  length_corr: [0, 0.1],
  dedup_gain: [0.02, 0.03],
  prominence_bin_edges: [0, 1, 2],
  null_band: { mean: 0, std: 1, n: 100, p95: 2, hist: [1, 2, 3] },
  observed_delta_range: [-1, 1],
  downdate_audit: { max_abs_overall: 0.01, n_docs_audited: 100 },
  scale: 1.0,
  n_docs: 100,
}

beforeEach(() => {
  bundle.set(null)
  tauThreshold.set(0.02)
})

it('presenceReader returns the hydrated presence value when present', () => {
  bundle.set({ phenotypes: { phenotypes: [], predictive_gain: PG } } as any)
  const reader = get(presenceReader)
  const p: Phenotype = { presence: 0.5 } as any
  expect(reader(p)).toBe(0.5)
})

it('presenceReader falls back to fractionAboveTau (theta-histogram based prevalence) when presence is absent', () => {
  bundle.set({
    phenotypes: {
      phenotypes: [],
      theta_histogram_bin_edges: [0, 0.5, 1],
    },
  } as any)
  const reader = get(presenceReader)
  // no presence field, no theta_histogram on the phenotype -> fractionAboveTau
  // falls back further to corpus_prevalence.
  const p: Phenotype = { corpus_prevalence: 0.42 } as any
  expect(reader(p)).toBeCloseTo(0.42, 6)
})

it('depthReader returns the hydrated depth value when present', () => {
  bundle.set({ phenotypes: { phenotypes: [], predictive_gain: PG } } as any)
  const reader = get(depthReader)
  expect(reader({ depth: 0.8 } as any)).toBe(0.8)
})

it('depthReader falls back to 0 when depth is null or undefined', () => {
  bundle.set({ phenotypes: { phenotypes: [] } } as any)
  const reader = get(depthReader)
  expect(reader({ depth: null } as any)).toBe(0)
  expect(reader({} as any)).toBe(0)
})

it('predictiveGain reads the bundle-level predictive_gain object', () => {
  bundle.set({ phenotypes: { phenotypes: [], predictive_gain: PG } } as any)
  expect(get(predictiveGain)).toEqual(PG)
})

it('predictiveGain is null when absent or when there is no bundle', () => {
  bundle.set({ phenotypes: { phenotypes: [] } } as any)
  expect(get(predictiveGain)).toBeNull()
  bundle.set(null)
  expect(get(predictiveGain)).toBeNull()
})
