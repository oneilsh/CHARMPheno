import { it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { bundle, covariateMode, covariateValues, prevalenceReader } from './store'

beforeEach(() => {
  bundle.set(null)
  covariateMode.set(false)
  covariateValues.set({})
})

it('covariate mode makes prevalenceReader use softmax(Gamma^T x)', () => {
  bundle.set({
    // minimal bundle with two phenotypes id 0,1 + covariate schema/effects
    phenotypes: { phenotypes: [{ id: 0 }, { id: 1 }] },
    covariateSchema: { k: 20, unsupported: [],
      controls: [{ name: 'age', type: 'continuous', range: [0, 100], default: 50 }],
      design_columns: [
        { name: 'Intercept', recipe: { kind: 'intercept' } },
        { name: 'age', recipe: { kind: 'main', var: 'age' } },
      ] },
    covariateEffects: [
      { covariate: 'Intercept', per_topic: [0, 0] },
      { covariate: 'age', per_topic: [1, 0] },
    ],
  } as any)
  covariateValues.set({ age: Math.log(2) })
  covariateMode.set(true)
  const reader = get(prevalenceReader)
  expect(reader({ id: 0 } as any)).toBeCloseTo(2 / 3, 6)
  expect(reader({ id: 1 } as any)).toBeCloseTo(1 / 3, 6)
})

it('covariate mode off restores fractionAboveTau behavior', () => {
  bundle.set({
    phenotypes: {
      phenotypes: [{ id: 0, corpus_prevalence: 0.5, theta_histogram: null }],
      theta_histogram_bin_edges: undefined,
    },
    covariateSchema: { k: 1, unsupported: [], controls: [], design_columns: [] },
    covariateEffects: [{ covariate: 'Intercept', per_topic: [0] }],
  } as any)
  covariateMode.set(false)
  const reader = get(prevalenceReader)
  // Without histogram, fractionAboveTau falls back to corpus_prevalence
  expect(reader({ id: 0, corpus_prevalence: 0.5, theta_histogram: null } as any)).toBeCloseTo(0.5, 6)
})

it('covariate mode on but schema.unsupported non-empty falls back to fractionAboveTau', () => {
  bundle.set({
    phenotypes: {
      phenotypes: [{ id: 0, corpus_prevalence: 0.7, theta_histogram: null }],
      theta_histogram_bin_edges: undefined,
    },
    covariateSchema: { k: 1, unsupported: ['some_feature'], controls: [], design_columns: [] },
    covariateEffects: [{ covariate: 'Intercept', per_topic: [0] }],
  } as any)
  covariateMode.set(true)
  const reader = get(prevalenceReader)
  expect(reader({ id: 0, corpus_prevalence: 0.7, theta_histogram: null } as any)).toBeCloseTo(0.7, 6)
})
