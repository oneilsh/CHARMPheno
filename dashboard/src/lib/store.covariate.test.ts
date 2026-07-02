import { it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import {
  bundle, conditioning, prevalenceReader,
  atlasConditioning, simulatorConditioning, patientConditioning,
  resetConditioningForCohort,
} from './store'

beforeEach(() => {
  bundle.set(null)
  resetConditioningForCohort()   // clears all three panel stores (atlas is the `conditioning` alias)
})

const COV_BUNDLE = {
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
}

it('covariateActive makes prevalenceReader use softmax(Gamma^T x)', () => {
  bundle.set(COV_BUNDLE as any)
  conditioning.set({ covariateActive: true, values: { age: Math.log(2) }, group: null })
  const reader = get(prevalenceReader)
  expect(reader({ id: 0 } as any)).toBeCloseTo(2 / 3, 6)
  expect(reader({ id: 1 } as any)).toBeCloseTo(1 / 3, 6)
})

it('gating-only quadrant masks the non-covariate base without covariate_effects', () => {
  // No covariateSchema / covariateEffects; gating present. corpus_prevalence is
  // the base (no theta histogram), masked by group.
  bundle.set({
    phenotypes: { phenotypes: [
      { id: 0, corpus_prevalence: 0.5 },
      { id: 1, corpus_prevalence: 0.3 },
    ] },
    gating: { group_var: 'g', groups: ['rare_dx'],
      topic_blocks: ['background', 'rare_dx'] },
  } as any)
  // Background only (group null): foreground topic 1 hidden.
  let reader = get(prevalenceReader)
  expect(reader({ id: 0, corpus_prevalence: 0.5 } as any)).toBeCloseTo(0.5, 6)
  expect(reader({ id: 1, corpus_prevalence: 0.3 } as any)).toBe(0)
  // Select rare_dx: foreground topic 1 revealed at its base value.
  conditioning.set({ covariateActive: false, values: {}, group: 'rare_dx' })
  reader = get(prevalenceReader)
  expect(reader({ id: 1, corpus_prevalence: 0.3 } as any)).toBeCloseTo(0.3, 6)
})

it('plain bundle uses the unchanged fractionAboveTau base', () => {
  bundle.set({
    phenotypes: { phenotypes: [{ id: 0, corpus_prevalence: 0.42 }] },
  } as any)
  const reader = get(prevalenceReader)
  expect(reader({ id: 0, corpus_prevalence: 0.42 } as any)).toBeCloseTo(0.42, 6)
})

it('panel conditioning stores are independent', () => {
  atlasConditioning.set({ covariateActive: true, values: { age: 70 }, group: 'cancer' })
  simulatorConditioning.set({ covariateActive: false, values: {}, group: null })
  expect(get(atlasConditioning).group).toBe('cancer')
  expect(get(simulatorConditioning).group).toBe(null)   // not shared
})

it('resetConditioningForCohort clears all panels', () => {
  atlasConditioning.set({ covariateActive: true, values: { age: 70 }, group: 'cancer' })
  patientConditioning.set({ covariateActive: true, values: { age: 40 }, group: 'dementia' })
  resetConditioningForCohort()
  expect(get(atlasConditioning)).toEqual({ covariateActive: false, values: {}, group: null })
  expect(get(patientConditioning)).toEqual({ covariateActive: false, values: {}, group: null })
})
