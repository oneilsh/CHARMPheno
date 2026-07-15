import { it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import {
  bundle, conditioning, coverageReader,
  atlasConditioning, simulatorConditioning, patientConditioning,
  resetConditioningForCohort,
} from './store'
import { makeStmBundleFixture } from './test-fixtures'

beforeEach(() => {
  bundle.set(null)
  resetConditioningForCohort()   // clears all three panel stores (atlas is the `conditioning` alias)
})

it('covariateActive drives coverageReader from the generative cohort (age-loaded topic wins)', () => {
  // Fixture Gamma: topic2 carries a strong positive age effect, topic1 a small
  // negative one. At age=100 topic2's expected mass — and thus its patient
  // coverage — exceeds topic1's. (Old behavior asserted an exact softmax(Gamma^T x)
  // point estimate; coverage is now a sampled fraction, so we assert the order.)
  bundle.set(makeStmBundleFixture())
  conditioning.set({ covariateActive: true, values: { age: 100 }, group: null })
  const reader = get(coverageReader)
  expect(reader({ id: 2 } as any)).toBeGreaterThan(reader({ id: 1 } as any))
})

it('gated bundle shows all cohorts in the display reader (no group masking)', () => {
  // The Phenotype Atlas encodes cohort as node COLOR, not a filter, so the
  // display reader never zeroes foreground topics by group — every cohort's
  // topics show their base prevalence regardless of any selected group.
  bundle.set({
    phenotypes: { phenotypes: [
      { id: 0, corpus_prevalence: 0.5 },
      { id: 1, corpus_prevalence: 0.3 },
    ] },
    gating: { group_var: 'g', groups: ['rare_dx'],
      topic_blocks: ['background', 'rare_dx'] },
  } as any)
  const reader = get(coverageReader)
  expect(reader({ id: 0, corpus_prevalence: 0.5 } as any)).toBeCloseTo(0.5, 6)
  // foreground topic 1 is NOT masked to 0 — it shows its base prevalence.
  expect(reader({ id: 1, corpus_prevalence: 0.3 } as any)).toBeCloseTo(0.3, 6)
})

it('plain bundle uses the unchanged fractionAboveTau base', () => {
  bundle.set({
    phenotypes: { phenotypes: [{ id: 0, corpus_prevalence: 0.42 }] },
  } as any)
  const reader = get(coverageReader)
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
