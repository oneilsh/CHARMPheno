import { describe, it, expect } from 'vitest'
import { generateCohort } from './cohort'
import type { Model } from './types'

const model: Model = {
  K: 3, V: 5,
  alpha: [0.1, 0.1, 0.1],
  beta: [
    [0.9, 0.025, 0.025, 0.025, 0.025],
    [0.025, 0.9, 0.025, 0.025, 0.025],
    [0.025, 0.025, 0.9, 0.025, 0.025],
  ],
}

describe('generateCohort', () => {
  it('deterministic given seed', () => {
    const a = generateCohort({ model, meanCodesPerDoc: 8, n: 10, seed: 42, nNeighbors: 3 })
    const b = generateCohort({ model, meanCodesPerDoc: 8, n: 10, seed: 42, nNeighbors: 3 })
    expect(a.patients.map((p) => p.code_bag)).toEqual(b.patients.map((p) => p.code_bag))
  })

  it('produces patients on the simplex', () => {
    const c = generateCohort({ model, meanCodesPerDoc: 5, n: 12, seed: 1, nNeighbors: 3 })
    expect(c.patients.length).toBe(12)
    for (const p of c.patients) {
      expect(p.theta.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
      expect(p.code_bag.length).toBeGreaterThan(0)
      expect(p.neighbors.length).toBe(3)
      expect(p.neighbors.includes(p.id)).toBe(false)
      expect(new Set(p.neighbors).size).toBe(3)
    }
  })

  it('zero-pads patient ids', () => {
    const c = generateCohort({ model, meanCodesPerDoc: 5, n: 5, seed: 1, nNeighbors: 2 })
    expect(c.patients[0].id).toBe('S0000')
    expect(c.patients[4].id).toBe('S0004')
  })

  it('set mode: all patients conditioned at the same group; per-patient group recorded', () => {
    const bundle: any = {
      model: { K: 4, V: 2, alpha: [1, 1, 1, 1], beta: [[.5, .5], [.5, .5], [.9, .1], [.1, .9]] },
      covariateSchema: { k: 1, controls: [], design_columns: [{ name: 'Intercept', recipe: { kind: 'intercept' } }], unsupported: [] },
      covariateEffects: [{ covariate: 'Intercept', per_topic: [0, 0, 0, 0] }],
      correlation: {
        topic_order: [1, 2, 3], block_labels: ['background', 'cancer', 'dementia'],
        R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        identified: [[true, true, true], [true, true, true], [true, true, true]],
        support: [[9, 9, 9], [9, 9, 9], [9, 9, 9]], reference_topic: 0,
      },
      gating: { group_var: 'g', groups: ['cancer', 'dementia'], topic_blocks: ['background', 'background', 'cancer', 'dementia'], group_proportions: { cancer: 0.8, dementia: 0.2 } },
      corpusStats: { mean_codes_per_doc: 10 },
    }
    const c = generateCohort({
      model: bundle.model, meanCodesPerDoc: 10, n: 20, seed: 1, nNeighbors: 3,
      conditioning: { mode: 'set', values: {}, group: 'cancer', bundle },
    })
    // set mode -> every patient is cancer; dementia foreground (topic 3) is masked.
    for (const p of c.patients) {
      expect(p.group).toBe('cancer')
      expect(p.theta[3]).toBe(0)
    }
  })
})
