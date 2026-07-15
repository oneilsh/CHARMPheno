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

  it('set mode with a prefix conditions theta toward the prefix-implicated topic (sampleRecordPosterior)', () => {
    // Same fixture as the group-conditioning test above, but with an
    // uncorrelated topic block structure (identity R) so the only signal
    // pulling mass toward topic 2 (cancer) is the prefix's observed codes,
    // not the covariate/group prior. beta is deliberately peaked so that
    // vocab word 0 is diagnostic of topic 2 - a prefix of word-0 codes
    // should therefore push posterior theta mass onto topic 2 relative to
    // the same cohort with no prefix.
    const bundle: any = {
      model: {
        K: 4, V: 4, alpha: [1, 1, 1, 1],
        beta: [
          [0.25, 0.25, 0.25, 0.25],
          [0.25, 0.25, 0.25, 0.25],
          [0.85, 0.05, 0.05, 0.05],
          [0.05, 0.05, 0.05, 0.85],
        ],
      },
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
    const N = 400
    // Prefix concentrated on vocab word 0, which beta makes diagnostic of
    // topic 2 ("cancer"). Group is fixed to 'cancer' in both runs so the
    // group mask (dementia = topic 3 zeroed) is identical; the only
    // difference between the two cohorts is the prefix.
    const prefixCounts = new Map<number, number>([[0, 6]])
    const withPrefix = generateCohort({
      model: bundle.model, meanCodesPerDoc: 10, n: N, seed: 7, nNeighbors: 3,
      conditioning: {
        mode: 'set', values: {}, group: 'cancer', bundle, prefixCounts, beta: bundle.model.beta,
      },
    })
    const noPrefix = generateCohort({
      model: bundle.model, meanCodesPerDoc: 10, n: N, seed: 7, nNeighbors: 3,
      conditioning: { mode: 'set', values: {}, group: 'cancer', bundle },
    })
    const meanMass = (c: typeof withPrefix, k: number) =>
      c.patients.reduce((s, p) => s + p.theta[k], 0) / c.patients.length
    const withMean = meanMass(withPrefix, 2)
    const noMean = meanMass(noPrefix, 2)
    expect(withMean).toBeGreaterThan(noMean + 0.1)
  })

  it('sample mode: each patient draws its own group + covariates; theta reflects the logistic-normal path (masked out-of-group topics are ~0)', () => {
    // Same shape of fixture as the 'set mode' test above (K=4: background
    // topics 0-1, cancer-only topic 2, dementia-only topic 3), but every
    // patient samples its OWN group from gating.group_proportions (via
    // sampleMarginalGroup) instead of sharing one fixed group. A patient
    // gated into 'cancer' should have its dementia-only topic (3) masked to
    // 0, and vice versa for a 'dementia' patient - the same allowed-set
    // masking sampleConditionedTheta applies in 'set' mode, just per-patient.
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
      model: bundle.model, meanCodesPerDoc: 10, n: 200, seed: 3, nNeighbors: 3,
      conditioning: { mode: 'sample', values: {}, group: null, bundle },
    })
    const groupsSeen = new Set(c.patients.map((p) => p.group))
    // With 200 draws at 80/20 proportions, both groups should show up.
    expect(groupsSeen.has('cancer')).toBe(true)
    expect(groupsSeen.has('dementia')).toBe(true)
    for (const p of c.patients) {
      expect(p.group === 'cancer' || p.group === 'dementia').toBe(true)
      if (p.group === 'cancer') expect(p.theta[3]).toBe(0)
      if (p.group === 'dementia') expect(p.theta[2]).toBe(0)
    }
  })
})
