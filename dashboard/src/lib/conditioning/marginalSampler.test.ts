import { describe, it, expect } from 'vitest'
import { createRng } from '../sampling'
import { sampleMarginalCovariates, sampleMarginalGroup } from './marginalSampler'
import type { CovariateSchema, GatingSpec } from '../types'

const schema: CovariateSchema = {
  k: 2,
  controls: [
    { name: 'age', type: 'continuous', range: [40, 80], default: 70 },
    { name: 'sex', type: 'categorical', reference: 'F',
      levels: ['F', 'M'], proportions: { F: 0.9, M: 0.1 } },
  ],
  design_columns: [],
  unsupported: [],
}

describe('sampleMarginalCovariates', () => {
  it('draws continuous within range and categorical from proportions', () => {
    const rng = createRng(1)
    let mCount = 0
    for (let i = 0; i < 4000; i++) {
      const v = sampleMarginalCovariates(schema, rng)
      expect(typeof v.age).toBe('number')
      expect(v.age as number).toBeGreaterThanOrEqual(40)
      expect(v.age as number).toBeLessThanOrEqual(80)
      if (v.sex === 'M') mCount++
    }
    // ~10% M, allow slack
    expect(mCount / 4000).toBeGreaterThan(0.05)
    expect(mCount / 4000).toBeLessThan(0.15)
  })
})

describe('sampleMarginalGroup', () => {
  it('respects group_proportions', () => {
    const gating: GatingSpec = {
      group_var: 'source_cohort', groups: ['cancer', 'dementia'],
      topic_blocks: [], group_proportions: { cancer: 0.8, dementia: 0.2 },
    }
    const rng = createRng(2)
    let cancer = 0
    for (let i = 0; i < 4000; i++)
      if (sampleMarginalGroup(gating, rng) === 'cancer') cancer++
    expect(cancer / 4000).toBeGreaterThan(0.7)
    expect(cancer / 4000).toBeLessThan(0.9)
  })

  it('falls back to uniform when group_proportions absent', () => {
    const gating: GatingSpec = {
      group_var: 'g', groups: ['a', 'b'], topic_blocks: [],
    }
    const rng = createRng(3)
    const seen = new Set<string>()
    for (let i = 0; i < 50; i++) seen.add(sampleMarginalGroup(gating, rng))
    expect(seen.has('a') && seen.has('b')).toBe(true)
  })
})
