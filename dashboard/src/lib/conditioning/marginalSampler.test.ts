import { describe, it, expect } from 'vitest'
import { createRng } from '../sampling'
import {
  sampleMarginalCovariates, sampleMarginalGroup, resolveGroup, ALL_SUBCOHORTS,
} from './marginalSampler'
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
    const seen = new Set<string | null>()
    for (let i = 0; i < 50; i++) seen.add(sampleMarginalGroup(gating, rng))
    expect(seen.has('a') && seen.has('b')).toBe(true)
    expect(seen.has(null)).toBe(false)   // no background_only_proportion -> never null
  })

  it('draws background-only (null) at background_only_proportion', () => {
    const gating: GatingSpec = {
      group_var: 'source_cohort', groups: ['cancer'], topic_blocks: [],
      group_proportions: { cancer: 0.15 },
      background_only_proportion: 0.85,
    }
    const rng = createRng(4)
    let nulls = 0
    let cancer = 0
    for (let i = 0; i < 4000; i++) {
      const g = sampleMarginalGroup(gating, rng)
      if (g === null) nulls++
      else if (g === 'cancer') cancer++
    }
    expect(nulls / 4000).toBeGreaterThan(0.80)
    expect(nulls / 4000).toBeLessThan(0.90)
    expect(cancer / 4000).toBeGreaterThan(0.10)
    expect(cancer / 4000).toBeLessThan(0.20)
  })
})

describe('resolveGroup', () => {
  const gating: GatingSpec = {
    group_var: 'source_cohort', groups: ['cancer', 'dementia'],
    topic_blocks: [], group_proportions: { cancer: 0.8, dementia: 0.2 },
  }

  it('passes a concrete group through unchanged', () => {
    const rng = createRng(1)
    expect(resolveGroup('cancer', gating, rng)).toBe('cancer')
  })

  it('passes null (background-only) through unchanged', () => {
    const rng = createRng(1)
    expect(resolveGroup(null, gating, rng)).toBe(null)
  })

  it('draws per-call from the population mix for the ALL_SUBCOHORTS sentinel', () => {
    const rng = createRng(2)
    const seen = new Map<string | null, number>()
    for (let i = 0; i < 4000; i++) {
      const g = resolveGroup(ALL_SUBCOHORTS, gating, rng)
      // Only real groups (or null) ever come back — never the sentinel itself.
      expect(g === null || g === 'cancer' || g === 'dementia').toBe(true)
      seen.set(g, (seen.get(g) ?? 0) + 1)
    }
    // Both subcohorts are represented, in proportion (cancer dominant).
    expect((seen.get('cancer') ?? 0)).toBeGreaterThan(seen.get('dementia') ?? 0)
    expect((seen.get('dementia') ?? 0)).toBeGreaterThan(0)
  })

  it('resolves the sentinel to null when there is no gating', () => {
    const rng = createRng(3)
    expect(resolveGroup(ALL_SUBCOHORTS, null, rng)).toBe(null)
    expect(resolveGroup(ALL_SUBCOHORTS, undefined, rng)).toBe(null)
  })
})
