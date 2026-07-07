import { describe, it, expect } from 'vitest'
import { cohortCoverage, sampleThetaCohort } from './coverage'
import { makeStmBundleFixture } from '../test-fixtures'

describe('cohortCoverage', () => {
  it('counts the fraction of patients with theta_k strictly greater than tau', () => {
    const thetas = [
      [0.5, 0.01, 0.0],
      [0.3, 0.03, 0.0],
    ]
    // tau=0.02: topic0 -> both>0.02 (1.0); topic1 -> only 0.03>0.02 (0.5);
    // topic2 -> none (0.0). Strict >: 0.02 itself would NOT count.
    expect(cohortCoverage(thetas, 0.02, 3)).toEqual([1.0, 0.5, 0.0])
  })

  it('treats theta_k == tau as NOT covered (strict inequality)', () => {
    expect(cohortCoverage([[0.02, 0.0, 0.0]], 0.02, 3)).toEqual([0.0, 0.0, 0.0])
  })

  it('returns all-zero coverage for an empty cohort', () => {
    expect(cohortCoverage([], 0.01, 4)).toEqual([0, 0, 0, 0])
  })
})

describe('sampleThetaCohort', () => {
  const base = { values: {}, n: 200, seed: 20260706 }

  it('returns n theta vectors of length K that sum to ~1 (softmax over free rows)', () => {
    const bundle = makeStmBundleFixture()
    const thetas = sampleThetaCohort({ bundle, active: false, ...base })
    expect(thetas).toHaveLength(200)
    for (const t of thetas) {
      expect(t).toHaveLength(3)
      expect(t.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
    }
  })

  it('is deterministic for a fixed seed', () => {
    const bundle = makeStmBundleFixture()
    const a = sampleThetaCohort({ bundle, active: false, ...base })
    const b = sampleThetaCohort({ bundle, active: false, ...base })
    expect(a).toEqual(b)
  })

  it('with active covariates fixed at high age, coverage shifts to the age-loaded topic', () => {
    // Fixture Gamma: topic1 eta = 1.0 - 0.02*age, topic2 eta = 0.5 + 0.03*age.
    // At age=100 topic2 dominates topic1, so its coverage is higher.
    const bundle = makeStmBundleFixture()
    const thetas = sampleThetaCohort({ bundle, active: true, values: { age: 100 }, n: 1500, seed: 20260706 })
    const cov = cohortCoverage(thetas, 0.01, 3)
    expect(cov[2]).toBeGreaterThan(cov[1])
  })
})
