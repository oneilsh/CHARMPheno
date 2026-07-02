import { describe, it, expect } from 'vitest'
import { createRng } from '../sampling'
import { cholesky, mvnDraw, sampleConditionedTheta } from './logisticNormal'
import type { Correlation, CovariateEffects } from '../types'

describe('cholesky', () => {
  it('reconstructs the matrix: L Lᵀ = A', () => {
    const A = [[4, 2, 0], [2, 5, 1], [0, 1, 3]]
    const L = cholesky(A)
    const K = A.length
    for (let i = 0; i < K; i++)
      for (let j = 0; j < K; j++) {
        let s = 0
        for (let k = 0; k < K; k++) s += L[i][k] * L[j][k]
        expect(s).toBeCloseTo(A[i][j], 10)
      }
    // lower-triangular
    expect(L[0][1]).toBe(0)
    expect(L[0][2]).toBe(0)
    expect(L[1][2]).toBe(0)
  })

  it('throws on a non-positive-definite matrix', () => {
    expect(() => cholesky([[1, 2], [2, 1]])).toThrow()
  })
})

describe('mvnDraw', () => {
  it('sample mean and covariance converge to (mean, Sigma)', () => {
    const mean = [1, -2]
    const Sigma = [[2, 0.8], [0.8, 1]]
    const L = cholesky(Sigma)
    const rng = createRng(123)
    const N = 40000
    const draws: number[][] = []
    for (let i = 0; i < N; i++) draws.push(mvnDraw(mean, L, rng))
    const m = [0, 0]
    for (const d of draws) { m[0] += d[0]; m[1] += d[1] }
    m[0] /= N; m[1] /= N
    expect(m[0]).toBeCloseTo(mean[0], 1)
    expect(m[1]).toBeCloseTo(mean[1], 1)
    let c00 = 0, c01 = 0, c11 = 0
    for (const d of draws) {
      c00 += (d[0] - m[0]) ** 2
      c01 += (d[0] - m[0]) * (d[1] - m[1])
      c11 += (d[1] - m[1]) ** 2
    }
    expect(c00 / N).toBeCloseTo(Sigma[0][0], 1)
    expect(c01 / N).toBeCloseTo(Sigma[0][1], 1)
    expect(c11 / N).toBeCloseTo(Sigma[1][1], 1)
  })

  it('is deterministic under a seeded RNG', () => {
    const L = cholesky([[1, 0], [0, 1]])
    const a = mvnDraw([0, 0], L, createRng(7))
    const b = mvnDraw([0, 0], L, createRng(7))
    expect(a).toEqual(b)
  })
})

function identityCorr(K1: number, order: number[]): Correlation {
  const R = Array.from({ length: K1 }, (_, i) =>
    Array.from({ length: K1 }, (_, j) => (i === j ? 1 : 0)))
  return {
    topic_order: order,
    block_labels: order.map(() => 'background'),
    R,
    identified: R.map((row) => row.map(() => true)),
    support: R.map((row) => row.map(() => 9)),
    reference_topic: 0,
  }
}

describe('sampleConditionedTheta', () => {
  it('returns a length-K distribution with reference topic drawn around eta=0', () => {
    // K=3: reference topic 0, free topics 1..2. Effects zero -> mean eta = 0.
    const effects: CovariateEffects = [
      { covariate: 'Intercept', per_topic: [0, 0, 0] },
    ]
    const corr = identityCorr(2, [1, 2])
    const theta = sampleConditionedTheta({
      effects, x: [1], correlation: corr,
      topicBlocks: null, group: null, rng: createRng(3),
    })
    expect(theta.length).toBe(3)
    const sum = theta.reduce((a, b) => a + b, 0)
    expect(sum).toBeCloseTo(1, 10)
    for (const p of theta) expect(p).toBeGreaterThan(0)
  })

  it('gives out-of-group foreground topics exactly zero mass', () => {
    // K=4: topic 0 reference(bg), 1 bg, 2 cancer, 3 dementia. Select cancer.
    const effects: CovariateEffects = [
      { covariate: 'Intercept', per_topic: [0, 0, 0, 0] },
    ]
    const corr = identityCorr(3, [1, 2, 3])
    const theta = sampleConditionedTheta({
      effects, x: [1], correlation: corr,
      topicBlocks: ['background', 'background', 'cancer', 'dementia'],
      group: 'cancer', rng: createRng(5),
    })
    expect(theta[3]).toBe(0)          // dementia foreground masked out
    expect(theta[2]).toBeGreaterThan(0) // cancer foreground allowed
    expect(theta.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
  })

  it('shifts the mean when a covariate effect is applied', () => {
    // Effect pushes free topic 2 up; its mean share should exceed topic 1's.
    const effects: CovariateEffects = [
      { covariate: 'Intercept', per_topic: [0, 0, 0] },
      { covariate: 'age', per_topic: [0, 0, 3] },
    ]
    const corr = identityCorr(2, [1, 2])
    const rng = createRng(11)
    let s1 = 0, s2 = 0
    for (let i = 0; i < 2000; i++) {
      const t = sampleConditionedTheta({
        effects, x: [1, 1], correlation: corr,
        topicBlocks: null, group: null, rng,
      })
      s1 += t[1]; s2 += t[2]
    }
    expect(s2).toBeGreaterThan(s1)
  })
})
