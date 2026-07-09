import { describe, it, expect } from 'vitest'
import { createRng } from '../sampling'
import { cholesky, mvnDraw, sampleConditionedTheta, buildGenerativeSigma } from './logisticNormal'
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

  it('does not throw when the Sigma sub-block is not positive-definite', () => {
    // 2 free topics with correlation 1.1 in magnitude -> indefinite 2x2 block.
    const corr: Correlation = {
      topic_order: [1, 2], block_labels: ['background', 'background'],
      R: [[1, 1.1], [1.1, 1]], identified: [[true, true], [true, true]],
      support: [[9, 9], [9, 9]], reference_topic: 0,
    }
    const effects: CovariateEffects = [{ covariate: 'Intercept', per_topic: [0, 0, 0] }]
    expect(() => sampleConditionedTheta({
      effects, x: [1], correlation: corr, topicBlocks: null, group: null, rng: createRng(1),
    })).not.toThrow()
  })
})

describe('buildGenerativeSigma eta_scale', () => {
  it('with eta_scale present, Sigma = eta_scale * R (s_k = sqrt(eta_scale) for every free row)', () => {
    const corr: Correlation = {
      topic_order: [1, 2], block_labels: ['background', 'background'],
      R: [[1, 0.5], [0.5, 1]],
      identified: [[true, true], [true, true]],
      support: [[9, 9], [9, 9]],
      reference_topic: 0,
      eta_scale: 4.0,
    }
    const freeIdx = [0, 1]
    const Sigma = buildGenerativeSigma(corr, freeIdx)
    // s_k = sqrt(4.0) = 2 for every row -> Sigma[a][b] = R[a][b] * 4.0
    expect(Sigma[0][0]).toBeCloseTo(1 * 4.0, 10)
    expect(Sigma[1][1]).toBeCloseTo(1 * 4.0, 10)
    expect(Sigma[0][1]).toBeCloseTo(0.5 * 4.0, 10)
    expect(Sigma[1][0]).toBeCloseTo(0.5 * 4.0, 10)
  })

  it('with eta_scale absent, Sigma is byte-identical to R (unit fallback)', () => {
    const corr: Correlation = {
      topic_order: [3, 1, 2], block_labels: ['background', 'background', 'background'],
      R: [
        [1, 0.4, -0.2],
        [0.4, 1, 0.15],
        [-0.2, 0.15, 1],
      ],
      identified: [[true, true, true], [true, true, true], [true, true, true]],
      support: [[9, 9, 9], [9, 9, 9], [9, 9, 9]],
      reference_topic: 0,
    }
    const freeIdx = [0, 1, 2]
    const Sigma = buildGenerativeSigma(corr, freeIdx)
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        expect(Sigma[i][j]).toBe(corr.R[i][j])
  })
})

describe('sampleConditionedTheta eta_scale', () => {
  // A K=3 fixture (reference 0, free topics 1,2 with positive correlation):
  // a large pooled eta_scale should visibly concentrate theta onto fewer
  // topics relative to eta_scale absent (unit fallback).
  function meanTopMass(etaScale: number | undefined, seed: number): number {
    const effects: CovariateEffects = [{ covariate: 'Intercept', per_topic: [0, 0, 0] }]
    const corr: Correlation = {
      topic_order: [1, 2], block_labels: ['background', 'background'],
      R: [[1, 0.3], [0.3, 1]],
      identified: [[true, true], [true, true]],
      support: [[9, 9], [9, 9]],
      reference_topic: 0,
      eta_scale: etaScale,
    }
    const rng = createRng(seed)
    const N = 3000
    let sumTop = 0
    for (let i = 0; i < N; i++) {
      const theta = sampleConditionedTheta({
        effects, x: [1], correlation: corr, topicBlocks: null, group: null,
        rng,
      })
      sumTop += Math.max(...theta)
    }
    return sumTop / N
  }

  it('a large pooled eta_scale produces more concentrated draws than eta_scale absent', () => {
    const noScale = meanTopMass(undefined, 101)
    const largeScale = meanTopMass(10, 101)
    expect(largeScale).toBeGreaterThan(noScale)
    expect(largeScale).toBeGreaterThan(0.6)
  })
})
