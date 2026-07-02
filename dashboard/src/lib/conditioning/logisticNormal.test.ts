import { describe, it, expect } from 'vitest'
import { createRng } from '../sampling'
import { cholesky, mvnDraw } from './logisticNormal'

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
