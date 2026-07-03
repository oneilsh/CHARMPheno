import { describe, it, expect } from 'vitest'
import { choleskyPD, solveSPD, invSPD } from './linalg'

describe('choleskyPD', () => {
  it('factors a PD matrix (L Lᵀ = A)', () => {
    const A = [[4, 2], [2, 3]]
    const L = choleskyPD(A)
    expect(L[0][0] * L[0][0]).toBeCloseTo(4, 9)
    expect(L[1][0] * L[0][0]).toBeCloseTo(2, 9)
    expect(L[1][0] ** 2 + L[1][1] ** 2).toBeCloseTo(3, 9)
  })
  it('regularizes a non-PD matrix instead of throwing', () => {
    const indefinite = [[1, 2], [2, 1]]   // eigenvalues 3, -1
    expect(() => choleskyPD(indefinite)).not.toThrow()
    const L = choleskyPD(indefinite)
    expect(Number.isFinite(L[0][0])).toBe(true)
  })
})

describe('solveSPD / invSPD', () => {
  it('solves A x = b', () => {
    const A = [[4, 1], [1, 3]]
    const x = solveSPD(A, [1, 2])
    // A x should be ~ [1,2]
    expect(A[0][0] * x[0] + A[0][1] * x[1]).toBeCloseTo(1, 8)
    expect(A[1][0] * x[0] + A[1][1] * x[1]).toBeCloseTo(2, 8)
  })
  it('inverts A (A · A⁻¹ = I)', () => {
    const A = [[4, 1], [1, 3]]
    const Ai = invSPD(A)
    const p00 = A[0][0] * Ai[0][0] + A[0][1] * Ai[1][0]
    const p01 = A[0][0] * Ai[0][1] + A[0][1] * Ai[1][1]
    expect(p00).toBeCloseTo(1, 8)
    expect(p01).toBeCloseTo(0, 8)
  })
})
