import { describe, it, expect } from 'vitest'
import { digamma, variationalEStep, relevance, topRelevantCodes } from './inference'

describe('digamma', () => {
  it('digamma(1) ≈ -0.5772 (Euler-Mascheroni)', () => {
    expect(digamma(1)).toBeCloseTo(-0.5772156649, 4)
  })
  it('digamma(10) ≈ 2.2517', () => {
    expect(digamma(10)).toBeCloseTo(2.251752589, 4)
  })
})

describe('variationalEStep', () => {
  it('empty doc returns prior mean', () => {
    const alpha = [0.5, 0.5]
    const { theta } = variationalEStep({ alpha, beta: [[0.5, 0.5], [0.5, 0.5]], codeCounts: new Map() })
    expect(theta[0]).toBeCloseTo(0.5)
    expect(theta[1]).toBeCloseTo(0.5)
  })

  it('shifts theta toward the topic owning a heavily observed code', () => {
    const alpha = [0.1, 0.1, 0.1]
    const beta = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
    const { theta } = variationalEStep({ alpha, beta, codeCounts: new Map([[0, 10]]) })
    expect(theta[0]).toBeGreaterThan(0.8)
  })

  it('theta sums to 1', () => {
    const alpha = [0.5, 0.5]
    const beta = [[0.3, 0.4, 0.3], [0.5, 0.2, 0.3]]
    const { theta } = variationalEStep({ alpha, beta, codeCounts: new Map([[0, 2], [2, 1]]) })
    expect(theta.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
  })
})

describe('relevance', () => {
  it('λ=1 returns log p(w|k)', () => {
    expect(relevance(0.4, 0.2, 1.0)).toBeCloseTo(Math.log(0.4))
  })
  it('λ=0 returns log lift', () => {
    expect(relevance(0.4, 0.2, 0.0)).toBeCloseTo(Math.log(0.4 / 0.2))
  })
  it('returns -Infinity for zero p(w|k)', () => {
    expect(relevance(0, 0.2, 0.5)).toBe(-Infinity)
  })
})

describe('topRelevantCodes', () => {
  it('orders by p(w|k) at λ=1, by lift at λ=0', () => {
    const pwk = [0.7, 0.2, 0.1]
    const pw  = [0.5, 0.4, 0.05]
    expect(topRelevantCodes({ pwk, pw, lambda: 1.0, n: 3 }).map((r) => r.index)).toEqual([0, 1, 2])
    expect(topRelevantCodes({ pwk, pw, lambda: 0.0, n: 3 })[0].index).toBe(2)
  })
})
