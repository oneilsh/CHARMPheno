import { describe, it, expect } from 'vitest'
import {
  digamma, variationalEStep, relevance, topRelevantCodes, jsd,
  phenotypesContainingCode,
} from './inference'

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

describe('phenotypesContainingCode', () => {
  // K=3 topics, V=4 codes. Code 3 is rare in the corpus but concentrated in
  // topic 2. With λ=0.6 (the default) it still ranks high in topic 2 but
  // very low in topic 0 where it has tiny weight.
  const beta = [
    [0.6, 0.3, 0.05, 0.05],  // topic 0: dominated by codes 0,1; touches 2,3
    [0.4, 0.4, 0.1, 0.1],    // topic 1: codes 0,1 dominant; some 2,3
    [0.2, 0.2, 0.2, 0.4],    // topic 2: code 3 leads
  ]
  const corpusFreq = [0.4, 0.4, 0.15, 0.05]

  it('includes a topic when the code is in its top-N relevance', () => {
    const containing = phenotypesContainingCode({
      beta, corpusFreq, codeIdx: 3, n: 2, lambda: 0.6,
    })
    // Topic 2's top-2 must include code 3 (highest weight).
    expect(containing.has(2)).toBe(true)
  })

  it('excludes a topic where the code falls outside the top-N', () => {
    // For topic 0 at top-N=2, the top-2 by relevance should be codes 0 and 1.
    const containing = phenotypesContainingCode({
      beta, corpusFreq, codeIdx: 3, n: 2, lambda: 0.6,
    })
    expect(containing.has(0)).toBe(false)
  })

  it('returns an empty set for null code index', () => {
    expect(
      phenotypesContainingCode({ beta, corpusFreq, codeIdx: -1, n: 5, lambda: 0.6 }).size,
    ).toBe(0)
  })
})

describe('jsd', () => {
  it('zero on identical', () => { expect(jsd([0.5, 0.5], [0.5, 0.5])).toBeCloseTo(0, 9) })
  it('symmetric', () => {
    const p = [0.7, 0.2, 0.1], q = [0.1, 0.2, 0.7]
    expect(jsd(p, q)).toBeCloseTo(jsd(q, p), 9)
  })
  it('bounded by log 2', () => {
    expect(jsd([1, 0], [0, 1])).toBeLessThanOrEqual(Math.log(2) + 1e-9)
  })
})
