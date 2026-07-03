import { describe, it, expect } from 'vitest'
import { topDifferentialCodes } from './difference'

describe('topDifferentialCodes', () => {
  // 4 terms. A loads term 0 heavily, B loads term 3 heavily.
  const betaA = [0.7, 0.2, 0.09, 0.01]
  const betaB = [0.01, 0.2, 0.09, 0.7]
  const pw = [0.25, 0.25, 0.25, 0.25]

  it('ranks A-distinctive terms on aSide and B-distinctive on bSide', () => {
    const { aSide, bSide } = topDifferentialCodes({ betaA, betaB, pw, lambda: 0.6, n: 2 })
    expect(aSide[0].index).toBe(0)   // term 0 most elevated in A
    expect(bSide[0].index).toBe(3)   // term 3 most elevated in B
  })

  it('is antisymmetric: swapping A/B negates deltas and swaps the sides', () => {
    const ab = topDifferentialCodes({ betaA, betaB, pw, lambda: 0.6, n: 4 })
    const ba = topDifferentialCodes({ betaA: betaB, betaB: betaA, pw, lambda: 0.6, n: 4 })
    expect(ba.aSide[0].index).toBe(ab.bSide[0].index)
    expect(ba.aSide[0].delta).toBeCloseTo(-ab.bSide[0].delta, 10)
  })

  it('excludes a term from the side where its beta is zero (relevance -Infinity)', () => {
    const bA = [0.5, 0.5, 0.0]   // term 2 absent from A
    const bB = [0.4, 0.3, 0.3]
    const { aSide } = topDifferentialCodes({ betaA: bA, betaB: bB, pw: [0.33, 0.33, 0.34], lambda: 0.6, n: 3 })
    expect(aSide.some((r) => r.index === 2)).toBe(false)   // -Inf delta not on A side
  })
})
