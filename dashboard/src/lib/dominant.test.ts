import { describe, it, expect } from 'vitest'
import { dominant, displayedDominant, dominantVote } from './dominant'
import type { Phenotype } from './types'

const ph = (id: number, quality: Phenotype['quality'] = 'phenotype'): Phenotype =>
  ({ id, quality } as Phenotype)

describe('dominant', () => {
  it('returns the argmax index', () => {
    expect(dominant([0.1, 0.7, 0.2])).toBe(1)
  })
})

describe('dominantVote', () => {
  const phenos = [ph(0), ph(1), ph(2)]

  it('tallies each draws dominant phenotype into a distribution summing to 1', () => {
    const draws = [
      [0.6, 0.3, 0.1], // dominant 0
      [0.5, 0.4, 0.1], // dominant 0
      [0.1, 0.8, 0.1], // dominant 1
      [0.2, 0.1, 0.7], // dominant 2
    ]
    const vote = dominantVote(draws, phenos, true)
    expect(vote).toEqual([0.5, 0.25, 0.25]) // 2/4, 1/4, 1/4
    expect(vote.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
  })

  it('concentrates when draws agree, spreads when they disagree', () => {
    const agree = Array.from({ length: 10 }, () => [0.6, 0.3, 0.1])
    expect(dominantVote(agree, phenos, true)).toEqual([1, 0, 0])
    const disagree = [[0.6, 0.3, 0.1], [0.1, 0.6, 0.3], [0.3, 0.1, 0.6]]
    const v = dominantVote(disagree, phenos, true)
    expect(Math.max(...v)).toBeLessThan(0.5) // no runaway winner
  })

  it('in basic mode votes for the displayed (non dead/mixed) dominant', () => {
    const phen = [ph(0, 'dead'), ph(1), ph(2)]
    // Draw is dominated by the DEAD topic 0; basic mode should vote for topic 1.
    const vote = dominantVote([[0.7, 0.2, 0.1]], phen, false)
    expect(vote[0]).toBe(0)
    expect(vote[1]).toBe(1)
  })

  it('returns all-zero for an empty sample set', () => {
    expect(dominantVote([], phenos, true)).toEqual([0, 0, 0])
  })
})

// Sanity that displayedDominant still behaves (used by dominantVote).
describe('displayedDominant', () => {
  it('skips dead/mixed in basic mode', () => {
    expect(displayedDominant([0.7, 0.3], [ph(0, 'dead'), ph(1)], false)).toBe(1)
    expect(displayedDominant([0.7, 0.3], [ph(0, 'dead'), ph(1)], true)).toBe(0)
  })
})
