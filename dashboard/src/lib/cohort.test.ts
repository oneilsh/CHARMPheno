import { describe, it, expect } from 'vitest'
import { generateCohort } from './cohort'
import type { Model } from './types'

const model: Model = {
  K: 3, V: 5,
  alpha: [0.1, 0.1, 0.1],
  beta: [
    [0.9, 0.025, 0.025, 0.025, 0.025],
    [0.025, 0.9, 0.025, 0.025, 0.025],
    [0.025, 0.025, 0.9, 0.025, 0.025],
  ],
}

describe('generateCohort', () => {
  it('deterministic given seed', () => {
    const a = generateCohort({ model, meanCodesPerDoc: 8, n: 10, seed: 42, nNeighbors: 3 })
    const b = generateCohort({ model, meanCodesPerDoc: 8, n: 10, seed: 42, nNeighbors: 3 })
    expect(a.patients.map((p) => p.code_bag)).toEqual(b.patients.map((p) => p.code_bag))
  })

  it('produces patients on the simplex', () => {
    const c = generateCohort({ model, meanCodesPerDoc: 5, n: 12, seed: 1, nNeighbors: 3 })
    expect(c.patients.length).toBe(12)
    for (const p of c.patients) {
      expect(p.theta.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
      expect(p.code_bag.length).toBeGreaterThan(0)
      expect(p.neighbors.length).toBe(3)
      expect(p.neighbors.includes(p.id)).toBe(false)
      expect(new Set(p.neighbors).size).toBe(3)
    }
  })

  it('zero-pads patient ids', () => {
    const c = generateCohort({ model, meanCodesPerDoc: 5, n: 5, seed: 1, nNeighbors: 2 })
    expect(c.patients[0].id).toBe('synth_0000')
    expect(c.patients[4].id).toBe('synth_0004')
  })
})
