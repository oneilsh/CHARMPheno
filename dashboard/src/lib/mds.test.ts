import { describe, it, expect } from 'vitest'
import { classicalMds, computeJsdMds } from './mds'

describe('classicalMds', () => {
  it('recovers relative distances for simple points', () => {
    // Triangle: points at (0,0), (1,0), (0,1)
    const pts = [[0, 0], [1, 0], [0, 1]]
    const D: number[][] = pts.map((a) => pts.map((b) => Math.hypot(a[0] - b[0], a[1] - b[1])))
    const coords = classicalMds(D, 3)
    // Check that distance ratios are preserved
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      const d = Math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
      // Allow for scaling and rounding
      if (D[i][j] > 0.01) expect(d).toBeCloseTo(D[i][j], 1)
    }
  })
})

describe('computeJsdMds', () => {
  it('returns one (x, y) per topic', () => {
    const beta = [
      [0.5, 0.5, 0, 0],
      [0, 0, 0.5, 0.5],
      [0.25, 0.25, 0.25, 0.25],
    ]
    const coords = computeJsdMds(beta)
    expect(coords.length).toBe(3)
    expect(coords[0].length).toBe(2)
  })
})
