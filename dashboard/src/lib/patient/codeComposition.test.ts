import { describe, it, expect } from 'vitest'
import { codeComposition, OTHER_ID } from './codeComposition'

// K=3, V=3. Topic 0 emits code 0, topic 1 emits code 1, topic 2 emits code 2.
const beta = [
  [0.8, 0.1, 0.1],
  [0.1, 0.8, 0.1],
  [0.1, 0.1, 0.8],
]

describe('codeComposition', () => {
  it('each code row segments sum to 1', () => {
    const theta = [0.5, 0.4, 0.1]
    const rows = codeComposition(theta, [0, 0, 1], beta, 3, 0.05)
    for (const row of rows) {
      const s = row.segments.reduce((a, b) => a + b.weight, 0)
      expect(s).toBeCloseTo(1, 10)
    }
  })

  it('counts repeated codes', () => {
    const rows = codeComposition([0.5, 0.4, 0.1], [0, 0, 1], beta, 3, 0.05)
    expect(rows.find((r) => r.w === 0)!.count).toBe(2)
    expect(rows.find((r) => r.w === 1)!.count).toBe(1)
  })

  it('buckets tail phenotypes (theta < threshold) into OTHER_ID', () => {
    // theta[2] = 0.02 < 0.05, so any weight on topic 2 goes to Other.
    const rows = codeComposition([0.5, 0.48, 0.02], [2], beta, 3, 0.05)
    const seg2 = rows[0].segments.find((s) => s.k === 2)
    expect(seg2).toBeUndefined()
    expect(rows[0].segments.some((s) => s.k === OTHER_ID)).toBe(true)
  })

  it('emits empty segments for a code no expressed topic can generate (z=0)', () => {
    const zeroBeta = [[0, 1], [0, 1]] // neither topic emits code 0
    const rows = codeComposition([0.5, 0.5], [0], zeroBeta, 2, 0.05)
    expect(rows[0].segments).toEqual([])
  })
})
