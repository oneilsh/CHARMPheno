import { describe, it, expect } from 'vitest'
import { codeComposition, OTHER_ID, explainedVsPrior, sortRowsForSelection } from './codeComposition'

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

describe('explainedVsPrior', () => {
  const beta = [
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
  ]

  it('explained + prior equals theta per phenotype', () => {
    const theta = [0.5, 0.4, 0.1]
    const { explained, prior } = explainedVsPrior(theta, [0, 1, 2], beta, 3)
    for (let k = 0; k < 3; k++) {
      expect(explained[k] + prior[k]).toBeCloseTo(theta[k], 10)
    }
  })

  it('clamps when codes over-explain a phenotype: prior=0, explained=theta', () => {
    // Only code 0 present → code evidence concentrates on topic 0, over-explaining it
    // relative to theta[0], while theta[1] gets no code support.
    const theta = [0.34, 0.33, 0.33]
    const { explained, prior } = explainedVsPrior(theta, [0], beta, 3)
    expect(prior[0]).toBe(0)
    expect(explained[0]).toBeCloseTo(theta[0], 10)
    expect(prior[1]).toBeGreaterThan(0) // no code speaks to topic 1 → prior-supported
  })

  it('returns zero explained when codeBag is empty', () => {
    const { explained, prior } = explainedVsPrior([0.5, 0.5, 0], [], beta, 3)
    expect(explained).toEqual([0, 0, 0])
    expect(prior).toEqual([0.5, 0.5, 0])
  })
})

describe('sortRowsForSelection', () => {
  const beta = [
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
  ]

  it('null selection sorts by occurrence count desc', () => {
    const rows = codeComposition([0.4, 0.4, 0.2], [0, 1, 1], beta, 3, 0.05)
    const sorted = sortRowsForSelection(rows, null)
    expect(sorted[0].count).toBeGreaterThanOrEqual(sorted[1].count)
    expect(sorted[0].w).toBe(1) // appears twice
  })

  it('a phenotype selection sorts by that phenotype weight desc', () => {
    const rows = codeComposition([0.4, 0.4, 0.2], [0, 1], beta, 3, 0.05)
    const sorted = sortRowsForSelection(rows, 1)
    // code 1 loads mostly on topic 1, so it ranks first when topic 1 is selected
    expect(sorted[0].w).toBe(1)
  })

  it('does not mutate the input array', () => {
    const rows = codeComposition([0.5, 0.5, 0], [0, 1], beta, 3, 0.05)
    const before = rows.map((r) => r.w)
    sortRowsForSelection(rows, 0)
    expect(rows.map((r) => r.w)).toEqual(before)
  })
})
