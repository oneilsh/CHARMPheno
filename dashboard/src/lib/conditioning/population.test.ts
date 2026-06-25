import { describe, it, expect } from 'vitest'
import { populationLines } from './population'

describe('populationLines', () => {
  it('summarizes continuous range+median and categorical proportions', () => {
    const schema = {
      k: 20, design_columns: [], unsupported: [],
      controls: [
        { name: 'age', type: 'continuous', range: [41, 68], default: 55 },
        { name: 'sex', type: 'categorical', reference: 'F', levels: ['F', 'M'],
          proportions: { F: 0.52, M: 0.48 } },
      ],
    } as any
    expect(populationLines(schema)).toEqual([
      { name: 'age', summary: '41-68 (med 55)' },
      { name: 'sex', summary: 'F 52% / M 48%' },
    ])
  })

  it('falls back to the level list when proportions are absent', () => {
    const schema = {
      k: 20, design_columns: [], unsupported: [],
      controls: [{ name: 'sex', type: 'categorical', reference: 'F', levels: ['F', 'M'] }],
    } as any
    expect(populationLines(schema)).toEqual([{ name: 'sex', summary: 'F / M' }])
  })

  it('returns [] for an undefined schema', () => {
    expect(populationLines(undefined)).toEqual([])
  })
})
