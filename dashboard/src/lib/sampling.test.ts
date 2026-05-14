import { describe, it, expect } from 'vitest'
import { createRng, sampleDirichlet, sampleCategorical, samplePoisson } from './sampling'

describe('createRng', () => {
  it('deterministic given seed', () => {
    const a = createRng(42), b = createRng(42)
    expect([a(), a(), a()]).toEqual([b(), b(), b()])
  })
})

describe('sampleDirichlet', () => {
  it('row on simplex', () => {
    const x = sampleDirichlet([1, 1, 1], createRng(1))
    expect(x.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
    expect(x.every((v) => v >= 0)).toBe(true)
  })
})

describe('sampleCategorical', () => {
  it('approx matches distribution', () => {
    const rng = createRng(3); const counts = [0, 0, 0]
    const p = [0.2, 0.3, 0.5]
    for (let i = 0; i < 20000; i++) counts[sampleCategorical(p, rng)]++
    expect(counts[0] / 20000).toBeCloseTo(0.2, 1)
    expect(counts[2] / 20000).toBeCloseTo(0.5, 1)
  })
})

describe('samplePoisson', () => {
  it('mean ≈ λ', () => {
    const rng = createRng(4); let s = 0
    for (let i = 0; i < 5000; i++) s += samplePoisson(5, rng)
    expect(s / 5000).toBeCloseTo(5, 0)
  })
})
