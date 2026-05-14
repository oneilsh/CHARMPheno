import { describe, it, expect } from 'vitest'
import { runSimulator, quantiles } from './runSamples'

describe('runSimulator', () => {
  it('returns N theta vectors on the simplex', () => {
    const alpha = [0.1, 0.1, 0.1]
    const beta = [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]]
    const out = runSimulator({ alpha, beta, meanCodesPerDoc: 5, prefix: [], nSamples: 20, seed: 0 })
    expect(out.thetaSamples.length).toBe(20)
    for (const t of out.thetaSamples) expect(t.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5)
  })
  it('prefix on code 0 biases theta toward topic 0', () => {
    const alpha = [0.1, 0.1, 0.1]
    const beta = [[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]]
    const out = runSimulator({ alpha, beta, meanCodesPerDoc: 1, prefix: Array(20).fill(0), nSamples: 50, seed: 1 })
    const meanT0 = out.thetaSamples.reduce((a, t) => a + t[0], 0) / out.thetaSamples.length
    expect(meanT0).toBeGreaterThan(0.7)
  })
})

describe('quantiles', () => {
  it('matches linear-interpolation', () => {
    expect(quantiles([1, 2, 3, 4, 5], [0, 0.5, 1])).toEqual([1, 3, 5])
  })
})
