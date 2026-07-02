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

describe('runSimulator conditioned θ', () => {
  it('uses the injected conditionedTheta for the no-prefix draw', () => {
    // A conditionedTheta that always puts all mass on topic 1 -> generated
    // codes come only from beta[1]; the reported theta concentrates on 1.
    const beta = [[0.5, 0.5], [0.0, 1.0]]   // topic 1 emits code 1 only
    const res = runSimulator({
      alpha: [1, 1], beta, meanCodesPerDoc: 20, prefix: [],
      nSamples: 5, seed: 1,
      conditionedTheta: () => [0, 1],
    })
    // All sampled codes should be code index 1.
    for (const bag of res.codeCountsSamples) {
      for (const [w] of bag) expect(w).toBe(1)
    }
  })

  it('without conditionedTheta behaves as before (Dirichlet path)', () => {
    const beta = [[0.5, 0.5], [0.5, 0.5]]
    const res = runSimulator({
      alpha: [1, 1], beta, meanCodesPerDoc: 10, prefix: [], nSamples: 3, seed: 1,
    })
    expect(res.thetaSamples.length).toBe(3)
  })
})
