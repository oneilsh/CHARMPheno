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

describe('runSimulator codeTopicCounts (generating-topic attribution)', () => {
  it('attributes every generated code to the topic that emitted it', () => {
    // Disjoint emissions: topic 0 emits only code 0, topic 1 only code 1.
    // Whatever topic generated a token, its code is that topic's code — so
    // codeTopicCounts[code c] must have ALL its mass on topic c.
    const beta = [[1.0, 0.0], [0.0, 1.0]]
    const res = runSimulator({
      alpha: [1, 1], beta, meanCodesPerDoc: 30, prefix: [], nSamples: 40, seed: 3,
    })
    for (const [w, topics] of res.codeTopicCounts) {
      for (const [z] of topics) expect(z).toBe(w) // code w only ever from topic w
    }
    expect(res.codeTopicCounts.size).toBeGreaterThan(0)
  })

  it('dominant generating topic of a code is its argmax over the topic counts', () => {
    // topic 0 emits code 0 (0.9) or 1 (0.1); topic 1 emits code 1 (0.9) or 0 (0.1).
    const beta = [[0.9, 0.1], [0.1, 0.9]]
    const res = runSimulator({
      alpha: [1, 1], beta, meanCodesPerDoc: 40, prefix: [], nSamples: 60, seed: 5,
    })
    const domTopic = (w: number) => {
      const tm = res.codeTopicCounts.get(w)!
      let best = -1, bestC = -1
      for (const [z, c] of tm) if (c > bestC) { bestC = c; best = z }
      return best
    }
    expect(domTopic(0)).toBe(0) // code 0 mostly from topic 0
    expect(domTopic(1)).toBe(1) // code 1 mostly from topic 1
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

  it('reports the conditioned draw directly even with a non-empty prefix (no re-diffusion)', () => {
    // conditionedTheta already incorporates the prefix (it's the posterior
    // draw from sampleRecordPosterior); re-inferring it via the Dirichlet
    // E-step over prefix+generated codes would re-diffuse it toward the
    // codes' own topic association, which is exactly the rainbow bug. With
    // a prefix present, the reported theta for every sample must equal the
    // fixed conditioned vector exactly.
    const alpha = [0.1, 0.1]
    const beta = [[0.9, 0.1], [0.1, 0.9]]
    const fixed = [0.05, 0.95]
    const res = runSimulator({
      alpha, beta, meanCodesPerDoc: 5, prefix: [0, 0, 0, 0, 0],
      nSamples: 5, seed: 3,
      conditionedTheta: () => fixed,
    })
    for (const t of res.thetaSamples) {
      expect(t).toEqual(fixed)
    }
  })
})
