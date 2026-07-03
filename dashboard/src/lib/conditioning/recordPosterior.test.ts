import { describe, it, expect } from 'vitest'
import { createRng } from '../sampling'
import { sampleConditionedTheta } from './logisticNormal'
import { sampleRecordPosterior } from './recordPosterior'
import type { Correlation, CovariateEffects } from '../types'

function identityCorr(order: number[]): Correlation {
  const K1 = order.length
  const R = Array.from({ length: K1 }, (_, i) => Array.from({ length: K1 }, (_, j) => (i === j ? 1 : 0)))
  return { topic_order: order, block_labels: order.map(() => 'background'),
    R, identified: R.map((r) => r.map(() => true)), support: R.map((r) => r.map(() => 9)),
    reference_topic: 0 }
}
const beta3 = [[0.34, 0.33, 0.33], [0.9, 0.05, 0.05], [0.05, 0.9, 0.05]]  // K=3, V=3

describe('sampleRecordPosterior', () => {
  it('empty prefix reduces exactly to the prior draw', () => {
    const effects: CovariateEffects = [{ covariate: 'Intercept', per_topic: [0, 0.3, -0.2] }]
    const corr = identityCorr([1, 2])
    const args = { effects, x: [1], correlation: corr, topicBlocks: null, group: null }
    const prior = sampleConditionedTheta({ ...args, rng: createRng(42) })
    const post = sampleRecordPosterior({ ...args, prefixCounts: new Map(), beta: beta3, rng: createRng(42) })
    expect(post).toEqual(prior)
  })

  it('a prefix loading topic 1 concentrates the posterior on topic 1', () => {
    const effects: CovariateEffects = [{ covariate: 'Intercept', per_topic: [0, 0, 0] }]
    const corr = identityCorr([1, 2])
    // term 0 is emitted almost only by topic 1 (beta3[1][0]=0.9)
    const rng = createRng(7)
    let s1 = 0, s2 = 0
    for (let i = 0; i < 400; i++) {
      const t = sampleRecordPosterior({ effects, x: [1], correlation: corr,
        topicBlocks: null, group: null, prefixCounts: new Map([[0, 15]]), beta: beta3, rng })
      s1 += t[1]; s2 += t[2]
    }
    expect(s1).toBeGreaterThan(s2 * 3)   // strongly concentrated on topic 1
  })

  it('keeps out-of-group foreground topics at exactly 0', () => {
    const effects: CovariateEffects = [{ covariate: 'Intercept', per_topic: [0, 0, 0, 0] }]
    const corr = identityCorr([1, 2, 3])
    const beta4 = [[0.5, 0.5], [0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]
    const t = sampleRecordPosterior({
      effects, x: [1], correlation: corr,
      topicBlocks: ['background', 'background', 'cancer', 'dementia'],
      group: 'cancer', prefixCounts: new Map([[0, 5]]), beta: beta4, rng: createRng(3),
    })
    expect(t[3]).toBe(0)                 // dementia masked
    expect(t.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
  })
})
