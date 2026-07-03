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

  it('a heavy prefix on a K=20-free-topic fixture still concentrates on the loaded topic (line-search regression)', () => {
    // Regression for the backtracking line search added to the Fisher-scoring
    // mode-finder. On this fixture the undamped full-Newton step provably
    // oscillates in a period-2 limit cycle (verified by instrumenting the
    // pre-fix loop's log-posterior objective: it alternates between roughly
    // -605 and -149 for the full 50-iteration cap and never settles), landing
    // on an essentially uniform theta (topic-1 mass ~= 0.049, i.e. ~1/20 -
    // the prefix has almost no effect despite count=200). With the line
    // search the iteration converges in ~8 steps to topic-1 mass ~= 0.97.
    // This test would FAIL against the pre-fix code (topic-1 mass would be
    // far below the K=20 uniform-plus-margin bar asserted here) and PASSES
    // with the line search.
    const K = 20
    const order = Array.from({ length: K }, (_, i) => i + 1) // K free topics, ref = 0
    const corr = identityCorr(order)
    const effects: CovariateEffects = [{ covariate: 'Intercept', per_topic: new Array(K + 1).fill(0) }]
    // V = K+1 codes; code 0 is emitted almost entirely by topic 1 (order[0]).
    const V = K + 1
    const beta = Array.from({ length: K + 1 }, (_, k) => {
      const row = new Array(V).fill(0.01 / (V - 1))
      row[0] = k === 1 ? 0.99 - 0.01 * (V - 2) : 0.01
      return row
    })
    const rng = createRng(11)
    let s1 = 0
    const trials = 20
    for (let i = 0; i < trials; i++) {
      const t = sampleRecordPosterior({
        effects, x: [1], correlation: corr, topicBlocks: null, group: null,
        prefixCounts: new Map([[0, 200]]), beta, rng,
      })
      expect(t.every((v) => Number.isFinite(v))).toBe(true)
      s1 += t[1]
    }
    const avgTop1 = s1 / trials
    // Uniform-over-21-allowed-topics would give ~0.048; the undamped mode
    // finder's limit cycle lands right around there. The damped fit should
    // concentrate the vast majority of mass on topic 1.
    expect(avgTop1).toBeGreaterThan(0.5)
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
