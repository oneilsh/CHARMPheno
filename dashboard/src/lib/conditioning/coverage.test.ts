import { describe, it, expect } from 'vitest'
import { cohortCoverage, sampleThetaCohort, withinCohortCoverage, thetaColumnDistribution } from './coverage'
import { sampleConditionedTheta } from './logisticNormal'
import { sampleMarginalCovariates, sampleMarginalGroup } from './marginalSampler'
import { buildDesignVector } from '../covariate'
import { createRng } from '../sampling'
import { makeStmBundleFixture } from '../test-fixtures'

describe('withinCohortCoverage', () => {
  const blocks = ['background', 'background', 'grp']
  const props = { grp: 0.2 }

  it('is a no-op when the bundle is not gated (null blocks or proportions)', () => {
    expect(withinCohortCoverage([0.1, 0.2, 0.05], null, props)).toEqual([0.1, 0.2, 0.05])
    expect(withinCohortCoverage([0.1, 0.2, 0.05], blocks, null)).toEqual([0.1, 0.2, 0.05])
  })

  it('leaves background topics unchanged and divides foreground topics by their group share', () => {
    // topic 2 is foreground in group 'grp' (share 0.2): whole-cohort 0.03 -> within-cohort 0.15.
    expect(withinCohortCoverage([0.5, 0.3, 0.03], blocks, props)).toEqual([0.5, 0.3, 0.15])
  })

  it('clamps a rescaled foreground coverage to 1', () => {
    // 0.25 / 0.2 = 1.25 -> clamped to 1 (a topic cannot exceed 100% within its cohort).
    expect(withinCohortCoverage([0.5, 0.3, 0.25], blocks, props)).toEqual([0.5, 0.3, 1])
  })

  it('leaves a foreground topic unchanged when its group share is missing/zero', () => {
    expect(withinCohortCoverage([0.5, 0.3, 0.03], blocks, {})).toEqual([0.5, 0.3, 0.03])
    expect(withinCohortCoverage([0.5, 0.3, 0.03], blocks, { grp: 0 })).toEqual([0.5, 0.3, 0.03])
  })
})

describe('thetaColumnDistribution', () => {
  it('bins one phenotype column into fractions over the bin edges (sums to 1)', () => {
    const cohort = [[0.1], [0.2], [0.6], [0.9]]
    const d = thetaColumnDistribution(cohort, 0, [0, 0.5, 1])
    expect(d.n).toBe(4)
    expect(d.histogram).toEqual([0.5, 0.5]) // 2 in [0,0.5), 2 in [0.5,1)
    expect(d.histogram.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
  })

  it('excludes masked (theta==0) out-of-group patients from the distribution', () => {
    // A foreground topic: two patients were masked to exactly 0 (out of group);
    // the distribution is over the eligible (theta>0) patients only, so its own
    // 0..1 scale — matching the within-cohort bubble.
    const cohort = [[0], [0.2], [0.6], [0]]
    const d = thetaColumnDistribution(cohort, 0, [0, 0.5, 1])
    expect(d.n).toBe(2)
    expect(d.histogram).toEqual([0.5, 0.5])
  })

  it('computes p5..p95 from the eligible values (linear interpolation)', () => {
    const cohort = [[0.1], [0.2], [0.3], [0.4]]
    const d = thetaColumnDistribution(cohort, 0, [0, 1])
    expect(d.percentiles.p50).toBeCloseTo(0.25, 10) // median of 4 pts, linear interp
    expect(d.percentiles.p95).toBeCloseTo(0.385, 10)
  })

  it('returns zeros for an empty cohort or an all-masked column', () => {
    expect(thetaColumnDistribution([], 0, [0, 0.5, 1])).toEqual({
      histogram: [0, 0], percentiles: { p5: 0, p25: 0, p50: 0, p75: 0, p95: 0 }, n: 0,
    })
    expect(thetaColumnDistribution([[0], [0]], 0, [0, 0.5, 1]).n).toBe(0)
  })

  it("the tail mass above tau equals the cohort's coverage (bubble = area above tau)", () => {
    // No masked patients, tau exactly on a bin edge, no value on the edge: the
    // fraction in bins at/above tau must equal cohortCoverage for that column.
    const cohort = [[0.05], [0.15], [0.25], [0.35], [0.85]]
    const edges = [0, 0.1, 0.2, 0.3, 0.4, 1.0]
    const tau = 0.2
    const d = thetaColumnDistribution(cohort, 0, edges)
    // bins with left edge >= tau: indices 2,3,4 -> theta in {0.25,0.35,0.85} = 3/5
    const tailIdx = edges.slice(0, -1).map((lo, i) => ({ lo, i })).filter((b) => b.lo >= tau)
    const tailMass = tailIdx.reduce((s, b) => s + d.histogram[b.i], 0)
    expect(tailMass).toBeCloseTo(cohortCoverage(cohort, tau, 1)[0], 10)
  })
})

describe('cohortCoverage', () => {
  it('counts the fraction of patients with theta_k strictly greater than tau', () => {
    const thetas = [
      [0.5, 0.01, 0.0],
      [0.3, 0.03, 0.0],
    ]
    // tau=0.02: topic0 -> both>0.02 (1.0); topic1 -> only 0.03>0.02 (0.5);
    // topic2 -> none (0.0). Strict >: 0.02 itself would NOT count.
    expect(cohortCoverage(thetas, 0.02, 3)).toEqual([1.0, 0.5, 0.0])
  })

  it('treats theta_k == tau as NOT covered (strict inequality)', () => {
    expect(cohortCoverage([[0.02, 0.0, 0.0]], 0.02, 3)).toEqual([0.0, 0.0, 0.0])
  })

  it('returns all-zero coverage for an empty cohort', () => {
    expect(cohortCoverage([], 0.01, 4)).toEqual([0, 0, 0, 0])
  })
})

describe('sampleThetaCohort', () => {
  const base = { values: {}, n: 200, seed: 20260706 }

  it('returns n theta vectors of length K that sum to ~1 (softmax over free rows)', () => {
    const bundle = makeStmBundleFixture()
    const thetas = sampleThetaCohort({ bundle, active: false, ...base })
    expect(thetas).toHaveLength(200)
    for (const t of thetas) {
      expect(t).toHaveLength(3)
      expect(t.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
    }
  })

  it('is deterministic for a fixed seed', () => {
    const bundle = makeStmBundleFixture()
    const a = sampleThetaCohort({ bundle, active: false, ...base })
    const b = sampleThetaCohort({ bundle, active: false, ...base })
    expect(a).toEqual(b)
  })

  it('the per-group Cholesky cache does not change the output (byte-identical to a naive per-patient draw)', () => {
    // sampleThetaCohort hoists the O(free³) factorization out of the per-patient
    // loop by caching it per group. This must not perturb the RNG stream: the
    // cached result has to equal a naive loop that rebuilds everything per
    // patient in the same draw order. Verify on both a fixed-covariate (active)
    // and a fully-marginal (inactive) cohort.
    const bundle = makeStmBundleFixture()
    const naive = (active: boolean, values: Record<string, number | string>): number[][] => {
      const rng = createRng(20260706)
      const out: number[][] = []
      const fixedX = active
        ? buildDesignVector(bundle.covariateSchema!.design_columns, values)
        : null
      for (let i = 0; i < 300; i++) {
        const group = bundle.gating ? sampleMarginalGroup(bundle.gating, rng) : null
        const x = fixedX ?? buildDesignVector(
          bundle.covariateSchema!.design_columns,
          sampleMarginalCovariates(bundle.covariateSchema!, rng),
        )
        out.push(sampleConditionedTheta({
          effects: bundle.covariateEffects!,
          x,
          correlation: bundle.correlation!,
          topicBlocks: bundle.gating?.topic_blocks ?? null,
          group,
          rng,
        }))
      }
      return out
    }
    expect(sampleThetaCohort({ bundle, active: true, values: { age: 50 }, n: 300, seed: 20260706 }))
      .toEqual(naive(true, { age: 50 }))
    expect(sampleThetaCohort({ bundle, active: false, values: {}, n: 300, seed: 20260706 }))
      .toEqual(naive(false, {}))
  })

  it('with active covariates fixed at high age, coverage shifts to the age-loaded topic', () => {
    // Fixture Gamma: topic1 eta = 1.0 - 0.02*age, topic2 eta = 0.5 + 0.03*age.
    // At age=100 topic2 dominates topic1, so its coverage is higher.
    const bundle = makeStmBundleFixture()
    const thetas = sampleThetaCohort({ bundle, active: true, values: { age: 100 }, n: 1500, seed: 20260706 })
    const cov = cohortCoverage(thetas, 0.01, 3)
    expect(cov[2]).toBeGreaterThan(cov[1])
  })

  it('on a GATED bundle, keeps the foreground topic represented but bounded by its group proportion', () => {
    // Same K=3 shape as makeStmBundleFixture(), but topic 2 is a foreground
    // ('grp') topic instead of background. group is ALWAYS a per-patient
    // marginal draw (see sampleThetaCohort's doc comment), so ~30% of
    // patients (the grp proportion) get topic 2 unmasked and ~70%
    // (background-only, group=null) get it masked to 0 via
    // allowedMaskForGroup/sampleConditionedTheta's `allowed()` check.
    const gated = {
      model: {
        K: 3,
        V: 4,
        alpha: [0.1, 0.1, 0.1],
        beta: [
          [0.7, 0.1, 0.1, 0.1],
          [0.1, 0.7, 0.1, 0.1],
          [0.1, 0.1, 0.1, 0.7],
        ],
      },
      phenotypes: {
        phenotypes: [
          { id: 0, label: 'Reference', description: '', quality: 'background', npmi: null, pair_coverage: null, corpus_prevalence: 0.5, original_topic_id: 0 },
          { id: 1, label: 'Topic A', description: '', quality: 'phenotype', npmi: 0.2, pair_coverage: 0.9, corpus_prevalence: 0.3, original_topic_id: 1 },
          { id: 2, label: 'Topic B (foreground)', description: '', quality: 'phenotype', npmi: 0.3, pair_coverage: 0.9, corpus_prevalence: 0.2, original_topic_id: 2 },
        ],
      },
      vocab: {
        codes: [
          { id: 0, code: 'C0', description: 'code 0', domain: 'condition', corpus_freq: 0.4 },
          { id: 1, code: 'C1', description: 'code 1', domain: 'condition', corpus_freq: 0.3 },
          { id: 2, code: 'C2', description: 'code 2', domain: 'condition', corpus_freq: 0.2 },
          { id: 3, code: 'C3', description: 'code 3', domain: 'condition', corpus_freq: 0.1 },
        ],
      },
      corpusStats: { corpus_size_docs: 1000, mean_codes_per_doc: 8, k: 3, v: 4, v_full: 4 },
      covariateSchema: {
        k: 3,
        controls: [{ name: 'age', type: 'continuous', range: [0, 100], default: 60 }],
        design_columns: [
          { name: 'Intercept', recipe: { kind: 'intercept' } },
          { name: 'age', recipe: { kind: 'main', var: 'age' } },
        ],
        unsupported: [],
      },
      covariateEffects: [
        { covariate: 'Intercept', per_topic: [0, 1.0, 0.5] },
        { covariate: 'age', per_topic: [0, -0.02, 0.03] },
      ],
      correlation: {
        topic_order: [0, 1, 2],
        block_labels: ['background', 'background', 'grp'],
        R: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        identified: [
          [true, true, true],
          [true, true, true],
          [true, true, true],
        ],
        support: [
          [100, 100, 100],
          [100, 100, 100],
          [100, 100, 100],
        ],
        reference_topic: 0,
      },
      gating: {
        group_var: 'g',
        groups: ['grp'],
        topic_blocks: ['background', 'background', 'grp'],
        group_proportions: { grp: 0.3 },
        background_only_proportion: 0.7,
      },
    } as any

    const thetas = sampleThetaCohort({ bundle: gated, active: false, values: {}, n: 2000, seed: 20260706 })
    const cov = cohortCoverage(thetas, 0.01, 3)
    // Some grp patients (~30% of the cohort) were sampled with topic 2
    // unmasked, so it's not entirely suppressed...
    expect(cov[2]).toBeGreaterThan(0)
    // ...but it's bounded well below 1: background-only patients (~70% of
    // the cohort, group=null) always have topic 2 masked to 0.
    expect(cov[2]).toBeLessThan(0.4)
  })
})
