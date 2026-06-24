import { describe, it, expect } from 'vitest'
import { evalRecipe, buildDesignVector, covariatePrevalence, covariatePrevalenceGated, allowedMaskForGroup } from './covariate'

const values = { age: 70, sex: 'M', source_cohort: 'dementia' }

describe('evalRecipe', () => {
  it('intercept -> 1', () => expect(evalRecipe({ kind: 'intercept' }, values)).toBe(1))
  it('main -> control value', () => expect(evalRecipe({ kind: 'main', var: 'age' }, values)).toBe(70))
  it('dummy match -> 1, else 0', () => {
    expect(evalRecipe({ kind: 'dummy', var: 'sex', level: 'M' }, values)).toBe(1)
    expect(evalRecipe({ kind: 'dummy', var: 'sex', level: 'F' }, values)).toBe(0)
  })
  it('interaction -> product', () => {
    const r = { kind: 'interaction', factors: [
      { kind: 'main', var: 'age' }, { kind: 'dummy', var: 'sex', level: 'M' }] }
    expect(evalRecipe(r, values)).toBe(70)
  })
})

it('buildDesignVector aligns to design_columns order', () => {
  const dc = [
    { name: 'Intercept', recipe: { kind: 'intercept' } },
    { name: 'C(sex)[T.M]', recipe: { kind: 'dummy', var: 'sex', level: 'M' } },
    { name: 'age', recipe: { kind: 'main', var: 'age' } },
  ] as const
  expect(buildDesignVector(dc as any, values)).toEqual([1, 1, 70])
})

it('covariatePrevalence softmaxes Gamma^T x', () => {
  // effects rows index-aligned with x; 2 topics.
  const effects = [
    { covariate: 'Intercept', per_topic: [0, 0] },
    { covariate: 'age', per_topic: [1, 0] },   // topic 0 gets +x_age
  ]
  const x = [1, Math.log(2)]   // eta = [log2, 0] -> softmax = [2/3, 1/3]
  const p = covariatePrevalence(effects, x)
  expect(p[0]).toBeCloseTo(2 / 3, 6)
  expect(p[1]).toBeCloseTo(1 / 3, 6)
})

describe('gated prevalence', () => {
  it('masks before softmax and zeros out-of-group foreground', () => {
    const effects = [{ covariate: 'Intercept', per_topic: [0, 0, 0] }]
    const x = [1]
    const blocks = ['background', 'rare_dx', 'other']
    const mask = allowedMaskForGroup(blocks, 'rare_dx')   // [true,true,false]
    expect(mask).toEqual([true, true, false])
    const prev = covariatePrevalenceGated(effects, x, mask)
    expect(prev[2]).toBe(0)
    expect(Math.abs(prev[0] + prev[1] - 1)).toBeLessThan(1e-9)
  })
  it('background-only selection zeros all foreground', () => {
    const blocks = ['background', 'rare_dx']
    expect(allowedMaskForGroup(blocks, null)).toEqual([true, false])
  })
})
