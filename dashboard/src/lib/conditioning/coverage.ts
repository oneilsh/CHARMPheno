import type { DashboardBundle } from '../types'
import { createRng } from '../sampling'
import { buildDesignVector } from '../covariate'
import { sampleConditionedTheta } from './logisticNormal'
import { sampleMarginalCovariates, sampleMarginalGroup } from './marginalSampler'

export interface ThetaCohortInput {
  bundle: DashboardBundle
  // true  -> covariates fixed to `values`, group still per-patient marginal.
  // false -> covariates AND group both per-patient marginal (corpus mix).
  active: boolean
  values: Record<string, number | string>
  n: number
  seed: number
}

// Draw `n` patient theta vectors from the faithful conditional logistic-normal
// (sampleConditionedTheta). STM bundles only — callers must check
// covariateEffects + correlation are present before invoking. Group is ALWAYS a
// per-patient marginal draw so gated foreground phenotypes remain represented.
export function sampleThetaCohort(input: ThetaCohortInput): number[][] {
  const { bundle: b, active, values, n, seed } = input
  const effects = b.covariateEffects!
  const correlation = b.correlation!
  const schema = b.covariateSchema!
  const topicBlocks = b.gating?.topic_blocks ?? null
  const rng = createRng(seed)
  const fixedX = active ? buildDesignVector(schema.design_columns, values) : null
  const thetas: number[][] = []
  for (let i = 0; i < n; i++) {
    const group = b.gating ? sampleMarginalGroup(b.gating, rng) : null
    const x = fixedX ?? buildDesignVector(schema.design_columns, sampleMarginalCovariates(schema, rng))
    thetas.push(sampleConditionedTheta({ effects, x, correlation, topicBlocks, group, rng }))
  }
  return thetas
}

// Per-topic fraction of the cohort with theta_k > tau (strict). K is the topic
// count; an empty cohort yields all-zero coverage.
export function cohortCoverage(thetas: number[][], tau: number, K: number): number[] {
  const cov = new Array<number>(K).fill(0)
  if (thetas.length === 0) return cov
  for (const theta of thetas)
    for (let k = 0; k < K; k++)
      if (theta[k] > tau) cov[k] += 1
  for (let k = 0; k < K; k++) cov[k] /= thetas.length
  return cov
}

// Rescale whole-cohort coverage to WITHIN-COHORT coverage. A foreground topic is
// masked to 0 for every patient outside its group, so its whole-cohort coverage
// is capped by that group's population share — every foreground bubble reads tiny
// next to background ones, hiding within-group diversity. Dividing each topic by
// the fraction of the population ELIGIBLE to express it — 1 for a background
// topic (everyone can), group_proportions[g] for a foreground topic in group g
// (only group-g patients can) — puts each cohort on its own 0..1 scale. Clamped
// to 1 (sampling noise can nudge a ratio just past its group share). No-op when
// the bundle is not gated (topicBlocks / groupProportions absent).
export function withinCohortCoverage(
  coverage: number[],
  topicBlocks: string[] | null,
  groupProportions: Record<string, number> | null,
): number[] {
  if (!topicBlocks || !groupProportions) return coverage
  return coverage.map((c, k) => {
    const block = topicBlocks[k]
    if (block === 'background') return c
    const p = groupProportions[block]
    return p && p > 0 ? Math.min(c / p, 1) : c
  })
}
