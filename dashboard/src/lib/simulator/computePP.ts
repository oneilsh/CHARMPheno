// Posterior-predictive draw shared by the two callers that produce the
// Simulator's sample-mix + predicted-record panels: the app's load routine
// (App.svelte, which seeds a default population so the tab is never empty) and
// the "Run the model" button (Simulator.svelte, which regenerates from the
// panel's conditions/conditioning). Factored out so both stay byte-identical
// and the button's isStm/conditioned-theta logic lives in one place.

import { runSimulator, type SimulatorRunResult } from './runSamples'
import { buildDesignVector } from '../covariate'
import { resolveGroup } from '../conditioning/marginalSampler'
import {
  prepareRecordPosterior, drawRecordPosterior, type RecordPosteriorPrep,
} from '../conditioning/recordPosterior'
import { createRng } from '../sampling'
import type { DashboardBundle } from '../types'
import type { Conditioning } from '../store'

// The default sample count: enough for stable occurrence-rate estimates in the
// posterior-predictive panel and a smooth per-sample strip, while the fast
// (non-autoregressive) path stays snappy at ~2 E-steps per sample.
export const DEFAULT_SIM_SAMPLES = 500

/**
 * Draw `opts.nSamples` posterior-predictive records for `b`, conditioned on the
 * starting-condition `prefix` and the source-cohort/covariate `conditioning`.
 *
 * STM bundles (covariate effects + a topic-correlation block) condition the
 * generative theta on the covariate values/group AND the prefix via the
 * logistic-normal posterior sampler; with an empty prefix and the default
 * conditioning this reduces to the covariate/group prior draw. Non-STM bundles
 * take the unchanged Dirichlet path (no conditionedTheta).
 */
export function computePosteriorPredictive(
  b: DashboardBundle,
  prefix: number[],
  conditioning: Pick<Conditioning, 'values' | 'group'>,
  opts: { nSamples: number; seed: number; autoregressive: boolean },
): SimulatorRunResult {
  const isStm = !!b.covariateEffects && !!b.correlation
  const prefixCounts = new Map<number, number>()
  for (const w of prefix) prefixCounts.set(w, (prefixCounts.get(w) ?? 0) + 1)

  let conditionedTheta: (() => number[]) | undefined
  if (isStm) {
    const schema = b.covariateSchema!
    const x = buildDesignVector(schema.design_columns, conditioning.values)
    const tRng = createRng(opts.seed ^ 0x9e3779b9)
    // Resolve the group per draw so an "all subcohorts" selection spreads each
    // sampled record across the population mix. The record-posterior prep
    // (mode + Laplace factor) is RNG-free and identical for every draw in a
    // group, so cache it per group and only draw() per sample.
    const topicBlocks = b.gating?.topic_blocks ?? null
    const prepCache = new Map<string | null, RecordPosteriorPrep>()
    conditionedTheta = () => {
      const g = resolveGroup(conditioning.group, b.gating, tRng)
      let prep = prepCache.get(g)
      if (!prep) {
        prep = prepareRecordPosterior({
          effects: b.covariateEffects!, x, correlation: b.correlation!,
          topicBlocks, group: g, prefixCounts, beta: b.model.beta,
        })
        prepCache.set(g, prep)
      }
      return drawRecordPosterior(prep, tRng)
    }
  }

  return runSimulator({
    alpha: b.model.alpha,
    beta: b.model.beta,
    meanCodesPerDoc: b.corpusStats.mean_codes_per_doc,
    prefix,
    nSamples: opts.nSamples,
    seed: opts.seed,
    autoregressive: opts.autoregressive,
    conditionedTheta,
  })
}
