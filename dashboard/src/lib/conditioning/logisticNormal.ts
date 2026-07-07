import { sampleStandardNormal } from '../sampling'
import type { CovariateEffects, Correlation } from '../types'
import { choleskyPD } from './linalg'

// Lower-triangular Cholesky factor L with L Lᵀ = A. Throws if A is not
// positive-definite (a non-positive pivot). Textbook Cholesky-Banachiewicz;
// the covariance sub-blocks it factors here are small (~40x40).
export function cholesky(A: number[][]): number[][] {
  const n = A.length
  const L: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i][j]
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k]
      if (i === j) {
        if (sum <= 0) throw new Error('cholesky: matrix is not positive-definite')
        L[i][j] = Math.sqrt(sum)
      } else {
        L[i][j] = sum / L[j][j]
      }
    }
  }
  return L
}

// One draw from Normal(mean, L Lᵀ): mean + L z, z standard-normal.
export function mvnDraw(mean: number[], L: number[][], rng: () => number): number[] {
  const n = mean.length
  const z = new Array<number>(n)
  for (let i = 0; i < n; i++) z[i] = sampleStandardNormal(rng)
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    let s = mean[i]
    for (let k = 0; k <= i; k++) s += L[i][k] * z[k]
    out[i] = s
  }
  return out
}

// Generative covariance sub-block over the free rows: the exported correlation R
// (unit-diagonal) rescaled to a covariance. Precedence:
//   1. correlation.eta_scale (scalar c): s_k = sqrt(eta_scale) for EVERY free
//      row -> Sigma = eta_scale * R. This is the current, preferred generation
//      input - a single pooled scale estimated at export with beta/R frozen
//      (corpus_eta_scale_gated_rdd / corpus_eta_scale_gated; ADR 0036
//      addendum), which supersedes the per-topic eta_var below (that per-topic
//      empirical variance came out ~10x too compressed, and a per-topic free
//      diagonal fit at fit time reopened the insight-0033 variance runaway).
//   2. correlation.eta_var (per-topic array, positional / aligned to R rows -
//      NOT display id): Sigma[a][b] = R[i][j] * s_a * s_b, s_k = sqrt(var_k),
//      var_k = eta_var[R-row r] ?? 1. Kept for back-compat with older bundles.
//   3. Neither present: s_k = 1 for every row -> Sigma = R exactly
//      (byte-identical to the original, pre-rescaling behavior).
// Scaling R up raises the eta variance, and softmax of higher-variance eta
// yields MORE peaked theta (more concentrated patients).
export function buildGenerativeSigma(
  correlation: Correlation,
  freeIdx: number[],
): number[][] {
  const es = correlation.eta_scale
  const ev = correlation.eta_var
  const s = freeIdx.map((r) =>
    es != null ? Math.sqrt(es) : Math.sqrt(ev ? (ev[r] ?? 1) : 1))
  return freeIdx.map((ri, a) =>
    freeIdx.map((rj, b) => (correlation.R[ri][rj] as number) * s[a] * s[b]))
}

// The group-only part of a conditional draw: which free R rows are sampled, the
// Cholesky factor L of their generative covariance, and whether the reference
// topic is included. This depends ONLY on the correlation, the gating blocks, and
// the selected group — NOT on the covariates x or the RNG — so it is identical
// for every patient in the same group and can be computed once and reused. It is
// also the EXPENSIVE part: choleskyPD is O(free³), and hoisting it out of a
// per-patient cohort loop (see coverage.ts) is what keeps the covariate sliders
// responsive — the covariates only shift the mean, never the covariance.
export interface GroupPrep {
  freeIdx: number[]   // indices into correlation.R / topic_order (the sampled rows)
  L: number[][]       // lower-triangular Cholesky factor of the free-row Sigma
  refIncluded: boolean // reference topic pinned to eta=0 (allowed in this group)
}

export function prepareConditionedGroup(args: {
  correlation: Correlation
  topicBlocks: string[] | null
  group: string | null
}): GroupPrep {
  const { correlation, topicBlocks, group } = args
  const ref = correlation.reference_topic ?? -1
  const order = correlation.topic_order        // display id per R row (free topics)

  // Allowed display-topic ids: all topics if not gated, else background plus the
  // selected group's foreground (null group = background only).
  const allowed = (k: number): boolean => {
    if (!topicBlocks) return true
    const b = topicBlocks[k]
    return b === 'background' || b === group
  }

  // Free R rows to sample: in topic_order, allowed, and not the reference.
  const freeIdx: number[] = []          // indices into correlation.R / order
  for (let r = 0; r < order.length; r++) {
    const k = order[r]
    if (k !== ref && allowed(k)) freeIdx.push(r)
  }

  // Cholesky of the Sigma sub-block over the free rows (guaranteed non-null / PD).
  const L = freeIdx.length ? choleskyPD(buildGenerativeSigma(correlation, freeIdx)) : []
  return { freeIdx, L, refIncluded: ref >= 0 && allowed(ref) }
}

// The per-patient part of a conditional draw, given a precomputed GroupPrep: form
// the covariate-dependent mean, draw eta ~ Normal(mean, L Lᵀ), and softmax to
// theta. This is the only part that varies with the covariates x and consumes the
// RNG (K standard-normals via mvnDraw), so the RNG stream is unchanged whether or
// not the prep was cached.
export function drawConditionedTheta(
  prep: GroupPrep,
  args: { effects: CovariateEffects; x: number[]; correlation: Correlation; rng: () => number },
): number[] {
  const { effects, x, correlation, rng } = args
  const { freeIdx, L, refIncluded } = prep
  const K = effects[0]?.per_topic.length ?? 0
  const ref = correlation.reference_topic ?? -1
  const order = correlation.topic_order

  // Mean eta over the free rows: mu_k = Gamma^T x (sum over covariate effects).
  const mean = freeIdx.map((r) => {
    const k = order[r]
    let m = 0
    for (let ci = 0; ci < effects.length; ci++) m += effects[ci].per_topic[k] * x[ci]
    return m
  })

  const etaFree = freeIdx.length ? mvnDraw(mean, L, rng) : []

  // Assemble eta over all K display topics: reference -> 0, free -> drawn,
  // masked -> -Infinity (exactly zero after softmax).
  const eta = new Array<number>(K).fill(-Infinity)
  if (refIncluded) eta[ref] = 0
  freeIdx.forEach((r, i) => { eta[order[r]] = etaFree[i] })

  const finite = eta.filter((e) => e !== -Infinity)
  const mx = finite.length ? Math.max(...finite) : 0
  const exp = eta.map((e) => (e === -Infinity ? 0 : Math.exp(e - mx)))
  const s = exp.reduce((a, b) => a + b, 0) || 1
  return exp.map((e) => e / s)
}

// Faithful STM forward draw: theta = softmax(eta), eta ~ Normal(Gamma^T x, Sigma)
// (logistic-normal prior; Blei & Lafferty 2007). The reference topic is pinned
// eta = 0 and excluded from Gamma's non-zero rows and from Sigma (correlation.R
// over the K-1 free topics). For a gated draw we restrict to the allowed set
// (background union the selected group) so the Sigma sub-block never includes a
// cross-group (unidentified/null) cell and is positive-definite by construction.
// Convenience wrapper: prepare the group then draw once. Cohort callers that
// sample many patients per group should instead cache prepareConditionedGroup and
// call drawConditionedTheta directly (see coverage.ts).
export function sampleConditionedTheta(args: {
  effects: CovariateEffects
  x: number[]
  correlation: Correlation
  topicBlocks: string[] | null
  group: string | null
  rng: () => number
}): number[] {
  const { effects, x, correlation, topicBlocks, group, rng } = args
  const prep = prepareConditionedGroup({ correlation, topicBlocks, group })
  return drawConditionedTheta(prep, { effects, x, correlation, rng })
}
