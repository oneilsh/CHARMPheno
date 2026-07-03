import { sampleConditionedTheta, mvnDraw, buildGenerativeSigma } from './logisticNormal'
import { choleskyPD, invSPD, solveSPD } from './linalg'
import type { CovariateEffects, Correlation } from '../types'

// Faithful STM record completion. The prefix (observed starting conditions)
// conditions theta via the posterior over eta under the covariate/group prior
// eta ~ Normal(mu, Sigma) (Blei & Lafferty 2007 logistic-normal). We find the
// posterior mode by Fisher scoring and draw from a Laplace Gaussian around it.
// An empty prefix has no likelihood term, so the mode is mu and the Laplace
// covariance is Sigma -> this reduces exactly to sampleConditionedTheta (the
// covariate/group prior draw), which is why cohort generation and record
// completion are a single code path.
export function sampleRecordPosterior(args: {
  effects: CovariateEffects
  x: number[]
  correlation: Correlation
  topicBlocks: string[] | null
  group: string | null
  prefixCounts: Map<number, number>
  beta: number[][]
  rng: () => number
}): number[] {
  const { effects, x, correlation, topicBlocks, group, prefixCounts, beta, rng } = args

  // Empty prefix: the posterior is the prior. Delegate so the draw is identical
  // — this keeps the "empty prefix == prior draw" invariant true.
  if (prefixCounts.size === 0) {
    return sampleConditionedTheta({ effects, x, correlation, topicBlocks, group, rng })
  }

  const K = effects[0]?.per_topic.length ?? 0
  const ref = correlation.reference_topic ?? -1
  const order = correlation.topic_order

  const allowed = (k: number): boolean => {
    if (!topicBlocks) return true
    const b = topicBlocks[k]
    return b === 'background' || b === group
  }

  // Free R rows: allowed, non-reference. `ids[i]` is the display topic id of free i.
  const freeIdx: number[] = []
  for (let r = 0; r < order.length; r++) {
    const k = order[r]
    if (k !== ref && allowed(k)) freeIdx.push(r)
  }
  const ids = freeIdx.map((r) => order[r])
  const F = ids.length

  const refAllowed = ref >= 0 && allowed(ref)

  // Prior mean mu and covariance Sigma over the free rows.
  const mu = freeIdx.map((r) => {
    const k = order[r]
    let m = 0
    effects.forEach((e, p) => { m += e.per_topic[k] * x[p] })
    return m
  })
  const Sigma = buildGenerativeSigma(correlation, freeIdx)

  if (F === 0) {
    // Only the reference is allowed: all mass on it (or empty -> uniform-safe).
    const theta = new Array<number>(K).fill(0)
    if (refAllowed) theta[ref] = 1
    return theta
  }

  const Sinv = invSPD(Sigma)

  // Prefix codes as arrays for the likelihood loop.
  const codes = [...prefixCounts.keys()]
  const counts = codes.map((w) => prefixCounts.get(w)!)

  // theta over the allowed set from free eta (reference eta = 0).
  const thetaFromEta = (eta: number[]): { thetaFree: number[]; thetaRef: number } => {
    const logits = refAllowed ? [0, ...eta] : [...eta]
    const mx = Math.max(...logits)
    const ex = logits.map((v) => Math.exp(v - mx))
    const s = ex.reduce((a, b) => a + b, 0) || 1
    if (refAllowed) {
      const thetaRef = ex[0] / s
      return { thetaFree: ex.slice(1).map((e) => e / s), thetaRef }
    }
    return { thetaFree: ex.map((e) => e / s), thetaRef: 0 }
  }

  // Log-posterior objective at z (over the free eta), used only by the line
  // search below: L(z) = -0.5 (z-mu)' Sinv (z-mu) + sum_w n_w log(s_w), where
  // s_w is the observation probability of code w under theta(z) (reference
  // included, free topics at z). log(max(s_w, 1e-300)) avoids -Infinity if a
  // trial step pushes s_w to (numerically) zero.
  const objective = (z: number[]): number => {
    const { thetaFree, thetaRef } = thetaFromEta(z)
    const dm = z.map((v, i) => v - mu[i])
    let quad = 0
    for (let i = 0; i < F; i++) for (let j = 0; j < F; j++) quad += dm[i] * Sinv[i][j] * dm[j]
    let ll = 0
    for (let c = 0; c < codes.length; c++) {
      const w = codes[c]
      const nw = counts[c]
      let sw = refAllowed ? thetaRef * (beta[ref]?.[w] ?? 0) : 0
      for (let i = 0; i < F; i++) sw += thetaFree[i] * (beta[ids[i]]?.[w] ?? 0)
      ll += nw * Math.log(Math.max(sw, 1e-300))
    }
    return -0.5 * quad + ll
  }

  // Fisher scoring to the posterior mode, with a backtracking (Armijo-style)
  // line search on the step length. The bare Fisher-scoring step solveSPD(H, g)
  // is a Newton step under the expected-information curvature; it converges
  // fast near the mode but with a strong likelihood (a large prefix) the full
  // step can overshoot the mode and oscillate. Halving alpha until the
  // objective does not decrease makes every accepted step monotone
  // non-decreasing in L, which guarantees convergence; well-behaved cases
  // still take the full step (alpha=1) on essentially every iteration, so
  // this leaves fast convergence and the mode itself unchanged.
  let eta = mu.slice()
  for (let iter = 0; iter < 50; iter++) {
    const { thetaFree, thetaRef } = thetaFromEta(eta)

    // Gradient of the log-likelihood: sum_w n_w (phi_free - thetaFree).
    const gLik = new Array<number>(F).fill(0)
    // Expected-information curvature accumulator starts from the prior precision.
    const H = Sinv.map((row) => row.slice())
    for (let c = 0; c < codes.length; c++) {
      const w = codes[c]
      const nw = counts[c]
      // s_w over allowed topics (reference included), phi over free topics.
      let sw = refAllowed ? thetaRef * (beta[ref]?.[w] ?? 0) : 0
      const phi = new Array<number>(F)
      for (let i = 0; i < F; i++) {
        const contrib = thetaFree[i] * (beta[ids[i]]?.[w] ?? 0)
        phi[i] = contrib
        sw += contrib
      }
      if (sw <= 0) continue
      for (let i = 0; i < F; i++) {
        phi[i] /= sw
        gLik[i] += nw * (phi[i] - thetaFree[i])
      }
      // Add n_w (diag(thetaFree) - thetaFree thetaFreeᵀ) to H (PSD data term).
      for (let i = 0; i < F; i++) {
        H[i][i] += nw * thetaFree[i]
        for (let j = 0; j < F; j++) H[i][j] -= nw * thetaFree[i] * thetaFree[j]
      }
    }
    // Prior gradient: -Sinv (eta - mu).
    const dm = eta.map((v, i) => v - mu[i])
    const grad = gLik.map((g, i) => {
      let pg = 0
      for (let j = 0; j < F; j++) pg += Sinv[i][j] * dm[j]
      return g - pg
    })
    const delta = solveSPD(H, grad)

    // Backtracking line search: accept the full Newton step (alpha=1) unless
    // it fails to improve L, in which case halve alpha until it does (or
    // alpha bottoms out, at which point we take the tiny/no-op step anyway).
    const L0 = objective(eta)
    let alpha = 1
    let trial = eta.map((v, i) => v + alpha * delta[i])
    while (alpha > 1e-4 && objective(trial) < L0) {
      alpha /= 2
      trial = eta.map((v, i) => v + alpha * delta[i])
    }

    let maxAbs = 0
    for (let i = 0; i < F; i++) {
      const stepI = alpha * delta[i]
      eta[i] += stepI
      maxAbs = Math.max(maxAbs, Math.abs(stepI))
    }
    if (maxAbs < 1e-6) break
  }

  // Laplace covariance = H⁻¹ at the mode; draw eta ~ Normal(eta*, H⁻¹).
  const { thetaFree } = thetaFromEta(eta)
  const H = Sinv.map((row) => row.slice())
  for (let c = 0; c < codes.length; c++) {
    const nw = counts[c]
    for (let i = 0; i < F; i++) {
      H[i][i] += nw * thetaFree[i]
      for (let j = 0; j < F; j++) H[i][j] -= nw * thetaFree[i] * thetaFree[j]
    }
  }
  const cov = invSPD(H)
  const etaDraw = mvnDraw(eta, choleskyPD(cov), rng)

  // Assemble theta over all K display topics.
  const logits = new Array<number>(K).fill(-Infinity)
  if (refAllowed) logits[ref] = 0
  ids.forEach((k, i) => { logits[k] = etaDraw[i] })
  const finite = logits.filter((e) => e !== -Infinity)
  const mx = finite.length ? Math.max(...finite) : 0
  const ex = logits.map((e) => (e === -Infinity ? 0 : Math.exp(e - mx)))
  const s = ex.reduce((a, b) => a + b, 0) || 1
  return ex.map((e) => e / s)
}
