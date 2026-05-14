export function digamma(x: number): number {
  let result = 0
  while (x < 6) { result -= 1 / x; x += 1 }
  result += Math.log(x) - 1 / (2 * x)
  const xx = 1 / (x * x)
  result -= xx * ((1 / 12) - xx * ((1 / 120) - xx * (1 / 252)))
  return result
}

export interface EStepInput {
  alpha: number[]
  beta: number[][]
  codeCounts: Map<number, number>
  maxIter?: number
  tol?: number
}
export interface EStepResult { theta: number[]; gamma: number[]; iterations: number }

export function variationalEStep(input: EStepInput): EStepResult {
  const { alpha, beta, codeCounts } = input
  const maxIter = input.maxIter ?? 50
  const tol = input.tol ?? 1e-4
  const K = alpha.length
  const entries = Array.from(codeCounts.entries())

  let gamma = alpha.slice()
  if (entries.length === 0) {
    const sum = gamma.reduce((a, b) => a + b, 0) || 1
    return { theta: gamma.map((g) => g / sum), gamma, iterations: 0 }
  }
  let prevGamma = gamma.slice()
  let it = 0
  for (; it < maxIter; it++) {
    const gammaSum = gamma.reduce((a, b) => a + b, 0)
    const psiSum = digamma(gammaSum)
    const eLogTheta = gamma.map((g) => digamma(g) - psiSum)
    const newGamma = alpha.slice()
    for (const [w, c] of entries) {
      const phi = new Array<number>(K)
      let phiSum = 0
      for (let k = 0; k < K; k++) {
        phi[k] = beta[k][w] * Math.exp(eLogTheta[k])
        phiSum += phi[k]
      }
      if (phiSum === 0) continue
      for (let k = 0; k < K; k++) newGamma[k] += c * (phi[k] / phiSum)
    }
    const delta = newGamma.reduce((a, g, k) => a + Math.abs(g - prevGamma[k]), 0)
    prevGamma = gamma
    gamma = newGamma
    if (delta < tol * K) { it++; break }
  }
  const gammaSum = gamma.reduce((a, b) => a + b, 0) || 1
  return { theta: gamma.map((g) => g / gammaSum), gamma, iterations: it }
}
