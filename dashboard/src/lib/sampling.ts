export function createRng(seed: number): () => number {
  let s = seed >>> 0
  return function () {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function sampleGamma(shape: number, rng: () => number): number {
  if (shape < 1) {
    const y = sampleGamma(shape + 1, rng)
    return y * Math.pow(rng(), 1 / shape)
  }
  const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d)
  while (true) {
    let u1: number, u2: number, v: number
    do { u1 = 2 * rng() - 1; u2 = 2 * rng() - 1; v = u1 * u1 + u2 * u2 } while (v >= 1 || v === 0)
    const x = u1 * Math.sqrt(-2 * Math.log(v) / v)
    const vv = 1 + c * x
    if (vv <= 0) continue
    const v3 = vv * vv * vv
    const u = rng()
    if (u < 1 - 0.0331 * x * x * x * x) return d * v3
    if (Math.log(u) < 0.5 * x * x + d * (1 - v3 + Math.log(v3))) return d * v3
  }
}

export function sampleDirichlet(alpha: number[], rng: () => number): number[] {
  const draws = alpha.map((a) => sampleGamma(a, rng))
  const sum = draws.reduce((a, b) => a + b, 0) || 1
  return draws.map((g) => g / sum)
}

export function sampleCategorical(p: number[], rng: () => number): number {
  const u = rng(); let cum = 0
  for (let i = 0; i < p.length; i++) { cum += p[i]; if (u < cum) return i }
  return p.length - 1
}

export function samplePoisson(lambda: number, rng: () => number): number {
  const L = Math.exp(-lambda); let k = 0; let p = 1
  do { k++; p *= rng() } while (p > L)
  return k - 1
}
