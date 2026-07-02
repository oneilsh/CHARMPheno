import { sampleCategorical } from '../sampling'
import type { CovariateSchema, GatingSpec } from '../types'

// Triangular draw on [a, b] with mode c (Stein & Keblis 2009 triangular
// inverse-CDF). A marginal-only approximation of a continuous covariate's
// spread; independent across covariates (no interactions) by design.
function sampleTriangular(a: number, b: number, c: number, rng: () => number): number {
  if (b <= a) return a
  const u = rng()
  const fc = (c - a) / (b - a)
  return u < fc
    ? a + Math.sqrt(u * (b - a) * (c - a))
    : b - Math.sqrt((1 - u) * (b - a) * (b - c))
}

// Draw a per-patient covariate value set from the model's reported marginals.
export function sampleMarginalCovariates(
  schema: CovariateSchema, rng: () => number,
): Record<string, number | string> {
  const values: Record<string, number | string> = {}
  for (const c of schema.controls) {
    if (c.type === 'continuous') {
      const [a, b] = c.range ?? [0, 1]
      const mode = c.default ?? (a + b) / 2
      values[c.name] = sampleTriangular(a, b, mode, rng)
    } else {
      const levels = c.levels ?? []
      const props = c.proportions
      if (props && levels.length) {
        const p = levels.map((l) => props[l] ?? 0)
        const s = p.reduce((x, y) => x + y, 0) || 1
        values[c.name] = levels[sampleCategorical(p.map((x) => x / s), rng)]
      } else if (levels.length) {
        values[c.name] = levels[Math.floor(rng() * levels.length)]
      }
    }
  }
  return values
}

// Draw a per-patient group from group_proportions; uniform fallback + warn.
export function sampleMarginalGroup(gating: GatingSpec, rng: () => number): string {
  const groups = gating.groups
  const props = gating.group_proportions
  if (props && groups.length) {
    const p = groups.map((g) => props[g] ?? 0)
    const s = p.reduce((x, y) => x + y, 0) || 1
    return groups[sampleCategorical(p.map((x) => x / s), rng)]
  }
  console.warn('[marginalSampler] gating.group_proportions absent; sampling groups uniformly')
  return groups[Math.floor(rng() * groups.length)]
}
