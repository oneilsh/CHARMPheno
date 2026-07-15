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

// Draw a per-patient group from group_proportions. Returns null for a
// background-only patient (no foreground group) — drawn with the reported
// background_only_proportion (the cohort share in no foreground group, e.g. a
// 'general' population plus any k-anon-suppressed group). group_proportions
// now sum to (1 - background_only_proportion). Older bundles that lack these
// fields fall back to a uniform draw over the foreground groups (never null),
// preserving prior behavior.
// Sentinel group value meaning "sample across all subcohorts": instead of
// pinning one foreground group (or null for background-only), each patient
// draws its own group from the population mix (sampleMarginalGroup). Kept
// distinct from null and from any real group name so a source-cohort picker
// can offer three intents — one subcohort, background-only, or the whole
// population — without overloading null.
export const ALL_SUBCOHORTS = '__all__'

// Resolve a conditioning group to the concrete group used for ONE draw. A
// real group name (or null for background-only) passes through unchanged; the
// ALL_SUBCOHORTS sentinel draws a fresh per-patient group from the population
// mix, so calling this once per sample yields a realistic cohort spread across
// subcohorts. Returns null for the sentinel when the bundle has no gating
// (there are no subcohorts to spread across).
export function resolveGroup(
  group: string | null,
  gating: GatingSpec | null | undefined,
  rng: () => number,
): string | null {
  if (group === ALL_SUBCOHORTS) return gating ? sampleMarginalGroup(gating, rng) : null
  return group
}

export function sampleMarginalGroup(gating: GatingSpec, rng: () => number): string | null {
  const groups = gating.groups
  const props = gating.group_proportions
  if (props && groups.length) {
    const bgOnly = gating.background_only_proportion ?? 0
    // Weights over [background-only (null), ...foreground groups].
    const weights = [bgOnly, ...groups.map((g) => props[g] ?? 0)]
    const s = weights.reduce((x, y) => x + y, 0) || 1
    const idx = sampleCategorical(weights.map((x) => x / s), rng)
    return idx === 0 ? null : groups[idx - 1]
  }
  console.warn('[marginalSampler] gating.group_proportions absent; sampling groups uniformly')
  return groups[Math.floor(rng() * groups.length)]
}
