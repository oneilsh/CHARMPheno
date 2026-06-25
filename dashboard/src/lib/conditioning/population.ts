import type { CovariateSchema } from '../types'

function pct(x: number): string {
  return `${Math.round(x * 100)}%`
}

export function populationLines(
  schema: CovariateSchema | undefined,
): { name: string; summary: string }[] {
  if (!schema) return []
  return schema.controls.map((c) => {
    if (c.type === 'continuous') {
      const [lo, hi] = c.range ?? [0, 0]
      return { name: c.name, summary: `${lo}-${hi} (med ${c.default ?? ''})` }
    }
    const levels = c.levels ?? []
    const summary = c.proportions
      ? levels.map((l) => `${l} ${pct(c.proportions![l] ?? 0)}`).join(' / ')
      : levels.join(' / ')
    return { name: c.name, summary }
  })
}
