import type { CovariateRecipe, CovariateEffects } from './types'

type Values = Record<string, number | string>

export function evalRecipe(recipe: CovariateRecipe, values: Values): number {
  switch (recipe.kind) {
    case 'intercept': return 1
    case 'main': return Number(values[recipe.var] ?? 0)
    case 'dummy': return values[recipe.var] === recipe.level ? 1 : 0
    case 'interaction':
      return recipe.factors.reduce((acc, f) => acc * evalRecipe(f, values), 1)
  }
}

export function buildDesignVector(
  designColumns: { name: string; recipe: CovariateRecipe }[],
  values: Values,
): number[] {
  return designColumns.map((d) => evalRecipe(d.recipe, values))
}

export function covariatePrevalence(effects: CovariateEffects, x: number[]): number[] {
  const K = effects[0]?.per_topic.length ?? 0
  const eta = new Array(K).fill(0)
  for (let p = 0; p < effects.length; p++) {
    const row = effects[p].per_topic
    for (let k = 0; k < K; k++) eta[k] += row[k] * x[p]
  }
  const m = Math.max(...eta)
  const exp = eta.map((e) => Math.exp(e - m))
  const s = exp.reduce((a, b) => a + b, 0) || 1
  return exp.map((e) => e / s)
}
