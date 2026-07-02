import { derived } from 'svelte/store'
import { phenotypeOrder, bundle } from './store'

// Phenotype hue assignment.
//
// Phenotypes are first laid out in a 1D similarity ordering (the
// `phenotypeOrder` store, which walks the K x K JSD distance matrix
// nearest-neighbor-style so adjacents are actually similar). Hues are
// then assigned with a golden-ratio stride around the wheel - adjacent
// positions in the similarity ordering land ~138 degrees apart on the
// wheel, so similar phenotypes are forced to look maximally DIFFERENT.
// That matters most on the patient atlas where two clusters with similar
// dominant phenotypes can sit close together.

const GOLDEN = 0.6180339887498949 // 1 / phi
const SATURATION = 65
const LIGHTNESS = 52

function hsl(h: number, s: number, l: number): string {
  return `hsl(${h.toFixed(1)}, ${s}%, ${l}%)`
}

const FALLBACK = [
  '#06b6d4', '#8b5cf6', '#10b981', '#f59e0b',
  '#ec4899', '#3b82f6', '#ef4444', '#64748b',
]

export const phenotypeHue = derived(phenotypeOrder, ($order) => {
  if (!$order || !$order.length) {
    return (k: number) => FALLBACK[((k % FALLBACK.length) + FALLBACK.length) % FALLBACK.length]
  }
  const colors = new Map<number, string>()
  $order.forEach((k, i) => {
    const h = ((i * GOLDEN) % 1) * 360
    colors.set(k, hsl(h, SATURATION, LIGHTNESS))
  })
  return (k: number) =>
    colors.get(k) ?? FALLBACK[((k % FALLBACK.length) + FALLBACK.length) % FALLBACK.length]
})

// Group hue assignment for the Patient atlas's color-by-group mode. Gated
// STM bundles have only a handful of groups (background + a few foreground
// conditions), so a direct index into the same categorical FALLBACK palette
// used by phenotypeHue's fallback is distinctive enough without needing the
// golden-ratio similarity ordering. Null group (background-only draw, or a
// non-gated bundle) gets a neutral gray rather than a palette color.
const NO_GROUP_COLOR = '#94a3b8'

export const groupHue = derived(bundle, ($b) => {
  const groups = $b?.gating?.groups ?? []
  const colors = new Map<string, string>()
  groups.forEach((g, i) => colors.set(g, FALLBACK[i % FALLBACK.length]))
  return (g: string | null | undefined) =>
    g == null ? NO_GROUP_COLOR : (colors.get(g) ?? NO_GROUP_COLOR)
})
