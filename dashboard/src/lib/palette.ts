import { derived } from 'svelte/store'
import { phenotypeOrder } from './store'

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
