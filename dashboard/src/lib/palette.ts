import { derived } from 'svelte/store'
import { hcl } from 'd3'
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
// golden-ratio similarity ordering. Null group (a background-only draw, or a
// non-gated bundle) gets a neutral slate rather than a palette color.
const NO_GROUP_COLOR = '#94a3b8'
// The shared "All" (background) block anchors the palette at the project teal
// (RGB 82,179,209). Foreground cohorts fan out around the hue wheel in HCL
// (perceptually uniform, the space ggplot2's hue_pal actually uses) starting at
// a +150° rose-red offset from the teal's hue — chosen over the strict
// complement (+180°), which lands in a muddy tan because the teal is low-chroma.
// Foreground uses a HIGHER chroma than the soft teal so specific cohorts pop
// against the shared background. Background is a real color, not gray: it spans
// the whole corpus and shouldn't read as de-emphasized.
const TEAL = '#52b3d1' // RGB 82,179,209 — the background/"All" anchor
const FG_HUE_OFFSET = 150 // rose-red start, relative to the teal hue
const FG_CHROMA = 62
const FG_LIGHTNESS = 64

export const groupHue = derived(bundle, ($b) => {
  const groups = $b?.gating?.groups ?? []
  const baseHue = hcl(TEAL).h
  // Spread k foreground cohorts evenly over the ring from the rose-red start
  // (step includes the background slot so they don't wrap back onto the teal).
  const step = 360 / (groups.length + 1)
  const colors = new Map<string, string>([['background', TEAL]])
  groups.forEach((g, i) => {
    const h = baseHue + FG_HUE_OFFSET + i * step
    colors.set(g, hcl(h, FG_CHROMA, FG_LIGHTNESS).formatHex())
  })
  return (g: string | null | undefined) =>
    g == null ? NO_GROUP_COLOR : (colors.get(g) ?? NO_GROUP_COLOR)
})
