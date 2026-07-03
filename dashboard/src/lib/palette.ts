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
// golden-ratio similarity ordering. Null group (a background-only draw, or a
// non-gated bundle) gets a neutral slate rather than a palette color.
const NO_GROUP_COLOR = '#94a3b8'
// The shared "All" (background) block anchors the hue wheel at the project teal
// (RGB 82,179,209 == HSL 194,58,57); the foreground cohorts then fan out around
// the ring in evenly-spaced hues — the ggplot2 hue_pal / scale_color_hue scheme
// (equal spacing over the wheel), but pinned so background is always the teal and
// every cohort shares the teal's saturation/lightness. Colors span N = 1 (teal
// background) + one per foreground group, so two cohorts land complementary,
// three at 120°, etc. Background gets a real color (not gray): it spans the whole
// corpus and shouldn't read as de-emphasized.
const BACKGROUND_HUE = 194 // teal RGB 82,179,209
const GROUP_SATURATION = 58
const GROUP_LIGHTNESS = 57

export const groupHue = derived(bundle, ($b) => {
  const groups = $b?.gating?.groups ?? []
  const n = groups.length + 1 // background + foreground cohorts, evenly spaced
  const colors = new Map<string, string>([
    ['background', hsl(BACKGROUND_HUE, GROUP_SATURATION, GROUP_LIGHTNESS)],
  ])
  groups.forEach((g, i) => {
    const h = (BACKGROUND_HUE + ((i + 1) * 360) / n) % 360
    colors.set(g, hsl(h, GROUP_SATURATION, GROUP_LIGHTNESS))
  })
  return (g: string | null | undefined) =>
    g == null ? NO_GROUP_COLOR : (colors.get(g) ?? NO_GROUP_COLOR)
})
