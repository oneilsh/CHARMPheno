// Stable label-set selection for the phenotype atlas.
//
// The atlas draws always-on text labels for a handful of nodes. The labeled
// SET must stay fixed as the user moves the covariate controls — otherwise the
// live prevalence reader changes bubble sizes, the "top N" membership reshuffles
// every frame, and labels flicker between nodes (jarring). We therefore pick the
// set from a STABLE metric (corpus_prevalence, which never changes with the
// conditioning state), not from the live reader.
//
// For gated bundles we pick top-`perGroup` WITHIN each topic block, so a small
// foreground block (e.g. the cancer topics) still gets its own textual anchors
// instead of being crowded out by the larger background block.

export interface LabelablePhenotype {
  id: number
  corpus_prevalence: number
}

export interface LabelOptions<T extends LabelablePhenotype> {
  // topic_blocks: phenotype id -> block name. Omit for ungated bundles, where
  // all phenotypes fall into one implicit group (=> a global top-`perGroup`).
  blocks?: string[]
  // How many labels per block (or, ungated, globally).
  perGroup: number
  // Optional view-mode filter (simple mode hides dead/mixed topics). Hidden
  // phenotypes are excluded before ranking. Generic over the element type so
  // callers can pass a predicate that reads richer fields (e.g. quality).
  isVisible?: (p: T) => boolean
}

export function labeledPhenotypeIds<T extends LabelablePhenotype>(
  phenotypes: T[],
  opts: LabelOptions<T>,
): Set<number> {
  const { blocks, perGroup, isVisible } = opts
  const visible = isVisible ? phenotypes.filter(isVisible) : phenotypes

  // Bucket by block (single '__all__' bucket when ungated).
  const groups = new Map<string, LabelablePhenotype[]>()
  for (const p of visible) {
    const key = blocks ? (blocks[p.id] ?? '__ungrouped__') : '__all__'
    const bucket = groups.get(key)
    if (bucket) bucket.push(p)
    else groups.set(key, [p])
  }

  const chosen = new Set<number>()
  for (const bucket of groups.values()) {
    bucket
      // prevalence desc, then id asc so ties are deterministic
      .slice()
      .sort((a, b) => b.corpus_prevalence - a.corpus_prevalence || a.id - b.id)
      .slice(0, perGroup)
      .forEach((p) => chosen.add(p.id))
  }
  return chosen
}

// Within-group prevalence rank for every (visible) phenotype: 0 = most prevalent
// in its block, 1 = next, ... Used for PROGRESSIVE label reveal on zoom — a node
// is labeled when its rank is below a zoom-dependent cutoff, so zooming in only
// ADDS lower-ranked labels (never reshuffles the ones already shown). Hidden
// phenotypes (failing isVisible) get no entry and are therefore never labeled.
export function labelRanks<T extends LabelablePhenotype>(
  phenotypes: T[],
  opts: { blocks?: string[]; isVisible?: (p: T) => boolean },
): Map<number, number> {
  const { blocks, isVisible } = opts
  const visible = isVisible ? phenotypes.filter(isVisible) : phenotypes

  const groups = new Map<string, T[]>()
  for (const p of visible) {
    const key = blocks ? (blocks[p.id] ?? '__ungrouped__') : '__all__'
    const bucket = groups.get(key)
    if (bucket) bucket.push(p)
    else groups.set(key, [p])
  }

  const ranks = new Map<number, number>()
  for (const bucket of groups.values()) {
    bucket
      .slice()
      .sort((a, b) => b.corpus_prevalence - a.corpus_prevalence || a.id - b.id)
      .forEach((p, i) => ranks.set(p.id, i))
  }
  return ranks
}

// ---------------------------------------------------------------------------
// Spatially-unbiased label sampling
//
// Ranking labels by prevalence (labelRanks above) makes the labeled set clump
// on the high-prevalence UMAP core — the most prevalent topics sit near each
// other, so their labels pile up in one region and the rest of the map is left
// unlabeled. selectLabels instead draws a STABLE RANDOM sample restricted to the
// current viewport: random order removes the prevalence/spatial bias, so labels
// spread across the whole map, and the viewport filter + zoom-scaled budget make
// zooming reveal labels for the region you are actually looking at.
//
// "Stable" is load-bearing: the sample must not reshuffle when the covariate
// controls move (that would flicker labels every frame). We therefore derive
// each label's priority from a deterministic hash of its id — a fixed function
// of identity alone, independent of the live coverage reader — so the only thing
// that changes the visible set is pan/zoom.

// Deterministic pseudo-random priority in [0, 1) from an integer id (+ optional
// seed). Uses the standard two-round integer bit-mix (a "fmix32"-style avalanche,
// as in MurmurHash3's finalizer, Appleby 2011) so consecutive ids scatter across
// the unit interval instead of tracking id order.
export function labelHash(id: number, seed = 0): number {
  let h = ((id | 0) ^ (seed | 0)) >>> 0
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b) >>> 0
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b) >>> 0
  h = (h ^ (h >>> 16)) >>> 0
  return h / 0x100000000
}

// A phenotype eligible for a text label, with its pre-zoom position in the map's
// g-coordinate space (cx, cy) — i.e. the node translate before the zoom
// transform is applied.
export interface LabelCandidate {
  id: number
  cx: number
  cy: number
}

export interface SelectLabelsOptions {
  // Current d3 zoom transform (screen = k * g + offset).
  transform: { k: number; x: number; y: number }
  // Viewport extent in screen/viewBox units (a candidate is "in view" when its
  // transformed position lands within [−margin, width+margin] × [−margin, ...]).
  width: number
  height: number
  // Base fraction of the in-view candidates to label at k=1. The effective
  // fraction is min(1, baseFraction * k), so zooming in labels a larger share of
  // the (shrinking) in-view set — "rounded up" via ceil so a sparse view always
  // keeps at least one label.
  baseFraction: number
  margin?: number
  seed?: number
}

// Choose which candidates to label: the lowest-hash in-view candidates, up to a
// zoom-scaled budget. The returned set is a pure function of (positions,
// transform, viewport, baseFraction, seed) — it does not depend on input order
// or on any live prevalence/coverage value.
export function selectLabels(
  candidates: LabelCandidate[],
  opts: SelectLabelsOptions,
): Set<number> {
  const { transform: t, width, height, baseFraction, margin = 0, seed = 0 } = opts
  const inView = candidates
    .filter((c) => {
      const sx = t.k * c.cx + t.x
      const sy = t.k * c.cy + t.y
      return sx >= -margin && sx <= width + margin && sy >= -margin && sy <= height + margin
    })
    .map((c) => ({ id: c.id, h: labelHash(c.id, seed) }))
    .sort((a, b) => a.h - b.h)

  const frac = Math.min(1, baseFraction * t.k)
  // Round the budget UP (a sparse view keeps ≥1 label), but subtract a tiny
  // epsilon first so floating-point dust (e.g. 0.2*3*10 = 6.0000000000000001)
  // doesn't spuriously bump an exact integer to the next whole label.
  const target = Math.ceil(frac * inView.length - 1e-9)
  return new Set(inView.slice(0, target).map((c) => c.id))
}
