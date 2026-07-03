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
