import { writable } from 'svelte/store'

// Two-level hash routing: `#/<top>/<sub>`. TOP_TABS drives the top nav
// (Tabs.svelte); SUBTABS drives the second-level nav (SubTabs.svelte). App.svelte
// renders the active top component; each top component renders its active subtab.
export const TOP_TABS = [
  { id: 'atlas', label: 'Phenotype Atlas' },
  { id: 'sim', label: 'Simulator' },
] as const

export type TopId = (typeof TOP_TABS)[number]['id']

export const SUBTABS: Record<TopId, readonly { id: string; label: string }[]> = {
  atlas: [
    { id: 'explore', label: 'Explore' },
    { id: 'compare', label: 'Compare' },
  ],
  sim: [
    { id: 'simulate', label: 'Simulate Cohort' },
    { id: 'explore', label: 'Explore Cohort' },
  ],
}

const TOP_IDS = TOP_TABS.map((t) => t.id) as readonly string[]

// Legacy single-segment hashes from the old three-tab layout.
const LEGACY: Record<string, { top: TopId; sub: string }> = {
  atlas: { top: 'atlas', sub: 'explore' },
  patient: { top: 'sim', sub: 'explore' },
  simulator: { top: 'sim', sub: 'simulate' },
}

export function parseRoute(hash: string): { top: TopId; sub: string } {
  const path = hash.replace(/^#\/?/, '')
  const [rawTop, rawSub] = path.split('/')
  if (rawTop && !rawSub && LEGACY[rawTop]) return LEGACY[rawTop]
  const top = (TOP_IDS.includes(rawTop) ? rawTop : 'atlas') as TopId
  const subs = SUBTABS[top].map((s) => s.id)
  const sub = subs.includes(rawSub) ? rawSub : subs[0]
  return { top, sub }
}

function current() {
  return parseRoute(typeof window === 'undefined' ? '' : window.location.hash)
}

const first = current()
export const topRoute = writable<TopId>(first.top)
export const subRoute = writable<string>(first.sub)

if (typeof window !== 'undefined') {
  window.addEventListener('hashchange', () => {
    const r = current()
    topRoute.set(r.top)
    subRoute.set(r.sub)
  })
}

export function go(top: TopId, sub?: string): void {
  const subs = SUBTABS[top].map((s) => s.id)
  const s = sub && subs.includes(sub) ? sub : subs[0]
  window.location.hash = `#/${top}/${s}`
}
