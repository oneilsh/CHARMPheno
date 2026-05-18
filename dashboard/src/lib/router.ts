import { writable } from 'svelte/store'

// Single source of truth for the dashboard's top-level tabs. Both
// Tabs.svelte (renders the nav) and App.svelte (renders the active
// tab's component) iterate over this list, so adding a new tab only
// requires extending TABS plus importing the component in App.svelte.
export const TABS = [
  { id: 'atlas',     label: 'Phenotype Atlas' },
  { id: 'patient',   label: 'Patient Atlas' },
  { id: 'simulator', label: 'Simulator' },
] as const

export type Route = (typeof TABS)[number]['id']
const ROUTE_IDS = TABS.map((t) => t.id) as readonly string[]

function parseHash(): Route {
  const h = window.location.hash.replace(/^#\//, '')
  return ROUTE_IDS.includes(h) ? (h as Route) : 'atlas'
}

export const route = writable<Route>(parseHash())
window.addEventListener('hashchange', () => route.set(parseHash()))
export function go(to: Route): void { window.location.hash = `#/${to}` }
