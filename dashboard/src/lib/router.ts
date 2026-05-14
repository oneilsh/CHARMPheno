import { writable } from 'svelte/store'

export type Route = 'atlas' | 'patient' | 'simulator'

function parseHash(): Route {
  const h = window.location.hash.replace(/^#\//, '')
  if (h === 'patient' || h === 'simulator') return h
  return 'atlas'
}

export const route = writable<Route>(parseHash())
window.addEventListener('hashchange', () => route.set(parseHash()))
export function go(to: Route): void { window.location.hash = `#/${to}` }
