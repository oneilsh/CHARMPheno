// Vitest global setup. jsdom lacks a few browser APIs that Svelte's binding
// machinery relies on; stub the minimum here so component tests can mount.

// `bind:clientWidth` / `bind:clientHeight` install a ResizeObserver, which jsdom
// does not implement. A no-op stub is enough — tests assert on structure, not on
// measured layout (clientWidth stays 0, so size-dependent code takes its default).
if (typeof globalThis.ResizeObserver === 'undefined') {
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  ;(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = ResizeObserverStub
}
