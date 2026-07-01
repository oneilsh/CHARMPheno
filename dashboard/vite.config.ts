import { defineConfig } from 'vitest/config'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  base: process.env.VITE_BASE ?? '/CHARMPheno/',
  // Forces Vite to resolve Svelte's client (DOM) build instead of the
  // server (SSR) build under vitest/jsdom. Without this, mounting a
  // component in a test (@testing-library/svelte's `render`) throws
  // "lifecycle_function_unavailable: `mount(...)` is not available on
  // the server" — Svelte 5 uses import-conditions, not file heuristics,
  // to pick between the two runtimes.
  resolve: {
    conditions: ['browser'],
  },
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
