import { defineConfig } from 'vitest/config'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  base: process.env.VITE_BASE ?? '/CHARMPheno/',
  // Vitest needs Svelte 5's browser (DOM) build so @testing-library/svelte's
  // render() works (SSR build throws lifecycle_function_unavailable). Scope the
  // override to the test run only; the production build keeps Vite's default
  // resolve conditions unchanged.
  ...(process.env.VITEST ? { resolve: { conditions: ['browser'] } } : {}),
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
