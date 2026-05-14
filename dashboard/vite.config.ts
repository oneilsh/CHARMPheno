import { defineConfig } from 'vitest/config'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  base: process.env.VITE_BASE ?? '/CHARMPheno/',
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
