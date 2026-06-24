import { describe, it, expect, vi, beforeEach } from 'vitest'
import { loadBundle, loadManifest } from './bundle'

describe('loadBundle', () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn((url: string) => {
      const stubs: Record<string, unknown> = {
        'data/cancer/model.json':         { K: 2, V: 3, alpha: [0.1, 0.1], beta: [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]] },
        'data/cancer/phenotypes.json':    { phenotypes: [] },
        'data/cancer/vocab.json':         { codes: [] },
        'data/cancer/corpus_stats.json':  { corpus_size_docs: 10, mean_codes_per_doc: 5, k: 2, v: 3, v_full: 3 },
        'data/cd/model.json':             { K: 2, V: 2, alpha: [0.5, 0.5], beta: [[0.6, 0.4], [0.3, 0.7]] },
        'data/cd/phenotypes.json':        { phenotypes: [] },
        'data/cd/vocab.json':             { codes: [] },
        'data/cd/corpus_stats.json':      { corpus_size_docs: 10, mean_codes_per_doc: 5, k: 2, v: 2, v_full: 2 },
        'data/cd/covariate_effects.json': [{ covariate: 'Intercept', per_topic: [0.1, 0.2] }],
        'data/cd/covariate_schema.json':  { k: 20, controls: [], design_columns: [], unsupported: [] },
        'data/manifest.json':             { default: 'cancer', cohorts: [{ id: 'cancer', label: 'Cancer', description: 'desc' }] },
      }
      const key = Object.keys(stubs).find((k) => url.endsWith(k))
      if (!key) return Promise.resolve({ ok: false, status: 404 } as Response)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(stubs[key]) } as Response)
    }) as any
  })

  it('loads all four files for a given cohort id', async () => {
    const b = await loadBundle('/', 'cancer')
    expect(b.model.K).toBe(2)
    expect(b.corpusStats.v_full).toBe(3)
  })

  it('loads the cohort manifest', async () => {
    const m = await loadManifest('/')
    expect(m.default).toBe('cancer')
    expect(m.cohorts[0].id).toBe('cancer')
  })

  it('loads covariate schema + effects when present', async () => {
    const b = await loadBundle('/', 'cd')
    expect(b.covariateSchema?.k).toBe(20)
    expect(b.covariateEffects?.length).toBe(1)
  })

  it('leaves covariate fields undefined for non-STM bundles (404)', async () => {
    const b = await loadBundle('/', 'cancer')
    expect(b.covariateSchema).toBeUndefined()
    expect(b.covariateEffects).toBeUndefined()
  })

  it('treats a SPA-fallback HTML body (ok:200, non-JSON) as an absent optional file', async () => {
    // Vite's dev server serves index.html with status 200 for a missing file
    // under public/, so an optional bundle file that doesn't exist arrives as
    // HTML rather than a 404. r.json() rejects on that body; loadBundle must
    // treat it as absent, not fail the whole bundle.
    globalThis.fetch = vi.fn((url: string) => {
      const required: Record<string, unknown> = {
        'data/cancer/model.json':        { K: 1, V: 1, alpha: [1], beta: [[1]] },
        'data/cancer/phenotypes.json':   { phenotypes: [] },
        'data/cancer/vocab.json':        { codes: [] },
        'data/cancer/corpus_stats.json': { corpus_size_docs: 1, mean_codes_per_doc: 1, k: 20, v: 1, v_full: 1 },
      }
      const key = Object.keys(required).find((k) => url.endsWith(k))
      if (key) return Promise.resolve({ ok: true, json: () => Promise.resolve(required[key]) } as Response)
      // Missing optional file -> SPA fallback: 200 OK with an HTML body.
      return Promise.resolve({
        ok: true,
        json: () => Promise.reject(new SyntaxError("Unexpected token '<'")),
      } as unknown as Response)
    }) as any
    const b = await loadBundle('/', 'cancer')
    expect(b.model.K).toBe(1)
    expect(b.covariateSchema).toBeUndefined()
    expect(b.covariateEffects).toBeUndefined()
    expect(b.gating).toBeUndefined()
  })
})

describe('loadBundle gating', () => {
  it('attaches gating when gating.json is present', async () => {
    const files: Record<string, unknown> = {
      'data/c/model.json': { K: 1, V: 1, alpha: [1], beta: [[1]] },
      'data/c/phenotypes.json': { phenotypes: [] },
      'data/c/vocab.json': { codes: [] },
      'data/c/corpus_stats.json': { corpus_size_docs: 0, mean_codes_per_doc: 0, k: 20, v: 1, v_full: 1 },
      'data/c/gating.json': { group_var: 'source_cohort', groups: ['rare_dx'], topic_blocks: ['background'] },
    }
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      const key = Object.keys(files).find((k) => url.endsWith(k))
      return key ? { ok: true, json: async () => files[key] } : { ok: false, status: 404 }
    }))
    const b = await loadBundle('', 'c')
    expect(b.gating?.groups).toEqual(['rare_dx'])
  })
})
