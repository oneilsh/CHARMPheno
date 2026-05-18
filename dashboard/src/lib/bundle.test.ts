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
        'data/manifest.json':             { default: 'cancer', cohorts: [{ id: 'cancer', label: 'Cancer', description: 'desc' }] },
      }
      const key = Object.keys(stubs).find((k) => url.endsWith(k))!
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
})
