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

describe('loadBundle predictive_gain hydration', () => {
  const K3_PREDICTIVE_GAIN = {
    presence: [0.1, 0.5, 0.9],
    mean_gain: [0.01, 0.05, 0.09],
    depth: [0.2, null, 0.8],
    prominence_hist: [[1, 2], [3, null], [4, 5]],
    length_corr: [0.0, 0.1, -0.1],
    dedup_gain: [0.02, 0.03, 0.04],
    prominence_bin_edges: [0, 1, 2],
    null_band: { mean: 0, std: 1, n: 100, p95: 2, hist: [1, 2, 3] },
    observed_delta_range: [-1, 1] as [number, number],
    downdate_audit: { max_abs_overall: 0.01, n_docs_audited: 100 },
    scale: 1.0,
    n_docs: 100,
  }

  it('hydrates presence/mean_gain/depth/length_corr/dedup_gain/prominence_hist onto each phenotype BY INDEX', async () => {
    const files: Record<string, unknown> = {
      'data/pg/model.json': { K: 3, V: 1, alpha: [1, 1, 1], beta: [[1], [1], [1]] },
      'data/pg/phenotypes.json': {
        phenotypes: [
          { id: 0, label: 'A', description: '', quality: null, npmi: null, pair_coverage: null, corpus_prevalence: 0.1, original_topic_id: 0 },
          { id: 1, label: 'B', description: '', quality: null, npmi: null, pair_coverage: null, corpus_prevalence: 0.2, original_topic_id: 1 },
          { id: 2, label: 'C', description: '', quality: null, npmi: null, pair_coverage: null, corpus_prevalence: 0.3, original_topic_id: 2 },
        ],
        predictive_gain: K3_PREDICTIVE_GAIN,
      },
      'data/pg/vocab.json': { codes: [] },
      'data/pg/corpus_stats.json': { corpus_size_docs: 0, mean_codes_per_doc: 0, k: 3, v: 1, v_full: 1 },
    }
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      const key = Object.keys(files).find((k) => url.endsWith(k))
      return key ? { ok: true, json: async () => files[key] } : { ok: false, status: 404 }
    }))
    const b = await loadBundle('', 'pg')
    expect(b.phenotypes.phenotypes[0].presence).toBe(0.1)
    expect(b.phenotypes.phenotypes[1].presence).toBe(0.5)
    expect(b.phenotypes.phenotypes[2].presence).toBe(0.9)
    expect(b.phenotypes.phenotypes[0].mean_gain).toBe(0.01)
    expect(b.phenotypes.phenotypes[1].depth).toBe(null)
    expect(b.phenotypes.phenotypes[2].depth).toBe(0.8)
    expect(b.phenotypes.phenotypes[0].length_corr).toBe(0.0)
    expect(b.phenotypes.phenotypes[1].dedup_gain).toBe(0.03)
    expect(b.phenotypes.phenotypes[0].prominence_hist).toEqual([1, 2])
    expect(b.phenotypes.phenotypes[1].prominence_hist).toEqual([3, null])
    // Bundle-level object (and its diagnostics) stays available too.
    expect(b.phenotypes.predictive_gain?.prominence_bin_edges).toEqual([0, 1, 2])
    expect(b.phenotypes.predictive_gain?.scale).toBe(1.0)
  })

  it('leaves per-phenotype predictive_gain fields undefined when predictive_gain is absent (backward-compat)', async () => {
    const files: Record<string, unknown> = {
      'data/nopg/model.json': { K: 1, V: 1, alpha: [1], beta: [[1]] },
      'data/nopg/phenotypes.json': {
        phenotypes: [
          { id: 0, label: 'A', description: '', quality: null, npmi: null, pair_coverage: null, corpus_prevalence: 0.1, original_topic_id: 0 },
        ],
      },
      'data/nopg/vocab.json': { codes: [] },
      'data/nopg/corpus_stats.json': { corpus_size_docs: 0, mean_codes_per_doc: 0, k: 1, v: 1, v_full: 1 },
    }
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      const key = Object.keys(files).find((k) => url.endsWith(k))
      return key ? { ok: true, json: async () => files[key] } : { ok: false, status: 404 }
    }))
    const b = await loadBundle('', 'nopg')
    expect(b.phenotypes.phenotypes[0].presence).toBeUndefined()
    expect(b.phenotypes.phenotypes[0].depth).toBeUndefined()
    expect(b.phenotypes.phenotypes[0].prominence_hist).toBeUndefined()
    expect(b.phenotypes.predictive_gain).toBeUndefined()
  })

  it('skips hydration and warns on a per-topic array length mismatch', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const files: Record<string, unknown> = {
      'data/mismatch/model.json': { K: 2, V: 1, alpha: [1, 1], beta: [[1], [1]] },
      'data/mismatch/phenotypes.json': {
        phenotypes: [
          { id: 0, label: 'A', description: '', quality: null, npmi: null, pair_coverage: null, corpus_prevalence: 0.1, original_topic_id: 0 },
          { id: 1, label: 'B', description: '', quality: null, npmi: null, pair_coverage: null, corpus_prevalence: 0.2, original_topic_id: 1 },
        ],
        // Only one entry per array while there are 2 phenotypes -> mismatch.
        predictive_gain: { ...K3_PREDICTIVE_GAIN, presence: [0.1] },
      },
      'data/mismatch/vocab.json': { codes: [] },
      'data/mismatch/corpus_stats.json': { corpus_size_docs: 0, mean_codes_per_doc: 0, k: 2, v: 1, v_full: 1 },
    }
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      const key = Object.keys(files).find((k) => url.endsWith(k))
      return key ? { ok: true, json: async () => files[key] } : { ok: false, status: 404 }
    }))
    const b = await loadBundle('', 'mismatch')
    expect(b.phenotypes.phenotypes[0].presence).toBeUndefined()
    expect(b.phenotypes.phenotypes[1].presence).toBeUndefined()
    expect(warn).toHaveBeenCalled()
    warn.mockRestore()
  })
})

describe('loadBundle correlation', () => {
  it('attaches correlation when correlation.json is present', async () => {
    const files: Record<string, unknown> = {
      'data/c/model.json': { K: 2, V: 1, alpha: [1, 1], beta: [[1], [1]] },
      'data/c/phenotypes.json': { phenotypes: [] },
      'data/c/vocab.json': { codes: [] },
      'data/c/corpus_stats.json': { corpus_size_docs: 0, mean_codes_per_doc: 0, k: 20, v: 1, v_full: 1 },
      'data/c/correlation.json': {
        topic_order: [0, 1],
        block_labels: ['background', 'cancer'],
        R: [[1, 0.5], [0.5, 1]],
        identified: [[true, true], [true, true]],
        support: [[10, 8], [8, 10]],
      },
    }
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      const key = Object.keys(files).find((k) => url.endsWith(k))
      return key ? { ok: true, json: async () => files[key] } : { ok: false, status: 404 }
    }))
    const b = await loadBundle('', 'c')
    expect(b.correlation?.R).toEqual([[1, 0.5], [0.5, 1]])
    expect(b.correlation?.identified).toEqual([[true, true], [true, true]])
  })

  it('leaves correlation undefined when correlation.json is absent', async () => {
    const files: Record<string, unknown> = {
      'data/c/model.json': { K: 1, V: 1, alpha: [1], beta: [[1]] },
      'data/c/phenotypes.json': { phenotypes: [] },
      'data/c/vocab.json': { codes: [] },
      'data/c/corpus_stats.json': { corpus_size_docs: 0, mean_codes_per_doc: 0, k: 20, v: 1, v_full: 1 },
    }
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      const key = Object.keys(files).find((k) => url.endsWith(k))
      return key ? { ok: true, json: async () => files[key] } : { ok: false, status: 404 }
    }))
    const b = await loadBundle('', 'c')
    expect(b.correlation).toBeUndefined()
  })
})
