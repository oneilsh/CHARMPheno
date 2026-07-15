import { describe, it, expect } from 'vitest'
import { parseRoute } from './router'

describe('parseRoute', () => {
  it('parses a two-level hash', () => {
    expect(parseRoute('#/atlas/compare')).toEqual({ top: 'atlas', sub: 'compare' })
    expect(parseRoute('#/sim/simulate')).toEqual({ top: 'sim', sub: 'simulate' })
  })
  it('defaults the subtab to the first for the top tab', () => {
    expect(parseRoute('#/atlas')).toEqual({ top: 'atlas', sub: 'explore' })
    expect(parseRoute('#/sim')).toEqual({ top: 'sim', sub: 'simulate' })
  })
  it('falls back to atlas/explore on empty or unknown top', () => {
    expect(parseRoute('')).toEqual({ top: 'atlas', sub: 'explore' })
    expect(parseRoute('#/nope/x')).toEqual({ top: 'atlas', sub: 'explore' })
  })
  it('redirects legacy single-segment hashes', () => {
    expect(parseRoute('#/patient')).toEqual({ top: 'sim', sub: 'explore' })
    expect(parseRoute('#/simulator')).toEqual({ top: 'sim', sub: 'simulate' })
  })
  it('falls back an unknown sub to the top tab default', () => {
    expect(parseRoute('#/atlas/bogus')).toEqual({ top: 'atlas', sub: 'explore' })
  })
})
