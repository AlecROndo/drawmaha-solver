import { describe, expect, it } from 'vitest'
import { addressOf, formatHash, parseHash, parseLocation } from './learn'

describe('parseLocation', () => {
  // Vercel rewrites /research, /learn and /play to this app's index.html, so
  // the path names the tab when the hash does not.
  it('opens the tab the path names', () => {
    expect(parseLocation('/play', '')).toMatchObject({ tab: 'play' })
    expect(parseLocation('/learn', '')).toMatchObject({ tab: 'learn' })
    expect(parseLocation('/research', '')).toMatchObject({ tab: 'research' })
    expect(parseLocation('/rung3', '')).toMatchObject({ tab: 'research' })
    expect(parseLocation('/rung3/', '')).toMatchObject({ tab: 'research' })
  })
  it('reads a trailing slash, and gives a bare /learn the Learn defaults', () => {
    expect(parseLocation('/play/', '')).toMatchObject({ tab: 'play' })
    expect(parseLocation('/learn', '')).toEqual({ tab: 'learn', view: 'range', strategy: 'tabular' })
  })
  it('lets the hash win when it names a tab', () => {
    expect(parseLocation('/rung3', '#play')).toMatchObject({ tab: 'play' })
    expect(parseLocation('/play', '#research')).toMatchObject({ tab: 'research' })
    expect(parseLocation('/research', '#learn/versus/deep-pi')).toEqual({ tab: 'learn', view: 'versus', strategy: 'deep-pi' })
  })
  it('keeps the path when the hash names no tab, such as a chapter anchor', () => {
    expect(parseLocation('/play', '#nope')).toMatchObject({ tab: 'play' })
    expect(parseLocation('/research', '#ch-03')).toMatchObject({ tab: 'research' })
  })
})

describe('addressOf', () => {
  const learn = { tab: 'learn', view: 'versus', strategy: 'deep-pi' } as const
  it('moves between the product paths when the page was opened at one', () => {
    expect(addressOf('/research', { ...learn, tab: 'play' })).toBe('/play')
    expect(addressOf('/play', learn)).toBe('/learn#learn/versus/deep-pi')
    expect(addressOf('/learn', { ...learn, tab: 'research' })).toBe('/research')
  })
  it('writes only the hash under /rung3, where the old links live', () => {
    expect(addressOf('/rung3/', { ...learn, tab: 'play' })).toBe('#play')
    expect(addressOf('/rung3', learn)).toBe('#learn/versus/deep-pi')
  })
  it('round-trips through parseLocation', () => {
    for (const path of ['/research', '/play', '/learn', '/rung3/']) {
      for (const tab of ['research', 'play', 'learn'] as const) {
        const page = { ...learn, tab }
        const url = new URL(addressOf(path, page), `https://drawmaha.app${path}`)
        expect(parseLocation(url.pathname, url.hash).tab).toBe(tab)
      }
    }
  })
})

describe('parseHash', () => {
  it('reads the two old tabs as before', () => {
    expect(parseHash('#research')).toEqual({ tab: 'research', view: 'range', strategy: 'tabular' })
    expect(parseHash('#play')).toEqual({ tab: 'play', view: 'range', strategy: 'tabular' })
    expect(parseHash('')).toEqual({ tab: 'research', view: 'range', strategy: 'tabular' })
  })
  it('reads the Learn tab with its view and strategy', () => {
    expect(parseHash('#learn')).toEqual({ tab: 'learn', view: 'range', strategy: 'tabular' })
    expect(parseHash('#learn/versus')).toEqual({ tab: 'learn', view: 'versus', strategy: 'tabular' })
    expect(parseHash('learn/range/deep-pi')).toEqual({ tab: 'learn', view: 'range', strategy: 'deep-pi' })
  })
  it('defaults every unknown segment rather than throwing', () => {
    expect(parseHash('#nope/nope/nope')).toEqual({ tab: 'research', view: 'range', strategy: 'tabular' })
    expect(parseHash('#learn/foo/bar')).toEqual({ tab: 'learn', view: 'range', strategy: 'tabular' })
  })
  it('drops a view and strategy after a tab that is not Learn, so parsing is idempotent under formatting', () => {
    const page = parseHash('#research/versus/deep-pi')
    expect(page).toEqual({ tab: 'research', view: 'range', strategy: 'tabular' })
    expect(parseHash(formatHash(page))).toEqual(page)
  })
})

describe('the page switching tabs', () => {
  // App.tsx's `show` writes formatHash of the next state and keeps the state
  // whole: leaving Learn for Play writes `#play` and loses nothing, so coming
  // back writes the same Learn hash.
  it('keeps the Learn view and strategy across a visit to another tab', () => {
    const learn = parseHash('#learn/versus/deep-pi')
    const play = { ...learn, tab: 'play' as const }
    expect(formatHash(play)).toBe('#play')
    expect(formatHash({ ...play, tab: 'learn' })).toBe('#learn/versus/deep-pi')
  })
})

describe('formatHash', () => {
  it('writes the tab alone for research and play', () => {
    expect(formatHash({ tab: 'play', view: 'versus', strategy: 'deep-pi' })).toBe('#play')
  })
  it('writes view and strategy for the Learn tab and round-trips', () => {
    const page = { tab: 'learn', view: 'versus', strategy: 'deep-exact' } as const
    expect(formatHash(page)).toBe('#learn/versus/deep-exact')
    expect(parseHash(formatHash(page))).toEqual(page)
  })
})
