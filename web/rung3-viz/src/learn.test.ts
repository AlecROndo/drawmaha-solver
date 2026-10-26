import { describe, expect, it } from 'vitest'
import { formatHash, parseHash } from './learn'

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
