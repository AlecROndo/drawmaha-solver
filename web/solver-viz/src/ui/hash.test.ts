import { describe, expect, it } from 'vitest'
import { formatHash, parseHash } from './hash'

describe('parseHash', () => {
  it('reads game, tab and strategy', () => {
    expect(parseHash('#mini/range/deep-pi')).toEqual({ game: 'mini', tab: 'range', strategy: 'deep-pi' })
    expect(parseHash('mini/versus/deep-exact')).toEqual({ game: 'mini', tab: 'versus', strategy: 'deep-exact' })
  })
  it('defaults a missing strategy to the tabular one, so links from before the switch still open', () => {
    expect(parseHash('#mini/versus')).toEqual({ game: 'mini', tab: 'versus', strategy: 'tabular' })
  })
  it('defaults every unknown segment rather than throwing', () => {
    expect(parseHash('#mini/range/foo')).toEqual({ game: 'mini', tab: 'range', strategy: 'tabular' })
    expect(parseHash('#nope/nope/nope')).toEqual({ game: 'mini', tab: 'range', strategy: 'tabular' })
    expect(parseHash('')).toEqual({ game: 'mini', tab: 'range', strategy: 'tabular' })
  })
})

describe('formatHash', () => {
  it('writes the strategy for mini-drawmaha and round-trips', () => {
    const view = { game: 'mini', tab: 'range', strategy: 'deep-pi' } as const
    expect(formatHash(view)).toBe('#mini/range/deep-pi')
    expect(parseHash(formatHash(view))).toEqual(view)
  })
  it('omits the strategy for the full game, which has no switch', () => {
    expect(formatHash({ game: 'full', tab: 'versus', strategy: 'deep-pi' })).toBe('#full/versus')
    expect(parseHash('#full/versus')).toEqual({ game: 'full', tab: 'versus', strategy: 'tabular' })
  })
})
