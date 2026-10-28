import { describe, expect, it } from 'vitest'

import { fraction, hoursAt, hoursOf, stackLabels } from './scales'

describe('stacked labels', () => {
  it('leaves labels that already clear each other where they are, in the input order', () => {
    expect(stackLabels([10, 60, 35], 14)).toEqual([10, 60, 35])
  })

  it('pushes a label down until it clears the one above by the gap, and the next down after it', () => {
    expect(stackLabels([100, 104, 106], 14)).toEqual([100, 114, 128])
    expect(stackLabels([50, 20, 22], 15)).toEqual([50, 20, 35])
  })

  it('slides a stack that would run past the floor back up, keeping its spacing', () => {
    expect(stackLabels([300, 304], 14, 306)).toEqual([292, 306])
    expect(stackLabels([100, 104], 14, 306)).toEqual([100, 114])
  })

  it('handles one label and none', () => {
    expect(stackLabels([7], 14)).toEqual([7])
    expect(stackLabels([], 14)).toEqual([])
  })
})

describe('hours from a wall clock', () => {
  it('prints one decimal under ten hours and a whole number from ten up', () => {
    expect(hoursOf(1440)).toBe('0.4')
    expect(hoursOf(9.94 * 3600)).toBe('9.9')
    expect(hoursOf(75_600)).toBe('21')
    expect(hoursAt(75_600)).toBe('21 h')
  })

  it('decides the branch on the rounded value, so 9.97 hours reads 10, never 10.0', () => {
    expect(hoursOf(35_900)).toBe('10')
    expect(hoursAt(35_900)).toBe('10 h')
  })
})

describe('a ratio as a fraction never overstates', () => {
  it('floors the reciprocal', () => {
    expect(fraction(0.117)).toBe('an eighth')
    expect(fraction(0.5)).toBe('a half')
    expect(fraction(0.08)).toBe('a 12th')
  })
})
