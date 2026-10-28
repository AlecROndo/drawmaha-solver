import { describe, expect, it } from 'vitest'

import { fraction, stackLabels } from './scales'

describe('stacked labels', () => {
  it('leaves labels that already clear each other where they are, in the input order', () => {
    expect(stackLabels([10, 60, 35], 14)).toEqual([10, 60, 35])
  })

  it('pushes a label down until it clears the one above by the gap, and the next down after it', () => {
    expect(stackLabels([100, 104, 106], 14)).toEqual([100, 114, 128])
    expect(stackLabels([50, 20, 22], 15)).toEqual([50, 20, 35])
  })

  it('handles one label and none', () => {
    expect(stackLabels([7], 14)).toEqual([7])
    expect(stackLabels([], 14)).toEqual([])
  })
})

describe('a ratio as a fraction never overstates', () => {
  it('floors the reciprocal', () => {
    expect(fraction(0.117)).toBe('an eighth')
    expect(fraction(0.5)).toBe('a half')
    expect(fraction(0.08)).toBe('a 12th')
  })
})
