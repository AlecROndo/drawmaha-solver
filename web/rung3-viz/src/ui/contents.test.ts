import { describe, expect, it } from 'vitest'
import { currentSection } from './contents'

describe('the contents rail picks the section the reader is at', () => {
  const line = 270 // 30% of a 900 px viewport

  it('marks nothing above the first section', () => {
    expect(currentSection([1000, 2200, 3100], line, false)).toBe(-1)
  })

  it('marks the last section whose top has crossed the reading line', () => {
    expect(currentSection([-400, 100, 900], line, false)).toBe(1)
    expect(currentSection([-2000, -600, 269], line, false)).toBe(2)
  })

  it('does not count a section whose top sits just below the line', () => {
    expect(currentSection([-100, 271], line, false)).toBe(0)
  })

  it('marks the last section once the page is at its end, even if its top never reached the line', () => {
    expect(currentSection([-5000, -900, 400], line, true)).toBe(2)
  })

  it('returns -1 for no sections, at the end or not', () => {
    expect(currentSection([], line, true)).toBe(-1)
    expect(currentSection([], line, false)).toBe(-1)
  })
})
