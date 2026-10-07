import { describe, expect, it } from 'vitest'
import { parseBoard, parseCard } from './cards'
import { EMPTY_FILTER, maskFor } from './filter'
import { sampleRange } from './sample'

const FLOP = parseBoard('Ks 9s 4d')
const s = sampleRange(FLOP, 2000, 5)
const AS = parseCard('As')

describe('maskFor', () => {
  it('the empty filter is no mask at all', () => {
    expect(maskFor(s, EMPTY_FILTER)).toBeNull()
    expect(maskFor(s, { rows: null, cols: null, held: [] })).toBeNull()
  })
  it('a held card keeps only the hands holding it', () => {
    const m = maskFor(s, { rows: null, cols: null, held: [AS] })
    expect(m).not.toBeNull()
    let kept = 0
    for (let i = 0; i < s.n; i++) {
      const has = s.cards.subarray(i * 5, i * 5 + 5).includes(AS)
      expect(m![i]).toBe(has ? 1 : 0)
      kept += m![i]
    }
    expect(kept).toBeGreaterThan(0)
    expect(kept).toBeLessThan(s.n)
  })
  it('a row set keeps only hands in those rows', () => {
    const m = maskFor(s, { rows: new Set([0, 8]), cols: null, held: [] })!
    for (let i = 0; i < s.n; i++) expect(m[i]).toBe(s.row[i] === 0 || s.row[i] === 8 ? 1 : 0)
  })
  it('rows, columns and held cards all have to pass', () => {
    const m = maskFor(s, { rows: new Set([3, 4, 5]), cols: new Set([3]), held: [AS] })!
    for (let i = 0; i < s.n; i++) {
      const ok = s.row[i] >= 3 && s.row[i] <= 5 && s.col[i] === 3 && s.cards.subarray(i * 5, i * 5 + 5).includes(AS)
      expect(m[i]).toBe(ok ? 1 : 0)
    }
  })
  it('an empty row set keeps nothing', () => {
    const m = maskFor(s, { rows: new Set(), cols: null, held: [] })!
    expect(m.reduce((a, b) => a + b, 0)).toBe(0)
  })
})
