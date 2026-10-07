import { describe, expect, it } from 'vitest'
import { deckStats, density, drawTable, examples, gridView, percentileOf, ribbon } from './aggregate'
import { parseBoard, parseCard, rankOf } from './cards'
import { maskFor } from './filter'
import { policyFor, throwCounts } from './policy'
import { sampleRange } from './sample'

const FLOP = parseBoard('Ks 9s 4d')
const s = sampleRange(FLOP, 5000, 11)
const pol = policyFor(s, 'facing')
const held = maskFor(s, { rows: null, cols: null, held: [parseCard('As')] })!

const sumMix = (m: { f: number; c: number; p: number }) => m.f + m.c + m.p

describe('gridView', () => {
  const g = gridView(s, pol, null)
  it('the whole range has share 1 and a mix that sums to 1', () => {
    expect(g.all.share).toBe(1)
    expect(g.all.n).toBe(s.n)
    expect(sumMix(g.all)).toBeCloseTo(1, 5)
  })
  it('is 9 rows by 8 columns', () => {
    expect(g.cells).toHaveLength(9)
    for (const r of g.cells) expect(r).toHaveLength(8)
    expect(g.rows).toHaveLength(9)
    expect(g.cols).toHaveLength(8)
  })
  it('cells add up to the whole, marginals add up to their lines', () => {
    let total = 0
    for (let r = 0; r < 9; r++) {
      let rowShare = 0
      for (let c = 0; c < 8; c++) {
        total += g.cells[r][c].share
        rowShare += g.cells[r][c].share
      }
      expect(rowShare).toBeCloseTo(g.rows[r].share, 9)
    }
    expect(total).toBeCloseTo(g.all.share, 9)
    for (let c = 0; c < 8; c++) {
      let colShare = 0
      for (let r = 0; r < 9; r++) colShare += g.cells[r][c].share
      expect(colShare).toBeCloseTo(g.cols[c].share, 9)
    }
  })
  it('an empty cell has share 0 and a zero mix, never NaN', () => {
    const empty = g.cells.flat().filter((c) => c.n === 0)
    expect(empty.length).toBeGreaterThan(0)
    for (const c of empty) {
      expect(c.share).toBe(0)
      expect(c.f).toBe(0)
      expect(c.c).toBe(0)
      expect(c.p).toBe(0)
    }
    for (const c of g.cells.flat()) if (c.n > 0) expect(sumMix(c)).toBeCloseTo(1, 5)
  })
  it('a mask shrinks the share of the whole but keeps it a share of the full sample', () => {
    const h = gridView(s, pol, held)
    let kept = 0
    for (let i = 0; i < s.n; i++) kept += held[i]
    expect(h.all.n).toBe(kept)
    expect(h.all.share).toBeCloseTo(kept / s.n, 9)
    let total = 0
    for (const c of h.cells.flat()) total += c.share
    expect(total).toBeCloseTo(h.all.share, 9)
  })
  it('still sums to one on a paired flop', () => {
    const paired = sampleRange(parseBoard('Ks Kd 4c'), 3000, 2)
    const g2 = gridView(paired, policyFor(paired, 'facing'), null)
    let total = 0
    for (const c of g2.cells.flat()) total += c.share
    expect(total).toBeCloseTo(1, 9)
    for (const c of g2.cells.flat()) expect(Number.isNaN(c.p)).toBe(false)
    for (const c of g2.cols) expect(Number.isNaN(c.p)).toBe(false)
  })
})

describe('ribbon', () => {
  it('returns one mix per bin, each summing to 1', () => {
    for (const key of ['total', 'inner', 'outer', 'scoop'] as const) {
      const r = ribbon(s, pol, null, key, 10)
      expect(r).toHaveLength(10)
      for (const m of r) expect(sumMix(m)).toBeCloseTo(1, 5)
    }
  })
  it('pots more at the strong end than the weak end', () => {
    const r = ribbon(s, pol, null, 'total', 10)
    expect(r[9].p).toBeGreaterThan(r[0].p)
    expect(r[0].f).toBeGreaterThan(r[9].f)
  })
  it('returns one bin per hand when the mask holds fewer hands than bins', () => {
    const m = new Uint8Array(s.n)
    m[3] = 1
    m[40] = 1
    const r = ribbon(s, pol, m, 'total', 10)
    expect(r).toHaveLength(2)
    expect(r[0].p).toBeCloseTo(Math.min(pol.p[3], pol.p[40]), 5)
  })
  it('is empty when no hand passes', () => {
    expect(ribbon(s, pol, new Uint8Array(s.n), 'total', 10)).toEqual([])
  })
})

describe('percentileOf', () => {
  it('lies in [0, 1] and is near 1 for a monster, near 0 for a blank', () => {
    const monster = percentileOf(s, pol, 'total', [parseCard('Kh'), parseCard('Kc'), parseCard('9h'), parseCard('9c'), parseCard('As')], FLOP)
    const blank = percentileOf(s, pol, 'inner', [parseCard('2h'), parseCard('3c'), parseCard('5d'), parseCard('7h'), parseCard('8c')], FLOP)
    expect(monster).toBeGreaterThan(0.95)
    expect(monster).toBeLessThanOrEqual(1)
    expect(blank).toBeLessThan(0.05)
    expect(blank).toBeGreaterThanOrEqual(0)
  })
  it('for a sampled hand, equals the fraction of the sample with a lower key', () => {
    const i = 77
    const hole = Array.from(s.cards.subarray(i * 5, i * 5 + 5))
    let below = 0
    for (let j = 0; j < s.n; j++) if (s.outerEquity[j] < s.outerEquity[i]) below++
    expect(percentileOf(s, pol, 'outer', hole, FLOP)).toBeCloseTo(below / s.n, 9)
  })
})

describe('density', () => {
  it('has bins² cells whose shares sum to 1', () => {
    const d = density(s, pol, null, 18)
    expect(d).toHaveLength(324)
    let total = 0
    for (const c of d) total += c.share
    expect(total).toBeCloseTo(1, 9)
    for (const c of d) if (c.n > 0) expect(sumMix(c)).toBeCloseTo(1, 5)
    for (const c of d) if (c.n === 0) expect(c.p).toBe(0)
  })
  it('puts a hand at x = outer, y = inner', () => {
    const d = density(s, pol, null, 4)
    const i = 0
    const x = Math.min(3, Math.floor(s.outerEquity[i] * 4))
    const y = Math.min(3, Math.floor(s.innerEquity[i] * 4))
    expect(d[y * 4 + x].n).toBeGreaterThan(0)
  })
})

describe('deckStats', () => {
  it('every hand holds five cards, so inRange sums to 5', () => {
    const d = deckStats(s, pol, null)
    expect(d.inRange).toHaveLength(52)
    expect(d.potLift).toHaveLength(52)
    let total = 0
    for (let c = 0; c < 52; c++) total += d.inRange[c]
    expect(total).toBeCloseTo(5, 4)
    for (const c of FLOP) {
      expect(d.inRange[c]).toBe(0)
      expect(d.potLift[c]).toBe(0)
    }
  })
  it('a held card is in every masked hand and lifts the pot share of its own rank', () => {
    const d = deckStats(s, pol, held)
    expect(d.inRange[parseCard('As')]).toBe(1)
    for (let c = 0; c < 52; c++) expect(Number.isNaN(d.potLift[c])).toBe(false)
  })
  it('an ace lifts the pot share above a deuce', () => {
    const d = deckStats(s, pol, null)
    expect(d.potLift[parseCard('Ah')]).toBeGreaterThan(d.potLift[parseCard('2h')])
  })
})

describe('drawTable', () => {
  it('has one line per inner row, shares summing to 1, counts summing to 1 where populated', () => {
    const t = drawTable(s, throwCounts(s), null)
    expect(t).toHaveLength(9)
    let total = 0
    for (const r of t) {
      total += r.share
      expect(r.counts).toHaveLength(6)
      const c = r.counts.reduce((a, b) => a + b, 0)
      if (r.share > 0) expect(c).toBeCloseTo(1, 5)
      else expect(c).toBe(0)
    }
    expect(total).toBeCloseTo(1, 9)
  })
  it('the made row stands pat', () => {
    const t = drawTable(s, throwCounts(s), null)
    expect(t[8].counts[0]).toBe(1)
  })
})

describe('examples', () => {
  it('returns the k hands of the cell that pot most, cards high to low', () => {
    const g = gridView(s, pol, null)
    let best: [number, number] = [0, 0]
    for (let r = 0; r < 9; r++) for (let c = 0; c < 8; c++) if (g.cells[r][c].n > g.cells[best[0]][best[1]].n) best = [r, c]
    const ex = examples(s, pol, null, best[0], best[1], 5)
    expect(ex).toHaveLength(5)
    for (let k = 1; k < ex.length; k++) expect(ex[k].mix.p).toBeLessThanOrEqual(ex[k - 1].mix.p)
    for (const e of ex) {
      expect(e.cards).toHaveLength(5)
      for (let k = 1; k < 5; k++) expect(rankOf(e.cards[k])).toBeLessThanOrEqual(rankOf(e.cards[k - 1]))
      expect(sumMix(e.mix)).toBeCloseTo(1, 5)
      expect(e.eqI).toBeGreaterThanOrEqual(0)
      expect(e.eqO).toBeLessThanOrEqual(1)
    }
  })
  it('returns fewer than k when the cell is smaller, and none for an empty cell', () => {
    const g = gridView(s, pol, null)
    const empty = g.cells.flat().findIndex((c) => c.n === 0)
    expect(examples(s, pol, null, Math.floor(empty / 8), empty % 8, 5)).toEqual([])
    const m = new Uint8Array(s.n)
    m[0] = 1
    expect(examples(s, pol, m, s.row[0], s.col[0], 5)).toHaveLength(1)
  })
})
