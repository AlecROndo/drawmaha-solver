/**
 * The exact ranges, against the real export. These tests read
 * `public/mini/` from disk through an injected fetcher, so they are the one
 * place the whole chain is exercised: manifest, key tables, chunks, the
 * canonical key, the reach weighting and the showdown equities. They are
 * skipped when the export has not been written (`uv run minidraw-ranges`).
 */

import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

import { canonicalPicture } from './canonical'
import { parseMini } from './cards'
import { MiniData, type Fetcher } from './data'
import { pointFor } from './points'
import { equities, rangeAt, reachRange } from './range'
import { outerCategory, outerScore } from './ranking'

const HERE = dirname(fileURLToPath(import.meta.url))
const EXPORT = join(HERE, '..', '..', 'public', 'mini')
const present = existsSync(join(EXPORT, 'index.json'))

const diskFetcher: Fetcher = async (url) => {
  const path = join(EXPORT, url.replace(/^mini\//, ''))
  if (!existsSync(path)) return new Response(null, { status: 404, statusText: 'Not Found' })
  return new Response(new Uint8Array(readFileSync(path)))
}

const load = () => MiniData.load('mini/', diskFetcher)
const sum = (a: Float32Array) => a.reduce((s, v) => s + v, 0)

describe.skipIf(!present)('the real export', () => {
  it('has 141 public points and the three key shapes of the census', async () => {
    const data = await load()
    expect(data.index.points).toHaveLength(141)
    expect(data.shapeSize('b1d0')).toBe(970)
    expect(data.shapeSize('b2d0')).toBe(10170)
    expect(data.shapeSize('b2d1')).toBe(100400)
  })
  it('every row of the opening point sums to the scale', async () => {
    const data = await load()
    const bytes = await data.point(0)
    const w = data.index.points[0].actions.length
    for (let i = 0; i < data.index.points[0].n; i++) {
      let s = 0
      for (let k = 0; k < w; k++) s += bytes[i * w + k]
      expect(s).toBe(data.index.scale)
    }
  })
})

describe.skipIf(!present)('reachRange', () => {
  const b1 = parseMini('4h')

  it('at the opening, seat 0 holds every 3-subset of the 14 live cards with equal weight', async () => {
    const data = await load()
    const r = await rangeAt(data, { board: [b1], lines: [''], draws: [], player: 0 }, [])
    expect(r.n).toBe(364)
    for (let i = 0; i < r.n; i++) expect(r.weight[i]).toBeCloseTo(1 / 364, 9)
    expect(sum(r.weight)).toBeCloseTo(1, 6)
    for (let i = 0; i < r.n; i++) {
      for (let k = 0; k < 3; k++) expect(r.cards[i * 4 + k]).not.toBe(b1)
      expect(r.cards[i * 4 + 3]).toBe(255)
    }
  })

  it('seat 1 at the opening is the non-actor and is uniform too, with no mix', async () => {
    const data = await load()
    const r = await reachRange(data, { board: [b1], lines: [''], draws: [], player: 0 }, 1, [])
    expect(r.n).toBe(364)
    expect(r.mix).toHaveLength(0)
    for (let i = 0; i < r.n; i++) expect(r.weight[i]).toBeCloseTo(1 / 364, 9)
  })

  it('after seat 0 bets, seat 0’s range at seat 1’s decision is its opening mix’s pot column, normalised', async () => {
    const data = await load()
    const open = await rangeAt(data, { board: [b1], lines: [''], draws: [], player: 0 }, [])
    const after = await reachRange(data, { board: [b1], lines: ['p'], draws: [], player: 1 }, 0, [])
    const pot = open.point.actions.indexOf('p')
    const w = open.point.actions.length
    let total = 0
    for (let i = 0; i < open.n; i++) total += open.mix[i * w + pot]
    expect(after.n).toBe(open.n)
    for (let i = 0; i < open.n; i++) expect(after.weight[i]).toBeCloseTo(open.mix[i * w + pot] / total, 6)
  })

  it('excluding a card removes every state holding it, for the other seat', async () => {
    const data = await load()
    const six = parseMini('6c')
    const r = await reachRange(data, { board: [b1], lines: ['p'], draws: [], player: 1 }, 0, [six])
    expect(r.n).toBe(286)
    for (let i = 0; i < r.n; i++) for (let k = 0; k < 3; k++) expect(r.cards[i * 4 + k]).not.toBe(six)
    expect(sum(r.weight)).toBeCloseTo(1, 6)
  })

  it('at round 2 a thrower has 2,860 states and a stand-pat seat 286, each summing to one', async () => {
    const data = await load()
    const spot = { board: [b1, parseMini('2c')], lines: ['pc', ''], draws: [1, 0], player: 0 as const }
    const thrower = await rangeAt(data, spot, [])
    const pat = await reachRange(data, spot, 1, [])
    expect(thrower.n).toBe(2860)
    expect(pat.n).toBe(286)
    expect(sum(thrower.weight)).toBeCloseTo(1, 5)
    expect(sum(pat.weight)).toBeCloseTo(1, 5)
    for (let i = 0; i < thrower.n; i++) expect(thrower.cards[i * 4 + 3]).not.toBe(255)
    for (let i = 0; i < pat.n; i++) expect(pat.cards[i * 4 + 3]).toBe(255)
  })

  it('a thrower’s post-draw state sums over the three pre-draw holes it can come from', async () => {
    const data = await load()
    const spot = { board: [b1, parseMini('2c')], lines: ['xx', ''], draws: [1, 0], player: 0 as const }
    const r = await rangeAt(data, spot, [])
    const open = pointFor(data.index, { board: [b1], lines: [''], draws: [], player: 0 })
    const draw = pointFor(data.index, { board: [b1], lines: ['xx'], draws: [], player: 0 })
    const openBytes = await data.point(open.id)
    const drawBytes = await data.point(draw.id)
    const scale = data.index.scale
    // A state's unnormalised reach by hand: for each of the three kept pairs,
    // the pre-draw hole's chance of checking at the opening times its chance
    // of throwing the discard at the draw, the discard's column being its
    // position in the pre-draw canonical order.
    const byHand = (i: number): number => {
      const hole = [r.cards[i * 4], r.cards[i * 4 + 1], r.cards[i * 4 + 2]]
      const discard = r.cards[i * 4 + 3]
      let reach = 0
      for (let drop = 0; drop < 3; drop++) {
        const pre = [discard, ...hole.filter((_, k) => k !== drop)]
        const { key, relabelling } = canonicalPicture({ hole: pre, thrown: [], board: [b1] })
        const row = data.keyIndex('b1d0', key)
        const pCheck = openBytes[row * open.actions.length + open.actions.indexOf('c')] / scale
        const order = [...pre].sort((a, c) => Math.floor(a / 3) - Math.floor(c / 3) || relabelling[a % 3] - relabelling[c % 3])
        const pThrow = drawBytes[row * draw.actions.length + draw.actions.indexOf('lmt'[order.indexOf(discard)])] / scale
        reach += pCheck * pThrow
      }
      return reach
    }
    // Every state carries the same chance factors, so the ratio of two weights is the ratio of the hand-built reaches.
    let a = -1
    let b = -1
    for (let i = 0; i < r.n && b < 0; i++) {
      if (byHand(i) > 0) {
        if (a < 0) a = i
        else b = i
      }
    }
    expect(a).toBeGreaterThanOrEqual(0)
    expect(b).toBeGreaterThan(a)
    expect(r.weight[a] / r.weight[b]).toBeCloseTo(byHand(a) / byHand(b), 4)
    expect(sum(r.weight)).toBeCloseTo(1, 5)
  })

  it('pointFor refuses a line that is not a decision', async () => {
    const data = await load()
    expect(() => pointFor(data.index, { board: [b1], lines: ['xf'], draws: [], player: 0 })).toThrow()
    expect(() => pointFor(data.index, { board: [b1], lines: ['xx'], draws: [], player: 1 })).toThrow()
  })

  it('the outer column at round 2 is the outer category of the best two hole cards with the board', async () => {
    const data = await load()
    // On 4♥ 2♣ an outer high card cannot exist (three distinct hole ranks from {3,5,6} always pair or run with the
    // board), so the board here is 6♦ 2♣: holes of 3, 4, 5 miss every pair, run and suit.
    const board = [parseMini('6d'), parseMini('2c')]
    const r = await rangeAt(data, { board, lines: ['xx', ''], draws: [0, 0], player: 0 }, [])
    let highCards = 0
    for (let i = 0; i < r.n; i++) {
      const hole = [r.cards[i * 4], r.cards[i * 4 + 1], r.cards[i * 4 + 2]]
      expect(r.col[i]).toBe(outerCategory(outerScore(hole, board)) - 1)
      if (r.col[i] === 0) highCards++
    }
    // Outer high card is rare but real: the deck has five ranks, so holes avoiding every pair and run with 4 and 2 exist.
    expect(highCards).toBeGreaterThan(0)
  })
})

describe.skipIf(!present)('equities', () => {
  const b1 = parseMini('4h')

  it('with one board card, only the inner half is scored, and win + lose + tie = 1', async () => {
    const data = await load()
    const spot = { board: [b1], lines: ['p'], draws: [], player: 1 as const }
    const mine = await rangeAt(data, spot, [])
    const theirs = await reachRange(data, spot, 0, [])
    const eq = equities(mine, theirs)
    expect(eq.outer).toBeNull()
    expect(eq.scoop).toBeNull()
    for (let i = 0; i < mine.n; i++) expect(eq.inner[i]).toBeGreaterThanOrEqual(0)
    for (let i = 0; i < mine.n; i++) expect(eq.inner[i]).toBeLessThanOrEqual(1)
    // Against itself, with the same weights, a range's mean inner equity is one half.
    const self = equities(mine, mine)
    let m = 0
    for (let i = 0; i < mine.n; i++) m += mine.weight[i] * self.inner[i]
    expect(m).toBeCloseTo(0.5, 2)
  })

  it('with both board cards, both halves and the scoop are scored and the scoop never exceeds either half', async () => {
    const data = await load()
    const spot = { board: [b1, parseMini('2c')], lines: ['pc', ''], draws: [1, 0], player: 0 as const }
    const mine = await rangeAt(data, spot, [])
    const theirs = await reachRange(data, spot, 1, [])
    const eq = equities(mine, theirs)
    expect(eq.outer).not.toBeNull()
    expect(eq.scoop).not.toBeNull()
    for (let i = 0; i < mine.n; i++) {
      if (Number.isNaN(eq.inner[i])) continue
      expect(eq.scoop![i]).toBeLessThanOrEqual(eq.inner[i] + 1e-6)
      expect(eq.scoop![i]).toBeLessThanOrEqual(eq.outer![i] + 1e-6)
    }
  })
})
