import { describe, expect, it } from 'vitest'
import { parseMini } from './cards'
import type { MiniIndex, PointMeta } from './data'
import { describeSpot, pointFor } from './points'

/** A hand-written slice of the 141 public points: enough of one line to walk round 1, the draw and round 2. */
const POINTS: PointMeta[] = [
  { id: 0, player: 0, boardCards: 1, draws: [], lines: [''], discards: 0, draw: false, actions: ['c', 'p'], shape: 'b1d0', file: 'point-000.bin.gz', n: 970 },
  { id: 1, player: 1, boardCards: 1, draws: [], lines: ['x'], discards: 0, draw: false, actions: ['c', 'p'], shape: 'b1d0', file: 'point-001.bin.gz', n: 970 },
  { id: 2, player: 1, boardCards: 1, draws: [], lines: ['p'], discards: 0, draw: false, actions: ['f', 'c', 'p'], shape: 'b1d0', file: 'point-002.bin.gz', n: 970 },
  { id: 3, player: 0, boardCards: 1, draws: [], lines: ['xx'], discards: 0, draw: true, actions: ['n', 'l', 'm', 't'], shape: 'b1d0', file: 'point-003.bin.gz', n: 970 },
  { id: 4, player: 1, boardCards: 1, draws: [0], lines: ['xx'], discards: 0, draw: true, actions: ['n', 'l', 'm', 't'], shape: 'b1d0', file: 'point-004.bin.gz', n: 970 },
  { id: 5, player: 1, boardCards: 1, draws: [1], lines: ['xx'], discards: 0, draw: true, actions: ['n', 'l', 'm', 't'], shape: 'b1d0', file: 'point-005.bin.gz', n: 970 },
  { id: 6, player: 0, boardCards: 2, draws: [1, 0], lines: ['xx', ''], discards: 1, draw: false, actions: ['c', 'p'], shape: 'b2d1', file: 'point-006.bin.gz', n: 100400 },
  { id: 7, player: 1, boardCards: 2, draws: [1, 0], lines: ['xx', 'p'], discards: 0, draw: false, actions: ['f', 'c', 'p'], shape: 'b2d0', file: 'point-007.bin.gz', n: 10170 },
  { id: 8, player: 0, boardCards: 2, draws: [1, 1], lines: ['xx', ''], discards: 1, draw: false, actions: ['c', 'p'], shape: 'b2d1', file: 'point-008.bin.gz', n: 100400 },
]

const index: MiniIndex = {
  strategy: { sha256: '', rule: 'lcfr', column: 'linear', iteration: 2_000_000, workers: 10, seed: 0 },
  deck: { ranks: '23456', suits: 'cdh' },
  scale: 250,
  shapes: {},
  points: POINTS,
}

const b1 = parseMini('4h')
const b2 = parseMini('2c')

describe('pointFor', () => {
  it("finds P0's opening and P1's reply by the line", () => {
    expect(pointFor(index, { board: [b1], lines: [''], draws: [], player: 0 }).id).toBe(0)
    expect(pointFor(index, { board: [b1], lines: ['x'], draws: [], player: 1 }).id).toBe(1)
    expect(pointFor(index, { board: [b1], lines: ['p'], draws: [], player: 1 }).id).toBe(2)
  })
  it("tells the two seats' draw points apart by the draws present", () => {
    expect(pointFor(index, { board: [b1], lines: ['xx'], draws: [], player: 0 }).id).toBe(3)
    expect(pointFor(index, { board: [b1], lines: ['xx'], draws: [0], player: 1 }).id).toBe(4)
    expect(pointFor(index, { board: [b1], lines: ['xx'], draws: [1], player: 1 }).id).toBe(5)
  })
  it('tells round-2 points apart by both draw counts and the second line', () => {
    expect(pointFor(index, { board: [b1, b2], lines: ['xx', ''], draws: [1, 0], player: 0 }).id).toBe(6)
    expect(pointFor(index, { board: [b1, b2], lines: ['xx', 'p'], draws: [1, 0], player: 1 }).id).toBe(7)
    expect(pointFor(index, { board: [b1, b2], lines: ['xx', ''], draws: [1, 1], player: 0 }).id).toBe(8)
  })
  it('throws on a fold: the hand is over', () => {
    expect(() => pointFor(index, { board: [b1], lines: ['xf'], draws: [], player: 0 })).toThrow(/no public decision point.*xf/)
  })
  it("throws on P1 at a closed round-1 line with no draws: it is P0's draw, not a bet", () => {
    expect(() => pointFor(index, { board: [b1], lines: ['xx'], draws: [], player: 1 })).toThrow(/no public decision point.*P1/)
  })
  it('throws when the deck is to act: two board cards but round 2 not opened', () => {
    expect(() => pointFor(index, { board: [b1, b2], lines: ['xx'], draws: [1, 1], player: 0 })).toThrow(/no public decision point/)
  })
  it('throws on a round-2 line nobody can stand on', () => {
    expect(() => pointFor(index, { board: [b1, b2], lines: ['xx', 'pc'], draws: [1, 0], player: 0 })).toThrow(/no public decision point.*xx\|pc/)
  })
  it('throws when the export lists one spot twice', () => {
    const doubled: MiniIndex = { ...index, points: [...POINTS, { ...POINTS[2], id: 99 }] }
    expect(() => pointFor(doubled, { board: [b1], lines: ['p'], draws: [], player: 1 })).toThrow(/2 public points match.*ids 2, 99/)
  })
})

describe('describeSpot', () => {
  it('spells the board, the lines and the draws', () => {
    expect(describeSpot({ board: [b1, b2], lines: ['xp', ''], draws: [1, 0], player: 1 })).toBe("P1 on 4h 2c after xp|'' with draws 1,0")
  })
  it('names an empty line and an empty board', () => {
    expect(describeSpot({ board: [b1], lines: [''], draws: [], player: 0 })).toBe("P0 on 4h after ''")
    expect(describeSpot({ board: [], lines: [], draws: [], player: 0 })).toBe('P0 on (no board) after (no betting)')
  })
})
