import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { gzipSync } from 'node:zlib'
import { describe, expect, it } from 'vitest'
import { MINI_DECK, parseMini, type MiniCard } from './cards'
import { canonicalPicture, keyString, type Picture } from './canonical'
import { MiniData, type Fetcher, type MiniIndex, type PointMeta } from './data'

const SCALE = 250

// ------------------------------------------------------------ a tiny world
//
// One shape (b1d0, built from the real enumeration so the count is a check
// on the canonicalisation port too), two points with uniform rows, served
// from a Map by an injected fetcher. Nothing here touches the network.

function combos3(cards: readonly MiniCard[]): [MiniCard, MiniCard, MiniCard][] {
  const out: [MiniCard, MiniCard, MiniCard][] = []
  for (let a = 0; a < cards.length; a++)
    for (let b = a + 1; b < cards.length; b++) for (let c = b + 1; c < cards.length; c++) out.push([cards[a], cards[b], cards[c]])
  return out
}

/** Every canonical one-board-card key, sorted the way Python sorts its tuples. */
function keysB1d0(): Picture[] {
  const seen = new Map<string, Picture>()
  for (const board of MINI_DECK) {
    for (const hole of combos3(MINI_DECK.filter((c) => c !== board))) {
      const { key } = canonicalPicture({ hole, thrown: [], board: [board] })
      seen.set(keyString(key), key)
    }
  }
  const flat = (k: Picture): number[] => [...k.hole, ...k.thrown, ...k.board]
  return [...seen.values()].sort((x, y) => {
    const a = flat(x)
    const b = flat(y)
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i]
    return 0
  })
}

const KEYS = keysB1d0()

function tableBytes(keys: Picture[]): Uint8Array {
  const rows = keys.map((k) => [...k.hole, ...k.thrown, ...k.board])
  return Uint8Array.from(rows.flat())
}

const POINTS: PointMeta[] = [
  { id: 0, player: 0, boardCards: 1, draws: [], lines: [''], discards: 0, draw: false, actions: ['c', 'p'], shape: 'b1d0', file: 'point-000.bin.gz', n: KEYS.length },
  { id: 1, player: 1, boardCards: 1, draws: [], lines: ['p'], discards: 0, draw: false, actions: ['f', 'c', 'p'], shape: 'b1d0', file: 'point-001.bin.gz', n: KEYS.length },
]

function uniformChunk(point: PointMeta): Uint8Array {
  const width = point.actions.length
  const row = Array.from({ length: width }, () => Math.floor(SCALE / width))
  for (let k = 0; k < SCALE - Math.floor(SCALE / width) * width; k++) row[k] += 1
  return Uint8Array.from(Array.from({ length: point.n }, () => row).flat())
}

interface World {
  files: Map<string, Uint8Array | string>
  fetches: string[]
  fetcher: Fetcher
}

function world(overrides: Partial<Record<string, Uint8Array | string>> = {}): World {
  const index: MiniIndex = {
    strategy: { sha256: 'synthetic', rule: 'lcfr', column: 'linear', iteration: 0, workers: 1, seed: 0 },
    deck: { ranks: '23456', suits: 'cdh' },
    scale: SCALE,
    shapes: { b1d0: { file: 'keys-b1d0.bin', n: KEYS.length, stride: 4 } },
    points: POINTS,
  }
  const files = new Map<string, Uint8Array | string>()
  files.set('mini/index.json', JSON.stringify(index))
  files.set('mini/keys-b1d0.bin', tableBytes(KEYS))
  for (const p of POINTS) files.set(`mini/${p.file}`, new Uint8Array(gzipSync(uniformChunk(p))))
  for (const [k, v] of Object.entries(overrides)) if (v !== undefined) files.set(k, v)
  const fetches: string[] = []
  const fetcher: Fetcher = async (url) => {
    fetches.push(url)
    const body = files.get(url)
    if (body === undefined) return new Response(null, { status: 404, statusText: 'Not Found' })
    return new Response(body instanceof Uint8Array ? new Uint8Array(body) : body)
  }
  return { files, fetches, fetcher }
}

// ------------------------------------------------------------------ tests

describe('the synthetic key table', () => {
  it('has the 970 one-board-card keys the Python census counts', () => {
    expect(KEYS).toHaveLength(970)
  })
})

describe('MiniData.load', () => {
  it('reads the manifest and indexes every key of every shape', async () => {
    const w = world()
    const data = await MiniData.load('mini/', w.fetcher)
    expect(data.index.scale).toBe(SCALE)
    expect(data.index.points).toHaveLength(2)
    expect(data.shapeSize('b1d0')).toBe(970)
    expect(w.fetches).toEqual(['mini/index.json', 'mini/keys-b1d0.bin'])
  })
  it('names a missing file', async () => {
    const w = world()
    w.files.delete('mini/keys-b1d0.bin')
    await expect(MiniData.load('mini/', w.fetcher)).rejects.toThrow(/mini\/keys-b1d0\.bin.*404/)
  })
  it('refuses a key table whose length disagrees with the manifest', async () => {
    const w = world({ 'mini/keys-b1d0.bin': tableBytes(KEYS).slice(0, 4 * 969) })
    await expect(MiniData.load('mini/', w.fetcher)).rejects.toThrow(/key table b1d0 has 3876 bytes, want 970 × 4/)
  })
  it('refuses a key table that lists a key twice', async () => {
    const doubled = tableBytes([...KEYS.slice(0, 969), KEYS[0]])
    const w = world({ 'mini/keys-b1d0.bin': doubled })
    await expect(MiniData.load('mini/', w.fetcher)).rejects.toThrow(/lists .* twice \(rows 0 and 969\)/)
  })
  it('refuses a shape whose stride does not fit its name', async () => {
    const w = world()
    const index = JSON.parse(w.files.get('mini/index.json') as string) as MiniIndex
    index.shapes = { b1d0: { file: 'keys-b1d0.bin', n: 970, stride: 5 } }
    w.files.set('mini/index.json', JSON.stringify(index))
    await expect(MiniData.load('mini/', w.fetcher)).rejects.toThrow(/shape b1d0 has stride 5/)
  })
})

describe('keyIndex', () => {
  it('returns the row of a canonical key, and that row spells the key', async () => {
    const data = await MiniData.load('mini/', world().fetcher)
    const { key } = canonicalPicture({ hole: [parseMini('6h'), parseMini('2d'), parseMini('6c')], thrown: [], board: [parseMini('3h')] })
    const row = data.keyIndex('b1d0', key)
    expect(keyString(KEYS[row])).toBe(keyString(key))
    expect(data.keyIndex('b1d0', KEYS[0])).toBe(0)
    expect(data.keyIndex('b1d0', KEYS[969])).toBe(969)
  })
  it('throws naming the key when it is not canonical or not of the shape', async () => {
    const data = await MiniData.load('mini/', world().fetcher)
    const physical: Picture = { hole: [parseMini('6h'), parseMini('2d'), parseMini('6c')], thrown: [], board: [parseMini('3h')] }
    expect(() => data.keyIndex('b1d0', physical)).toThrow(/key 14,1,12\/\|5 is not in shape b1d0/)
    expect(() => data.keyIndex('b2d0', KEYS[0])).toThrow(/b2d0 is not a key shape/)
  })
})

describe('point', () => {
  it('this runtime has DecompressionStream, so the loader needs no zlib fallback', () => {
    expect(typeof DecompressionStream).toBe('function')
  })
  it('inflates a chunk to n × width bytes whose rows sum to the scale', async () => {
    const data = await MiniData.load('mini/', world().fetcher)
    const bytes = await data.point(1)
    expect(bytes.byteLength).toBe(970 * 3)
    for (let r = 0; r < 970; r++) expect(bytes[r * 3] + bytes[r * 3 + 1] + bytes[r * 3 + 2]).toBe(SCALE)
  })
  it('fetches each chunk once', async () => {
    const w = world()
    const data = await MiniData.load('mini/', w.fetcher)
    const [a, b] = await Promise.all([data.point(0), data.point(0)])
    await data.point(0)
    expect(a).toBe(b)
    expect(w.fetches.filter((u) => u === 'mini/point-000.bin.gz')).toHaveLength(1)
  })
  it('throws, naming the point and both lengths, on a short chunk — before any mix is read', async () => {
    const short = new Uint8Array(gzipSync(uniformChunk(POINTS[0]).slice(0, 970 * 2 - 7)))
    const data = await MiniData.load('mini/', world({ 'mini/point-000.bin.gz': short }).fetcher)
    await expect(data.point(0)).rejects.toThrow(/point 0 \(point-000\.bin\.gz\) inflated to 1933 bytes, want 970 × 2 = 1940/)
  })
  it('throws on a chunk that is not gzip', async () => {
    const data = await MiniData.load('mini/', world({ 'mini/point-000.bin.gz': uniformChunk(POINTS[0]) }).fetcher)
    await expect(data.point(0)).rejects.toThrow()
  })
  it('does not cache a failure: the next call fetches again', async () => {
    const w = world()
    const data = await MiniData.load('mini/', w.fetcher)
    const good = w.files.get('mini/point-000.bin.gz') as Uint8Array
    w.files.delete('mini/point-000.bin.gz')
    await expect(data.point(0)).rejects.toThrow(/404/)
    w.files.set('mini/point-000.bin.gz', good)
    expect((await data.point(0)).byteLength).toBe(970 * 2)
  })
  it('throws on a point the manifest does not list', async () => {
    const data = await MiniData.load('mini/', world().fetcher)
    expect(() => data.point(7)).toThrow(/public point 7 is not in this export/)
  })
})

// -------------------------------------------------------- the real export

const HERE = dirname(fileURLToPath(import.meta.url))
const EXPORT = join(HERE, '..', '..', 'public', 'mini')
const BASE = 'export/'

/** Serves `public/mini/` from disk under the `export/` base. */
const diskFetcher: Fetcher = async (url) => {
  if (!url.startsWith(BASE)) return new Response(null, { status: 404, statusText: 'Not Found' })
  const path = join(EXPORT, url.slice(BASE.length))
  if (!existsSync(path)) return new Response(null, { status: 404, statusText: 'Not Found' })
  return new Response(new Uint8Array(readFileSync(path)))
}

describe.skipIf(!existsSync(join(EXPORT, 'index.json')))('the real export under public/mini', () => {
  it('has 141 points and key tables of 970 / 10,170 / 100,400', async () => {
    const data = await MiniData.load(BASE, diskFetcher)
    expect(data.index.points).toHaveLength(141)
    expect(data.index.scale).toBe(SCALE)
    expect(data.shapeSize('b1d0')).toBe(970)
    expect(data.shapeSize('b2d0')).toBe(10_170)
    expect(data.shapeSize('b2d1')).toBe(100_400)
    expect(data.index.points.map((p) => p.id)).toEqual(data.index.points.map((_, i) => i))
  })
  it("point 0 is P0's opening and every row of it sums to the scale", async () => {
    const data = await MiniData.load(BASE, diskFetcher)
    const p0 = data.index.points[0]
    expect(p0).toMatchObject({ player: 0, boardCards: 1, draws: [], lines: [''], shape: 'b1d0', actions: ['c', 'p'] })
    const bytes = await data.point(0)
    const width = p0.actions.length
    for (let r = 0; r < p0.n; r++) {
      let s = 0
      for (let a = 0; a < width; a++) s += bytes[r * width + a]
      expect(s).toBe(SCALE)
    }
  })
  it('its b1d0 table is the synthetic enumeration in the same order', async () => {
    const data = await MiniData.load(BASE, diskFetcher)
    for (let r = 0; r < KEYS.length; r += 97) expect(data.keyIndex('b1d0', KEYS[r])).toBe(r)
  })
})
