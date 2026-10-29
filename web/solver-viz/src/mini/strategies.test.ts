import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { parseMini } from './cards'
import type { Fetcher, MiniIndex } from './data'
import { reachRange } from './range'
import { DEFAULT_STRATEGY, STRATEGIES, Strategies, describeStrategy, isStrategyId, strategyChoice } from './strategies'

// ------------------------------------------------------------ the words

function manifest(strategy: Partial<MiniIndex['strategy']>, grade: MiniIndex['grade']): MiniIndex {
  return {
    strategy: { sha256: 'x'.repeat(64), release: 'tag', asset: 'file.npz', rule: 'lcfr', column: 'linear', iteration: 1, workers: 1, seed: 0, ...strategy },
    grade,
    deck: { ranks: '23456', suits: 'cdh' },
    scale: 250,
    shapes: { b1d0: { file: 'keys-b1d0.bin', n: 970, stride: 4 } },
    points: [{ id: 0, player: 0, boardCards: 1, draws: [], lines: [''], discards: 0, draw: false, actions: ['c', 'p'], shape: 'b1d0', file: 'point-000.bin.gz', n: 970 }],
  }
}

describe('describeStrategy', () => {
  it('names the tabular rule, its lockstep size and its grade', () => {
    const words = describeStrategy(
      manifest({ rule: 'lcfr', column: 'linear', iteration: 2_000_000, workers: 10 }, { exploitability: 0.06134626307457512, valueP0: -0.0906451773437125, handsPerSeat: 20_000_000, file: 'f', key: 'k' }),
    )
    expect(words).toEqual({ name: 'tabular LCFR', detail: '2,000,000 iterations × 10 workers · 20M hands per seat', grade: '0.061' })
  })
  it('names Π and the exact average by their column', () => {
    const pi = describeStrategy(manifest({ rule: 'deep', column: 'pi', iteration: 500, workers: 0 }, { exploitability: 0.13634085589731734, valueP0: -0.1, handsPerSeat: 5_000_000, file: 'f', key: 'k' }))
    expect(pi).toEqual({ name: 'Deep CFR · Π, the policy net', detail: 'iteration 500 · 5M hands per seat', grade: '0.136' })
    const exact = describeStrategy(manifest({ rule: 'deep', column: 'exact', iteration: 500, workers: 0 }, { exploitability: 0.10794019398465793, valueP0: -0.1, handsPerSeat: 5_000_000, file: 'f', key: 'k' }))
    expect(exact.name).toBe('Deep CFR · the exact average')
    expect(exact.grade).toBe('0.108')
  })
  it('reads a deal-pack manifest, which carries only the two blocks', () => {
    const { strategy, grade } = manifest({ rule: 'deep', column: 'exact', iteration: 200, workers: 0 }, { exploitability: 0.08591359915074376, valueP0: -0.10580907636152471, handsPerSeat: 2_000_000, file: 'f', key: 'k' })
    expect(describeStrategy({ strategy, grade })).toEqual({ name: 'Deep CFR · the exact average', detail: 'iteration 200 · 2M hands per seat', grade: '0.086' })
  })
  it('has no grade to quote when the manifest carries none', () => {
    const words = describeStrategy(manifest({ rule: 'test', iteration: 1, workers: 1 }, null))
    expect(words.grade).toBeNull()
    expect(words.detail).toBe('1 iterations × 1 workers')
  })
})

describe('the strategy list', () => {
  it('has three distinct ids and directories, the tabular one first and default', () => {
    expect(STRATEGIES.map((s) => s.id)).toEqual(['tabular', 'deep-pi', 'deep-exact'])
    expect(new Set(STRATEGIES.map((s) => s.dir)).size).toBe(3)
    expect(DEFAULT_STRATEGY).toBe('tabular')
    expect(isStrategyId('deep-pi')).toBe(true)
    expect(isStrategyId('deep')).toBe(false)
    expect(strategyChoice('deep-exact').dir).toBe('mini-deep-exact/')
  })
})

// ------------------------------------------------------------- the cache

describe('Strategies', () => {
  it('fetches a manifest once per strategy and retries after a failure', async () => {
    let calls = 0
    let fail = true
    const fetcher: Fetcher = async () => {
      calls++
      if (fail) return new Response(null, { status: 500, statusText: 'boom' })
      return new Response(JSON.stringify(manifest({ rule: 'deep', column: 'pi' }, null)), { headers: { 'content-type': 'application/json' } })
    }
    const set = new Strategies('base/', fetcher)
    await expect(set.manifest('deep-pi')).rejects.toThrow('base/mini-deep-pi/index.json')
    fail = false
    const first = await set.manifest('deep-pi')
    const second = await set.manifest('deep-pi')
    expect(second).toBe(first)
    expect(calls).toBe(2)
  })
})

// -------------------------------------------------------- the real exports

const HERE = dirname(fileURLToPath(import.meta.url))
const PUBLIC = join(HERE, '..', '..', 'public')
const REPO = join(HERE, '..', '..', '..', '..')
const BASE = 'export/'

const diskFetcher: Fetcher = async (url) => {
  if (!url.startsWith(BASE)) return new Response(null, { status: 404, statusText: 'Not Found' })
  const path = join(PUBLIC, url.slice(BASE.length))
  if (!existsSync(path)) return new Response(null, { status: 404, statusText: 'Not Found' })
  return new Response(new Uint8Array(readFileSync(path)))
}

const present = STRATEGIES.filter((s) => existsSync(join(PUBLIC, s.dir, 'index.json')))

describe.skipIf(present.length < STRATEGIES.length)('the three committed exports', () => {
  const set = new Strategies(BASE, diskFetcher)

  it('each names its release and quotes the grade its figures file records', async () => {
    for (const s of STRATEGIES) {
      const index = await set.manifest(s.id)
      expect(index.strategy.release, s.id).toBeTruthy()
      expect(index.grade, s.id).not.toBeNull()
      const recorded = JSON.parse(readFileSync(join(REPO, index.grade!.file), 'utf8')) as Record<string, { exploitability: number }>
      expect(index.grade!.exploitability).toBe(recorded[index.grade!.key].exploitability)
      expect(describeStrategy(index).grade).toMatch(/^\d\.\d{3}$/)
    }
  })

  it('share the point walk and the key tables byte for byte', async () => {
    const tabular = await set.manifest('tabular')
    for (const s of STRATEGIES) {
      const index = await set.manifest(s.id)
      expect(index.points, s.id).toEqual(tabular.points)
      expect(index.shapes, s.id).toEqual(tabular.shapes)
      for (const shape of Object.values(index.shapes)) {
        expect(readFileSync(join(PUBLIC, s.dir, shape.file)).equals(readFileSync(join(PUBLIC, 'mini', shape.file))), `${s.id} ${shape.file}`).toBe(true)
      }
    }
  })

  it('enumerate the same states at a spot, so a switch keeps the spot and only the weights move', async () => {
    const spot = { board: [parseMini('4h')], lines: ['p'], draws: [], player: 1 as const }
    const ranges = await Promise.all(STRATEGIES.map(async (s) => reachRange(await set.data(s.id), spot, 1, [])))
    for (const r of ranges) {
      expect(r.n).toBe(ranges[0].n)
      expect(Array.from(r.cards)).toEqual(Array.from(ranges[0].cards))
    }
    const mixes = ranges.map((r) => Array.from(r.mix.slice(0, 30)).join(','))
    expect(new Set(mixes).size).toBe(3)
  })
})
