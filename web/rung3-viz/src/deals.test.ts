/// <reference types="node" />
// The one test that reads the committed pack off disk; the app itself fetches it.
import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

import { checkChunk, checkManifest, expectedDeals, loadFirst, loadNext } from './deals'
import type { Deal, Manifest } from './play'

const read = (name: string) => JSON.parse(readFileSync(new URL(`../public/deals/${name}`, import.meta.url), 'utf8'))

/** A `fetch` over a handful of named files: a JSON body for each, 404 for the rest. */
const serving = (files: Record<string, unknown>): typeof fetch =>
  (async (url: string | URL | Request) => {
    const name = String(url).split('/').pop() ?? ''
    const body = files[name]
    return body === undefined ? new Response(null, { status: 404 }) : new Response(JSON.stringify(body), { status: 200 })
  }) as typeof fetch

describe('a chunk against the manifest', () => {
  const manifest: Manifest = {
    seed: 9,
    deals: 5,
    chunk: 2,
    chunks: ['pack-00.json', 'pack-01.json', 'pack-02.json'],
    strategy: { sha256: 'ab'.repeat(32), rule: 'test', column: 'linear', iteration: 1, workers: 1, seed: 0 },
  }

  it('expects a full chunk everywhere but the last, which holds what is left', () => {
    expect([0, 1, 2].map((i) => expectedDeals(manifest, i))).toEqual([2, 2, 1])
  })

  it('refuses a manifest whose deals, chunk size and chunk count do not agree', () => {
    expect(() => checkManifest(manifest)).not.toThrow()
    expect(() => checkManifest({ ...manifest, deals: 7 })).toThrow('describes 7 deals in chunks of 2 but names 3 chunks')
    expect(() => checkManifest({ ...manifest, chunks: [] })).toThrow('names 0 chunks')
    expect(() => checkManifest({ ...manifest, chunk: 0 })).toThrow()
    expect(() => checkManifest({ ...manifest, deals: 0, chunks: [] })).toThrow()
  })

  it('holds every chunk name to pack-NN.json before it goes into a URL', () => {
    const odd = (name: unknown) => ({ ...manifest, chunks: ['pack-00.json', 'pack-01.json', name as string] })
    expect(() => checkManifest(odd('../index.json'))).toThrow('not a pack-NN.json: ../index.json')
    expect(() => checkManifest(odd('pack-2.json'))).toThrow('not a pack-NN.json: pack-2.json')
    expect(() => checkManifest(odd('https://elsewhere/pack-02.json'))).toThrow('not a pack-NN.json')
    expect(() => checkManifest(odd(2))).toThrow('not a pack-NN.json: 2')
    expect(() => checkManifest(odd('pack-02.json'))).not.toThrow()
  })

  it('refuses a count the manifest did not describe, and names a malformed deal', () => {
    expect(() => checkChunk(manifest, 0, [])).toThrow('pack-00.json holds 0 deals; the manifest says 2')
    expect(() => checkChunk(manifest, 2, [{} as Deal])).toThrow('pack-02.json, deal 0: the deck is not a permutation')
  })
})

describe('the committed pack', () => {
  const manifest = read('index.json') as Manifest

  it('is 600 deals in six chunks of 100, from the frozen LCFR strategy', () => {
    expect(() => checkManifest(manifest)).not.toThrow()
    expect(manifest.deals).toBe(600)
    expect(manifest.chunk).toBe(100)
    expect(manifest.chunks.length).toBe(6)
    expect(manifest.strategy.rule).toBe('lcfr')
    expect(manifest.strategy.sha256).toMatch(/^[0-9a-f]{64}$/)
  })

  it('passes every chunk through the grammar the browser plays by', () => {
    for (const [index, name] of manifest.chunks.entries()) {
      expect(() => checkChunk(manifest, index, read(name) as Deal[])).not.toThrow()
    }
  })
})

describe('loading the pack', () => {
  const manifest = read('index.json') as Manifest
  const first = read('pack-00.json') as Deal[]

  it('sits down on the manifest and a checked chunk 0', async () => {
    const got = await loadFirst(serving({ 'index.json': manifest, 'pack-00.json': first }))
    expect(got.manifest).toEqual(manifest)
    expect(got.deals.length).toBe(100)
  })

  it('refuses to sit down on a missing manifest, one that does not add up, or a chunk 0 that disagrees with it', async () => {
    await expect(loadFirst(serving({}))).rejects.toThrow("the deal pack's manifest did not load (404)")
    await expect(loadFirst(serving({ 'index.json': { ...manifest, deals: 601 } }))).rejects.toThrow('names 6 chunks')
    await expect(loadFirst(serving({ 'index.json': manifest }))).rejects.toThrow('deal chunk pack-00.json did not load (404)')
    await expect(loadFirst(serving({ 'index.json': manifest, 'pack-00.json': first.slice(0, 99) }))).rejects.toThrow(
      'pack-00.json holds 99 deals; the manifest says 100',
    )
  })

  it('refuses a later chunk the manifest did not describe, or with a deal the grammar cannot play', async () => {
    const third = read('pack-03.json') as Deal[]
    expect(await loadNext(manifest, 3, serving({ 'pack-03.json': third }))).toEqual(third)
    await expect(loadNext(manifest, 3, serving({ 'pack-03.json': third.slice(1) }))).rejects.toThrow(
      'pack-03.json holds 99 deals; the manifest says 100',
    )
    const broken = third.map((deal) => ({ ...deal, r1: { ...deal.r1 } }))
    delete broken[7].r1['xp']
    await expect(loadNext(manifest, 3, serving({ 'pack-03.json': broken }))).rejects.toThrow(
      "pack-03.json, deal 7: round 1 at 'xp' is missing or malformed",
    )
    await expect(loadNext(manifest, 5, serving({}))).rejects.toThrow('deal chunk pack-05.json did not load (404)')
  })
})
