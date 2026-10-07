/// <reference types="node" />
// The one test that reads the committed pack off disk; the app itself fetches it.
import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

import { checkChunk, checkManifest, expectedDeals } from './deals'
import type { Deal, Manifest } from './play'

const read = (name: string) => JSON.parse(readFileSync(new URL(`../public/deals/${name}`, import.meta.url), 'utf8'))

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
