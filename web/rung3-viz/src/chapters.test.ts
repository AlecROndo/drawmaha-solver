import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

/**
 * The page's chapter ids and figure numbers are page order, read from the
 * source of `ui/ResearchTab.tsx`: the contents rail links to `#ch-NN`, so a
 * chapter whose id repeats an earlier one is unreachable from the rail and
 * prints the wrong number, and a figure out of sequence is cited wrongly by
 * every caption after it. A renumbering done by search-and-replace produced
 * exactly that once, which is why this reads the source rather than trusting it.
 */

const source = readFileSync(new URL('./ui/ResearchTab.tsx', import.meta.url), 'utf8')

describe('the research page is numbered in page order', () => {
  it('declares its chapters once each, in order, and renders them under the same ids in the same order', () => {
    const declared = [...source.matchAll(/\{ id: '(ch-\d\d)', n: '(\d\d)'/g)].map((m) => ({ id: m[1], n: m[2] }))
    const rendered = [...source.matchAll(/<Chapter id="(ch-\d\d)"/g)].map((m) => m[1])
    expect(declared.length).toBeGreaterThan(0)
    expect(declared.map((c) => c.id)).toEqual(declared.map((_, i) => `ch-${String(i + 1).padStart(2, '0')}`))
    expect(declared.map((c) => c.n)).toEqual(declared.map((c) => c.id.slice(3)))
    expect(rendered).toEqual(declared.map((c) => c.id))
  })

  it('numbers its figures 1..N in the order they appear', () => {
    const figures = [...source.matchAll(/<Figure\s+n=\{(\d+)\}/g)].map((m) => Number(m[1]))
    expect(figures.length).toBeGreaterThan(0)
    expect(figures).toEqual(figures.map((_, i) => i + 1))
  })

  it('cites no chapter number past the last one', () => {
    const last = [...source.matchAll(/<Chapter id="ch-(\d\d)"/g)].length
    const cited = [...source.matchAll(/[Cc]hapters? (\d\d)/g)].map((m) => Number(m[1]))
    for (const n of cited) expect(n).toBeLessThanOrEqual(last)
  })
})
