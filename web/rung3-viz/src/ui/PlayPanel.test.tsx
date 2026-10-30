/**
 * The card face is the one piece of the Play tab whose markup carries logic:
 * the suit index picks the ink class, the rank and glyph are split into two
 * elements the stylesheet places, and the whole span is one labelled image.
 * Rendered to static markup (no DOM), so this pins structure, not pixels.
 */

import { readFileSync } from 'node:fs'
import { renderToStaticMarkup } from 'react-dom/server'
import { expect, test } from 'vitest'
import { SUITS, sym } from '../minidraw'
import { CardGlyph } from './PlayPanel'

/** A card is rank × 3 + suit, so 4h is 2 × 3 + 2 = 8. */
const index = (rank: string, suit: string) => '23456'.indexOf(rank) * SUITS.length + SUITS.indexOf(suit)

test("a face is a small rank and a hidden glyph in the suit's class, labelled with its name", () => {
  const html = renderToStaticMarkup(<CardGlyph card={index('4', 'h')} />)
  expect(html).toMatch(/^<span class="pcard face s2" role="img" aria-label="4h">/)
  expect(html).toContain('<b>4</b>')
  expect(html).toContain('<i aria-hidden="true">♥</i>')
})

test('the three suits map to s0 clubs, s1 diamonds, s2 hearts, in that order', () => {
  const seen = ['c', 'd', 'h'].map((suit) => {
    const c = index('6', suit)
    const html = renderToStaticMarkup(<CardGlyph card={c} />)
    expect(html).toContain(`aria-label="${sym(c)}"`)
    // the glyph is everything inside the <i>, not one code unit, so a glyph that grew a variation selector would still match
    const m = html.match(/class="pcard face s(\d)".*<i aria-hidden="true">([^<]+)<\/i>/)
    return m ? `${m[1]}${m[2]}` : html
  })
  expect(seen).toEqual(['0♣', '1♦', '2♥'])
})

test('face down and not-yet-dealt are empty labelled images; small only adds a class', () => {
  expect(renderToStaticMarkup(<CardGlyph hidden />)).toBe(
    '<span class="pcard back" role="img" aria-label="a card, face down"></span>',
  )
  expect(renderToStaticMarkup(<CardGlyph />)).toBe('<span class="pcard empty" role="img" aria-label="no card yet"></span>')
  expect(renderToStaticMarkup(<CardGlyph card={index('2', 'c')} small />)).toMatch(/^<span class="pcard small face s0"/)
})

/** The one mapping the markup tests cannot see: which ink each suit class wears, on the face and in a line of type. */
test('the stylesheet gives each suit the same ink on the face and in text: call-green, check-blue, oxblood', () => {
  const css = readFileSync(new URL('../index.css', import.meta.url), 'utf8')
  const ink = (selector: string) => {
    const m = css.match(new RegExp(`${selector.replace(/[.$]/g, '\\$&')} \\{\\s*color: var\\((--[a-z0-9-]+)\\);`))
    return m ? m[1] : `no rule for ${selector}`
  }
  const faces = [0, 1, 2].map((n) => ink(`.pcard.s${n} i`))
  const names = [0, 1, 2].map((n) => ink(`.ctext.s${n}`))
  expect(faces).toEqual(['--act-call', '--act-check', '--accent'])
  expect(names).toEqual(faces)
})
