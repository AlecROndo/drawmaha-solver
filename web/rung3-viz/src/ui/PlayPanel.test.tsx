/**
 * The card face is the one piece of the Play tab whose markup carries logic:
 * the suit picks the ink class, the rank and glyph are split into two
 * elements the theme places, and the whole span is one labelled image.
 * Rendered to static markup (no DOM), so this pins structure, not pixels.
 */

import { readFileSync } from 'node:fs'
import { renderToStaticMarkup } from 'react-dom/server'
import { expect, test } from 'vitest'
import { SUITS, sym } from '../minidraw'
import { CardGlyph } from './PlayPanel'

/** A card is rank × 3 + suit, so 4h is 2 × 3 + 2 = 8. */
const index = (rank: string, suit: string) => '23456'.indexOf(rank) * SUITS.length + SUITS.indexOf(suit)

test("a face is the theme's cardface: a rank and a hidden glyph in the suit's class, labelled with its name", () => {
  const html = renderToStaticMarkup(<CardGlyph card={index('4', 'h')} />)
  expect(html).toMatch(/^<span class="pcard cardface su-h" role="img" aria-label="4h">/)
  expect(html).toContain('<b>4</b>')
  expect(html).toContain('<i aria-hidden="true">♥</i>')
})

test('the three suits map to su-c clubs, su-d diamonds, su-h hearts', () => {
  const seen = ['c', 'd', 'h'].map((suit) => {
    const c = index('6', suit)
    const html = renderToStaticMarkup(<CardGlyph card={c} />)
    expect(html).toContain(`aria-label="${sym(c)}"`)
    // the glyph is everything inside the <i>, not one code unit, so a glyph that grew a variation selector would still match
    const m = html.match(/class="pcard cardface su-([cdh])".*<i aria-hidden="true">([^<]+)<\/i>/)
    return m ? `${m[1]}${m[2]}` : html
  })
  expect(seen).toEqual(['c♣', 'd♦', 'h♥'])
})

test('face down and not-yet-dealt are empty labelled images; small only adds a class', () => {
  expect(renderToStaticMarkup(<CardGlyph hidden />)).toBe(
    '<span class="pcard cardface back" role="img" aria-label="a card, face down"></span>',
  )
  expect(renderToStaticMarkup(<CardGlyph />)).toBe('<span class="pcard cardface empty" role="img" aria-label="no card yet"></span>')
  expect(renderToStaticMarkup(<CardGlyph card={index('2', 'c')} small />)).toMatch(/^<span class="pcard small cardface su-c"/)
})

/** The one mapping the markup tests cannot see: which ink each suit wears, on the theme's face and in this sheet's line of type. */
test('the theme and the sheet give each suit the same ink on the face and in text: call-green, check-blue, oxblood', () => {
  const theme = readFileSync(new URL('../theme.css', import.meta.url), 'utf8')
  const sheet = readFileSync(new URL('../index.css', import.meta.url), 'utf8')
  const ink = (css: string, selector: string) => {
    const m = css.match(new RegExp(`${selector.replace(/[.$]/g, '\\$&')} \\{\\s*color: var\\((--[a-z0-9-]+)\\);`))
    return m ? m[1] : `no rule for ${selector}`
  }
  // the pip is the card's direct child: `> i`, never ` i` (tests/test_ui_standard.py holds that side)
  const faces = ['c', 'd', 'h'].map((k) => ink(theme, `.cardface.su-${k} > i`))
  const names = ['c', 'd', 'h'].map((k) => ink(sheet, `.ctext.su-${k}`))
  expect(faces).toEqual(['--act-call', '--act-check', '--accent'])
  expect(names).toEqual(faces)
})
