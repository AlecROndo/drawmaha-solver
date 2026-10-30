/**
 * The shared chrome says what the site is now: five product surfaces and one
 * call to action in the top bar, three columns and the chip in the footer.
 * Rendered to static markup (no DOM), so the test reads the same file every
 * app carries and nothing about the browser.
 */

import { renderToStaticMarkup } from 'react-dom/server'
import { expect, test } from 'vitest'
import { Footer, TopBar } from './site'

/** The method's words stay on the Benchmarks page and in the README; the chrome never says them. */
const BANNED = /Rungs|rung 0|ladder|climb/i

test('the top bar names the five surfaces and marks the current one', () => {
  const html = renderToStaticMarkup(<TopBar here="benchmarks" />)
  for (const w of ['Research', 'Solver', 'Benchmarks', 'Play', 'Rules', 'Open the solver']) expect(html).toContain(w)
  expect(html).toMatch(/href="https:\/\/drawmaha.app\/benchmarks"[^>]*aria-current="page"/)
  expect(html).not.toMatch(BANNED)
})

test('the footer has three columns and the chip', () => {
  const html = renderToStaticMarkup(<Footer />)
  for (const w of ['Range explorer', 'Releases', 'Learn', 'ML survey', 'GitHub']) expect(html).toContain(w)
  expect(html).toContain('class="chip-end"')
  expect(html).not.toMatch(BANNED)
})
