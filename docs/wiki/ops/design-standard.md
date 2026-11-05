---
title: The design standard
description: Paper, ink and one oxblood accent; two type voices; motion read from three tokens; and a test that enforces the mechanical half on every stylesheet.
---

[`web/DESIGN.md`](repo:web/DESIGN.md) is the UI standard, and it is enforced. Every surface is held to it: the static pages, the archived Cover, and the five apps. The standard treats taste the way Emil Kowalski describes design engineering, as a set of decisions made the same way every time. Most of them nobody consciously notices, and together they make software feel right.

The standard has two halves. The **mechanical half** (which curve, how long, what a press does, what reduced motion keeps) is checked by [`tests/test_ui_standard.py`](repo:tests/test_ui_standard.py) on every stylesheet. The **judgement half** is a review table at the end of `DESIGN.md`. Before you touch a stylesheet, a `<style>` block or a component's classes, read `DESIGN.md`. Afterwards, run:

```bash
uv run pytest tests/test_ui_standard.py tests/test_shared_chrome.py tests/test_cover_page.py
```

## The visual system in six decisions

| Decision | What the theme declares |
| --- | --- |
| **Paper, ink, one accent** | `--paper` `#f7f5f0` is the page and `--ink` `#16140f` the text. `--accent` `#8e2038` is oxblood, the deeper shade of the chip's red, and the *only* accent: the winning series in a chart, bets and raises, links, the focus ring |
| **Semantic chart and action tokens** | `--series-1/2/3` are ink, accent and muted ink, with a dash pattern for a fourth series and a direct label at the end of every line. The action triplet is `--act-bet` (accent), `--act-call` `#2f7d4a` green and `--act-check` `#2f66a3` blue (check and fold) |
| **Two voices and a mono** | Newsreader (`--serif`) for display, headings and research prose; Geist (`--sans`) for UI, tables, charts and numbers, with tabular numerals; IBM Plex Mono (`--mono`) only for the chip's glyph grid and for code |
| **One scale** | display 64, `h1` 52, `h2` 32, `h3` 22, lede 21, research body 18 serif, UI 15, captions 13 / 12 / 11 |
| **One grid** | 1160 px of content (`--max`) inside a 40 px gutter, 12 columns on a 24 px gap |
| **Hairlines, not boxes** | every container is a top or bottom hairline, never a shadowed box; the only radii are 6 px on buttons and controls |

Playing cards follow one drawing too: the theme's `.cardface`, the "bleed pip". The rank sits top-left in Geist 600, and one oversized suit glyph is cropped by the card's lower-right edge. Spades wear ink, hearts the oxblood, diamonds the check-blue and clubs the call-green. A surface sets only the card's width, `--w`.

:::note
`--yellow` and `--pink` are retired, and `test_retired_tokens_are_gone` keeps them out of every sheet but the archived Cover's. `--call-fill`, `--pot-fill` and `--mix-*` are gone too; mix bands read the action triplet.
:::

## Motion reads three tokens

```css
--ease-out: cubic-bezier(0.23, 1, 0.32, 1);     /* anything entering or settling */
--ease-in-out: cubic-bezier(0.77, 0, 0.175, 1); /* a thing that moves while on screen */
--t-press: 160ms;                               /* the press */
```

The built-in keywords are too weak to read as intended. `ease` is kept for exactly one job: a colour, background or border changing under the pointer, at 150 ms. `ease-in` is never used. It delays the first frame of movement, which is the frame the user is watching, so 300 ms of `ease-in` feels slower than 300 ms of `ease-out`.

Before animating anything, decide in this order:

1. **How often will the user see it?** A keyboard action, or a bar that updates a thousand times a second, never animates; the number printed beside it would disagree with the bar. A hover gets 150 ms of colour. A sheet opening is occasional and gets the full treatment.
2. **What is it for?** Press feedback, state, spatial consistency, or preventing a jarring swap. "It looks cool" is not a purpose for anything seen more than once.
3. **Which curve?** Entering or settling uses `--ease-out`. Moving while on screen uses `--ease-in-out`. Colour under the pointer uses `ease`. Constant motion uses `linear`.
4. **How long?** A press is 160 ms, hover colour 150 ms, a popover or sheet 150–250 ms. Nothing in the chrome runs over 300 ms per declaration, delays included. Exits are faster than entrances.

## What the test enforces

| Rule | Example on the site |
| --- | --- |
| everything pressable shrinks to `scale(0.97)` for 160 ms on `:active` | `.btn`, `.navbtn`, `.view`, `.seg button`, `.radios label`, `.rail a`, and the solver's cells, keys and deck cards (`PRESSABLES` in the test) |
| a disabled control does not answer a press | `.btn:active:not(:disabled)` |
| no `transition: all`, no `ease-in`, no hard-coded curve in the chrome | `.sweep` uses `0.4s var(--ease-out)`; it used to be `0.55s ease-in` |
| only `transform` and `opacity` animate in the chrome | a bar that moves uses `transform: scaleX()`, never `width` |
| nothing moves on load | the turning chip is the one thing that moves without being asked |
| a hover that moves something is gated on a real pointer | inside `@media (hover: hover) and (pointer: fine)`, because touch fires hover on tap and never un-fires it |
| reduced motion means fewer and gentler, not none, named by selector | the theme quiets `.btn:active:not(:disabled)`, not bare `.btn:active`, because `:not()` adds specificity |
| a one-shot that code waits on still ends under reduced motion | the swept chips switch to a fade-only keyframe, so `animationend` still fires and unmounts them |

A motion that explains rather than decorates can be exempted. `EXPLANATORY` in the test lists each such selector with its reason: today `.sweep` (chips crossing the felt to the pot, 400 ms) and the archived Cover's three. `test_every_allowlisted_selector_names_a_rule_that_moves` fails on an entry that exempts nothing, so a stale exemption cannot hide the next real one.

## Reviewing UI

A UI review in this repo is a single markdown table with `Before | After | Why` columns, one row per finding:

| Before | After | Why |
| --- | --- | --- |
| `transition: all 300ms` | `transition: transform var(--t-press) var(--ease-out)` | name the property; `all` animates things you did not mean |
| `.foo:hover { }` and no `:active` | add `.foo:active { transform: scale(0.97) }` | a press has to be heard |

Review a change the next day, at 4× slow motion in the DevTools Animations panel, and on a phone over the LAN.

:::details Where the CSS lives
- **The theme** (tokens, top bar, page head, buttons, figures, hairline tables, `.views`, `.seg`, `.radios`, `.stats`, `.rail`, footer): `web/rung0-viz/src/theme.css` and its four byte-identical copies. Edit rung 0's and `cp` it across.
- **Each app's own figures and layout**: `web/<app>/src/index.css`, on top of the theme.
- **The generated pages**: one inline `<style>` each, written by `scripts/site/make_static_pages.py`. Change the generator, never the HTML.
- **The archived Cover**: its old chrome inlined, minified, one rule per line.
:::
