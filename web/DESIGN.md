# The UI standard for drawmaha.app

Every page of the site is held to this document: the five static pages
(the homepage, Rules, Benchmarks, Releases and the archived Cover) and the
five apps (the three benchmark apps, the research app and the range
explorer). It is the design engineering stance in Emil Kowalski's sense:
taste is a set of decisions made the same way every time, most of which
nobody consciously notices, and which compound into software that feels
right. The visual system is summarised below and argued in
`docs/superpowers/specs/2026-10-29-paper-lab-site-design.md`; the rest of
this document is about how things move, how they answer a press, and how
that is kept true.

`tests/test_ui_standard.py` enforces the mechanical half of it on every
stylesheet in `web/`; the judgement half is the review table at the end.

## The visual system

The spec holds the reasoning; this is what the theme
(`web/rung0-viz/src/theme.css`, whose header states the same five points)
declares.

- **Paper, ink and one accent.** `--paper` `#f7f5f0` is the page;
  `--paper-2` / `--paper-3` are recessed cells, bar tracks and code;
  `--rule` / `--rule-2` are hairlines and control borders. `--ink`
  `#16140f` is text and the primary series, `--ink-2` / `--ink-3` secondary
  text and muted series. `--accent` `#8e2038` is oxblood, the deeper shade of
  the chip's red (`#c8102e`), and the only accent: the winning series in a
  chart, bet and raise, links, the focus ring; `--accent-tint` is a link's underline and the text selection,
  and on the homepage's opening matrix alone it fills the bet-dominant cells.
- **Charts and actions read semantic tokens.** `--series-1/2/3` are ink,
  accent and muted ink, with a dash pattern for a fourth series and a direct
  label at the end of every line; `--grid` is the rule. The action triplet
  is `--act-bet` (the accent), `--act-call` `#2f7d4a` and `--act-check`
  `#2f66a3` (check and fold).
- **Two voices and the chip's mono.** Newsreader (`--serif`) for display,
  headings and the research body; Geist (`--sans`) for the UI, tables,
  charts and captions, with tabular numerals. IBM Plex Mono (`--mono`)
  draws the chip's glyph grid (600) and sets inline code and commands (400);
  it is never a prose or UI voice.
- **The scale.** Display 64 (the home hero), `h1` 52, `h2` 32 (30 inside
  the publication), `h3` 22, lede 21, research body 18 serif, UI 15,
  captions 13 / 12 / 11, the eyebrow 12 uppercase at 0.08 em.
- **The grid.** 1160 px of content (`--max`) inside 40 px of page padding
  (`--gutter`), 12 columns on a 24 px gap; a home section is a 4 + 8
  `.split`.
- **The hairline rule.** Every container is a top or bottom hairline, never
  a box with a shadow: a figure (`figure.fig`) sits on a top rule, anything
  tabular is `table.hair` or `.rows`. The only radii are 6 px on buttons
  and controls.
- **One card, the bleed pip.** Every playing card on the site is the theme's
  `.cardface`: the rank in the top-left in Geist 600, sized to read beside
  the pip, one suit glyph too big for the card cropped by its own
  lower-right edge, in the suit's ink at full strength, on `--card-face`
  card stock. Spades wear ink, hearts the oxblood, and the minor suits
  borrow the action triplet, diamonds the check-blue and clubs the
  call-green. A surface sets only `--w` (the felt's 52 px, the coach's
  28 px, a hand in a line of text 24 px); Kuhn and Leduc, games of ranks
  without suits, set `.letter` and bleed the rank in the serif. Face down is
  the chip's pinstripe, not yet dealt a dashed recess. The rank is the
  card's own `<b>` and the pip its own `<i>`, direct children: every rank or
  pip rule is `card > b` / `card > i`, so an element a surface nests inside
  a card for its own purpose (the solver's deck tile and its pot-lift bar)
  is never drawn as a pip; `test_a_card_reaches_only_its_own_rank_and_pip`
  holds it.

## Where the CSS lives

| Surface | File | Notes |
| --- | --- | --- |
| The theme: tokens, top bar, page head, buttons, figures, hairline tables, `.views`, `.seg`, `.radios`, `.stats`, `.rail`, footer | `web/rung0-viz/src/theme.css` and its four copies in the other apps | Byte-identical in all five apps; edit rung 0's and `cp` across (`tests/test_shared_chrome.py`) |
| Each app's own figures and layout | `web/<app>/src/index.css` | On top of the theme; reads its tokens |
| The four product pages | `web/cover/{index,rules,benchmarks,releases}.html` | Link `/theme.css` (the build copies the theme there) and `/fonts.css`; each carries one inline `<style>` of its own layout and chart classes. Written by `scripts/site/make_static_pages.py`; `web/cover/README.md` describes them |
| The archived Cover | `web/cover/cover.html` | Inlines the old chrome, minified, one rule per line; edited by hand if at all |

A chrome rule therefore lands in one place, the theme, and is copied four
times; the pages pick it up through the link. The motion tokens below are
declared in the theme's `:root` and in the Cover's, and the test checks they
agree.

## The motion tokens

```css
--ease-out: cubic-bezier(0.23, 1, 0.32, 1);     /* anything entering or settling */
--ease-in-out: cubic-bezier(0.77, 0, 0.175, 1); /* a thing that moves while on screen */
--t-press: 160ms;                               /* the press */
```

The built-in keywords are too weak to read as intended. `ease` (the
keyword) is kept for exactly one job: a colour, background or border
changing under the pointer, at 150 ms. `ease-in` is never used: it delays
the first frame of movement, which is the frame the user is watching, so the
same 300 ms feels slower than 300 ms of `ease-out`. `linear` is for constant
motion only.

A figure may own a curve when the curve is what the figure says; it is then
allowlisted by name in the test. No sheet on the site hard-codes one today.
Chrome never does.

## Decide in this order before animating anything

1. **How often will the user see it?** A keyboard action (rung 0's Space to
   run, → to step) or a bar that updates a thousand times a second never
   animates; the number printed beside it would disagree with the bar. A
   hover sees 150 ms of colour. Opening a sheet is occasional and gets the
   full treatment. A figure drawing itself once per visit may take seconds.
2. **What is it for?** Press feedback, state indication, spatial consistency
   (the solver's panes open from under the mix bar, the chips slide toward
   the pot), or preventing a jarring swap. "It looks cool" is not a purpose
   for anything seen more than once.
3. **Which curve?** Entering or settling → `--ease-out`. Moving while on
   screen → `--ease-in-out`. Colour under the pointer → `ease`. Constant →
   `linear`.
4. **How long?** Press 160 ms. Hover colour 150 ms. Popover and sheet
   150–250 ms. Nothing in the chrome over 300 ms per declaration — and the
   test reads every time in the declaration, delays included, because a
   300 ms delay is a wait the user feels just as a 300 ms slide is. Exits
   faster than entrances.

## The rules, each with the site's own before and after

| Rule | Before (on this site) | After | Why |
| --- | --- | --- | --- |
| Everything pressable shrinks to `scale(0.97)` for 160 ms on `:active` | `.btn`, `.navbtn`, `.view`, the mode tabs, the timeline stations and the lock buttons had hover colour and nothing on press | one `:active` rule per pressable, `transform var(--t-press) var(--ease-out)` in its transition: the theme's six (`.btn`, `.navbtn`, `.view`, `.seg button`, `.radios label`, `.rail a`), rung 2's timeline stations (`.tl button.stn`), the solver's cells, ledger rows, keys, deck cards and hand-so-far steps (`.cell`, `.lrow`, `.key`, `.dcard`, `.sofar .tok` and the rest of `PRESSABLES`) | the page confirms it heard the press before anything else happens; one value everywhere so the whole site presses alike |
| A disabled control does not answer a press | — | `.btn:active:not(:disabled)`, `.seg button:active:not(:disabled)`, the solver's `.dcard:active:not(:disabled)` | feedback on a dead button is a lie |
| Entrances use `--ease-out`, never `ease-in` | rung 2's chips (and rung 3's) swept to the pot with `0.55s ease-in` | `.sweep`: `0.4s var(--ease-out)` | a chip pushed across felt leaves fast and settles; ease-in made it hesitate then lurch |
| Chrome motion stays under 300 ms per declaration | — | the presses at 160 ms, hover colour at 150 ms, the solver's panes sheet (`.sheet`) arriving over 200 ms | a 180 ms dropdown feels more responsive than a 400 ms one, at the same frame rate |
| Exits are faster than entrances; a cascade staggers 30–80 ms and never blocks | — | nothing on the site exits or staggers today; the first one added follows this | the user has decided; the page just answers |
| Nothing moves on load | the homepage's sections rose in on scroll (`.reveal`, 700 ms) | the static pages render still; the chip turning is the one thing that moves without being asked | a page that performs on arrival reads as a demo |
| Hover transforms only where a pointer can hover | an arrow on the old front page nudged 4 px on `:hover`, which a tap leaves stuck out | no hover on the site moves anything; one that does goes inside `@media (hover: hover) and (pointer: fine)` | touch fires hover on tap and never un-fires it |
| Reduced motion means fewer and gentler, not none | `* { animation: none !important; transition: none !important }` in the theme | the presses, the sweep and the sheet are quieted by name; colour transitions stay; the chip is drawn once (`mark.ts` and `chip.js` read the same query) | a state change still has to be legible to someone who turned motion off |
| Reduced motion names the state's own selector, not just the base one | — | the theme quiets `.btn:active:not(:disabled)`, not `.btn:active` | `:not(:disabled)` adds specificity, so quieting the bare `:active` would leave the press in place |
| A one-shot that the code waits on still ends under reduced motion | — | rung 2's and rung 3's chips switch to `sweep-fade`, a fade-only keyframe of the same length | the token unmounts on `animationend`; `animation: none` would strand it on the felt |
| Popovers come from their trigger; modals stay centred | — | the solver's panes sheet (`sheet-in`) drops 4 px into place under the mix bar that opens it, with `--ease-out` | spatial consistency: the thing came from where you were pointing |
| Nothing appears from `scale(0)` | — | no element on the site scales from zero | things in the world have a shape before they arrive |
| Transitions for interruptible UI, keyframes only for one-shot tokens | — | the presses and the colour changes on `.views`, `.seg` and `.rail` are transitions; the swept chips and the panes sheet are keyframes, each played once | keyframes restart from zero when interrupted |
| Only `transform` and `opacity` animate in the chrome | the old top bar's menu animated `width` | nothing on the live site animates a layout property; the archived Cover's menu still does, allowlisted | a layout animation repaints the page around it every frame |

### The allowlist, as it stands

`EXPLANATORY` in the test names the selectors whose motion is explanatory
rather than chrome, each with its reason, and every entry matches a rule
that moves today (`test_every_allowlisted_selector_names_a_rule_that_moves`
fails on an entry that exempts nothing):

- `.sweep`: rungs 2 and 3's chips crossing 58 px of felt to the pot,
  one-shot, 400 ms;
- `.rungs-menu`, `.live i`, `.st::before`: the archived Cover's menu
  width, live dot and in-progress pulse, kept as they were.

Only `.rungs-menu` is a width exception, and only on the archived Cover. No
sheet on the live site animates `width`: a bar that updates while the user
acts is drawn with `transform: scaleX()` from `transform-origin: left`, or
not animated at all (rung 0's `.bar-fill`).
New UI does not get the width exception.

## Adding motion to the site

- Read the token, do not write a curve: `var(--ease-out)`, `var(--t-press)`.
- A new pressable gets its `:active` rule in the same commit as its `:hover`
  rule, `transform var(--t-press) var(--ease-out)` in its transition list,
  and `transform: none` for its `:active` in the same stylesheet's
  reduced-motion block — the theme only knows its own pressables. Add its
  selector to `PRESSABLES` in the test.
- Anything that pulses (`animation`) or travels (a `transform` transition
  other than the press) gets, in the same stylesheet's reduced-motion block,
  a rule for the same selector: `animation: none`, or a fade-only keyframe
  where the code waits on `animationend`; `transition: none` or
  `transform: none` for a travel. The test refuses a pulse or a travel it
  cannot find there by name.
- A new entrance over 300 ms, a figure with its own curve, or a width
  animation is an exception: add the selector to `EXPLANATORY` in the test
  with a one-line reason, the way the existing ones are.
- A chrome change goes into rung 0's theme and is copied to the other four
  apps; the static pages read it through `/theme.css`. A page's own motion
  goes in that page's inline `<style>` in the generator, and the pages are
  regenerated.
- Review it the next day, at 4× slow motion in DevTools (Animations panel),
  and on a phone over the LAN for anything touched.

## Reviewing UI in this repo

Review output is a single markdown table with `Before | After | Why`
columns, one row per finding, never a list of Before:/After: lines:

| Before | After | Why |
| --- | --- | --- |
| `transition: all 300ms` | `transition: transform var(--t-press) var(--ease-out)` | name the property; `all` animates things you did not mean |
| `animation: slide 0.55s ease-in` | `animation: slide 0.4s var(--ease-out)` | ease-in hesitates on the frame the user is watching |
| `.foo:hover { }` and no `:active` | add `.foo:active { transform: scale(0.97) }` | a press has to be heard |
