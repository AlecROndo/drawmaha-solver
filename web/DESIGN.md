# The UI standard for drawmaha.app

Every page of the site — the homepage, the Rules tab, the Cover and the
three rung visualizers — is held to this document. It is the design
engineering stance in Emil Kowalski's sense: taste is a set of decisions
made the same way every time, most of which nobody consciously notices, and
which compound into software that feels right. The visual theme (field,
inks, three voices, the chip, windows, hairline rows) is written at the top
of `web/rung0-viz/src/theme.css` and is not repeated here. This document is
about how things move, how they answer a press, and how that is kept true.

`tests/test_ui_standard.py` enforces the mechanical half of it on every
stylesheet in `web/`; the judgement half is the review table at the end.

## Where the CSS lives, and why there are three copies of the chrome

| Surface | File | Notes |
| --- | --- | --- |
| Shared chrome (top bar, Rungs menu, buttons, hero, windows, views, footer) | `web/rung{0,1,2}-viz/src/theme.css` | Byte-identical in all three apps; edit rung 0's and `cp` across (`tests/test_shared_chrome.py`) |
| Each rung's own figures | `web/rung{0,1,2}-viz/src/index.css` | On top of the theme; reads its tokens |
| Homepage, Rules, Cover | `web/cover/{index,rules,cover}.html` | Each inlines its own copy of the chrome CSS, minified, one rule per line |

A rule that belongs to the chrome therefore lands in **four** places: the
theme (then copied twice) and each of the three pages. The tokens below are
declared in all four `:root` blocks and the test checks they agree.

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
motion only (a progress bar filling, the live dot's pulse).

Figures may own a curve. The ladder's line draws with
`cubic-bezier(.12,.8,.2,1)` because "draws on fast, then decelerates over
the last two thirds" is what that figure is saying; a figure's curve is an
explanatory choice and is allowlisted by name in the test. Chrome never
hard-codes a curve.

## Decide in this order before animating anything

1. **How often will the user see it?** A keyboard action (rung 0's Space to
   run, → to step) or a bar that updates a thousand times a second never
   animates; the number printed beside it would disagree with the bar. A
   hover sees 150 ms of colour. Opening a menu is occasional and gets the
   full treatment. A figure drawing itself once per visit may take seconds.
2. **What is it for?** Press feedback, state indication, spatial consistency
   (the rung actions drop from their rung, the chips slide toward the pot),
   or preventing a jarring swap. "It looks cool" is not a purpose for
   anything seen more than once.
3. **Which curve?** Entering or settling → `--ease-out`. Moving while on
   screen → `--ease-in-out`. Colour under the pointer → `ease`. Constant →
   `linear`.
4. **How long?** Press 160 ms. Hover colour 150 ms. Popover and menu
   150–250 ms, with the whole menu settled by 400 ms. Nothing in the chrome
   over 300 ms per declaration — and the test reads every time in the
   declaration, delays included, because a 300 ms delay is a wait the user
   feels just as a 300 ms slide is. Exits faster than entrances.

## The rules, each with the site's own before and after

| Rule | Before (on this site) | After | Why |
| --- | --- | --- | --- |
| Everything pressable shrinks to `scale(0.97)` for 160 ms on `:active` | `.btn`, `.navbtn`, `.view`, the rungs in the menu, the mode tabs, the timeline stations, the lock buttons, the console and seat buttons had hover colour and nothing on press | one `:active` rule per pressable, `transform var(--t-press) var(--ease-out)` added to its transition | the page confirms it heard the press before anything else happens; one value everywhere so the whole site presses alike |
| A disabled control does not answer a press | — | `.btn:active:not(:disabled)`, `.seg button:active:not([disabled])` | feedback on a dead button is a lie |
| Entrances use `--ease-out`, never `ease-in` | rung 2's chips swept to the pot with `0.55s ease-in` | `0.4s var(--ease-out)` | a chip pushed across felt leaves fast and settles; ease-in made it hesitate then lurch |
| Chrome motion stays under 300 ms per declaration | Rungs menu: width 500 ms, rope 550 ms, cells 340 ms with a 75 ms stagger (last cell lands at 640 ms) | width 300 ms, rope 300 ms, cells 240 ms with a 40 ms stagger (settled by 400 ms) | a 180 ms dropdown feels more responsive than a 400 ms one, at the same frame rate |
| Exits are faster than entrances | cells closed in 140 ms | 120 ms, menu's visibility delay matched to it | the user has decided; the page just answers |
| Stagger 30–80 ms between siblings, never blocking | 75 ms (wide) / 55 ms (narrow) | 40 ms / 35 ms | a cascade reads as one gesture, not a queue |
| Scroll reveals are entrances too | `.reveal`: 700 ms `ease` on opacity and travel | opacity 450 ms `ease`, travel 550 ms `--ease-out` | `ease` starts slow; a section arriving should start fast and settle |
| Hover transforms only where a pointer can hover | the ladder's arrow nudged 4 px on `:hover`, which a tap leaves stuck out | wrapped in `@media (hover: hover) and (pointer: fine)` | touch fires hover on tap and never un-fires it |
| Reduced motion means fewer and gentler, not none | `* { animation: none !important; transition: none !important }` in the theme | the travel, scale and pulse are removed by name; colour and opacity transitions stay | a state change still has to be legible to someone who turned motion off |
| Reduced motion names the state's own selector, not just the base one | — | `.rungs-menu, .rungs.open .rungs-menu { transition: none }` | `.rungs.open .rungs-menu` outranks `.rungs-menu`, so quieting only the base rule leaves the open slide in place |
| A one-shot that the code waits on still ends under reduced motion | — | rung 2's chips switch to a fade-only keyframe of the same length | the token unmounts on `animationend`; `animation: none` would strand it on the felt |
| Popovers come from their trigger; modals stay centred | the rung's action strip translates down 4 px from under its rung | kept, with `--ease-out` | spatial consistency: the thing came from where you were pointing |
| Nothing appears from `scale(0)` | — | the one `scaleX(0)` on the site is the menu's rope **drawing** across, a line reveal, not an element popping | things in the world have a shape before they arrive |
| Transitions for interruptible UI, keyframes only for one-shot tokens and the live pulse | — | the menu is all transitions (hover in, hover out mid-open retargets); the swept chips and the blink are keyframes | keyframes restart from zero when interrupted |
| Only `transform` and `opacity` animate in the chrome | the Rungs menu animates `width` | kept, documented | the menu is *meant* to push "Rules" aside; the reflow is one flex row in the top bar |

### The width bars, the one tolerated exception

The homepage's action bars, the Leduc meter, the Rules tab's size bars and
the ladder's bar figures animate `width`. Two of them are gradient fills
that `scaleX` would squash, and all of them are figures a visitor sees once
as they scroll in. They are allowlisted by name in the test with that
reason. New UI does not get this exception: a bar that updates while the
user acts is drawn with `transform: scaleX()` from `transform-origin: left`,
or not animated at all (rung 0's `.bar-fill`).

## Adding motion to the site

- Read the token, do not write a curve: `var(--ease-out)`, `var(--t-press)`.
- A new pressable gets its `:active` rule in the same commit as its `:hover`
  rule, `transform var(--t-press) var(--ease-out)` in its transition list,
  and `transform: none` for its `:active` in the same stylesheet's
  reduced-motion block — the theme only knows its own pressables. Add its
  selector to `PRESSABLES` in the test.
- A new entrance over 300 ms, a figure with its own curve, or a width
  animation is an exception: add the selector to `EXPLANATORY` in the test
  with a one-line reason, the way the existing ones are.
- A chrome change goes into the theme, is copied to the other two apps, and
  is made again in each of the three pages' inline `<style>`.
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
