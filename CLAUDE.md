# Drawmaha solver — instructions for agents working in this repo

## Anything under `web/` follows `web/DESIGN.md`

That file is the UI standard for drawmaha.app: the motion tokens
(`--ease-out`, `--ease-in-out`, `--t-press`), the rule that every pressable
thing answers a press, the 300 ms ceiling on chrome motion, what reduced
motion keeps, and the review-table format for UI feedback. Read it before
touching a stylesheet, a `<style>` block or a component's classes, and run
`uv run pytest tests/test_ui_standard.py tests/test_shared_chrome.py`
afterwards — the standard is enforced, not advisory.

Two structural facts it depends on:

- The four visualizers each carry a byte-identical copy of the chrome
  (`src/theme.css`, `src/main.tsx`, `src/ui/{site.tsx,mark.ts,ascii.ts}`).
  Edit rung 0's copy, `cp` it to rungs 1, 2 and 3, `cmp` them.
- The homepage, Rules and Cover (`web/cover/*.html`) each inline their own
  copy of the chrome CSS. A chrome change is made in all three as well. See
  `web/cover/README.md` for the font rule and the hand-edit pins.

## Tests

`uv run pytest` runs everything, including the static-page checks and the
UI standard. The visualizers' own suites are `npm test` inside each
`web/rung*-viz`.
