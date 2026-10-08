# Range-UI studies — ten forms for "everything I could have here"

Design explorations for the solver's range interface (the view that answers
"here is the spot, here is everything I could be holding, what does the
solver do with each"), plus two replacements for the rung-2 station-rail
lead-up. Not product code: nothing here is served or tested, and the page
follows the site's theme by copying its tokens, not by importing them.

- `index.html` — the artifact. Open it over HTTP from the repo root so the
  relative font paths resolve (`python3 -m http.server 8793`, then
  `/research/range-ui-studies/index.html`); `file://` works with system fonts.
- `compose.py` — samples 200,000 five-card holdings on K♠ 9♠ 4♦, classifies
  each as inner (five hole cards) and outer (two hole + three board) with draw
  flags, and writes `data.js`. The composition is exact to sampling error; the
  action mix is an illustrative function of the two equities, because rung 4
  is not trained. `uv run python research/range-ui-studies/compose.py 200000`,
  from the repo root.
- `data.js` — the generated data the page reads. The sampler is seeded
  (`random.seed(7)`), so re-running the command above rewrites the file
  byte-identically; the committed copy is that output.
