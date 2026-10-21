# `rung3-viz/` — `/rung3`: the research page and the trainer

- `src/data/grades.json` and `src/data/answer_sheet.json` are copies of
  `figures/rung3/`; `minidraw-analysis` regenerates those, and the copies are
  refreshed by hand in the same change.
- `public/deals/` is the pack written by `uv run minidraw-deals`: 600 deals
  in six chunks, each with the frozen strategy's mix at every node either
  player can reach, and `public/deals/index.json` recording the seed and the
  strategy's digest. The page refuses a chunk whose deal count disagrees with
  the manifest, or a deal missing a node its grammar can reach. A re-export
  keeps the seed; the digest changes only when the frozen strategy does.
- `src/minidraw.ts` transcribes only the pot-limit betting grammar, and
  `src/minidraw.test.ts` pins it against the same round-1 lines the Python
  suite pins. Everything else on the page is read from the pack or the data
  copies; nothing looks the strategy up.
- The pack holds both holes, so the bot's cards are readable in the network
  tab. That is accepted for a trainer and is not a bug to fix.
