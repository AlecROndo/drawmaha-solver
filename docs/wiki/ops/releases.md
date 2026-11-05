---
title: Releases: frozen strategies the site ships
description: Each shipped strategy is a GitHub release asset pinned by SHA-256 in one catalogue, with its grade read from a committed file, so every export is reproducible from a clean checkout.
---

A trained strategy is far too big to commit. The frozen LCFR table alone is an 18 MB file. Yet the site has to say exactly which strategy a page plays and how exploitable it is. Releases are how both are done. The strategy file is published as a **GitHub release asset**, and the repository commits only its **name and SHA-256 digest**, plus a pointer to the committed file that holds its grade.

The one place this is all written down is the catalogue, `RELEASES` in [`src/drawmaha_solver/minidrawmaha/answer/releases.py`](repo:src/drawmaha_solver/minidrawmaha/answer/releases.py). The [Releases page](/releases) is generated from it.

## The catalogue, entry by entry

| `--release` name | Tag | What it is | Shipped on |
| --- | --- | --- | --- |
| `tabular` | `rung3-lcfr-2m` | the tabular LCFR average at 2M lockstep iterations, the race's winner | `/solver` (`public/mini/`), `/learn` |
| `deep-pi` | `rung4-deep-t500` | Deep CFR 4 × 256 at iteration 500, as Π (the policy net) | `/solver` (`public/mini-deep-pi/`) |
| `deep-exact` | `rung4-deep-t500` | the same run's exact average | `/solver` (`public/mini-deep-exact/`) |
| `deep-pot-exact` | `rung4-deep-pot0.5-t200` | the ÷ √pot arm's exact average at iteration 200 | `/play` (the deal pack) |
| `deep-pot-pi` | `rung4-deep-pot0.5-t200` | that arm's Π | catalogued, not exported |

Each entry is a `Release` with seven fields: `name`, `directory` (where `/solver`'s export lands, or `None` for a strategy shipped another way), `tag`, `asset`, `sha256`, `grade_file` and `grade_key` (the object in the grades file holding its `exploitability` and `value_p0`).

:::note
The ÷ pot arm has since graded lower than anything released (0.068 chips per hand at 10M hands, in `figures/rung4/pot_1-256x4-s16000-lr5e-4.json`). It is not released yet. The page says which strategy it plays, and the catalogue says what that strategy is.
:::

## Why each field exists

- **Published by tag and asset, refused unless the digest matches.** `fetch_strategy` downloads the asset once into `~/.cache/drawmaha-solver` (or `$XDG_CACHE_HOME/drawmaha-solver`) and re-hashes the cached copy on every call. A clean checkout can therefore reproduce every export, and a local copy is identified by its digest (`release_of`) rather than by its file name.
- **The grade is read, never typed.** At export time, `read_grade` copies the strategy's verdict from its committed grades file into the export's manifest. The tests hold the manifest to the file, so a page quotes a number that has a committed source and never one typed into TypeScript.
- **One provenance block for both exporters.** `provenance_json` writes the `strategy` block that both `/solver`'s `index.json` and the deal pack's manifest open with: digest, tag, asset, rule, column, iteration, workers and seed. The two pages can therefore share the words they use about a strategy (`describeStrategy` in `solver-viz`).
- **A strategy without a `directory` cannot reach `/solver` by accident.** `minidraw-ranges` refuses to export it without an explicit `--out`.

## Cutting a release

1. **Freeze the strategy into a file.** For a tabular checkpoint:
   ```bash
   uv run python scripts/minidrawmaha/freeze_strategy.py \
       runs/modal/lcfr/iter-002000000.npz runs/strategy/strategy-lcfr-2m.npz --grade
   ```
   This prints the file's size and SHA-256. `--grade` re-grades the written file itself (about two minutes, 2 GB), and a faithful freeze matches the checkpoint's own exploitability to the sixth decimal. For Deep CFR, the grader has already saved the three graded objects under `frozen/` on the run's Volume ([Training runs](wiki:ops/modal)).
2. **Make sure the grade is committed** under `figures/`. The release will point at it.
3. **Publish the GitHub release** with the file as its asset. The tags so far are `rung3-lcfr-2m` (2026-10-02), `rung4-deep-t500` (2026-10-26) and `rung4-deep-pot0.5-t200` (2026-10-28).
4. **Add the `Release` entry** with its digest, grade file and key.
5. **Export it** to wherever it ships:
   ```bash
   uv run minidraw-ranges --release <name>   # /solver: web/solver-viz/public/<directory>/
   uv run minidraw-deals                     # /play: the pack for deal_pack.PACK_RELEASE
   ```
6. **Regenerate the static pages** (`uv run python scripts/site/make_static_pages.py`), so `/releases` lists it, and commit everything together.

:::details What an export holds
- **`/solver`'s export** cuts the strategy by public decision point. Each of the 141 points is one column of the table, and every point of a shape (board cards × discards) stands on one sorted key list written once: 970 keys at `b1d0`, 10,170 at `b2d0` and 100,400 at `b2d1`. Probabilities are stored as bytes that sum to a scale of 250, so each entry is within 0.4% of the strategy's own number. The whole strategy is 14 MB raw and about 4 MB gzipped, against 57 MB of float32.
- **The deal pack** is 600 deals dealt in advance, in six chunks. Each deal carries the strategy's per-mille mix at every node either player can reach, plus all 16 post-draw showdowns. The browser follows the path it is on and reads the vector, so it never has to transcribe the infoset key. The pack holds both holes, so the bot's cards can be read from the network tab. That is accepted for a trainer.
:::
