# The wiki

The source of the site's `/wiki` page: everything a new reader needs to come
up to speed on the project, laid out the way a docs site is — tabs across the
top, groups of pages down the left, the page in the middle, its contents on
the right. Its format follows the Laurence docs (a Mintlify site): an
introduction with a primer and glossary, a key-concepts table that links each
idea to its deep dive, theory pages that derive, implementation pages that
map the code, and cards at the foot of a page pointing at what to read next.

`scripts/site/wiki.py` renders it and `scripts/site/make_static_pages.py`
writes the result to `web/cover/wiki.html`, so after editing anything here run

```bash
uv run python scripts/site/make_static_pages.py
```

and commit the regenerated page with the change; `tests/test_cover_page.py`
fails while they differ.

## Layout

- `nav.json` — the tabs, their groups and each group's pages, in order. A page
  is named by its path under `docs/wiki/` without `.md` (`theory/cfr`). A
  page not named in `nav.json` is not rendered, and a name with no file fails
  the build.
- `<tab>/<page>.md` — one file per page.

## A page

```markdown
---
title: Counterfactual regret minimisation
description: One line, shown under the title and in the sidebar's tooltip.
---

Opening paragraph — no heading before it.

## A heading that states the finding

Body.
```

Headings are `##` and `###` (both listed in the page's contents rail) and
`####` (not listed). There is no `#`: the title is the page's `h1`.

## What the renderer understands

| Markdown | Renders as |
| --- | --- |
| paragraphs, `**bold**`, `*italic*`, `` `code` `` | the same |
| `- item` / `1. item`, nested by two spaces | lists |
| GitHub pipe tables (header row, `---` row) | a hairline table |
| fenced code with a language | a code block |
| ```` ```eq ```` fenced block | a display equation, one per line, in the serif |
| `![alt](fig:rung1/exploitability.png)` | a figure from `figures/`, served at `/figures/` |
| `[text](wiki:theory/cfr)`, `[text](wiki:theory/cfr#a-heading)` | a link to another wiki page or a heading on it |
| `[text](repo:src/drawmaha_solver/kuhn/cfr.py)` | a link to the file on GitHub |
| `[text](/research)` | a link to a route of the site |
| `:::note` … `:::` (also `info`, `tip`, `warning`) | a callout |
| `:::details A title` … `:::` | a closed accordion |
| `:::cards` then `- [Title](wiki:x) One line.` items, then `:::` | a grid of link cards |

A heading's anchor is its text lower-cased with every run of other characters
turned into one hyphen (`## The ÷ pot net` → `the-pot-net`).

## Writing it

- **Math is unicode, never LaTeX**: `σᵢ(I, a) = R⁺(I, a) / Σₐ′ R⁺(I, a′)`.
  A standalone equation goes in an `eq` block.
- **Headings state the finding or the object**, not a placeholder: "The
  average strategy converges, the current one cycles", not "Results".
- **Dissect, top-down.** Before a formula or an algorithm step: what we are
  trying to understand and why it exists, in plain words. Then what each piece
  is, why it has this shape, and what moves when a piece grows or shrinks.
- **Every number is one the repository holds** — the README, `figures/`, the
  run log — and names where it comes from. The wiki explains; it does not
  become a second source of results.
- Link the code with `repo:` links and the live surfaces with site routes.
