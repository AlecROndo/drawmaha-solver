"""The wiki: `docs/wiki/` rendered into the body of `/wiki`.

The wiki is a docs site folded into one static page. `docs/wiki/nav.json`
names the tabs, their groups and each group's pages; each page is a Markdown
file with a title and a one-line description in its frontmatter. This module
turns them into one `<main>` body: the tabs across the top, the current tab's
groups down the left, the page in the middle with its headings on the right.
Every page is in the HTML; a few lines of script show the one the hash names
(`/wiki#theory/cfr`), and with no script every page simply reads in order.

The Markdown is a small dialect, documented in `docs/wiki/README.md`, parsed
here with the standard library alone because the generator that calls this
needs nothing else. It is deliberately strict: an unknown `wiki:` link, a page
named in `nav.json` without a file, or an unclosed block fails the build
rather than shipping a dead link.

Ids are global across the page, so a page's id is its path
(`theory/cfr`) and a heading's is the page's id, two hyphens, and the
heading's slug (`theory/cfr--the-walk`). A `wiki:` link is rewritten to that
fragment.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIKI = ROOT / "docs" / "wiki"
REPO = "https://github.com/AlecROndo/drawmaha-solver"


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def slug(text: str) -> str:
    """A heading's anchor: its plain text lower-cased, every other run one hyphen."""
    plain = re.sub(r"[`*_]", "", re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text))
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-") or "section"


@dataclass
class Page:
    """One wiki page: where it sits in the nav, its frontmatter and its Markdown."""

    path: str
    tab: str
    group: str
    title: str
    description: str
    body: str
    headings: list[tuple[int, str, str]] = field(default_factory=list)  # (level, text html, id)


# ------------------------------------------------------------------ loading


def load_nav() -> list[dict]:
    return json.loads((WIKI / "nav.json").read_text())["tabs"]


def parse_frontmatter(text: str, where: str) -> tuple[dict[str, str], str]:
    """Split `---` frontmatter of `key: value` lines from the body."""
    m = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        raise ValueError(f"{where}: no frontmatter")
    meta = {}
    for line in m.group(1).splitlines():
        if line.strip():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"')
    for key in ("title", "description"):
        if not meta.get(key):
            raise ValueError(f"{where}: frontmatter has no {key}")
    return meta, text[m.end():]


def load_pages() -> list[Page]:
    pages = []
    for tab in load_nav():
        for group in tab["groups"]:
            for path in group["pages"]:
                file = WIKI / f"{path}.md"
                if not file.exists():
                    raise FileNotFoundError(f"docs/wiki/nav.json names {path}, but {file.relative_to(ROOT)} does not exist")
                meta, body = parse_frontmatter(file.read_text(), str(file.relative_to(ROOT)))
                pages.append(Page(path, tab["tab"], group["group"], meta["title"], meta["description"], body))
    return pages


# ------------------------------------------------------------------- inline


class Renderer:
    """Markdown to HTML for one page, resolving links against every page."""

    def __init__(self, page: Page, known: dict[str, set[str]]) -> None:
        self.page = page
        self.known = known  # page path -> its heading slugs

    # links -------------------------------------------------------------

    def href(self, target: str) -> tuple[str, bool]:
        """The URL a link points at, and whether it leaves the site."""
        if target.startswith("wiki:"):
            path, _, anchor = target[5:].partition("#")
            if path not in self.known:
                raise ValueError(f"{self.page.path}: link to unknown wiki page {path!r}")
            if anchor:
                if anchor not in self.known[path]:
                    raise ValueError(f"{self.page.path}: link to {path}#{anchor}, a heading that page does not have")
                return f"#{path}--{anchor}", False
            return f"#{path}", False
        if target.startswith("#"):
            anchor = target[1:]
            if anchor not in self.known[self.page.path]:
                raise ValueError(f"{self.page.path}: link to #{anchor}, a heading this page does not have")
            return f"#{self.page.path}--{anchor}", False
        if target.startswith("repo:"):
            rel = target[5:]
            if not (ROOT / rel.split("#")[0]).exists():
                raise ValueError(f"{self.page.path}: repo link to {rel}, which does not exist")
            kind = "tree" if (ROOT / rel.split("#")[0]).is_dir() else "blob"
            return f"{REPO}/{kind}/main/{rel}", True
        if target.startswith("/"):
            return target, False
        if target.startswith(REPO):
            return target, True
        # The site's one outside destination is the repository (tests/test_cover_page.py).
        raise ValueError(f"{self.page.path}: link {target!r} is not wiki:, repo:, a route or the repository")

    def inline(self, text: str) -> str:
        """Code spans first (their contents are literal), then links, then emphasis."""
        out, codes = [], []

        def stash_code(m: re.Match) -> str:
            codes.append(f"<code>{esc(m.group(1))}</code>")
            return f"\x00{len(codes) - 1}\x00"

        text = re.sub(r"`([^`]+)`", stash_code, text)

        def link(m: re.Match) -> str:
            label, target = m.group(1), m.group(2)
            url, outside = self.href(target)
            rel = ' rel="noopener"' if outside else ""
            return f'<a href="{esc(url)}"{rel}>\x01{label}\x02</a>'

        text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, text)
        # escape everything outside the tags just built
        parts = re.split(r"(<a [^>]*>|</a>|\x01|\x02)", text)
        for part in parts:
            if part.startswith("<a ") or part == "</a>":
                out.append(part)
            elif part in ("\x01", "\x02"):
                continue
            else:
                out.append(esc(part).replace("&#x27;", "'"))
        s = "".join(out)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
        return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], s)

    # blocks ------------------------------------------------------------

    def render(self, text: str) -> str:
        lines = text.splitlines()
        return self.blocks(lines, 0, len(lines))

    def blocks(self, lines: list[str], i: int, end: int) -> str:
        out: list[str] = []
        para: list[str] = []

        def flush() -> None:
            if para:
                out.append(f"<p>{self.inline(' '.join(s.strip() for s in para))}</p>")
                para.clear()

        while i < end:
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                flush()
                i += 1
            elif stripped.startswith("```"):
                flush()
                lang = stripped[3:].strip()
                j = i + 1
                while j < end and not lines[j].strip().startswith("```"):
                    j += 1
                if j >= end:
                    raise ValueError(f"{self.page.path}: unclosed code fence at line {i + 1}")
                body = lines[i + 1 : j]
                if lang == "eq":
                    rows = "".join(f'<div class="wk-eq-row">{self.inline(r.strip())}</div>' for r in body if r.strip())
                    out.append(f'<div class="wk-eq" role="math">{rows}</div>')
                else:
                    cls = f' class="lang-{esc(lang)}"' if lang else ""
                    out.append(f"<pre><code{cls}>{esc(chr(10).join(body))}</code></pre>")
                i = j + 1
            elif stripped.startswith(":::"):
                flush()
                i = self.directive(lines, i, end, out)
            elif m := re.match(r"(#{2,4})\s+(.*)", stripped):
                flush()
                level = len(m.group(1))
                hid = f"{self.page.path}--{slug(m.group(2))}"
                inner = self.inline(m.group(2))
                if level <= 3:
                    self.page.headings.append((level, inner, hid))
                out.append(f'<h{level} id="{esc(hid)}">{inner}</h{level}>')
                i += 1
            elif stripped.startswith("|") and i + 1 < end and re.match(r"\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
                flush()
                i = self.table(lines, i, end, out)
            elif re.match(r"\s*([-*]|\d+\.)\s+", line):
                flush()
                i = self.list_(lines, i, end, out)
            elif m := re.match(r"!\[([^\]]*)\]\(fig:([^)]+)\)", stripped):
                flush()
                rel = m.group(2)
                if not (ROOT / "figures" / rel).exists():
                    raise ValueError(f"{self.page.path}: figure figures/{rel} does not exist")
                alt = m.group(1)
                out.append(
                    f'<figure class="wk-fig"><img src="/figures/{esc(rel)}" alt="{esc(alt)}" loading="lazy">'
                    f"<figcaption>{self.inline(alt)}</figcaption></figure>"
                )
                i += 1
            else:
                para.append(line)
                i += 1
        flush()
        return "\n".join(out)

    def directive(self, lines: list[str], i: int, end: int, out: list[str]) -> int:
        head = lines[i].strip()[3:].strip()
        kind, _, arg = head.partition(" ")
        depth, j = 1, i + 1
        while j < end:
            s = lines[j].strip()
            if s == ":::":
                depth -= 1
                if depth == 0:
                    break
            elif s.startswith(":::") and len(s) > 3:
                depth += 1
            j += 1
        if j >= end:
            raise ValueError(f"{self.page.path}: unclosed :::{kind} at line {i + 1}")
        if kind in ("note", "info", "tip", "warning"):
            inner = self.blocks(lines, i + 1, j)
            label = {"note": "Note", "info": "Info", "tip": "Tip", "warning": "Warning"}[kind]
            out.append(f'<aside class="wk-call {kind}"><p class="wk-call-k">{label}</p>{inner}</aside>')
        elif kind == "details":
            inner = self.blocks(lines, i + 1, j)
            out.append(f'<details class="wk-det"><summary>{self.inline(arg or "Details")}</summary><div class="wk-det-body">{inner}</div></details>')
        elif kind == "cards":
            cards = []
            for raw in lines[i + 1 : j]:
                s = raw.strip()
                if not s:
                    continue
                m = re.match(r"[-*]\s+\[([^\]]+)\]\(([^)\s]+)\)\s*[—–-]?\s*(.*)", s)
                if not m:
                    raise ValueError(f"{self.page.path}: a :::cards line must be '- [Title](target) text', got {s!r}")
                url, outside = self.href(m.group(2))
                rel = ' rel="noopener"' if outside else ""
                cards.append(
                    f'<a class="wk-card" href="{esc(url)}"{rel}><b>{self.inline(m.group(1))}</b>'
                    f"<span>{self.inline(m.group(3))}</span></a>"
                )
            out.append(f'<div class="wk-cards">{"".join(cards)}</div>')
        else:
            raise ValueError(f"{self.page.path}: unknown block :::{kind}")
        return j + 1

    def table(self, lines: list[str], i: int, end: int, out: list[str]) -> int:
        def cells(row: str) -> list[str]:
            row = row.strip().strip("|")
            parts, cur, in_code = [], "", False
            for ch in row:
                if ch == "`":
                    in_code = not in_code
                if ch == "|" and not in_code:
                    parts.append(cur)
                    cur = ""
                else:
                    cur += ch
            parts.append(cur)
            return [p.strip() for p in parts]

        head = cells(lines[i])
        aligns = ["r" if c.strip().endswith(":") and not c.strip().startswith(":") else "" for c in cells(lines[i + 1])]
        j = i + 2
        rows = []
        while j < end and lines[j].strip().startswith("|"):
            rows.append(cells(lines[j]))
            j += 1

        def cls(k: int, first: bool) -> str:
            names = [n for n in ("k" if first and k == 0 else "", aligns[k] if k < len(aligns) else "") if n]
            return f' class="{" ".join(names)}"' if names else ""

        th = "".join(f"<th{cls(k, False)}>{self.inline(c)}</th>" for k, c in enumerate(head))
        body = "".join(
            "<tr>" + "".join(f"<td{cls(k, True)}>{self.inline(c)}</td>" for k, c in enumerate(r)) + "</tr>" for r in rows
        )
        out.append(f'<div class="wk-table"><table class="hair"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>')
        return j

    def list_(self, lines: list[str], i: int, end: int, out: list[str]) -> int:
        """A list and its nested lists, by indentation; continuation lines join their item."""
        base = len(lines[i]) - len(lines[i].lstrip())
        ordered = bool(re.match(r"\s*\d+\.", lines[i]))
        items: list[list[str]] = []
        j = i
        while j < end:
            line = lines[j]
            if not line.strip():
                # a blank line ends the list unless the next line continues it
                if j + 1 < end and re.match(r"\s*([-*]|\d+\.)\s+", lines[j + 1]) and len(lines[j + 1]) - len(lines[j + 1].lstrip()) >= base:
                    j += 1
                    continue
                break
            indent = len(line) - len(line.lstrip())
            m = re.match(r"\s*([-*]|\d+\.)\s+(.*)", line)
            if m and indent == base:
                if bool(re.match(r"\d+\.", m.group(1))) != ordered:
                    break  # a numbered list after a bulleted one is a new list
                items.append([m.group(2)])
            elif m is None and items and len(items[-1]) == 1 and indent > base:
                items[-1][0] += " " + line.strip()  # a wrapped first line, indented
            elif indent > base and items:
                items[-1].append(line[base + 2 :] if len(line) > base + 2 else line.strip())
            elif m is None and items and indent >= base and len(items[-1]) == 1:
                items[-1][0] += " " + line.strip()  # a wrapped first line
            else:
                break
            j += 1
        tag = "ol" if ordered else "ul"
        rendered = []
        for item in items:
            first, rest = item[0], item[1:]
            inner = self.inline(first)
            if rest:
                nested_lines = rest
                inner += self.blocks(nested_lines, 0, len(nested_lines))
            rendered.append(f"<li>{inner}</li>")
        out.append(f"<{tag}>{''.join(rendered)}</{tag}>")
        return j


# ------------------------------------------------------------------- the page


def known_headings(pages: list[Page]) -> dict[str, set[str]]:
    """Every page's heading slugs, read before any page renders so links can be checked."""
    known = {}
    for p in pages:
        slugs = set()
        fence = False
        for line in p.body.splitlines():
            if line.strip().startswith("```"):
                fence = not fence
            elif not fence and (m := re.match(r"\s*#{2,4}\s+(.*)", line)):
                slugs.add(slug(m.group(1)))
        known[p.path] = slugs
    return known


CSS = """
/* The wiki: tabs, then the tab's groups | the page | its headings. */
.wk-head { padding: 48px 0 0; }
.wk-head .eyebrow { margin-bottom: 10px; }
.wk-head h1 { font-size: 40px; }
.wk-head .views { margin-top: 24px; }
.wk-grid { display: grid; grid-template-columns: 220px minmax(0, 1fr) 200px; gap: 48px; padding-top: 40px; }
.wk-side .k { margin-top: 26px; }
.wk-side .k:first-child { margin-top: 0; }
.wk-side ul { margin-bottom: 4px; }
.wk-toc { font-size: 12px; }
.wk-toc li.l3 a { padding-left: 26px; }
.wk-page { min-width: 0; max-width: 760px; }
.wk-page + .wk-page { margin-top: 96px; padding-top: 48px; border-top: 1px solid var(--rule); }
.wk-page > header .eyebrow { margin-bottom: 12px; }
.wk-page > header h1 { font-size: 44px; max-width: 22ch; }
.wk-page > header .lede { font-size: 19px; margin-top: 14px; }
.wk-page > header { padding-bottom: 28px; border-bottom: 1px solid var(--rule); }
.wk-body { font-family: var(--serif); font-size: 18px; line-height: 1.62; color: var(--ink); }
.wk-body p, .wk-body ul, .wk-body ol { margin: 18px 0 0; }
.wk-body ul, .wk-body ol { padding-left: 22px; }
.wk-body li { margin: 6px 0 0; }
.wk-body li > ul, .wk-body li > ol { margin-top: 6px; }
.wk-body strong { font-weight: 500; }
.wk-body h2 { margin: 56px 0 0; scroll-margin-top: 24px; }
.wk-body h3 { margin: 36px 0 0; scroll-margin-top: 24px; }
.wk-body h4 { margin: 28px 0 0; font-family: var(--sans); font-size: 15px; font-weight: 600; scroll-margin-top: 24px; }
.wk-body > :first-child { margin-top: 32px; }
.wk-body a { color: var(--accent); text-decoration: underline; text-decoration-color: var(--accent-tint); text-underline-offset: 3px; transition: text-decoration-color 0.15s; }
.wk-body a:hover { text-decoration-color: var(--accent); }
.wk-body pre { margin: 20px 0 0; padding: 16px 18px; background: var(--paper-2); border-top: 1px solid var(--rule); overflow-x: auto; font-size: 13px; line-height: 1.55; }
.wk-body pre code { background: none; padding: 0; font-size: inherit; }
.wk-table { margin: 22px 0 0; overflow-x: auto; }
.wk-table table.hair { font-family: var(--sans); }
.wk-table td { line-height: 1.5; }
.wk-eq { margin: 22px 0 0; padding: 14px 0; border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); text-align: center; font-size: 20px; overflow-x: auto; }
.wk-eq-row + .wk-eq-row { margin-top: 8px; }
.wk-call { margin: 24px 0 0; padding: 14px 18px 16px; background: var(--paper-2); border-top: 2px solid var(--ink-3); font-size: 16px; }
.wk-call.info { border-top-color: var(--act-check); }
.wk-call.tip { border-top-color: var(--act-call); }
.wk-call.warning { border-top-color: var(--accent); }
.wk-call > p.wk-call-k { margin: 0; font-family: var(--sans); font-size: 12px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); }
.wk-call > :nth-child(2) { margin-top: 6px; }
.wk-det { margin: 24px 0 0; border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }
.wk-det summary { cursor: pointer; padding: 12px 0; font-family: var(--sans); font-size: 15px; font-weight: 500; color: var(--ink); transition: color 0.15s; }
.wk-det summary:hover { color: var(--accent); }
.wk-det-body { padding-bottom: 18px; }
.wk-det-body > :first-child { margin-top: 0; }
.wk-fig { margin: 26px 0 0; border-top: 1px solid var(--rule); padding-top: 14px; }
.wk-fig img { display: block; max-width: 100%; height: auto; }
.wk-fig figcaption { margin-top: 10px; font-family: var(--sans); font-size: 13px; line-height: 1.5; color: var(--ink-3); }
.wk-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 0 24px; margin: 28px 0 0; }
.wk-card { display: block; padding: 16px 0 18px; border-top: 1px solid var(--rule); color: var(--ink); text-decoration: none; font-family: var(--sans); transition: color 0.15s, border-color 0.15s, transform var(--t-press) var(--ease-out); }
.wk-body a.wk-card { color: var(--ink); text-decoration: none; }
.wk-card b { display: block; font-family: var(--serif); font-weight: 400; font-size: 20px; line-height: 1.25; }
.wk-card span { display: block; margin-top: 6px; font-size: 14px; line-height: 1.45; color: var(--ink-3); }
.wk-card:hover { border-color: var(--ink); }
.wk-body a.wk-card:hover b { color: var(--accent); }
.wk-card:active { transform: scale(0.97); }
.wk-pager { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 64px; }
.wk-pager a { display: block; padding-top: 14px; border-top: 1px solid var(--rule); text-decoration: none; color: var(--ink); font-family: var(--sans); transition: color 0.15s, border-color 0.15s, transform var(--t-press) var(--ease-out); }
.wk-pager a:hover { border-color: var(--ink); }
.wk-pager a:active { transform: scale(0.97); }
.wk-pager a span { display: block; font-size: 12px; color: var(--ink-3); letter-spacing: 0.08em; text-transform: uppercase; }
.wk-pager a b { display: block; margin-top: 4px; font-family: var(--serif); font-weight: 400; font-size: 19px; }
.wk-pager a.next { text-align: right; grid-column: 2; }
/* With the script, only the current tab's groups, the current page and its headings show. */
.wk-js .wk-side[hidden], .wk-js .wk-page[hidden], .wk-js .wk-toc[hidden] { display: none; }
.wk-js .wk-page + .wk-page { margin-top: 0; padding-top: 0; border-top: 0; }
@media (max-width: 1180px) {
  .wk-grid { grid-template-columns: 200px minmax(0, 1fr); }
  .wk-tocs { display: none; }
}
@media (max-width: 860px) {
  .wk-grid { grid-template-columns: minmax(0, 1fr); gap: 28px; }
  .wk-cards, .wk-pager { grid-template-columns: 1fr; }
  .wk-pager a.next { grid-column: 1; }
  .wk-page > header h1 { font-size: 34px; }
  .wk-body { font-size: 17px; }
}
@media (prefers-reduced-motion: reduce) {
  .wk-card:active, .wk-pager a:active { transform: none; }
}
"""

SCRIPT = """<script>
(function () {
  var root = document.documentElement;
  root.classList.add('wk-js');
  var pages = Array.prototype.slice.call(document.querySelectorAll('.wk-page'));
  var sides = document.querySelectorAll('.wk-side');
  var tocs = document.querySelectorAll('.wk-toc');
  var tabs = document.querySelectorAll('.wk-tab');
  var first = pages[0].id;
  function show() {
    var target = decodeURIComponent(location.hash.slice(1)) || first;
    var id = target.split('--')[0];
    var page = document.getElementById(id);
    if (!page || !page.classList.contains('wk-page')) { id = first; page = document.getElementById(first); }
    var tab = page.getAttribute('data-tab');
    pages.forEach(function (p) { p.hidden = p !== page; });
    Array.prototype.forEach.call(sides, function (s) { s.hidden = s.getAttribute('data-tab') !== tab; });
    Array.prototype.forEach.call(tocs, function (t) { t.hidden = t.getAttribute('data-page') !== id; });
    Array.prototype.forEach.call(tabs, function (t) {
      if (t.getAttribute('data-tab') === tab) { t.classList.add('current'); t.setAttribute('aria-current', 'true'); }
      else { t.classList.remove('current'); t.removeAttribute('aria-current'); }
    });
    Array.prototype.forEach.call(document.querySelectorAll('.wk-side a'), function (a) {
      if (a.getAttribute('href') === '#' + id) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current');
    });
    document.title = page.getAttribute('data-title') + ' \\u00b7 Drawmaha wiki';
    var heading = target !== id ? document.getElementById(target) : null;
    if (heading) heading.scrollIntoView(); else window.scrollTo(0, 0);
    spy();
  }
  function spy() {
    var toc = document.querySelector('.wk-toc:not([hidden])');
    if (!toc) return;
    var links = toc.querySelectorAll('a');
    var current = null;
    Array.prototype.forEach.call(links, function (a) {
      var h = document.getElementById(a.getAttribute('href').slice(1));
      if (h && h.getBoundingClientRect().top < 120) current = a;
    });
    Array.prototype.forEach.call(links, function (a) {
      if (a === current) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current');
    });
  }
  window.addEventListener('hashchange', show);
  window.addEventListener('scroll', spy, { passive: true });
  show();
})();
</script>"""


def render() -> tuple[str, str, str, str]:
    """The wiki's body, its page CSS, its script, and the sources it was written from."""
    pages = load_pages()
    known = known_headings(pages)
    bodies = {p.path: Renderer(p, known).render(p.body) for p in pages}
    tabs = load_nav()

    first_of = {tab["tab"]: tab["groups"][0]["pages"][0] for tab in tabs}
    tab_links = "\n".join(
        f'    <a class="view wk-tab" data-tab="{esc(t["tab"])}" href="#{esc(first_of[t["tab"]])}"><b>{esc(t["tab"])}</b></a>'
        for t in tabs
    )
    by_path = {p.path: p for p in pages}

    sides = []
    for t in tabs:
        groups = []
        for g in t["groups"]:
            items = "".join(
                f'<li><a href="#{esc(path)}" title="{esc(by_path[path].description)}">{esc(by_path[path].title)}</a></li>'
                for path in g["pages"]
            )
            groups.append(f'<p class="k">{esc(g["group"])}</p><ul>{items}</ul>')
        sides.append(f'<nav class="rail wk-side" data-tab="{esc(t["tab"])}" aria-label="{esc(t["tab"])} pages">{"".join(groups)}</nav>')

    articles, tocs = [], []
    for k, p in enumerate(pages):
        prev = pages[k - 1] if k > 0 else None
        nxt = pages[k + 1] if k + 1 < len(pages) else None
        pager = ""
        if prev or nxt:
            left = f'<a class="prev" href="#{esc(prev.path)}"><span>Previous</span><b>{esc(prev.title)}</b></a>' if prev else ""
            right = f'<a class="next" href="#{esc(nxt.path)}"><span>Next</span><b>{esc(nxt.title)}</b></a>' if nxt else ""
            pager = f'<nav class="wk-pager" aria-label="Previous and next page">{left}{right}</nav>'
        articles.append(
            f'<article class="wk-page" id="{esc(p.path)}" data-tab="{esc(p.tab)}" data-title="{esc(p.title)}">\n'
            f'<header><p class="eyebrow">{esc(p.tab)} · {esc(p.group)}</p><h1>{esc(p.title)}</h1>'
            f'<p class="lede">{esc(p.description)}</p></header>\n'
            f'<div class="wk-body">\n{bodies[p.path]}\n</div>\n{pager}\n</article>'
        )
        if p.headings:
            items = "".join(
                f'<li class="l{level}"><a href="#{esc(hid)}">{re.sub(r"<[^>]+>", "", text)}</a></li>'
                for level, text, hid in p.headings
            )
            tocs.append(f'<nav class="rail wk-toc" data-page="{esc(p.path)}" aria-label="On this page"><p class="k">On this page</p><ul>{items}</ul></nav>')

    body = f"""<section class="wk-head">
  <p class="eyebrow">Wiki</p>
  <h1>Everything about the solver, in one place</h1>
  <nav class="views" aria-label="Wiki sections">
{tab_links}
  </nav>
</section>
<div class="wk-grid">
<div class="wk-sides">
{chr(10).join(sides)}
</div>
<div class="wk-pages">
{chr(10).join(articles)}
</div>
<div class="wk-tocs">
{chr(10).join(tocs)}
</div>
</div>"""
    return body, CSS, SCRIPT, "docs/wiki/ (nav.json and every page it names)"
