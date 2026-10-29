"""The static pages only promise what the deploy actually serves.

Five plain HTML files under `web/cover/` sit at the root of the site: the
homepage, Rules, Benchmarks and Releases, written by
`scripts/site/make_static_pages.py` from committed data, and the Cover, the
old front page kept as an archive (see the README there). Nothing here
re-checks their wording beyond the one rule the product copy keeps. What it
checks is the set of things a static page gets wrong silently, and that no
reviewer catches by reading:

- an `href` to a route that does not exist — the temptation each round is to
  link a page "for later", which ships a page whose every third click is a 404;
- a font the pages declare but the build never puts under `/fonts/`, which
  falls back to a system face and nobody notices until a screenshot;
- an inline script that no longer parses;
- a request leaving the host — the pages self-host everything, no CDN;
- a page that drifted from what its generator writes, i.e. a hand edit to a
  file that is regenerated;
- the method's words ("rung", "ladder", "climb") in the product's copy.

The set of real destinations is read out of `vercel.json`, so adding a route
there is what unlocks linking to it.
"""

import html
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
COVER = ROOT / "web" / "cover"
PAGES = ("index.html", "rules.html", "benchmarks.html", "releases.html", "cover.html")
# The four written by the generator; the Cover is the archive and keeps its old chrome.
GENERATED = ("index.html", "rules.html", "benchmarks.html", "releases.html")
GENERATOR = ROOT / "scripts" / "site" / "make_static_pages.py"

# The site's own origin; absolute links to it are the same as root-relative ones.
ORIGIN = "https://drawmaha.app"
# The one other place a page may send a reader: the repository, its files and its releases.
REPO = "https://github.com/AlecROndo/drawmaha-solver"

# Routes served by an app that reads its own fragment (a tab, a chapter).
APP_ROUTES = {"/rung0", "/rung1", "/rung2", "/rung3", "/research", "/learn", "/play", "/solver"}


@pytest.fixture(scope="module")
def pages() -> dict[str, str]:
    return {name: (COVER / name).read_text() for name in PAGES}


@pytest.fixture(scope="module")
def routes() -> set[str]:
    """Every path the deploy answers, from vercel.json plus the filesystem.

    `public/` is served filesystem-first, so each page is reachable by file
    name and `/` is `index.html` with no rewrite; everything else needs a
    rewrite source.
    """
    config = json.loads((ROOT / "vercel.json").read_text())
    return {rewrite["source"] for rewrite in config["rewrites"]} | {"/"} | {
        f"/{name}" for name in PAGES
    }


def _ids(page: str) -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', page))


def _anchor_hrefs(page: str) -> set[str]:
    hrefs = set(re.findall(r'<a\b[^>]*\bhref="([^"]+)"', page))
    assert hrefs, "the page has no links at all — the regex stopped matching"
    return hrefs


@pytest.mark.parametrize("name", PAGES)
def test_every_link_goes_somewhere_that_exists(
    name: str, pages: dict[str, str], routes: set[str]
) -> None:
    page = pages[name]
    for href in _anchor_hrefs(page):
        if href.startswith(REPO):
            continue  # the repository on GitHub: the code, the figure data, the release assets
        target, _, fragment = href.partition("#")

        if target == "":
            # A same-page anchor: the id has to be on this page.
            assert fragment in _ids(page), f"{name}: {href} points at no id on the page"
            continue

        if target in PAGES:
            # A relative link between the pages: the other file has to be
            # checked in beside this one, and the anchor has to exist in it.
            other = pages[target]
            assert not fragment or fragment in _ids(other), (
                f"{name}: {href} anchors into {target} at an id it does not have"
            )
            continue

        if target.startswith(ORIGIN):
            target = target[len(ORIGIN) :] or "/"
        assert target.startswith("/"), f"{name}: {href} is neither a route nor a page"
        assert target in routes, f"{name}: {href} is not a route in vercel.json"
        if fragment and target.lstrip("/") + ".html" in PAGES:
            other = pages[target.lstrip("/") + ".html"]
            assert fragment in _ids(other), f"{name}: {href} anchors into {target} at an id it does not have"
        elif fragment:
            # Cross-page fragments into an app are the app's business; the
            # route existing is all this page can promise.
            assert target in APP_ROUTES, f"{name}: {href} anchors into a page with no app"


@pytest.mark.parametrize("name", PAGES)
def test_nothing_is_fetched_from_anywhere_but_this_host(
    name: str, pages: dict[str, str]
) -> None:
    """No CDN: every src, every stylesheet and every link is same-page, same-site, or a named outside door.

    What a page *loads* (`src`, `<link href>`) is same-host or the site's
    origin, always. What a reader can *follow* (`<a href>`) may also go to
    the repository on GitHub, and nowhere else.
    """
    page = pages[name]
    loaded = set(re.findall(r'\bsrc="([^"]+)"', page)) | set(
        re.findall(r'<link\b[^>]*\bhref="([^"]+)"', page)
    )
    followed = _anchor_hrefs(page)
    for url in loaded | followed:
        if url.startswith(ORIGIN) or (url in followed and url.startswith(REPO)):
            continue
        assert not re.match(r"^(?:[a-z]+:)?//", url), (
            f"{name}: {url} leaves the host — the pages self-host everything"
        )
    assert "url(http" not in page and "@import" not in page, f"{name} pulls CSS from somewhere"


def _copied_fonts() -> set[str]:
    """Every woff2 `vercel_build.sh` copies into public/fonts/, read from its copy_font lines."""
    script = (ROOT / "scripts" / "site" / "vercel_build.sh").read_text()
    copied = set()
    for helper, style in (("copy_font", "normal"), ("copy_font_italic", "italic")):
        for weights, family in re.findall(
            rf"for weight in ([\d ]+); do {helper} ([a-z-]+) \"\$weight\"", script
        ):
            copied |= {f"{family}-latin-{w}-{style}.woff2" for w in weights.split()}
        for family, weight in re.findall(rf"^{helper} ([a-z-]+) (\d+)$", script, re.MULTILINE):
            copied.add(f"{family}-latin-{weight}-{style}.woff2")
    assert copied, "no copy_font line parsed — the regex stopped matching"
    return copied


def test_the_fonts_it_loads_are_the_ones_the_build_serves() -> None:
    """`fonts.css` declares the eight files the build copies, and only those.

    Newsreader 400 / 500 / 400 italic, Geist Sans 400 / 500 / 600 and IBM
    Plex Mono 400 (code) and 600 (the chip's glyph grid). The build copies
    one more, Instrument Serif 400, for the hero studies at /hero, which load
    it themselves; no static page names it.
    A file declared and not copied is a 404 the browser hides behind a
    fallback face.
    """
    declared = re.findall(r"url\(/fonts/([a-z0-9-]+\.woff2)\)", (COVER / "fonts.css").read_text())
    assert len(declared) == 8 == len(set(declared)), f"fonts.css declares {declared}"
    copied = _copied_fonts()
    assert set(declared) <= copied, f"fonts.css declares {set(declared) - copied}, which the build never copies"
    assert copied - set(declared) == {"instrument-serif-latin-400-normal.woff2"}, (
        f"the build copies {copied - set(declared)} that fonts.css does not declare"
    )


@pytest.mark.parametrize("name", GENERATED)
def test_each_page_links_the_theme_the_fonts_and_the_chip(name: str, pages: dict[str, str]) -> None:
    """Fonts come through /fonts.css alone, the chrome through /theme.css, the mark through /chip.js."""
    page = pages[name]
    for asset in ('href="/fonts.css"', 'href="/theme.css"', 'src="/chip.js"'):
        assert asset in page, f"{name} does not load {asset}"
    assert "/fonts/" not in page, f"{name} names a font file directly; fonts.css is the one place"
    assert page.count("mountChip(") == 2, f"{name} should mount the chip twice, in the bar and the footer"


APPS = ("rung0", "rung1", "rung2", "rung3", "solver")


def test_the_favicon_every_page_links_is_one_the_build_serves(pages: dict[str, str]) -> None:
    """Every page and every app links `/favicon.svg`, and the build copies the chip's to exactly that path.

    A linked icon the build never copies is a 404 in every page's console.
    """
    icon = re.compile(r'<link rel="icon" href="([^"]+)"')
    linked = {name: icon.findall(pages[name]) for name in GENERATED}
    for app in APPS:
        linked[f"{app}-viz"] = icon.findall((ROOT / "web" / f"{app}-viz" / "index.html").read_text())
    assert all(hrefs == ["/favicon.svg"] for hrefs in linked.values()), f"icon links: {linked}"
    script = (ROOT / "scripts" / "site" / "vercel_build.sh").read_text()
    assert re.search(r"^cp web/cover/favicon\.svg public/favicon\.svg$", script, re.MULTILINE), (
        "vercel_build.sh does not copy web/cover/favicon.svg to public/favicon.svg"
    )
    svg = (COVER / "favicon.svg").read_text()
    assert svg.lstrip().startswith("<svg") and "#8e2038" in svg, "favicon.svg is not the oxblood chip"


@pytest.mark.parametrize("name", PAGES)
def test_every_inline_script_parses(name: str, pages: dict[str, str]) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH, so the inline scripts cannot be syntax-checked")

    scripts = re.findall(r"<script>(.*?)</script>", pages[name], re.DOTALL)
    assert scripts, f"{name} has no inline <script> blocks — the regex stopped matching"

    with tempfile.TemporaryDirectory() as tmp:
        for index, source in enumerate(scripts):
            path = Path(tmp) / f"{name}-{index}.js"
            path.write_text(source)
            result = subprocess.run(
                [node, "--check", str(path)], capture_output=True, text=True, check=False
            )
            assert result.returncode == 0, (
                f"{name}: inline script #{index} does not parse:\n{result.stderr}"
            )


def _copy(page: str) -> str:
    """What a reader reads: the title, the description, and the text between tags."""
    head = " ".join(re.findall(r"<title>(.*?)</title>", page, re.DOTALL))
    head += " " + " ".join(re.findall(r'<meta name="description" content="([^"]*)"', page))
    body = re.sub(r"<(script|style)\b.*?</\1>", " ", page, flags=re.DOTALL)
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.DOTALL)
    body = re.sub(r"<[^>]+>", " ", body)
    return html.unescape(f"{head} {body}")


@pytest.mark.parametrize("name", GENERATED)
def test_no_page_says_rung_or_ladder_in_its_copy(name: str, pages: dict[str, str]) -> None:
    """The ladder is the method, told in the README; the product's pages speak the product's words."""
    copy = _copy(pages[name])
    assert len(copy.split()) > 100, f"{name}: the copy extractor found almost nothing"
    found = re.findall(r"\b(rungs?|ladders?|climb\w*)\b", copy, re.IGNORECASE)
    assert not found, f"{name} says {sorted(set(found))} in its copy"


def _generator():
    spec = importlib.util.spec_from_file_location("make_static_pages", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_pages_are_what_the_generator_writes(pages: dict[str, str]) -> None:
    """The four pages are regenerated, never hand-edited: a hand edit shows up here as a diff."""
    rendered = _generator().render()
    assert sorted(rendered) == sorted(GENERATED)
    for name in GENERATED:
        assert "scripts/site/make_static_pages.py" in pages[name].split("<head>")[0], (
            f"{name} does not name its generator in its head comment"
        )
        assert rendered[name] == pages[name], (
            f"{name} is not what make_static_pages.py writes; edit the generator and run "
            "`uv run python scripts/site/make_static_pages.py`"
        )


SITE_TSX = ROOT / "web" / "rung0-viz" / "src" / "ui" / "site.tsx"


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _tsx_href(expr: str) -> str:
    """A JSX href (`REPO`, `` `${HOME}/solver` ``) as the generator writes it: root-relative or the repo URL."""
    expr = expr.strip().strip("`'\"")
    if expr == "REPO":
        return REPO
    expr = expr.replace("${REPO}", REPO).replace("${HOME}", "")
    return expr or "/"


def _chrome_of_site_tsx() -> dict[str, object]:
    """The nav, the footer's project paragraph, its column headings and its links, read out of site.tsx."""
    source = SITE_TSX.read_text()
    nav = [
        (_tsx_href(href), label)
        for label, href in re.findall(
            r"\{\s*id:\s*'\w+',\s*label:\s*'([^']+)',\s*href:\s*(`[^`]+`|'[^']+')\s*\}", source
        )
    ]
    footer = re.search(r"<footer>(.*?)</footer>", source, re.DOTALL)
    assert nav and footer, "site.tsx's NAV or <footer> stopped matching"
    body = footer.group(1)
    paragraph = re.search(r"<p>(.*?)</p>", body, re.DOTALL)
    links = [
        (_tsx_href(href), _squash(label))
        for href, label in re.findall(r"<li>\s*<a\s+href=\{(`[^`]*`|\w+)\}\s*>(.*?)</a>\s*</li>", body, re.DOTALL)
    ]
    headings = [_squash(h) for h in re.findall(r"<h4>(.*?)</h4>", body, re.DOTALL)]
    assert paragraph and links and headings, "site.tsx's footer paragraph, links or headings stopped matching"
    return {"nav": nav, "paragraph": _squash(paragraph.group(1)), "links": links, "headings": headings}


@pytest.mark.parametrize("name", GENERATED)
def test_the_pages_chrome_says_what_site_tsx_says(name: str, pages: dict[str, str]) -> None:
    """The top bar and the footer are site.tsx's markup written out twice; the two copies say the same words.

    The generator writes the chrome by hand from site.tsx, so a wording change
    made in one and not the other drifts silently: the nav labels and hrefs,
    the footer's project paragraph, its column headings and its links (label
    and destination, in order) are held equal here.
    """
    want = _chrome_of_site_tsx()
    page = pages[name]
    nav = re.search(r'<nav class="nav"[^>]*>(.*?)</nav>', page, re.DOTALL)
    footer = re.search(r"<footer>(.*?)</footer>", page, re.DOTALL)
    assert nav and footer, f"{name}: no top-bar nav or footer found"
    got_nav = [(href, _squash(label)) for href, label in re.findall(r'<a\b[^>]*\bhref="([^"]+)"[^>]*>(.*?)</a>', nav.group(1))]
    assert got_nav == want["nav"], f"{name}: nav {got_nav} != site.tsx {want['nav']}"
    body = footer.group(1)
    paragraph = re.search(r"<p>(.*?)</p>", body, re.DOTALL)
    assert paragraph and _squash(paragraph.group(1)) == want["paragraph"], (
        f"{name}: the footer paragraph differs from site.tsx's:\n{paragraph and _squash(paragraph.group(1))}\n{want['paragraph']}"
    )
    got_links = [(href, _squash(label)) for href, label in re.findall(r'<li><a href="([^"]+)">(.*?)</a></li>', body, re.DOTALL)]
    assert got_links == want["links"], f"{name}: footer links {got_links} != site.tsx {want['links']}"
    got_headings = [_squash(h) for h in re.findall(r"<h4>(.*?)</h4>", body, re.DOTALL)]
    assert got_headings == want["headings"], f"{name}: footer headings {got_headings} != {want['headings']}"
