"""The static pages only promise what the deploy actually serves.

The homepage, its Rules tab and the Cover are three self-contained HTML files
under `web/cover/` — the first two designed outside the repo and exported, the
Cover authored here (see the README there) — so nothing here re-checks their
wording. What it checks is the set of things a static page gets wrong
silently, and that no reviewer catches by reading:

- an `href` to a route that does not exist — the nav lists rungs that have no
  page yet, and the temptation each round is to link them "for later", which
  ships a page whose every third click is a 404;
- a font the page loads but the build never puts under `/fonts/`, which falls
  back to a system face and nobody notices until a screenshot;
- an inline script that no longer parses after a hand edit;
- a request leaving the host — the pages self-host everything, no CDN.

The set of real destinations is read out of `vercel.json`, so adding a route
there is what unlocks linking to it.
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
COVER = ROOT / "web" / "cover"
PAGES = ("index.html", "rules.html", "cover.html")

# The site's own origin; absolute links to it are the same as root-relative ones.
ORIGIN = "https://drawmaha.app"


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


def _hrefs(page: str) -> set[str]:
    hrefs = set(re.findall(r'\bhref="([^"]+)"', page))
    assert hrefs, "the page has no links at all — the regex stopped matching"
    return hrefs


@pytest.mark.parametrize("name", PAGES)
def test_every_link_goes_somewhere_that_exists(
    name: str, pages: dict[str, str], routes: set[str]
) -> None:
    page = pages[name]
    for href in _hrefs(page):
        target, _, fragment = href.partition("#")

        if target == "":
            # A same-page anchor: the id has to be on this page.
            assert fragment in _ids(page), f"{name}: {href} points at no id on the page"
            continue

        if target in PAGES:
            # A relative link between the two pages: the other file has to be
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
        if fragment:
            # Cross-page fragments are the visualizers' business, not ours; the
            # route existing is all this page can promise.
            assert target in {"/rung0", "/rung1", "/rung2"}, (
                f"{name}: {href} anchors into a page with no app"
            )


@pytest.mark.parametrize("name", PAGES)
def test_nothing_is_fetched_from_anywhere_but_this_host(
    name: str, pages: dict[str, str]
) -> None:
    """No CDN: every src and href is same-page, same-site, or the site's origin."""
    page = pages[name]
    urls = set(re.findall(r'\b(?:href|src)="([^"]+)"', page))
    for url in urls:
        if url.startswith(ORIGIN):
            continue
        assert not re.match(r"^(?:[a-z]+:)?//", url), (
            f"{name}: {url} leaves the host — the pages self-host everything"
        )


@pytest.mark.parametrize("name", PAGES)
def test_the_fonts_it_loads_are_the_ones_the_build_serves(
    name: str, pages: dict[str, str]
) -> None:
    """Every /fonts/ URL is copied by `copy_font` or checked in under web/cover/fonts/.

    The build copies the families the visualizers already bundle out of
    @fontsource; the cover page's own voices (Instrument Sans, the serif's
    italic) are files in this directory. A font that neither route produces
    is a 404 the browser hides behind a fallback face.
    """
    page = pages[name]
    script = (ROOT / "scripts" / "vercel_build.sh").read_text()
    checked_in = {path.name for path in (COVER / "fonts").glob("*.woff2")}

    wanted = set(re.findall(r"/fonts/([a-z0-9-]+\.woff2)", page))
    assert wanted, f"{name} loads no fonts from /fonts/ — the regex stopped matching"

    for file in wanted:
        if file in checked_in:
            continue
        # copy_font's argument pattern is `<family>-latin-<weight>-normal.woff2`,
        # so only an upright weight can come from the build script.
        match = re.fullmatch(r"([a-z-]+)-latin-(\d+)-normal\.woff2", file)
        assert match, f"{name} loads {file}, which is neither checked in nor a copy_font shape"
        family, weight = match.groups()
        looped = re.search(rf"for weight in ([\d ]+); do copy_font {family}\b", script)
        single = re.search(rf"copy_font {family} {weight}\b", script)
        assert single or (looped and weight in looped.group(1).split()), (
            f"{name} loads {file} but vercel_build.sh never copies it"
        )


def test_the_checked_in_fonts_are_not_ones_the_build_already_copies() -> None:
    """web/cover/fonts/ holds only what copy_font cannot reach; no duplicates."""
    script = (ROOT / "scripts" / "vercel_build.sh").read_text()
    for path in (COVER / "fonts").glob("*.woff2"):
        match = re.fullmatch(r"([a-z-]+)-latin-(\d+)-normal\.woff2", path.name)
        if not match:
            continue
        family, weight = match.groups()
        looped = re.search(rf"for weight in ([\d ]+); do copy_font {family}\b", script)
        single = re.search(rf"copy_font {family} {weight}\b", script)
        assert not (single or (looped and weight in looped.group(1).split())), (
            f"{path.name} is checked in under web/cover/fonts/ but the build already copies it"
        )


# The fixes that post-date the studies source. `index.html` and `rules.html`
# are exports (see the README), but the source they come from predates #42:
# the bowl settling in the Solve widget (#42), the rung-0 triangle drawing that
# same bowl and the Rules tab's beat anchor (#43) are hand-edits to the exported
# files. A re-export of the stale source would put the old code back and
# nothing would fail, so each edit is pinned here by a line only the new code
# has. Porting the edits into the source and re-exporting retires this.
HAND_EDITS = {
    "index.html": (
        "function bowlArc(tab, u)",  # the ball's clock converted to arc length, shared by both triangles (#43)
        "function bowlTable(m, R0, phase)",  # the bowl path in a triangle's pixels (#43)
        "bowlTable(m, geo.P0[1] - m[1], phase)",  # the rung-0 figure draws the bowl, not its own run (#43)
        "acc = BOWL.acc",  # the figure's clock honours ?bowl= like the Solve widget (#44)
        "ctx.fillText('rung 0 lands at'",  # the end label says the numbers are the real run's (#44)
    ),
    "rules.html": (
        "(el.lastElementChild || el).getBoundingClientRect().bottom",  # a beat's anchor is its text (#43)
        "Math.min(5, Math.max(0, Math.round(feltP)));",  # the lit beat is the one whose text is nearest mid-screen (#43)
    ),
}


@pytest.mark.parametrize("name", sorted(HAND_EDITS))
def test_the_hand_edits_survived_the_last_export(
    name: str, pages: dict[str, str]
) -> None:
    for marker in HAND_EDITS[name]:
        assert marker in pages[name], (
            f"{name} no longer contains `{marker}`: a re-export of the studies "
            "source has reverted a hand edit (see web/cover/README.md)"
        )


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
