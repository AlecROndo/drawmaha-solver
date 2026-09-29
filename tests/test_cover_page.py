"""The cover page only promises links the deploy actually serves.

The page is one static HTML string, so nothing here re-checks its wording. What
it does check is the one thing a static string gets wrong silently: an `href`
to a route that does not exist. The redesign's nav and footer list rungs 2 to 4
and a set of writeups that have no pages yet, and the temptation each time is
to make them links "for later" — which ships a page whose every third click is
a 404.

The set of real destinations is read out of `vercel.json`, so adding a route
there is what unlocks linking to it.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SITE = "https://drawmaha.app"


def _load_cover_page() -> str:
    """The page's HTML, from the Vercel entrypoint itself."""
    namespace: dict[str, object] = {}
    exec(compile((ROOT / "api" / "index.py").read_text(), "api/index.py", "exec"), namespace)
    page = namespace["PAGE"]
    assert isinstance(page, str)
    return page


@pytest.fixture(scope="module")
def page() -> str:
    return _load_cover_page()


@pytest.fixture(scope="module")
def routes() -> set[str]:
    config = json.loads((ROOT / "vercel.json").read_text())
    return {rewrite["source"] for rewrite in config["rewrites"]}


def test_every_link_goes_somewhere_that_exists(page: str, routes: set[str]) -> None:
    hrefs = set(re.findall(r'href="([^"]+)"', page))
    assert hrefs, "the cover page has no links at all — the regex stopped matching"

    for href in hrefs:
        path, _, fragment = href.partition("#")
        if path == "":
            # A same-page anchor: the id has to be on this page.
            assert f'id="{fragment}"' in page, f"{href} points at no id on the page"
            continue
        if path.startswith(SITE):
            # The rail's chip links home by the canonical address. That is this
            # same deploy, so the path after the origin still has to be a route.
            path = path[len(SITE) :] or "/"
        assert path in routes, f"{href} is not a route in vercel.json"
        if fragment:
            # Cross-page fragments are the visualizers' business, not ours; the
            # route existing is all this page can promise.
            assert path in {"/rung0", "/rung1", "/rung2"}, (
                f"{href} anchors into a page with no app"
            )


def test_every_canvas_on_the_page_has_a_scene_in_the_script(page: str) -> None:
    """The markup and the script are two strings glued together; a canvas whose
    scene the script never defines is skipped at mount and stays blank."""
    scenes = set(re.findall(r'<canvas data-scene="([^"]+)"', page))
    assert scenes == {"logo", "D1", "G5"}, f"the cover page mounts {sorted(scenes)}"
    assert "<script>\n" in page and "requestAnimationFrame(tick);" in page, (
        "HERO_JS is not embedded in the page"
    )
    for name in scenes:
        assert f"SCENES.{name} = {{" in page, f"canvas {name} has no SCENES.{name} in the script"


def test_the_fonts_it_self_hosts_are_the_ones_the_build_copies(page: str) -> None:
    """Every @font-face URL has a matching `copy_font` line in the build script."""
    script = (ROOT / "scripts" / "vercel_build.sh").read_text()

    wanted: dict[str, set[str]] = {}
    for family, weight in re.findall(r"/fonts/([a-z-]+)-latin-(\d+)-normal\.woff2", page):
        wanted.setdefault(family, set()).add(weight)

    # The design runs on three voices and stops there; a fourth family loaded
    # by the page but never copied is the failure this is really guarding.
    assert set(wanted) == {"instrument-serif", "ibm-plex-mono", "kalam"}, (
        f"the cover page loads {sorted(wanted)}, not the three voices the system defines"
    )

    for family, weights in wanted.items():
        for weight in weights:
            looped = re.search(rf"for weight in ([\d ]+); do copy_font {family}\b", script)
            single = re.search(rf"copy_font {family} {weight}\b", script)
            assert single or (looped and weight in looped.group(1).split()), (
                f"the page loads {family} {weight} but vercel_build.sh never copies it"
            )
