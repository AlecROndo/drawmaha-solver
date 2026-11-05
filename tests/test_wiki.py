"""The wiki's renderer (`scripts/site/wiki.py`) reads the dialect `docs/wiki/README.md` documents.

The page it writes is held byte for byte by `tests/test_cover_page.py`; this
file holds the renderer itself: that every page `docs/wiki/nav.json` names
renders with every link resolved, and that the dialect's constructs come out
as the README says.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "scripts" / "site"
spec = importlib.util.spec_from_file_location("wiki", SITE / "wiki.py")
wiki = importlib.util.module_from_spec(spec)
sys.modules["wiki"] = wiki
spec.loader.exec_module(wiki)


def _render(md: str, path: str = "t/x") -> str:
    page = wiki.Page(path, "T", "G", "Title", "d", md)
    return wiki.Renderer(page, wiki.known_headings([page])).render(md)


def test_every_page_in_the_nav_renders_with_its_links_resolved() -> None:
    body, _, _, _ = wiki.render()
    for tab in wiki.load_nav():
        for group in tab["groups"]:
            for path in group["pages"]:
                assert f'id="{path}"' in body, f"{path} is in nav.json but not on the page"


def test_a_heading_anchor_is_its_slug_under_the_pages_id() -> None:
    assert '<h2 id="t/x--the-pot-net">The ÷ pot net</h2>' in _render("## The ÷ pot net")


def test_wiki_and_repo_links_resolve_and_a_bad_one_fails() -> None:
    out = _render("## A\n\n[a](wiki:t/x#a) [r](repo:README.md)")
    assert 'href="#t/x--a"' in out
    assert f'href="{wiki.REPO}/blob/main/README.md"' in out
    with pytest.raises(ValueError, match="unknown wiki page"):
        _render("[a](wiki:nowhere)")
    with pytest.raises(ValueError, match="does not have"):
        _render("[a](wiki:t/x#nothing)")
    with pytest.raises(ValueError, match="does not exist"):
        _render("[a](repo:no/such/file.py)")


def test_code_is_literal_and_tables_keep_a_pipe_inside_code() -> None:
    out = _render("| A | B |\n|---|--:|\n| `x|y` | **2** |")
    assert "<code>x|y</code>" in out and '<td class="r"><strong>2</strong></td>' in out
    assert "<pre><code>a &lt; b</code></pre>" in _render("```\na < b\n```")


def test_blocks_render_and_an_unclosed_one_fails() -> None:
    out = _render(":::note\nhi\n:::\n\n:::details Why\nbody\n:::\n\n:::cards\n- [T](wiki:t/x) Desc.\n:::")
    assert '<aside class="wk-call note">' in out
    assert "<summary>Why</summary>" in out
    assert '<a class="wk-card" href="#t/x"><b>T</b><span>Desc.</span></a>' in out
    with pytest.raises(ValueError, match="unclosed"):
        _render(":::note\nnever closed")


def test_a_numbered_list_after_a_bulleted_one_is_its_own_list() -> None:
    out = _render("- a\n  wrapped\n1. b")
    assert "<ul><li>a wrapped</li></ul>" in out and "<ol><li>b</li></ol>" in out
