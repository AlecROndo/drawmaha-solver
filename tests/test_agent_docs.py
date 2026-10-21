"""The instruction files agents read stay true to the tree they describe.

Every `AGENTS.md` in the repo is loaded by walk-up when an agent works inside
its directory, and an agent acts on what it reads without re-checking the
code. A path that no longer exists is worse than no instruction at all: it
sends the agent confidently to the wrong place. So this test walks every
`AGENTS.md`, pulls each backticked token that looks like a repo path, and
demands it resolve — against the repo root first, then against the file's
own directory. Globs (`figures/rung*/`) must match something.

Two more things it holds. Each `AGENTS.md` has a `CLAUDE.md` beside it whose
whole content is `@AGENTS.md`: `AGENTS.md` is the canonical file, and the
one-line import is what makes Claude Code's walk-up (which keys on
`CLAUDE.md`) load it. And the root `AGENTS.md` names every nested one in
its subtree map or resolver, directly or through a parent, so nothing is
orphaned.

Tokens that are not paths are skipped by shape: no slash and no file
extension, a command with spaces, a URL or a site route (`/rung3`), a
`$VAR`, a `<placeholder>`, a path under the home directory (`~/…`, on Alec's
machine and not in the repo), a gitignored build output (`public/`), or a
bare data file name (`state.json`, a run's artifact rather than a source).
A bare module name (`game.py`) or a relative sub-path (`src/index.css`)
is accepted when it exists anywhere under the file's own directory, so a
file can speak of "each rung's `game.py`" without listing five paths.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"node_modules", ".git", "dist", ".venv"}
EXTENSIONS = (
    ".py", ".md", ".json", ".sh", ".toml", ".ts", ".tsx", ".css", ".html",
    ".tex", ".yml", ".yaml", ".txt", ".png", ".pdf", ".npz", ".bin",
)
BACKTICKED = re.compile(r"`([^`\n]+)`")
# Written by a build or a run, gitignored, and named in prose anyway.
BUILD_OUTPUTS = {"public", "dist", "runs", "node_modules", "checkpoints"}
# A bare name with one of these extensions is a run's artifact, not a source.
DATA_EXTENSIONS = (".json", ".npz", ".bin", ".png", ".pdf")


def agent_files() -> list[Path]:
    found = []
    for path in ROOT.rglob("AGENTS.md"):
        if SKIP_DIRS.isdisjoint(path.relative_to(ROOT).parts):
            found.append(path)
    return sorted(found)


def looks_like_repo_path(token: str) -> bool:
    if " " in token or token.startswith(("~", "$", "<", "/", "http", "--", "@", "#", ".")):
        return False
    if "<" in token or "…" in token or "{" in token or "::" in token:
        return False
    stripped = token.rstrip(".,;:")
    if stripped.rstrip("/") in BUILD_OUTPUTS:
        return False
    if "/" not in stripped and stripped.endswith(DATA_EXTENSIONS):
        return False
    return "/" in stripped or stripped.endswith(EXTENSIONS)


def resolves(token: str, here: Path) -> bool:
    stripped = token.rstrip(".,;:").rstrip("/") or token
    # Exact, from the root or from the file's directory; a file may also name
    # its own directory (`deepcfr/` at the top of deepcfr/AGENTS.md).
    for base in (ROOT, here, here.parent):
        if "*" not in stripped and (base / stripped).exists():
            return True
    # Otherwise anywhere under the file's own tree, globs included.
    return any(here.rglob(stripped))


@pytest.mark.parametrize("agent_file", agent_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_path_an_agents_file_names_exists(agent_file: Path) -> None:
    here = agent_file.parent
    missing = sorted(
        {
            token
            for token in BACKTICKED.findall(agent_file.read_text())
            if looks_like_repo_path(token) and not resolves(token, here)
        }
    )
    assert not missing, (
        f"{agent_file.relative_to(ROOT)} names paths that do not exist "
        f"(relative to the repo root or to its own directory): {missing}"
    )


@pytest.mark.parametrize("agent_file", agent_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_agents_file_has_its_claude_shim(agent_file: Path) -> None:
    shim = agent_file.parent / "CLAUDE.md"
    assert shim.is_file(), (
        f"{agent_file.relative_to(ROOT)} has no CLAUDE.md beside it; Claude Code's "
        "walk-up keys on CLAUDE.md, so write one containing only `@AGENTS.md`"
    )
    assert shim.read_text().strip() == "@AGENTS.md", (
        f"{shim.relative_to(ROOT)} must contain only `@AGENTS.md`: AGENTS.md is "
        "the canonical file, and a second body here would drift from it"
    )


def test_the_root_file_reaches_every_nested_one() -> None:
    """Every nested AGENTS.md sits under a directory the root file names."""
    root_text = (ROOT / "AGENTS.md").read_text()
    named = {
        token.rstrip("/")
        for token in BACKTICKED.findall(root_text)
        if looks_like_repo_path(token)
    }
    for agent_file in agent_files():
        rel = agent_file.parent.relative_to(ROOT)
        if rel == Path("."):
            continue
        reached = any(
            name == str(rel) or name.startswith(f"{rel}/") or str(rel).startswith(f"{name}/")
            for name in named
        )
        assert reached, (
            f"{agent_file.relative_to(ROOT)} lives under {rel}/, which the root "
            "AGENTS.md never names; add it to the subtree map"
        )


# ---------------------------------------------------------------------------
# The paths the rename moved, named outside the AGENTS.md files
# ---------------------------------------------------------------------------

# The walk above reads only `AGENTS.md`. A docstring, a visualizer's header
# comment, a fixture's `_source` string or a README that still names a file
# at its old place is just as wrong, and an agent follows it just as
# trustingly. The old names are derived from the tree rather than listed:
# every module now under `rules/`, `solver/` or `answer/` once lived flat in
# `minidrawmaha/`, and every script now under `scripts/<group>/` once lived
# flat in `scripts/`. The dated specs and plans under `docs/superpowers/` are
# history and keep their old paths on purpose; `docs/AGENTS.md` states the
# mapping.
MINIDRAWMAHA = ROOT / "src" / "drawmaha_solver" / "minidrawmaha"
SPLIT = ("rules", "solver", "answer")
HISTORY = ROOT / "docs" / "superpowers"
PROSE_EXTENSIONS = (".py", ".md", ".ts", ".tsx", ".toml", ".sh", ".json", ".html", ".css")
# The deal packs and the strategy exports are data, megabytes of it.
PROSE_SIZE_CAP = 1_000_000


def moved_modules() -> set[str]:
    return {
        path.stem
        for group in SPLIT
        for path in (MINIDRAWMAHA / group).glob("*.py")
        if path.stem != "__init__"
    }


def prose_files() -> list[Path]:
    found = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if not SKIP_DIRS.isdisjoint(rel.parts) or HISTORY in path.parents:
            continue
        if (
            path.is_file()
            and path.suffix in PROSE_EXTENSIONS
            and path.stat().st_size <= PROSE_SIZE_CAP
        ):
            found.append(path)
    return sorted(found)


def test_the_split_left_no_flat_module_behind() -> None:
    """`rules/`, `solver/` and `answer/` hold every module that moved."""
    moved = moved_modules()
    assert len(moved) >= 15, moved
    assert not [p.name for p in MINIDRAWMAHA.glob("*.py") if p.stem in moved]


def test_scripts_has_no_flat_entry_points() -> None:
    """Every script sits in a group, so a file directly under `scripts/` is always stale."""
    flat = [p.name for p in (ROOT / "scripts").iterdir() if p.suffix in (".py", ".sh")]
    assert not flat, flat


def test_no_prose_names_a_moved_file_at_its_old_place() -> None:
    old_module = re.compile(
        r"\bminidrawmaha[./](?:" + "|".join(sorted(moved_modules())) + r")\b(?![-\w])"
    )
    old_script = re.compile(r"\bscripts/[A-Za-z0-9_]+\.(?:py|sh)\b")
    stale: list[str] = []
    for path in prose_files():
        text = path.read_text(errors="replace")
        for pattern in (old_module, old_script):
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                stale.append(f"{path.relative_to(ROOT)}:{line}: {match.group(0)}")
    assert not stale, "these name a moved file at its old place:\n" + "\n".join(stale)
