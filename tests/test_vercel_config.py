import json
import re
import shlex
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERCEL_JSON = REPO_ROOT / "vercel.json"

# ---------------------------------------------------------------------------
# The deploy config's one load-bearing, invisible field
# ---------------------------------------------------------------------------
#
# Vercel auto-detects a framework preset when vercel.json does not pin one, and
# the pyproject.toml at the repo root makes it detect "python". A backend
# framework preset appends a `/(.*) -> /python` catch-all to the route table,
# which swallows every path below the explicit rewrites and hands it to a
# serverless function — so /, /rules, /rung0, /rung1, their /assets/* bundles
# and /fonts/* all build correctly into public/ and then never get served.
# Nothing fails: the build is green, the deploy is green, and the site is gone.
#
# `"framework": null` is what turns that off, and it is a single line in a JSON
# file that no other code references. Dropping it during a merge or a config
# regeneration re-breaks the live demo silently, so it gets an explicit test.

def _vercel_config() -> dict:
    return json.loads(VERCEL_JSON.read_text())

def test_framework_detection_is_pinned_off():
    config = _vercel_config()
    # `in` before the value check on purpose: a missing key is the regression,
    # and `config.get("framework") is None` would pass for it.
    assert "framework" in config, (
        'vercel.json must pin "framework": null; without it Vercel detects the '
        '"python" preset from pyproject.toml and routes every path to a function'
    )
    assert config["framework"] is None


def test_the_cover_page_is_served_from_the_filesystem():
    # The cover page used to be a python function behind a "/" -> "/api/index"
    # rewrite. It is public/index.html now and the function is gone, so a "/"
    # rewrite coming back (say, from a merge) would point the root at nothing.
    # The Rules tab is the other half: a bare file is not an extensionless
    # URL, so /rules needs its rewrite spelled out the way each rung's is.
    rewrites = {r["source"]: r["destination"] for r in _vercel_config()["rewrites"]}
    assert "/" not in rewrites, 'a "/" rewrite would point the root at a function that no longer exists'
    assert rewrites.get("/rules") == "/rules.html"
    # The Cover — the site's original front page, kept — is the third file.
    assert rewrites.get("/cover") == "/cover.html"

def test_static_output_is_served_from_public():
    # The other half of the same contract: the catch-all only mattered because
    # scripts/site/vercel_build.sh writes the visualizers into public/, which Vercel
    # serves filesystem-first once no framework preset outranks it.
    assert _vercel_config()["outputDirectory"] == "public"

def test_every_rung_the_build_script_emits_has_a_rewrite():
    # scripts/site/vercel_build.sh builds `rungs=(rung0 rung1)` into public/<rung>,
    # but a directory is not a URL: /rung0 with no trailing slash needs an
    # explicit rewrite to its index.html. Adding a rung to the script without
    # adding its rewrite here is the next version of this same silent break.
    build_script = (REPO_ROOT / "scripts" / "site" / "vercel_build.sh").read_text()
    # Matched, not split: `split("rungs=(")[1]` raises IndexError before the
    # assert below can fire, so renaming the variable in the shell script would
    # fail this test with a traceback about list indices instead of the message
    # written for exactly that case. shlex, not str.split, so the idiomatic
    # `rungs=("rung0" "rung1")` yields rung0 rather than a quote-wrapped token
    # that then fails the rewrite lookup on a config that is actually correct.
    declaration = re.search(r"^rungs=\(([^)]*)\)", build_script, re.MULTILINE)
    assert declaration, "could not find a rungs=(...) list in scripts/site/vercel_build.sh"
    rungs = shlex.split(declaration.group(1))
    assert rungs, "the rungs=(...) list in scripts/site/vercel_build.sh is empty"

    rewrites = {r["source"]: r["destination"] for r in _vercel_config()["rewrites"]}
    for rung in rungs:
        assert rewrites.get(f"/{rung}") == f"/{rung}/index.html"


# ---------------------------------------------------------------------------
# rung3 builds before solver, so what it imports from solver-viz needs only react
# ---------------------------------------------------------------------------
#
# /rung3's Learn tab imports the range interface from web/solver-viz/src, and
# scripts/site/vercel_build.sh runs `npm ci && npm run build` app by app in the
# rungs=(...) order, so when rung3-viz builds, solver-viz/node_modules does not
# exist yet. `react` is the one package those modules import, and rung3-viz
# resolves it from its own node_modules (vite.config.ts `dedupe`,
# tsconfig.app.json `paths`). Any other package imported by a solver-viz module
# reachable from rung3 resolves nowhere on Vercel and fails the deploy, while
# every local check passes because both apps were installed first.

WEB = REPO_ROOT / "web"
HOST_SRC = WEB / "rung3-viz" / "src"
GUEST_SRC = WEB / "solver-viz" / "src"
RESOLVED_BY_THE_HOST = {"react", "react/jsx-runtime"}
IMPORT = re.compile(
    r"""^\s*(?:import|export)\b[^'"\n]*?\bfrom\s*['"]([^'"]+)['"]|^\s*import\s*['"]([^'"]+)['"]""",
    re.MULTILINE,
)
CSS_IMPORT = re.compile(r"""@import\s+(?:url\()?['"]?([^'")\s;]+)""")


def _imports(path: Path) -> list[str]:
    text = path.read_text()
    if path.suffix == ".css":
        return CSS_IMPORT.findall(text)
    return [a or b for a, b in IMPORT.findall(text)]


def _resolve(importer: Path, spec: str) -> Path:
    target = importer.parent / spec
    candidates = (
        target,
        *(target.with_name(target.name + ext) for ext in (".ts", ".tsx", ".css")),
        target / "index.ts",
        target / "index.tsx",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise AssertionError(f"{importer} imports {spec!r}, which resolves to no file")


def _sources(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix in (".ts", ".tsx") and ".test." not in p.name]


def test_rung3_builds_before_solver_in_the_build_script():
    build_script = (REPO_ROOT / "scripts" / "site" / "vercel_build.sh").read_text()
    declaration = re.search(r"^rungs=\(([^)]*)\)", build_script, re.MULTILINE)
    assert declaration, "could not find a rungs=(...) list in scripts/site/vercel_build.sh"
    rungs = shlex.split(declaration.group(1))
    assert rungs.index("rung3") < rungs.index("solver"), (
        "if solver now builds first the import rule below is no longer load-bearing; keep it anyway, "
        "it is what lets each app build alone"
    )


def test_what_rung3_imports_from_solver_viz_imports_only_react():
    reached: set[Path] = set()
    queue = [
        _resolve(importer, spec)
        for importer in _sources(HOST_SRC)
        for spec in _imports(importer)
        if spec.startswith(".") and "solver-viz" in spec
    ]
    assert queue, "rung3-viz no longer imports anything from solver-viz; retire this test with the Learn tab"
    packages: dict[str, Path] = {}
    while queue:
        path = queue.pop()
        if path in reached:
            continue
        reached.add(path)
        assert GUEST_SRC in path.parents, f"{path} is outside {GUEST_SRC}"
        for spec in _imports(path):
            if spec.startswith("."):
                queue.append(_resolve(path, spec))
            else:
                packages.setdefault(spec, path)
    foreign = {spec: str(path.relative_to(WEB)) for spec, path in packages.items() if spec not in RESOLVED_BY_THE_HOST}
    assert not foreign, (
        f"solver-viz modules reachable from rung3-viz import {foreign}; rung3-viz builds before "
        "solver-viz/node_modules exists on Vercel, so these resolve to nothing there"
    )
