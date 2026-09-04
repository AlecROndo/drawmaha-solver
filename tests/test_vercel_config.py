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
# which swallows every path below the three explicit rewrites and hands it to
# api/index.py — so /rung0, /rung1, their /assets/* bundles and /fonts/* all
# build correctly into public/ and then serve the cover page instead. Nothing
# fails: the build is green, the deploy is green, and the visualizers are gone.
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
        '"python" preset from pyproject.toml and routes every path to api/index.py'
    )
    assert config["framework"] is None

def test_static_output_is_served_from_public():
    # The other half of the same contract: the catch-all only mattered because
    # scripts/vercel_build.sh writes the visualizers into public/, which Vercel
    # serves filesystem-first once no framework preset outranks it.
    assert _vercel_config()["outputDirectory"] == "public"

def test_every_rung_the_build_script_emits_has_a_rewrite():
    # scripts/vercel_build.sh builds `rungs=(rung0 rung1)` into public/<rung>,
    # but a directory is not a URL: /rung0 with no trailing slash needs an
    # explicit rewrite to its index.html. Adding a rung to the script without
    # adding its rewrite here is the next version of this same silent break.
    build_script = (REPO_ROOT / "scripts" / "vercel_build.sh").read_text()
    # Matched, not split: `split("rungs=(")[1]` raises IndexError before the
    # assert below can fire, so renaming the variable in the shell script would
    # fail this test with a traceback about list indices instead of the message
    # written for exactly that case. shlex, not str.split, so the idiomatic
    # `rungs=("rung0" "rung1")` yields rung0 rather than a quote-wrapped token
    # that then fails the rewrite lookup on a config that is actually correct.
    declaration = re.search(r"^rungs=\(([^)]*)\)", build_script, re.MULTILINE)
    assert declaration, "could not find a rungs=(...) list in scripts/vercel_build.sh"
    rungs = shlex.split(declaration.group(1))
    assert rungs, "the rungs=(...) list in scripts/vercel_build.sh is empty"

    rewrites = {r["source"]: r["destination"] for r in _vercel_config()["rewrites"]}
    for rung in rungs:
        assert rewrites.get(f"/{rung}") == f"/{rung}/index.html"
