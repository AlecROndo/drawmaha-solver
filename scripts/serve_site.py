"""Serve the whole site locally, the way vercel.json wires it in production.

`vercel.json` serves `public/` filesystem-first (the cover page is its
`index.html`) and rewrites the extensionless routes — `/rules`, `/rung0` and
the rest — onto files inside it. There is no single command that reproduces
that locally, so reviewing a change to the cover page and the visualizers
together meant three terminals and a guess. This is that one command:

    bash scripts/vercel_build.sh          # build public/
    uv run python scripts/serve_site.py   # http://localhost:4321

Stdlib only, so it can never break on the solver's numeric stack.
"""

import argparse
import json
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"

# Read from vercel.json rather than mirrored by hand: a hand copy silently
# missed each new rung's rewrite, which is the one thing this server exists
# to reproduce. "/" needs no rule: SimpleHTTPRequestHandler serves a
# directory's index.html, the same filesystem-first default Vercel applies.
REWRITES = {
    rule["source"]: rule["destination"]
    for rule in json.loads((ROOT / "vercel.json").read_text())["rewrites"]
}


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path in REWRITES:
            self.path = REWRITES[path]
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4321)
    args = parser.parse_args()

    if not PUBLIC.is_dir():
        sys.exit("public/ is missing — run `bash scripts/vercel_build.sh` first.")

    handler = partial(Handler, directory=str(PUBLIC))
    with HTTPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"serving the site on http://localhost:{args.port}", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
