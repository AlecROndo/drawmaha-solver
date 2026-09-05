"""Serve the whole site locally, the way vercel.json wires it in production.

`vercel.json` rewrites `/` to the stdlib handler in `api/index.py` and serves
`public/` for everything else. There is no single command that reproduces that
locally, so reviewing a change to the cover page and the two visualizers
together meant three terminals and a guess. This is that one command:

    bash scripts/vercel_build.sh          # build public/
    uv run python scripts/serve_site.py   # http://localhost:4321

Stdlib only, same as the page it serves.
"""

import argparse
import importlib.util
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"

# Mirrors the "rewrites" block of vercel.json.
REWRITES = {"/rung0": "/rung0/index.html", "/rung1": "/rung1/index.html"}


def cover_page() -> bytes:
    """The cover page's HTML, read from the real Vercel entrypoint."""
    spec = importlib.util.spec_from_file_location("api_index", ROOT / "api" / "index.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PAGE.encode("utf-8")


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path == "/":
            body = cover_page()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # Re-read on every request so an edit to api/index.py shows up on
            # reload; a cached cover page is the one thing that would make this
            # server misleading.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
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
