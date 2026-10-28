#!/usr/bin/env python3
"""Read a platform's renditions the way a browser reads them.

`platforms.py` answers what is in the bytes it downloaded, using c2pa-python.
What a person on the other end sees is decided by whichever verifier their
application embeds, so this serves `verifier/` -- a page that runs the
Content Authenticity Initiative's own browser SDK -- together with the files
this lab has already fetched.

    .venv/bin/python verifier.py                     # serves out/, opens a browser
    .venv/bin/python verifier.py --out out/photo-discord
    .venv/bin/python verifier.py --port 0 --no-open  # pick a free port, stay put

The page itself is static, and this server exists only to save you dragging
files out of `out/` by hand: every `served/` directory under the roots you give
it becomes a list of chips, one per rendition. Dropping your own files on the
page works too.

Nothing is uploaded. The browser fetches the 9 MB verifier from a pinned CDN
URL and checks the bytes in the tab; this server never sends a file anywhere
but to that tab.

``pages.py build`` stages the same page as ``<site>/verifier/`` -- the files
plus a ``samples.json`` and the bytes it names -- so the published site carries
both the trend page and this one. Everything there is a relative URL, because
Pages serves the site from a project path rather than a host root.
"""

from __future__ import annotations

import argparse
import functools
import json
import mimetypes
import shutil
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
PAGE = HERE / "verifier"
DEFAULT_OUT = "out"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}

# The three files that make up the page. The server maps them to routes and the
# static build copies them; one list keeps the two from drifting apart.
PAGE_FILES = ("index.html", "app.js", "style.css")

_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


def _label(name: str) -> str:
    """`discord__proxy_webp.webp` is the lab's `platform__variant` naming."""
    stem = Path(name).stem
    if "__" in stem:
        platform, variant = stem.split("__", 1)
        return f"{platform} · {variant.replace('_', ' ')}"
    return stem


def served_dirs(roots):
    """(run name, directory) for every `served/` directory worth listing."""
    for root in roots:
        root = Path(root)
        if root.is_file():
            if root.suffix.lower() in SUFFIXES:
                # A single file names its run the way a `served/` directory
                # does: the run, not the folder the file happens to sit in.
                directory = root.parent
                name = directory.parent.name if directory.name == "served" else directory.name
                yield name, directory
            continue
        for served in sorted(root.glob("**/served")):
            if served.is_dir():
                yield served.parent.name, served


def catalogue(roots):
    """Runs newest first, each with the files the page may ask for.

    Returns ``[(run_name, [(label, url, path), ...])]``. The URL is the one path
    the handler accepts, so serving is an exact lookup rather than a guess at
    what a request meant.
    """
    groups: dict[str, list[Path]] = {}
    seen: dict[str, int] = {}
    for name, directory in served_dirs(roots):
        files = [p for p in directory.iterdir()
                 if p.is_file() and p.suffix.lower() in SUFFIXES]
        if not files:
            continue
        seen[name] = seen.get(name, 0) + 1
        unique = name if seen[name] == 1 else f"{name} ({seen[name]})"
        groups[unique] = files

    runs = []
    for name, files in groups.items():
        files = sorted(files, key=lambda path: path.name)
        runs.append((
            name,
            [
                (_label(path.name), f"samples/{name}/{path.name}", path)
                for path in files
            ],
            max(path.stat().st_mtime for path in files),
        ))
    runs.sort(key=lambda run: run[2], reverse=True)
    return [(name, files) for name, files, _ in runs]


def _payload(runs, home=None) -> dict:
    """The catalogue the page reads, from either the server or a build."""
    payload: dict = {"runs": runs}
    if home:
        payload["home"] = dict(home)
    return payload


def listing(roots) -> dict:
    """What the page sees: names, labels and sizes, but no host file paths.

    No ``home``: the server serves the verifier alone, with no trend page above
    it, and a link back would land on the page the reader is already on.
    """
    return _payload([
        {
            "name": name,
            "files": [
                {"label": label, "url": url, "bytes": path.stat().st_size}
                for label, url, path in files
            ],
        }
        for name, files in catalogue(roots)
    ])


def build_site(roots, site_dir, *, home=None) -> Path:
    """Stage the page and the renditions it lists as ``<site_dir>/verifier``.

    This is the published copy: the same three files the local server serves,
    plus a ``samples.json`` and the bytes it names, so the page that goes up
    next to the trend page has renditions to open instead of only a drop zone.

    A root that does not exist is skipped rather than refused. A page with
    nothing to list still verifies dropped files, and a caller that points at a
    run directory the measurement never wrote should get the page, not an
    error -- that is also what keeps the trend page's link from dangling.

    Every URL written here is relative (``samples/<run>/<file>``), because
    Pages serves the site under a project path; a leading slash would resolve
    against the host root and 404. ``home`` names the trend page, which sits one
    directory up, so the two pages link to each other rather than one of them
    being a dead end.
    """
    target = Path(site_dir) / "verifier"
    target.mkdir(parents=True, exist_ok=True)
    for name in PAGE_FILES:
        shutil.copy2(PAGE / name, target / name)

    runs = []
    for name, files in catalogue(roots):
        entries = []
        for label, url, path in files:
            destination = target / url
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            entries.append(
                {"label": label, "url": url, "bytes": path.stat().st_size}
            )
        runs.append({"name": name, "files": entries})

    # Same shape the server answers on /samples.json, because the page reads
    # whichever one it finds without knowing which served it -- plus `home`,
    # which only a build knows: the local server has no trend page to link to.
    payload = json.dumps(_payload(runs, home=home), indent=2) + "\n"
    (target / "samples.json").write_text(payload, encoding="utf-8")
    return target


class Handler(BaseHTTPRequestHandler):
    server_version = "provenance-lab-verifier"

    def __init__(self, *args, roots=(), **kwargs):
        self._roots = [Path(root) for root in roots]
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):  # quieter than the stdlib default
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self):
        self._send(b"not found\n", "text/plain; charset=utf-8", status=404)

    def _sample(self, path: str) -> Path | None:
        """The file behind `/samples/<run>/<file>`, if the lab listed it."""
        if not path.startswith("/samples/"):
            return None
        wanted = path[1:]  # catalogue URLs have no leading slash
        for _, files in catalogue(self._roots):
            for _, url, file in files:
                if url == wanted:
                    return file
        return None

    def do_GET(self):  # noqa: N802 - stdlib names this
        path = urlparse(self.path).path
        if path == "/samples.json":
            body = json.dumps(listing(self._roots), indent=2).encode()
            return self._send(body, "application/json; charset=utf-8")
        if path in _STATIC:
            name, content_type = _STATIC[path]
            file = PAGE / name
            if file.is_file():
                return self._send(file.read_bytes(), content_type)
            return self._not_found()
        sample = self._sample(path)
        if sample is not None:
            content_type, _ = mimetypes.guess_type(sample.name)
            return self._send(sample.read_bytes(), content_type or "application/octet-stream")
        return self._not_found()


def serve(roots, host=DEFAULT_HOST, port=DEFAULT_PORT, open_browser=True):
    """Start the server and return it, without blocking."""
    if not (PAGE / "index.html").is_file():
        raise SystemExit(f"the page is missing from {PAGE}")
    handler = functools.partial(Handler, roots=[str(root) for root in roots])
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        raise SystemExit(
            f"cannot listen on {host}:{port}: {exc}\n"
            f"another port: --port {port + 1}") from None

    url = f"http://{host}:{server.server_address[1]}/"
    found = sum(len(files) for _, files in catalogue(roots))
    print(f"verifier : {url}  ({found} rendition(s) to check, Ctrl-C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", action="append", default=None,
                        help=f"run directory to list renditions from "
                             f"(default {DEFAULT_OUT}; repeatable)")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"address to serve on (default {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port to serve on, 0 picks a free one "
                             f"(default {DEFAULT_PORT})")
    parser.add_argument("--no-open", action="store_true",
                        help="serve without opening a browser")
    args = parser.parse_args(argv)

    # A missing run directory is not an error: the page verifies dropped files
    # perfectly well with nothing to list.
    roots = [Path(root) for root in (args.out or [DEFAULT_OUT]) if Path(root).exists()]
    server = serve(roots, args.host, args.port, open_browser=not args.no_open)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
