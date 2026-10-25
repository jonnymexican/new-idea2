#!/usr/bin/env python3
"""One command to measure the local arm, build the page and look at it.

The platform arm is only interesting next to the page that remembers it, and
wiring the two by hand takes three commands and a static server. This is that
wiring in one place:

    .venv/bin/python demo.py
    .venv/bin/python demo.py --port 9000 --no-open
    .venv/bin/python demo.py --no-serve          # files only

It measures the `local` adapter -- a real HTTP server that re-encodes the
signed fixture under four policies -- writes the report to `out/demo/`, appends
it as a column of the trend page in `out/site/`, and serves that page until
Ctrl-C. The verifier page staged beside it lists the arm's four renditions, so
the local run looks like the published one. Nothing leaves the machine: the
live adapters are not involved, and none of them is configured here.

Both output directories are gitignored. Repeating the command adds a column per
measurement, which is the page doing its job rather than a bug; `--no-serve`
leaves the files without starting a server.
"""

from __future__ import annotations

import argparse
import functools
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pages

DEFAULT_OUT = "out/demo"
DEFAULT_SITE = "out/site"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8788
DEFAULT_TITLE = "Platform survival"


def measure(out_dir):
    """Run the local adapter and return the report it wrote.

    `platforms` is imported here rather than at module import: the arm needs
    c2pa-python and Pillow, while building and serving the page are standard
    library only, and a reader who only wants the page should not have to have
    the signing stack installed to run `--help`.
    """
    import platforms

    out_dir = Path(out_dir)
    code = platforms.main(["run", "--platforms", "local", "--out", str(out_dir)])
    if code:
        raise SystemExit(f"the local arm exited {code}")
    report = out_dir / "platforms.json"
    if not report.exists():
        raise SystemExit(f"the local arm wrote no report at {report}")
    return report


def build_page(report, site, title=DEFAULT_TITLE, run=None):
    """Append `report` as one column and render the page. Returns the record.

    `run` is the directory the arm wrote, so the verifier page staged next to
    the trend page lists this measurement's renditions rather than only
    offering a drop zone.
    """
    return pages.build(
        Path(report),
        Path(site),
        title=title,
        verifier_roots=[Path(run)] if run else [],
    )


def serve(site, host=DEFAULT_HOST, port=DEFAULT_PORT, open_browser=True):
    """Serve `site` until interrupted; it is static, so the stdlib is enough."""
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(site))
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        # A port already in use is the likeliest failure by far, and a traceback
        # would bury the one word that fixes it.
        raise SystemExit(
            f"cannot listen on {host}:{port}: {exc}\n"
            f"another port: --port {port + 1}") from None

    url = f"http://{host}:{server.server_address[1]}/"
    print(f"page     : {url}  (Ctrl-C to stop)", flush=True)
    if open_browser:
        # Returns False, rather than raising, on a machine with no browser; the
        # URL above is then the only way in, which is fine.
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"where the measurement writes (default {DEFAULT_OUT})")
    parser.add_argument("--site", default=DEFAULT_SITE,
                        help=f"where the page is built (default {DEFAULT_SITE})")
    parser.add_argument("--title", default=DEFAULT_TITLE,
                        help="page title")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"address to serve on (default {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="port to serve on, 0 picks a free one "
                             f"(default {DEFAULT_PORT})")
    parser.add_argument("--no-serve", action="store_true",
                        help="measure and build only; do not start a server")
    parser.add_argument("--no-open", action="store_true",
                        help="serve without opening a browser")
    args = parser.parse_args(argv)

    report = measure(args.out)
    try:
        record = build_page(report, args.site, args.title, run=args.out)
    except ValueError as exc:
        # A history file that is not JSON would otherwise be a traceback; pages
        # raises ValueError with the path and the reason.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"built    : {Path(args.site) / 'index.html'} "
          f"({len(record['platforms'])} platform(s))", flush=True)
    if args.no_serve:
        return 0
    serve(args.site, args.host, args.port, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
