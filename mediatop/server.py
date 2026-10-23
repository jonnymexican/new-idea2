#!/usr/bin/env python3
"""
mediatop: the point of the news, one screen, four genres.

A single standard-library process serves one page and one JSON endpoint. The
work is: read twelve feeds, fetch the article behind each headline, keep the
paragraphs that are the story, and score those sentences down to one point and
a handful of facts. Nothing is written by a model — every line on the board is
a sentence from the article, chosen rather than composed.

    python mediatop/server.py
    python mediatop/server.py --port 8770 --per-genre 8
    python mediatop/server.py --print          # the board, as text, then exit

Then open http://127.0.0.1:8770/ .

Endpoints:

    GET /            the page (HTML, CSS and JS in one file)
    GET /api/top     the board as JSON; `?genre=tech` for one genre,
                     `?refresh=1` to ignore the cache

Caching is the contract with the publishers: a feed is read at most once per
`--ttl` seconds (default 600), and each article page is fetched once per build,
for a few dozen requests every ten minutes. If what comes back is stale-but-usable
it is served, and the page shows when it was assembled.

Anything the page cannot read is labelled on the card, not hidden: a paywalled
article whose text never arrived is shown from its feed summary, and a feed that
failed is listed at the top. A board that quietly dropped half its sources would
be worse than one that admits it.
"""

import argparse
import json
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import feeds, summary

HERE = Path(__file__).resolve().parent
PAGE = HERE / "page.html"
FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#0a0b0d"/>'
    '<path d="M8 22V10h3l3 7 3-7h3v12h-2.6v-7.4L14.8 22h-1.6L10.6 14.6V22z"'
    ' fill="#e8eaed"/></svg>'
)


def interleave(records, limit, key):
    """Take up to `limit` records, round-robin across whatever `key` groups by.

    One feed updates three times an hour and another once a day, so the newest
    six stories are routinely the same publisher six times. Sorting by time and
    then walking the sources in turn keeps the board a board rather than one
    outlet's front page, while each source's records stay newest first.
    """
    queues = {}
    for record in records:
        queues.setdefault(key(record), []).append(record)
    ordered, groups = [], list(queues)
    while len(ordered) < limit and groups:
        for group in list(groups):
            queue = queues[group]
            ordered.append(queue.pop(0))
            if not queue:
                groups.remove(group)
            if len(ordered) >= limit:
                break
    return ordered


def spread(cards, limit):
    """Cards from different sources before cards from the same one."""
    return interleave(cards, limit, lambda card: card["source"])


class Board:
    """The board, built at most once per genre per TTL."""

    def __init__(self, per_genre=6, ttl=600.0, timeout=12.0, workers=8):
        self.per_genre = per_genre
        self.ttl = ttl
        self.timeout = timeout
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="mediatop")
        self._cache = {}                       # genre -> {"at", "cards", "failures"}
        self._locks = {genre: threading.Lock() for genre in feeds.GENRES}

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- building ---------------------------------------------------------
    def _build_genre(self, genre):
        started = time.time()
        items, failures = feeds.load_genre(genre, timeout=self.timeout)
        # Fetch a few more pages than the board shows: some will be unreadable,
        # and a short genre should still fill its column. Interleave the items
        # first, or the newest feed alone supplies every candidate.
        candidates = interleave(items, self.per_genre * 3, lambda item: item.source)
        cards = list(self._pool.map(
            lambda item: summary.card(item, timeout=self.timeout), candidates))
        cards = [card for card in cards if card["point"]]
        cards.sort(key=lambda card: card["published"] or "", reverse=True)
        return {"at": started, "cards": spread(cards, self.per_genre),
                "failures": failures, "considered": len(items)}

    def _fresh(self, genre, now):
        entry = self._cache.get(genre)
        return entry is not None and now - entry["at"] < self.ttl

    def _ensure(self, genre, refresh=False):
        now = time.time()
        if not refresh and self._fresh(genre, now):
            return self._cache[genre]
        with self._locks[genre]:
            # Another thread may have finished this genre while we waited.
            if not refresh and self._fresh(genre, time.time()):
                return self._cache[genre]
            try:
                entry = self._build_genre(genre)
            except Exception as error:          # a build must never 500 the page
                entry = {"at": now, "cards": [],
                         "failures": [{"source": feeds.LABELS[genre],
                                       "error": f"{type(error).__name__}: "
                                                f"{str(error)[:100]}"}],
                         "considered": 0}
            self._cache[genre] = entry
            return entry

    def board(self, genre=None, refresh=False):
        wanted = [genre] if genre else list(feeds.GENRES)
        unknown = [key for key in wanted if key not in feeds.GENRES]
        if unknown:
            raise KeyError(unknown[0])

        entries = {key: self._ensure(key, refresh) for key in wanted}
        failures = [dict(failure, genre=key)
                    for key, entry in entries.items()
                    for failure in entry["failures"]]
        assembled = {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "genres": [
                {
                    "key": key,
                    "label": feeds.LABELS[key],
                    "built": datetime.fromtimestamp(
                        entries[key]["at"], timezone.utc
                    ).isoformat(timespec="seconds"),
                    "age_seconds": int(time.time() - entries[key]["at"]),
                    "considered": entries[key]["considered"],
                    "cards": entries[key]["cards"],
                }
                for key in wanted
            ],
            "failures": failures,
        }
        assembled["counts"] = {
            "stories": sum(len(genre["cards"]) for genre in assembled["genres"]),
            "sources": len({card["source"] for genre in assembled["genres"]
                            for card in genre["cards"]}),
        }
        return assembled


class Handler(BaseHTTPRequestHandler):
    server_version = "mediatop/0.1"
    board = None

    def log_message(self, fmt, *args):       # quieter than the default
        if "api/top" in (self.path or ""):
            print(f"  {self.address_string()} {fmt % args}")

    def _send(self, code, body, content_type, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"

        if route == "/":
            try:
                body = PAGE.read_bytes()
            except OSError:
                self._json(500, {"error": "page.html is missing"})
                return
            self._send(200, body, "text/html; charset=utf-8")
            return

        if route == "/api/top":
            query = parse_qs(parsed.query)
            genre = (query.get("genre") or [None])[0]
            refresh = (query.get("refresh") or ["0"])[0] not in ("0", "", "false")
            try:
                payload = self.board.board(genre=genre, refresh=refresh)
            except KeyError as error:
                self._json(400, {"error": f"unknown genre: {error.args[0]}",
                                 "genres": list(feeds.GENRES)})
                return
            except Exception as error:        # never leave the page hanging
                self._json(500, {"error": f"{type(error).__name__}: {error}"})
                return
            self._json(200, payload)
            return

        if route == "/favicon.svg":
            self._send(200, FAVICON.encode("utf-8"), "image/svg+xml")
            return

        self._json(404, {"error": "not found", "paths": ["/", "/api/top"]})


def print_board(board):
    """The board as plain text, for a terminal check without a browser."""
    payload = board.board(refresh=True)
    for genre in payload["genres"]:
        print(f"\n== {genre['label'].upper()} "
              f"({len(genre['cards'])} of {genre['considered']})")
        for card in genre["cards"]:
            print(f"\n  [{card['source']}] {card['title']}")
            print(f"    POINT: {card['point']}")
            for fact in card["facts"]:
                print(f"      - {fact}")
            tag = card["method"] if not card["note"] else f"{card['method']}: {card['note']}"
            print(f"    ({tag})")
    for failure in payload["failures"]:
        print(f"\n  ! {failure['genre']}/{failure['source']}: {failure['error']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--per-genre", type=int, default=6,
                        help="stories shown per genre (default 6)")
    parser.add_argument("--ttl", type=float, default=600.0,
                        help="seconds a built board stays fresh (default 600)")
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--print", action="store_true",
                        help="build once, print the board as text, exit")
    parser.add_argument("--open", action="store_true", help="open a browser")
    args = parser.parse_args(argv)

    board = Board(per_genre=args.per_genre, ttl=args.ttl, timeout=args.timeout)
    if args.print:
        try:
            print_board(board)
        finally:
            board.close()
        return 0

    Handler.board = board
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}/"
    print(f"mediatop on {url}  ({args.per_genre}/genre, ttl {args.ttl:.0f}s)")
    print("building the first board...")
    started = time.time()
    board.board()
    print(f"ready in {time.time() - started:.1f}s — Ctrl-C to stop")
    if args.open:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
        board.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
