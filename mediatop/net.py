"""
The only place mediatop touches the network.

One plain GET per resource, a timeout, a size ceiling, and gzip in case a server
sends it anyway. Nothing is retried: a feed that fails is reported as failed,
which is how the page can say so honestly instead of quietly showing one genre
short.
"""

import gzip
import urllib.error
import urllib.request
import zlib

# A browser string, not a product string. CBC's edge drops anything that does
# not look like a browser (a request as `mediatop/0.1` times out every time,
# the same URL as Chrome returns in 200 ms), and the reader asks for no more
# than a browser would: one feed every ten minutes, one page per story.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


class FetchError(RuntimeError):
    """A resource could not be fetched, with a reason fit to show a reader."""


def fetch(url, timeout=12.0, max_bytes=4_000_000):
    """Fetch `url` and return (bytes, charset, final_url)."""
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, identity",
        "Accept-Language": "en-CA,en;q=0.9",
        "Connection": "close",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(max_bytes)
            encoding = (response.headers.get("Content-Encoding") or "").lower()
            if encoding == "gzip":
                raw = gzip.decompress(raw)
            elif encoding == "deflate":
                try:
                    raw = zlib.decompress(raw)
                except zlib.error:
                    raw = zlib.decompress(raw, -zlib.MAX_WBITS)
            charset = response.headers.get_content_charset() or "utf-8"
            return raw, charset, response.geturl()
    except urllib.error.HTTPError as error:
        raise FetchError(f"HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, OSError, EOFError,
            gzip.BadGzipFile, zlib.error) as error:
        reason = getattr(error, "reason", None) or error
        raise FetchError(f"{type(error).__name__}: {reason}") from error


def decode(raw, charset):
    """Decode bytes from a web page, falling back rather than raising.

    Serving pages claim `utf-8` and then contain Windows-1252 punctuation, so a
    strict decode is not an option. `errors="replace"` keeps the readable text
    and costs one character where the bytes were wrong.
    """
    for codec in (charset, "utf-8", "cp1252"):
        if not codec:
            continue
        try:
            return raw.decode(codec)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", "replace")


def fetch_text(url, timeout=12.0, max_bytes=4_000_000):
    """Fetch a page and return decoded text plus the URL it finally landed on."""
    raw, charset, final_url = fetch(url, timeout=timeout, max_bytes=max_bytes)
    return decode(raw, charset), final_url
