"""
Where mediatop's stories come from: four genres, three feeds each.

Every feed here was checked by hand against a real fetch before it was written
down, and every one of them serves article pages that actually contain the text
(a `<p>`-per-paragraph body a reader can extract). That is the only qualification
that matters: a feed whose pages cannot be read is a headline list, and mediatop
is not a headline list.

Parse both dialects that are still in use — RSS 2.0 and Atom — plus the RDF
flavour Nature still serves. Namespaces are stripped first, so `dc:date`,
`content:encoded` and their friends become plain names and one code path reads
them all.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
import xml.etree.ElementTree as ET

from . import net


@dataclass(frozen=True)
class Feed:
    genre: str
    source: str
    url: str


@dataclass
class Item:
    title: str
    url: str
    source: str
    genre: str
    published: datetime | None
    description: str


GENRES = ("tech", "world", "canada", "science")

LABELS = {
    "tech": "Tech",
    "world": "Geopolitics",
    "canada": "Canadian government",
    "science": "Science",
}

FEEDS = (
    # technology
    Feed("tech", "Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
    Feed("tech", "The Verge", "https://www.theverge.com/rss/index.xml"),
    Feed("tech", "The Register", "https://www.theregister.com/headlines.atom"),
    # geopolitics
    Feed("world", "BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    Feed("world", "The Guardian", "https://www.theguardian.com/world/rss"),
    Feed("world", "Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    # Canadian government
    Feed("canada", "CBC Politics", "https://www.cbc.ca/webfeed/rss/rss-politics"),
    Feed("canada", "Global News", "https://globalnews.ca/politics/feed/"),
    Feed("canada", "National Post",
         "https://nationalpost.com/category/news/politics/feed/"),
    # science
    Feed("science", "Nature", "https://www.nature.com/nature.rss"),
    Feed("science", "ScienceDaily",
         "https://www.sciencedaily.com/rss/top/science.xml"),
    Feed("science", "Quanta", "https://api.quantamagazine.org/feed/"),
)

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def text_of(markup):
    """Plain text from a fragment of feed HTML, with entities decoded."""
    if not markup:
        return ""
    return _SPACE.sub(" ", unescape(_TAG.sub(" ", markup))).strip()


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _when(value):
    """Parse either an RFC 822 (`pubDate`) or an ISO 8601 (`dc:date`) stamp."""
    value = (value or "").strip()
    if not value:
        return None
    moment = None
    try:
        moment = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _child(element, *names):
    for child in element:
        if _local(child.tag) in names:
            return child
    return None


def _link(element):
    """The URL of an item: RSS text content, or Atom's `href` attribute."""
    best = ""
    for child in element:
        if _local(child.tag) != "link":
            continue
        rel = (child.get("rel") or "alternate").lower()
        href = (child.text or "").strip() or (child.get("href") or "").strip()
        if not href:
            continue
        if rel == "alternate" and href.startswith("http"):
            return href
        if not best and href.startswith("http"):
            best = href
    if best:
        return best
    guid = _child(element, "guid", "id")
    if guid is not None:
        candidate = (guid.text or "").strip()
        if candidate.startswith("http"):
            return candidate
    return ""


def parse(data, feed):
    """Turn one feed document into items, newest first. Never raises on content."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        raise net.FetchError(f"not XML: {error}") from error
    # Strip namespaces in place: `{http://purl.org/dc/elements/1.1/}date` -> `date`.
    for element in root.iter():
        if isinstance(element.tag, str):
            element.tag = _local(element.tag)

    items = []
    for element in root.iter():
        if element.tag not in ("item", "entry"):
            continue
        title_el = _child(element, "title")
        title = text_of("".join(title_el.itertext()) if title_el is not None else "")
        url = _link(element)
        if not title or not url:
            continue
        summary_el = _child(element, "encoded", "description", "summary", "content")
        description = text_of("".join(summary_el.itertext())
                              if summary_el is not None else "")
        date_el = _child(element, "pubDate", "date", "published", "updated",
                         "created", "issued")
        published = _when("".join(date_el.itertext())
                          if date_el is not None else "")
        items.append(Item(title=title, url=url, source=feed.source,
                          genre=feed.genre, published=published,
                          description=description))

    floor = datetime.min.replace(tzinfo=timezone.utc)
    items.sort(key=lambda item: item.published or floor, reverse=True)
    return items


def load_genre(genre, timeout=12.0):
    """Fetch every feed for one genre. Returns (items, failures).

    Failures are returned rather than raised: one dead feed should cost its own
    stories, not the whole board, and the page reports which one it was.
    """
    items, failures = [], []
    for feed in FEEDS:
        if feed.genre != genre:
            continue
        try:
            raw, _, _ = net.fetch(feed.url, timeout=timeout)
            items.extend(parse(raw, feed))
        except (net.FetchError, ValueError) as error:
            failures.append({"source": feed.source, "error": str(error)[:120]})
    return dedupe(items), failures


def title_key(title):
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def url_key(url):
    key = re.sub(r"^https?://(www\.)?", "", url.strip().lower())
    key = key.split("#", 1)[0].split("?", 1)[0]
    return key.rstrip("/")


def dedupe(items):
    """One card per story, keeping the first (newest) copy of a duplicate.

    Cross-posted stories appear in two feeds under the same headline or the same
    canonical URL; either match is enough.
    """
    seen_titles, seen_urls, kept = set(), set(), []
    for item in items:
        tkey, ukey = title_key(item.title), url_key(item.url)
        if not tkey or tkey in seen_titles or ukey in seen_urls:
            continue
        seen_titles.add(tkey)
        seen_urls.add(ukey)
        kept.append(item)
    return kept
