"""
The page, reduced to the paragraphs that are actually the story.

News pages are mostly furniture: menus, a cookie bar, "most read" rails, a
newsletter pitch, a photo credit, three paragraphs of comment policy. Only the
`<p>` elements inside the article are the story, and even those need filtering,
so this module does three things in order:

1. parse the page and keep only paragraph text, remembering whether each
   paragraph sat inside `<article>` or `<main>` and how much of it was a link;
2. discard the furniture — short paragraphs, link-dense ones (a rail of
   headlines), and ones that match a boilerplate phrase;
3. split what survives into sentences that a reader can scan.

All of it is standard library: `html.parser` is tolerant of the markup that
actual publishers ship, which is the property that matters here.
"""

import re
from html.parser import HTMLParser

SKIP_TAGS = {
    "script", "style", "noscript", "template", "svg", "iframe", "form", "nav",
    "header", "footer", "aside", "button", "select", "option", "figure",
    "figcaption", "menu", "dialog",
}

# Containers that look like page furniture by name alone: "related stories",
# "latest", a jobs widget, a cookie bar. Everything inside them is dropped even
# when it is marked up as a paragraph, which is how a Nature job advert stops
# being one of the facts of a coral-reef story.
NOISE_ATTR = re.compile(
    r"(?<![a-z])(related|recommend|latest|promo|newsletter|signup|sign-up|"
    r"subscribe|advert|sponsor|sidebar|most-read|mostread|trending|popular|share|"
    r"social|comments?|cookies?|consent|banner|jobs?|careers?|recruit|breadcrumb|"
    r"byline|caption|credit|paywall|regwall|pay-barrier)(?![a-z])",
    re.I)

STRUCTURAL = {"html", "body", "main", "article"}

VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
}

SPACE = re.compile(r"\s+")


class _Page(HTMLParser):
    """Collect `<p>` text with the little bit of context the filters need."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paragraphs = []
        self.page_title = ""
        self._skip = []            # tags we are inside that hold no story text
        self._article = 0          # depth of <article>
        self._main = 0             # depth of <main>
        self._links = 0            # depth of <a>
        self._nest = []            # (tag, contributed to noise) for balance
        self._noise = 0            # depth of furniture containers
        self._collecting = None    # dict for the paragraph being read
        self._in_title = False

    # -- state ------------------------------------------------------------
    def _skipping(self):
        return bool(self._skip)

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
            return
        if tag in VOID_TAGS:
            return
        if tag in SKIP_TAGS:
            self._skip.append(tag)
            return
        if self._skipping():
            # An <a> inside a skipped <nav> must not leave the link counter up:
            # its closing tag is swallowed by the skip rule, so counting it here
            # would make every later paragraph look link-dense.
            return

        # `html`, `body`, `main` and `article` are structure, not furniture, and
        # their class lists can contain the article's own slug — one of those
        # words ("latest", "share", "jobs") would otherwise erase the story.
        attributes = dict(attrs)
        noisy = tag not in STRUCTURAL and bool(
            NOISE_ATTR.search(attributes.get("class") or "")
            or NOISE_ATTR.search(attributes.get("id") or ""))
        self._nest.append((tag, noisy))
        if noisy:
            self._noise += 1

        if tag == "article":
            self._article += 1
        elif tag == "main":
            self._main += 1
        elif tag == "a":
            self._links += 1
        elif tag == "p" and self._collecting is None:
            self._collecting = {
                "parts": [],
                "link_chars": 0,
                "link_base": self._links,   # drift from before this paragraph
                "chars": 0,
                "article": self._article > 0,
                "main": self._main > 0,
                "noise": self._noise > 0,
            }

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
            return
        if self._skip:
            if tag in self._skip:
                while self._skip:
                    if self._skip.pop() == tag:
                        break
            return
        while self._nest:
            nested, contributed = self._nest.pop()
            if contributed:
                self._noise = max(0, self._noise - 1)
            if nested == tag:
                break

        if tag == "article":
            self._article = max(0, self._article - 1)
        elif tag == "main":
            self._main = max(0, self._main - 1)
        elif tag == "a":
            self._links = max(0, self._links - 1)
        elif tag == "p" and self._collecting is not None:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.page_title += data
            return
        if self._skipping() or self._collecting is None:
            return
        stripped = data.strip()
        if not stripped:
            self._collecting["parts"].append(" ")
            return
        self._collecting["parts"].append(data)
        self._collecting["chars"] += len(stripped)
        if self._links > self._collecting["link_base"]:
            self._collecting["link_chars"] += len(stripped)

    def _flush(self):
        record = self._collecting
        self._collecting = None
        text = SPACE.sub(" ", "".join(record["parts"])).strip()
        text = text.replace("\u00ad", "")
        if text:
            self.paragraphs.append({"text": text,
                                    "chars": len(text),
                                    "link_chars": record["link_chars"],
                                    "article": record["article"],
                                    "main": record["main"],
                                    "noise": record["noise"]})


BOILERPLATE = (
    re.compile(r"^(subscribe|sign up|sign in|log in|register)\b"),
    re.compile(r"\bnewsletter\b"),
    re.compile(r"all rights reserved"),
    re.compile(r"\bcookies?\b"),
    re.compile(r"^advertisement\b"),
    re.compile(r"^(read|see) (more|also)\b"),
    re.compile(r"^follow us\b"),
    re.compile(r"\bdownload (the|our) app\b"),
    re.compile(r"^share (this|on)\b"),
    re.compile(r"^(photo|image|photograph|illustration|video|graphic)"
               r"(graph)?\s*[:\-]"),
    re.compile(r"\bgetty images\b"),
    re.compile(r"\bsupport our journalism\b"),
    re.compile(r"\b(terms of (service|use)|privacy policy)\b"),
    re.compile(r"^(comments?|related|more from|trending|recommended)\b"),
    re.compile(r"^[\u00a9\u00ae]"),
    re.compile(r"^by [a-z]+ [a-z]+$"),
    re.compile(r"\b(subscriber|paywall)\b.*\b(only|exclusive)\b"),
    re.compile(r"^(postdoctoral|postdoc|phd (position|student)|apply )"),
    re.compile(r"^thumbnail\b"),
    re.compile(r"\b(image|photo|photograph)s? (from|credit|courtesy)\b"),
    re.compile(r"^doi:\s"),
    re.compile(r"\b(affiliate link|may earn a commission|we earn a commission)\b"),
    re.compile(r"\b(thanks for (reading|visiting)|you are using an?)\b"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),   # a contact address is not a fact
    re.compile(r"\b(job alert|vacanc(y|ies)|job title|salary range)\b"),
)

MIN_PARAGRAPH = 55           # shorter than this is a label, a credit or a hint
MIN_SENTENCE = 24            # a snippet sentence can be shorter than a body one
MAX_LINK_SHARE = 0.5         # more than half link text is a rail of headlines


def is_boilerplate(text):
    lowered = text.lower()
    return any(pattern.search(lowered) for pattern in BOILERPLATE)


def article_paragraphs(html, min_len=MIN_PARAGRAPH):
    """The story's paragraphs, best-effort, from one page of HTML.

    Returns a list of strings in document order. Empty means "this page did not
    give up its text", which callers must handle rather than paper over.
    """
    parser = _Page()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # html.parser can still trip on pathological markup
        pass

    records = parser.paragraphs
    if not records:
        return []
    # Prefer the smallest scope that still holds the story: <article>, then
    # <main>, then the whole page.
    for flag in ("article", "main"):
        scoped = [r for r in records if r[flag]]
        if sum(len(r["text"]) for r in scoped) >= 400:
            records = scoped
            break

    paragraphs = []
    for record in records:
        text = record["text"]
        if len(text) < min_len:
            continue
        if record["noise"]:
            continue
        if record["chars"] and record["link_chars"] / record["chars"] > MAX_LINK_SHARE:
            continue
        if is_boilerplate(text):
            continue
        paragraphs.append(text)
    return paragraphs


def page_title(html):
    parser = _Page()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass
    return SPACE.sub(" ", parser.page_title).strip()


# Abbreviations whose full stop does not end a sentence. Lower case, no dots.
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc",
    "inc", "ltd", "co", "corp", "gov", "sen", "rep", "gen", "col", "lt", "sgt",
    "capt", "no", "vol", "fig", "al", "approx", "dept", "est",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "e.g", "i.e", "u.s", "u.k", "u.n", "a.m", "p.m", "ph.d",
}

_BOUNDARY = re.compile(r"([.!?])[\"'\u201d\u2019)\]]*\s+")
_WORD_TAIL = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_DATELINE = re.compile(r"^[A-Z][A-Z0-9 .]{1,24}[\u2014\u2013-]\s*")
_TRAILING_BOILER = re.compile(
    r"^(this (report|article|story)|with files from|reporting by|"
    r"the canadian press|files from)", re.I)


def strip_dateline(sentence):
    """Drop a leading `OTTAWA — ` style dateline, which carries no fact."""
    match = _DATELINE.match(sentence)
    if not match:
        return sentence
    head = match.group(0)
    if not any(word.isupper() and len(word) > 2
               for word in head.split() if word.isalpha()):
        return sentence
    return sentence[match.end():].lstrip()


def split_sentences(text):
    """Split prose into sentences, without breaking on `Dr.` or `U.S.`."""
    pieces, start = [], 0
    for match in _BOUNDARY.finditer(text):
        tail = text[start:match.start()]
        following = text[match.end():match.end() + 1]
        word = _WORD_TAIL.search(tail.strip())
        if word:
            token = word.group(1).lower().rstrip(".")
            if token in ABBREVIATIONS:
                continue
            if len(token) == 1 and word.group(1)[0].isupper():
                continue           # an initial: "J. Smith"
        if following.islower():
            continue               # a decimal or a typo, not a boundary
        pieces.append(text[start:match.end()].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return [piece for piece in pieces if piece]


def sentences_indexed(paragraphs):
    """Sentences of an article, in order, each with its paragraph's length.

    The length matters to the scorer: a one-line paragraph on a news page is
    usually a photo caption or a related-story teaser, not a fact of the story.
    """
    out = []
    for paragraph in paragraphs:
        for sentence in split_sentences(paragraph):
            sentence = strip_dateline(sentence).strip()
            if len(sentence) < MIN_SENTENCE:
                continue
            if _TRAILING_BOILER.match(sentence):
                continue
            if is_boilerplate(sentence):
                continue
            out.append((sentence, len(paragraph)))
    return out


def sentences(paragraphs):
    """Sentences of an article, in order, with boilerplate and datelines gone."""
    return [sentence for sentence, _ in sentences_indexed(paragraphs)]
