"""
The point of a story, and the facts around it.

Everything here is extractive: mediatop quotes, it does not paraphrase. That is
a deliberate limit — a four-million-parameter model asked to rewrite a news
paragraph invents details, and the whole promise of a one-screen summary is
that nothing on it is made up. So the job is selection, not generation:

* sentence score = how much of the headline it carries, how much specific
  detail it holds (numbers, names), whether it reports an event rather than a
  mood, and where it sits — with position *earned*, because a scene-setter in
  the lead is not a point;
* the point is the best sentence, with a mild preference for the earlier of two
  near-equal candidates;
* the facts are the next few highest-scoring sentences that are not near
  duplicates and not scene-setting, put back into document order so the card
  reads top-down — a sentence in the past perfect, or a first-person quotation,
  belongs to the paragraph before the story and is skipped unless it carries a
  result.

The scoring is the part that took work, so it is worth saying what it is trying
to recognise. A sentence that is the point of a story usually does three things
at once: it repeats the headline's distinctive terms (the editor already wrote
the one-line answer to "what happened"), it carries something specific — a
twice-checked number, a named actor, a result — and it reports an event rather
than setting a scene. Three failure modes follow from that and are penalised by
name below: the **scene-setter** ("On a frigid February morning in Edinburgh…"),
the **throat-clearer** ("Among the 377 results…, one stood out"), and the
**topic sentence** that restates the headline without adding a single fact.

When the page gives up no text — a paywall, a 403, a JavaScript shell — the card
falls back to the feed's own summary and says so, rather than pretending to have
read the article.
"""

import math
import re
from collections import Counter

from . import extract, net

MAX_FACTS = 4
MIN_ARTICLE_CHARS = 350
FACT_WEIGHT = 0.3        # per distinct figure in a sentence
NAME_WEIGHT = 0.15       # per named thing

STOPWORDS = set("""
a about after all also an and any are as at be because been before being but by
can could did do does for from had has have he her here his how i if in into is
it its just like may more most much my no not now of on once only or other our
out over said say says she should so some such than that the their them then
there these they this those though through to too under up us was we were what
when where which while who why will with would you your
""".split())

# The verb of consequence: a sentence holding one of these is reporting an
# event, which is what a point does. `discovered`/`found` are here because
# science copy says "researchers found that…" where news copy says "the court
# ruled that…".
REPORTING = re.compile(
    r"\b(said|says|told|tells|announced|announces|confirmed|revealed|reported|warned|"
    r"ordered|ruled|found|find|discovered|showed|shows|suggests?|approved|"
    r"rejected|banned|launched|unveiled|signed|agreed|resigned|sued|won|lost|"
    r"cut|raised|plans|planning|expects|expected|introduced)\b", re.I)
# An event: something happened, said in the *present* perfect — "has been used",
# "have agreed". This is what separates the lede from the paragraph under it.
# "The milestone treatment is part of a clinical trial, a technique that can
# alter DNA…" shares every technical word with the headline and still reports
# nothing; the sentence beside it — "has been used to edit DNA directly in a
# person's body" — is the news.
#
# Present perfect only, and only the participle. "Had" is the tense of backfill
# ("the rain had stopped", "his garden had dried up") and "was planning" is an
# intention; both are how a scene is set, not how a development is reported.
EVENT = re.compile(
    r"\b(?:(?:has|have)\s+been\s+[a-z]{2,}(?:ed|en|wn|ne|lt|ld|nt|de)"
    r"|(?:has|have)\s+(?!been\b)[a-z]{2,}(?:ed|en|wn|ne|lt|ld|nt|de))\b",
    re.I)
# "found that", "showed how": the shape of a result being stated.
FINDING = re.compile(
    r"\b(found|find|discovered|showed|shows|revealed|suggests?|concluded|"
    r"reported|ruled|confirmed)\b\s+(that|how|the|a|an)\b", re.I)
# A sentence opening with a time or place is licensed to be a scene-setter: the
# actual claim has not started yet when the sentence begins.
FRAME_START = re.compile(
    r"^(on|in|at|among|during|after|before|when|as|while|for|since|by|with|"
    r"under|across|over|through|throughout|once|long before|now|today|"
    r"this week|last (year|month|week))\b", re.I)
# Throat-clearing: prose about the telling of the story rather than the story.
HOOK = re.compile(
    r"\b(one stood out|stood out|here's|here is|there's a|this is the story|"
    r"to understand|that's where|it began|turns out|let's|the question is|"
    r"our gallery|the full list|we've chosen|below|scroll|this piece)\b", re.I)
# Colour in place of a fact, usually the paragraph that says a mystery remains.
META = re.compile(
    r"\b(struggling to explain|no idea why|remains a mystery|remains unclear|"
    r"couldn't be reached|could not be reached|did not respond|didn't respond|"
    r"did not immediately|is unclear|it's unclear)\b", re.I)
# Novelty is news value: "the first known instance" is usually the point of the
# story, and a card that drops it leaves the reader without the reason it is a
# story at all. Bare "first" is left out on purpose: "a listing's first photo"
# is not a first anything.
NOVELTY = re.compile(
    r"\b(for the first time|the first (known|ever|time|person|human)|a first for|"
    r"unprecedented|never before|first-of-its-kind)\b", re.I)
# A related-story headline or section label that the extractor kept. Title case
# with no half of the sentence doing any work: "Scientists Examine Chornobyl's
# Radioactive Dust in Unprecedented Detail" is a link to another story, not a
# fact of this one.
HEADING_STOP = set("""
a an the of in on at to for and or by with from as is are was were be been
it its his her their this that these those who what why how about into over
after before than with without
""".split())
# The writer talking about the piece, or about themselves: "My second thought:
# Microsoft knows people won't pay that much." Capitalised on purpose, because
# "US" and "us" are different words.
FIRST_PERSON = re.compile(r"^(I|I'm|I've|My|We|We're|Our)\b")
HEDGE = re.compile(r"\b(appears?|appeared|seems?|seemed|may|might)\b", re.I)
# A sentence that opens with a quotation is somebody's remark. It can be true
# and still not be what happened, so `pick_point` treats it as a last resort.
REMARK = re.compile(r"^[\"'\u201c\u2018]\s*")
# Narrative colour: the tense of the paragraph *before* the story starts. The
# past perfect ("had run out") and the progressive past ("was planning") say
# what was already true when the news began, which is exactly the scene-setting
# a card has no room for. The forms left out are the ones that carry a result:
# "had been given the drug" is backfill around a fact, not the scenery, and
# "had been trying" is somebody's afternoon.
BACKFILL = re.compile(
    r"\bhad\s+(?!been\b)(?:[a-z]{2,}(?:ed|en|wn|ne|lt|ld|nt|de)\b|"
    r"(?:run|come|gone|seen|made|taken|given|written|known|begun|found|"
    r"brought|thought|kept|held|left|lost|built|sent|spent|grown|drawn|"
    r"shown|told|sold|paid|met|led|won|become|done|got|gotten)\b)"
    r"|\bhad\s+been\s+[a-z]{2,}ing\b"
    r"|\b(?:was|were)\s+[a-z]{2,}ing\b", re.I)
# A quotation in the first person is somebody remembering. It can be true, and it
# is still not a fact of the story: "I snipped off an inch or so, put it in my
# pocket…" says how a photograph came about, not what was photographed.
QUOTED_SELF = re.compile(r"[\"\u201c]\s*(?:I|I'm|I've|I'd|I'll|we|we're|we've|"
                         r"my|our)\b")
WEAK_START = re.compile(r"^(and|but|so|yet|still|however|meanwhile|also)\b", re.I)
# A sentence that opens with a pronoun or a bare quotation depends on the one
# before it: "He just had a lot of Yemeni Jewish friends..." is not a summary of
# anything until you know who "he" is.
DEPENDENT_START = re.compile(
    r"^(he|she|they|it|this|that|these|those|his|her|its|their|we|our|i)\b",
    re.I)
NUMERIC = re.compile(
    r"(\d|\b(million|billion|trillion|thousand|per cent|percent|"
    r"half|double|triple|dozen)\b)", re.I)
# Figures written as words. "per cent" is a unit, not a figure: the number it
# belongs to is already counted, so including it would count one fact twice.
NUMBER_WORDS = re.compile(
    r"\b(million|billion|trillion|thousand|half|double|triple|dozen)\b", re.I)
MONTHS = ("january|february|march|april|may|june|july|august|september|"
          "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|"
          "oct|nov|dec")
# "October 6, 2026" as a whole is one date, not three facts: a scene-setter
# stuffed with timestamps must not outscore a sentence with one real figure.
DATE = re.compile(r"\b(?:%s)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b"
                  % MONTHS, re.I)
YEAR = re.compile(r"\b(?:1[89]|20)\d\d\b")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def content_words(text):
    return [word for word in words(text) if word not in STOPWORDS and len(word) > 1]


def jaccard(left, right):
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def stem(word):
    """A crude suffix stripper, enough to compare a headline with a sentence.

    `listing`/`listings`, `proved`/`proves` and `delivery`/`delivering` should
    count as the same word; a real stemmer is not needed for that, and this
    stays in the standard library with the rest of the module.
    """
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    for suffix in ("ing", "ed", "es"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def numeric_facts(sentence):
    """The distinct figures in a sentence, with dates taken out first.

    A date is one fact (when it happened), not the three numbers it is written
    with, and the year is almost never the point. Counting it as three would let
    "On 6 October 2026…" outscore "…by 42 percent".
    """
    stripped = YEAR.sub(" ", DATE.sub(" ", sentence))
    figures = {number for number in NUMBER.findall(stripped)}
    figures.update(word.lower() for word in NUMBER_WORDS.findall(sentence))
    return figures


def named(sentence):
    """Named things, however loosely: capitalised words and acronyms."""
    found = set(re.findall(r"\b[A-Z][a-z]{2,}\b|\b[A-Z]{2,}\b", sentence))
    if found and sentence[:1].isupper():
        found.discard(sentence.split()[0].strip(".,\"'\u201c\u201d"))
    return found


def is_heading(text):
    """Whether a line reads like a headline rather than a sentence of the story.

    Two things together, because either alone is wrong. It must have no closing
    punctuation — link titles and labels do not end in a full stop, while prose
    does — and nearly every content word must be capitalised. Capitalisation
    alone would catch a sentence that is mostly names ("Miller spoke at McGill
    University's Democracy, Rebuilt conference in Montreal on Thursday"), and
    missing punctuation alone would catch the paragraph the publisher forgot to
    finish. Numbers count as capitalised, since "40-Year-Old" is title case.
    """
    if text.rstrip()[-1:] in (".", "?", "!"):
        return False
    found = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\u2019-]*", text)
    content = [word for word in found if word.lower() not in HEADING_STOP]
    if len(content) < 4:
        return False
    capitalised = sum(1 for word in content
                      if word[0].isupper() or word[0].isdigit())
    return capitalised / len(content) >= 0.85


def quote_marks(text):
    """How many double quotation marks a sentence holds, of any style."""
    return text.count('"') + text.count("\u201c") + text.count("\u201d")


def is_colour(sentence):
    """Whether a sentence sets a scene instead of stating a fact.

    Not a tone problem: the sentence is usually true, often the warmest line on
    the page, and says nothing a reader can carry away. "He was planning to
    sprout mustard seeds for fresh samples, but he was impatient" is the
    paragraph before the story, and on today's Quanta card it landed among the
    facts beside a point about mitochondria.

    Two shapes are treated as colour, each with a guard so that a real result
    written in the same grammar survives:

    * **Backfill** — past perfect or progressive past. Guarded by figures, the
      present perfect, a reported finding and novelty, so "Prime Medicine
      announced that it had treated the first person in its trial" is a fact.
    * **A quotation in the first person** — somebody remembering. Guarded the
      same way, so "I've never seen anything like it in 30 years" keeps its
      number and its place.
    """
    settled = bool(numeric_facts(sentence) or EVENT.search(sentence)
                   or FINDING.search(sentence) or NOVELTY.search(sentence))
    if settled:
        return False
    backfill = BACKFILL.search(sentence)
    # Backfill only counts as the sentence's own tense when it is not tucked
    # into a relative clause: "at a rate the researchers had not expected" is a
    # detail *inside* a finding, not a scene being set around one. Nor when it
    # sits inside somebody's statement: "a lawyer said in court that Trump was
    # campaigning in Georgia" reports an allegation, which is the fact.
    report = REPORTING.search(sentence)
    if backfill and not (report and report.start() < backfill.start()) \
            and (FRAME_START.match(sentence)
                 or "," not in sentence[:backfill.start()]):
        return True
    return bool(QUOTED_SELF.search(sentence) and not quote_marks(sentence) % 2)


def usable_as_fact(sentence):
    """Whether a selected sentence can stand as one of the card's facts.

    Being specific is not enough. Four shapes read as noise next to a point:
    a related-story headline, a first-person aside from the writer, a
    fragment that begins or ends inside a quotation — which the sentence
    splitter cannot see, and which reads as nonsense without the words it was
    cut from ("And I think very widely respected through his whole
    organization," said the chairman of the wineries association) — and a
    sentence that is scene-setting rather than news.
    """
    if is_heading(sentence):
        return False
    if FIRST_PERSON.match(sentence):
        return False
    if quote_marks(sentence) % 2:
        return False
    if is_colour(sentence):
        return False
    return True


def headline_share(stems, headline_stems, frequencies):
    """How much of the headline this sentence carries, rare terms weighted.

    The headline's common words ("fire", "company") appear in half the article
    and say little about which sentence is the point; its distinctive ones
    ("grenade", "Wainwright") say almost everything. A headline word the article
    never uses is left out rather than counted as a miss: nothing can cover it,
    so it cannot tell two sentences apart.
    """
    weights = {term: 1.0 / math.log(2 + frequencies[term])
               for term in headline_stems if frequencies[term]}
    total = sum(weights.values())
    if not total:
        return 0.0
    return sum(weight for term, weight in weights.items() if term in stems) / total


def score(sentence, index, count, headline_stems, frequencies,
          headline_numbers=frozenset(), paragraph_length=1000):
    tokens = words(sentence)
    length = len(tokens)
    stems = {stem(token) for token in tokens}
    facts = numeric_facts(sentence)
    names = named(sentence)
    shared = headline_share(stems, headline_stems, frequencies)
    reports = bool(REPORTING.search(sentence))
    event = bool(EVENT.search(sentence))

    # Position, earned rather than given. News is an inverted pyramid, but the
    # first sentence of a feature is as likely to be weather as news, so a lead
    # only keeps its full credit if it says something specific: a headline term,
    # a figure, or a verb of consequence. This is what makes "On a frigid
    # February morning…" lose to the sentence eight paragraphs down that says
    # what happened. The event test below deliberately does not earn lead credit:
    # the past perfect is how a scene is set, not how a lead is written.
    value = (2.2 if shared or facts or reports else 0.88) * \
        (1 - index / max(count, 1)) ** 2

    value += 2.4 * shared
    if headline_numbers:
        # The number in the headline is usually the point of the story: "22".
        hit = headline_numbers & set(NUMBER.findall(sentence))
        value += 1.0 if hit else 0.0
    value += min(0.6, FACT_WEIGHT * len(facts))
    value += min(0.75, NAME_WEIGHT * len(names))
    if reports:
        value += 0.45
    if event:
        value += 0.5
    if FINDING.search(sentence):
        value += 0.7
    if NOVELTY.search(sentence):
        value += 0.5

    if HOOK.search(sentence):
        value -= 0.9
    if META.search(sentence):
        value -= 0.7
    if FRAME_START.match(sentence):
        value -= 0.35
    if HEDGE.search(sentence):
        value -= 0.25
    if length < 8:
        value -= 1.2
    elif length > 45:
        value -= 0.4
    if WEAK_START.match(sentence):
        value -= 0.35
    if DEPENDENT_START.match(sentence):
        value -= 0.6
    if REMARK.match(sentence):
        value -= 0.55
    if paragraph_length < 140:
        # Captions and related-story heads sit alone in a short paragraph.
        value -= 0.8
    # A sentence that only restates the headline — no figure, no name the
    # headline does not already have, no event — is a topic sentence, not a
    # point. This is the one that used to beat the actual finding.
    if (shared >= 0.55 and not facts and not (reports or event)
            and not any(stem(name.lower()) not in headline_stems for name in names)):
        value -= 0.5
    return value


def pick_point(ranked, sentences):
    """The best sentence, with two rules stated rather than buried in weights.

    **A remark is a last resort.** A sentence that opens with a quotation is a
    person's opinion, not the story's point — "Being the ambassador to Russia
    was probably the hardest assignment of my career" says nothing about what
    happened. Such a sentence is skipped unless nothing else comes within 40%
    of it, which keeps a story that really is a quotation working: a page with
    nothing but remarks has no other candidate and falls back to the ranking.

    **Between two near-equal candidates, earlier wins.** Within 5% of each other
    the earlier sentence usually leads the story and reads better alone; a later
    sentence tends to refer back to something already said. Anything further
    apart is a real difference in what the sentence says.
    """
    candidates = [pair for pair in ranked if not REMARK.match(sentences[pair[0]])]
    if not candidates:
        candidates = list(ranked)
    best_index, best_score = candidates[0]
    if best_score < 0.6 * ranked[0][1]:
        best_index, best_score = ranked[0]
        candidates = list(ranked)
    for index, value in candidates[:3]:
        if value >= 0.95 * best_score and index < best_index:
            return index
    return best_index


def rank(sentences, headline, paragraph_lengths=None):
    headline_stems = {stem(word) for word in content_words(headline)}
    numbers = frozenset(NUMBER.findall(headline))
    lengths = paragraph_lengths or [1000] * len(sentences)
    frequencies = Counter()
    for text in sentences:
        frequencies.update({stem(word) for word in content_words(text)})
    scored = [(index, score(text, index, len(sentences), headline_stems,
                            frequencies, numbers, lengths[index]))
              for index, text in enumerate(sentences)]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


def top_of(paragraphs, headline="", description=""):
    """The card's text: one point, a few facts, and how they were obtained."""
    body = [paragraph for paragraph in paragraphs if paragraph.strip()]
    characters = sum(len(paragraph) for paragraph in body)
    indexed = extract.sentences_indexed(body)
    story = [sentence for sentence, _ in indexed]
    paragraph_lengths = [length for _, length in indexed]

    if characters < MIN_ARTICLE_CHARS or len(story) < 2:
        return _from_description(headline, description)

    ranked = rank(story, headline, paragraph_lengths)
    point_index = pick_point(ranked, story)
    point = story[point_index]

    chosen = []
    for index, _ in ranked:
        if index == point_index or len(chosen) >= MAX_FACTS:
            continue
        candidate = story[index]
        if not usable_as_fact(candidate):
            continue
        if jaccard(content_words(candidate), content_words(point)) >= 0.45:
            continue
        if any(jaccard(content_words(candidate), content_words(other)) >= 0.45
               for other in chosen):
            continue
        chosen.append(candidate)

    chosen.sort(key=lambda text: story.index(text))
    return {"point": point, "facts": chosen, "method": "article",
            "chars": characters, "sentences": len(story)}


def _from_description(headline, description):
    """Fallback: the feed's own summary, which is short but honest.

    The same fact filter applies here. A feed summary is written by the
    publisher and is often the only text a card gets, so it was the one path
    that could still put a scene on the board — "officials were preparing for a
    difficult winter" is nobody's news. The point stays the summary's first
    sentence even when it reads as colour: it is the sentence the publisher
    chose to stand for the story, and the alternative is the headline.
    """
    story = extract.sentences([description]) if description else []
    story = [sentence for sentence in story
             if jaccard(content_words(sentence), content_words(headline)) < 0.6]
    if not story:
        return {"point": headline, "facts": [], "method": "headline",
                "chars": len(description or ""), "sentences": 0}
    facts = [sentence for sentence in story[1:] if usable_as_fact(sentence)]
    return {"point": story[0], "facts": facts[:MAX_FACTS],
            "method": "snippet", "chars": len(description),
            "sentences": len(story)}


def card(item, timeout=12.0, fetch=net.fetch_text):
    """One story as the board shows it: headline, point, facts, provenance."""
    paragraphs, note = [], ""
    try:
        html, _ = fetch(item.url, timeout=timeout)
        paragraphs = extract.article_paragraphs(html)
        if not paragraphs:
            note = "article text not readable"
    except net.FetchError as error:
        note = str(error)[:80]

    summary = top_of(paragraphs, item.title, item.description)
    return {
        "title": item.title,
        "url": item.url,
        "source": item.source,
        "genre": item.genre,
        "published": item.published.isoformat() if item.published else None,
        "point": summary["point"],
        "facts": summary["facts"],
        "method": summary["method"],
        "note": note,
    }
