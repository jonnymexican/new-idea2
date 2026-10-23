# mediatop

The point of the news, one screen, four genres.

Every story on the board is reduced to **one sentence that is the point** and
**three or four facts around it**, all quoted from the article — never
paraphrased. A card is meant to be read in about fifteen seconds, and to leave
you knowing whether you want the full story, not to replace it.

Genres: **Tech**, **Geopolitics**, **Canadian government**, **Science**.

```bash
python -m mediatop.server            # http://127.0.0.1:8770/
python -m mediatop.server --per-genre 8 --open
python -m mediatop.server --print    # the same board, as text, no browser
python -m mediatop.server --save     # the board as mediatop-2026-10-23.md
python -m unittest mediatop.test_mediatop
```

`--save` writes the board as Markdown, named for the day it was built, so a
morning run leaves one file per morning rather than a growing list. No browser
and no server are involved: it builds once, writes, and exits. `--save boards/`
puts them in a directory instead, and `--save notes/today.md` uses an exact
name. The file says the same things the page does — each card's point, its
facts, the source, and `summary only` where the article could not be read —
plus any feeds that failed, at the bottom.

No dependencies beyond the standard library, no build step, no API key, no
model. `python -m mediatop.server --print` is the fastest way to see what the
summariser actually does to a day's stories.

## How a card is made

1. **Read twelve feeds** — three per genre, all checked by hand for one
   property: their article pages contain the story as real paragraphs. A feed
   whose pages cannot be read is a headline list, and a headline list is not
   what this is.
2. **Fetch each article** and keep the `<p>` text inside `<article>`/`<main>`,
   discarding menus, cookie bars, link rails, captions, credits, "related"
   modules and job adverts.
3. **Split it into sentences**, without breaking on `Dr.`, `U.S.` or `3.5`.
4. **Score the sentences against the headline**, which is the editor's own
   one-line answer to "what happened". A sentence carrying the headline's
   *distinctive* words counts for far more than one carrying its common ones:
   "grenade" says which sentence is the point where "fire" appears everywhere.
   On top of that: figures (a date counts once, as a date, and a year is not a
   fact), named things, a verb of consequence (`announced`, `found`, `ruled`),
   an **event** in the present perfect (`has been used`, `have agreed`) — which
   is the difference between a lede and the paragraph under it that explains
   its subject — novelty (`the first known`), and position, which is
   **earned**, because a lead keeps its full credit only if it says something
   specific. Three named
   failure modes are penalised directly: the **scene-setter** ("On a frigid
   February morning…"), the **throat-clearer** ("Among the 377 results…, one
   stood out") and the **topic sentence** that restates the headline without
   adding a fact. A sentence opening with a quotation is somebody's remark, and
   becomes the point only if nothing else comes within 40% of it.
5. **Take the point and a few non-duplicate facts**, then put the facts back in
   document order so the card reads top-down. A candidate has to be a fact of
   the story, not merely a specific sentence: related-story headlines
   ("Scientists Examine Chornobyl's Radioactive Dust in Unprecedented Detail"),
   the writer's asides ("My second thought: …") and fragments that begin or end
   inside somebody's quotation are skipped, and the next sentence down takes the
   place.

### What it refuses to do

- **It does not rewrite.** There is no model in this project, and that is a
  design decision: a summary that reads smoothly but invents a figure is worse
  than a quoted sentence, because a reader cannot tell which happened.
- **It does not pretend.** When a page is paywalled, blocked, or a JavaScript
  shell, the card says `summary only` and shows the feed's own description
  instead of the article — with a tooltip saying why. Feeds that fail are listed
  at the top of the board, not silently dropped.
- **It does not hammer anyone.** Every feed is fetched at most once per ten
  minutes (`--ttl`), and each article page once per build, for roughly 30
  requests per refresh. The user agent is a browser string because CBC's edge
  drops anything else — the same URL that hangs for 20 s as `mediatop/0.1`
  returns in 200 ms as Chrome — and it asks for no more than a browser would.

## Endpoints

| Route | Purpose |
| --- | --- |
| `GET /` | the board (HTML, CSS and JS in one file) |
| `GET /api/top` | the board as JSON |
| `GET /api/top?genre=tech` | one genre |
| `GET /api/top?refresh=1` | rebuild instead of serving the cache |

`--port`, `--per-genre`, `--ttl` and `--timeout` are the knobs; the server binds
to loopback and serves whatever it fetched to anyone who can reach the port.

## Files

| File | Purpose |
| --- | --- |
| [`net.py`](net.py) | one GET per resource: timeout, size ceiling, gzip, tolerant decoding |
| [`feeds.py`](feeds.py) | the twelve feeds, RSS/Atom/RDF parsing, dedupe by title or URL |
| [`extract.py`](extract.py) | HTML → story paragraphs → sentences, and the furniture filters |
| [`summary.py`](summary.py) | sentence scoring, the point, the facts, the fallback |
| [`server.py`](server.py) | cache, board assembly, HTTP routes, `--print`, `--save` |
| [`digest.py`](digest.py) | the board as a dated Markdown file |
| [`page.html`](page.html) | the single-screen board, and its Auto / Day / Night background switcher |
| [`test_mediatop.py`](test_mediatop.py) | 48 tests: fixtures only, no network |

## Where it is weak

These are measured or observed, not hypothetical:

- **An event is recognised by grammar, not by meaning.** On 19 live articles
  labelled by hand the point now matches the label 17 times, up from 12, and no
  scene-setter, throat-clearer or quoted remark won a point it used to win. The
  present perfect is what tells the lede from the explainer beneath it — but it
  is a proxy: "the company has been clear about the price" is not an event, and
  a plain-past event with no reporting verb ("the fire killed four soldiers")
  gets no credit for being one.
- **A fact can still be colour.** "He was planning to sprout mustard seeds for
  fresh samples, but he was impatient" can survive the filter when it is the
  most specific line left on the page, because telling a scene from a fact is
  the same judgement the point picker makes — and it makes it with a scoring
  model, not one rule.
- **A surname with no context can land among the facts.** "Rutland said 27
  soldiers are still in hospital" is true and specific, but a reader who
  skipped the paragraph above does not know who Rutland is. Headline anchoring
  keeps it out of the point now; it is still a fine fact.
- **The one-line-paragraph rule is a guess.** It drops most captions and
  related-story heads, and occasionally a real sentence that the publisher put
  in a paragraph of its own.
- **English only**, and only the twelve feeds listed in
  [`feeds.py`](feeds.py) — adding one means checking that its article pages are
  readable, which is the whole test.
