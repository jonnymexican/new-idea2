"""
Tests for mediatop. Standard library only, no network: every page and feed here
is a string written in this file.

    python -m unittest mediatop.test_mediatop -v
"""

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock
from urllib.error import HTTPError
from urllib.request import urlopen

from . import extract, feeds, net, server, summary


RSS = """<?xml version="1.0"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/"
     xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Feed</title>
    <item>
      <title><![CDATA[Ottawa signs <b>new</b> deal]]></title>
      <link>https://example.test/a?utm_source=rss</link>
      <pubDate>Thu, 22 Oct 2026 09:00:00 +0000</pubDate>
      <description><![CDATA[<p>Ministers agreed on Thursday.</p>]]></description>
    </item>
    <item>
      <title>Older story</title>
      <link>https://example.test/b</link>
      <pubDate>Wed, 21 Oct 2026 09:00:00 +0000</pubDate>
      <content:encoded><![CDATA[<p>Longer body &amp; details.</p>]]></content:encoded>
    </item>
  </channel>
</rss>"""

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Atom story</title>
    <link rel="alternate" href="https://example.test/atom"/>
    <published>2026-10-22T12:30:00Z</published>
    <summary>Short summary of the atom story.</summary>
  </entry>
</feed>"""

RDF = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns="http://purl.org/rss/1.0/"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
  <item>
    <title>RDF story</title>
    <link>https://example.test/rdf</link>
    <dc:date>2026-10-20T08:00:00Z</dc:date>
    <description>RDF description.</description>
  </item>
</rdf:RDF>"""

PAGE = """<!doctype html><html><head><title>Test article — Publisher</title>
<style>.x{color:red}</style><script>var tracking = "should not appear";</script></head>
<body>
<nav><p>Home | Politics | World | Subscribe to our newsletter</p></nav>
<header><p>By A Reporter</p></header>
<div class="rail"><p><a href="/1">One headline</a> <a href="/2">Two headlines</a>
<a href="/3">Three headlines</a> <a href="/4">Four headlines</a></p></div>
<article>
  <p>OTTAWA — Parliament approved the bill 214 to 98 on Thursday, sending it to
     the Senate for a final vote next week.</p>
  <p>The rain had stopped by the time the vote was called, and the chamber was
     unusually quiet.</p>
  <p>Finance Minister Jane Doe said the measure would cost $4.2 billion over
     four years, according to the government's own estimate.</p>
  <p>Subscribe to our newsletter for the latest politics coverage.</p>
  <p>Critics argued that the costing was optimistic, because the U.S. tariff
     dispute could add roughly $900 million in the first year alone.</p>
  <p>Mr. Smith, who has covered the file since 2019, said the vote was closer
     than expected.</p>
  <p>Photograph: Reuters</p>
</article>
<footer><p>Copyright 2026. All rights reserved. Terms of service.</p></footer>
</body></html>"""


class FeedParsing(unittest.TestCase):
    def test_text_of_strips_markup_and_entities(self):
        self.assertEqual(feeds.text_of("<p>a &amp; b <b>c</b></p>"), "a & b c")

    def test_rss_items_newest_first(self):
        items = feeds.parse(RSS.encode(), feeds.FEEDS[0])
        self.assertEqual([item.title for item in items],
                         ["Ottawa signs new deal", "Older story"])
        self.assertEqual(items[0].url, "https://example.test/a?utm_source=rss")
        self.assertEqual(items[0].published.year, 2026)
        self.assertEqual(items[0].description, "Ministers agreed on Thursday.")
        self.assertEqual(items[1].description, "Longer body & details.")
        self.assertEqual(items[0].source, feeds.FEEDS[0].source)

    def test_atom_uses_href_and_published(self):
        items = feeds.parse(ATOM.encode(), feeds.FEEDS[0])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].url, "https://example.test/atom")
        self.assertEqual(items[0].description, "Short summary of the atom story.")
        self.assertEqual(items[0].published.hour, 12)

    def test_rdf_with_dc_date(self):
        items = feeds.parse(RDF.encode(), feeds.FEEDS[0])
        self.assertEqual(items[0].url, "https://example.test/rdf")
        self.assertEqual(items[0].published.day, 20)

    def test_broken_feed_is_a_fetch_error_not_a_crash(self):
        with self.assertRaises(net.FetchError):
            feeds.parse(b"<rss><channel><item>", feeds.FEEDS[0])

    def test_item_without_link_is_skipped(self):
        broken = RSS.replace("<link>https://example.test/a?utm_source=rss</link>", "")
        items = feeds.parse(broken.encode(), feeds.FEEDS[0])
        self.assertEqual([item.title for item in items], ["Older story"])

    def test_dedupe_by_title_or_url(self):
        item = feeds.Item(title="Same Story!", url="https://example.test/x?a=1",
                          source="A", genre="tech", published=None, description="")
        same_title = feeds.Item(title="same story", url="https://other.test/y",
                                source="B", genre="tech", published=None,
                                description="")
        same_url = feeds.Item(title="Different headline",
                              url="http://www.example.test/x?b=2", source="C",
                              genre="tech", published=None, description="")
        kept = feeds.dedupe([item, same_title, same_url])
        self.assertEqual([entry.source for entry in kept], ["A"])


class Extraction(unittest.TestCase):
    def test_article_paragraphs_keep_only_the_story(self):
        paragraphs = extract.article_paragraphs(PAGE)
        joined = " ".join(paragraphs)
        self.assertNotIn("tracking", joined)          # script
        self.assertNotIn("Home | Politics", joined)   # nav
        self.assertNotIn("One headline", joined)      # link rail
        self.assertNotIn("newsletter", joined.lower())
        self.assertNotIn("All rights reserved", joined)  # footer
        self.assertNotIn("Photograph", joined)        # credit
        self.assertNotIn("By A Reporter", joined)     # byline
        self.assertEqual(len(paragraphs), 5)
        self.assertTrue(paragraphs[0].startswith("OTTAWA"))

    def test_page_title(self):
        self.assertEqual(extract.page_title(PAGE), "Test article — Publisher")

    def test_page_without_paragraphs_is_empty(self):
        self.assertEqual(extract.article_paragraphs("<html><body><div>x</div>"
                                                    "</body></html>"), [])

    def test_abbreviations_do_not_end_sentences(self):
        parts = extract.split_sentences(
            "Dr. Smith met the U.S. envoy on Monday. They agreed.")
        self.assertEqual(parts, ["Dr. Smith met the U.S. envoy on Monday.",
                                 "They agreed."])

    def test_decimals_do_not_end_sentences(self):
        parts = extract.split_sentences("It rose 3.5 per cent in a year. Then it fell.")
        self.assertEqual(len(parts), 2)

    def test_dateline_is_stripped_but_names_are_not(self):
        self.assertEqual(extract.strip_dateline("OTTAWA — The vote passed."),
                         "The vote passed.")
        self.assertEqual(extract.strip_dateline("Parliament — not a dateline."),
                         "Parliament — not a dateline.")

    def test_sentences_drop_reporting_credit(self):
        sentences = extract.sentences(
            ["The vote passed on Thursday after a long debate in the House.",
             "This report by The Canadian Press was first published Oct. 22, 2026."])
        self.assertEqual(sentences, ["The vote passed on Thursday after a long "
                                     "debate in the House."])


class Summary(unittest.TestCase):
    def top(self, paragraphs=None, headline="Parliament approves bill",
            description=""):
        if paragraphs is None:
            paragraphs = extract.article_paragraphs(PAGE)
        return summary.top_of(paragraphs, headline, description)

    def test_point_is_the_lead_fact(self):
        result = self.top()
        self.assertEqual(result["method"], "article")
        self.assertIn("214 to 98", result["point"])
        self.assertNotIn("OTTAWA", result["point"])

    def test_facts_exclude_the_point_and_stay_in_order(self):
        result = self.top()
        self.assertTrue(result["facts"])
        self.assertNotIn(result["point"], result["facts"])
        positions = [PAGE.index(fact[:25]) for fact in result["facts"]]
        self.assertEqual(positions, sorted(positions))

    def test_scene_setter_does_not_become_the_point(self):
        paragraphs = [
            "The rain had stopped by the time the vote was called, and the "
            "chamber was unusually quiet.",
            "Parliament approved the bill 214 to 98 on Thursday, sending it to "
            "the Senate for a final vote.",
        ]
        result = summary.top_of(paragraphs * 3, "Vote")
        self.assertIn("214 to 98", result["point"])

    def test_a_named_scene_setter_does_not_keep_the_lead(self):
        # The Quanta shape: a warm, specific opener with names but no headline
        # term, no figure and no event verb. Position alone used to win here.
        paragraphs = [
            "On a frigid February morning in Edinburgh, Scotland, Patrick Hickey "
            "had run out of living things to put under his microscope.",
            "His time-lapse video of the dance inside a single cell was honoured "
            "in this year's Nikon Small World in Motion competition.",
            "The footage shows mitochondria moving around the cell far more than "
            "the textbooks describe, crowding past the chloroplasts.",
            "Hickey had been trying to photograph plant leaves when he noticed "
            "the movement by accident.",
        ]
        result = summary.top_of(
            paragraphs,
            "See inside cells like never before in these microscopic images")
        self.assertIn("Nikon", result["point"])
        self.assertNotIn("frigid", result["point"])

    def test_an_event_beats_the_explainer_beneath_it(self):
        # The Nature prime-editing shape. Both sentences carry the headline's
        # technical words; only the first reports that something happened.
        paragraphs = [
            "A powerful gene-editing technique has been used to edit DNA "
            "directly in a person's body - the first known instance of such a "
            "treatment.",
            "The milestone treatment is part of a clinical trial of prime "
            "editing, a technique that can alter DNA with more precision than "
            "standard CRISPR-Cas9 methods can.",
            "Prime Medicine, a biotechnology company in Cambridge, "
            "Massachusetts, announced that it had treated the first person in "
            "its trial earlier this month.",
            "In this case, prime editing was used to treat a condition called "
            "Wilson disease by rewriting DNA in liver cells, to erase a "
            "mutation that prevents them from eliminating excess copper.",
            "The company has not reported how the patient is doing, and the "
            "trial is still recruiting participants at sites in the United "
            "States and Britain.",
        ]
        result = summary.top_of(
            paragraphs,
            "A first for CRISPR: powerful prime editing alters DNA directly in "
            "the human body")
        self.assertTrue(result["point"].startswith("A powerful gene-editing"),
                        result["point"])
        self.assertNotIn("is part of", result["point"])

    def test_throat_clearing_does_not_become_the_point(self):
        # The Quanta shape again, with the hook in front of a real fact.
        paragraphs = [
            "Among the 377 results that OpenAI announced on October 6, 2026, "
            "one stood out.",
            "It claimed to settle the quasi-Riemann hypothesis, a weaker version "
            "of the Riemann hypothesis and arguably the biggest open question in "
            "mathematics.",
            "Mathematicians have spent 160 years on the stronger problem and had "
            "expected a machine to be of little help with it for a decade yet.",
            "The argument was checked by hand at three universities before the "
            "announcement, and the checking found no gaps in it.",
        ]
        result = summary.top_of(paragraphs, "AI proves the quasi-Riemann hypothesis")
        self.assertNotIn("one stood out", result["point"])
        self.assertIn("quasi-Riemann", result["point"])

    def test_topic_sentence_loses_to_the_finding(self):
        paragraphs = [
            "Birds living in cities appear to react differently to men and women, "
            "and scientists are struggling to explain why.",
            "An international research team discovered that urban birds tend to "
            "fly away sooner when approached by women than by men.",
            "The study took place in five European countries, where participants "
            "walked directly toward birds in city parks and other green spaces.",
            "On average, men were able to approach about one meter closer than "
            "women before the birds flew away.",
        ]
        result = summary.top_of(paragraphs,
                                "Birds are more afraid of women than men")
        self.assertIn("fly away sooner", result["point"])
        self.assertNotIn("struggling to explain", result["point"])

    def test_a_quotation_is_a_last_resort(self):
        paragraphs = [
            "\"Being the ambassador to Russia was probably the hardest "
            "assignment of my career,\" Lynne Tracy, who represented the US in "
            "Moscow until last year, tells the BBC.",
            "Deemed \"unfriendly\" by President Vladimir Putin's government, "
            "Western ambassadors are under constant surveillance, isolated and "
            "increasingly cut off from the country they were sent to understand.",
            "Hundreds of Russian diplomats were expelled from European capitals "
            "after the war began, and Moscow responded in kind within weeks.",
            "US, UK and European diplomats once played a crucial role in Russia, "
            "defusing tensions even at the worst moments of the Cold War.",
            "They compare the current pressure to that era, when Western "
            "embassies were monitored through physical surveillance and "
            "listening devices hidden inside gifts.",
            "Several ambassadors describe embassy staff being followed to "
            "restaurants, and phones that stop working when a delegation arrives.",
            "Moscow has declared hundreds of Western diplomats persona non "
            "grata, leaving many embassies short-staffed and unable to travel.",
            "The result is a posting in which diplomats see less of the country "
            "they are meant to explain than at any time since 1991.",
        ]
        result = summary.top_of(
            paragraphs,
            "Surveillance and 'constant paranoia': Western diplomats tell BBC "
            "of life in war-time Moscow")
        self.assertTrue(result["point"].startswith("Deemed"), result["point"])
        self.assertNotIn("hardest assignment", result["point"])

    def test_a_date_is_one_fact_and_a_year_is_not_a_fact(self):
        self.assertEqual(
            summary.numeric_facts("On October 6, 2026, 377 results appeared."),
            {"377"})
        self.assertEqual(summary.numeric_facts("It cut the risk by 42 percent."),
                         {"42"})
        self.assertEqual(summary.numeric_facts("The trial enrolled 1,137 patients."),
                         {"1,137"})

    def test_short_body_falls_back_to_the_feed_summary(self):
        result = self.top(paragraphs=["Too short to be a story."],
                          description="The government said it would act. "
                                      "Opposition parties disagreed.")
        self.assertEqual(result["method"], "snippet")
        self.assertEqual(result["point"], "The government said it would act.")
        self.assertEqual(result["facts"], ["Opposition parties disagreed."])

    def test_no_description_reports_the_headline_only(self):
        result = summary.top_of([], headline="Something happened", description="")
        self.assertEqual(result["method"], "headline")
        self.assertEqual(result["point"], "Something happened")
        self.assertEqual(result["facts"], [])

    def test_card_falls_back_when_the_page_cannot_be_read(self):
        item = feeds.Item(title="Headline here", url="https://example.test/a",
                          source="Publisher", genre="tech",
                          published=None,
                          description="First sentence of the summary is this one. "
                                      "A second sentence carries another fact.")
        def failing(url, timeout=12.0):
            raise net.FetchError("HTTP 403")
        card = summary.card(item, fetch=failing)
        self.assertEqual(card["method"], "snippet")
        self.assertIn("403", card["note"])
        self.assertEqual(card["facts"], ["A second sentence carries another fact."])

    def test_card_reads_the_article_when_it_can(self):
        item = feeds.Item(title="Parliament approves bill",
                          url="https://example.test/a", source="Publisher",
                          genre="canada", published=None, description="")
        card = summary.card(item, fetch=lambda url, timeout=12.0: (PAGE, url))
        self.assertEqual(card["method"], "article")
        self.assertEqual(card["note"], "")
        self.assertIn("214 to 98", card["point"])


class ServerRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class StubBoard:
            def board(self, genre=None, refresh=False):
                if genre and genre not in feeds.GENRES:
                    raise KeyError(genre)
                keys = [genre] if genre else list(feeds.GENRES)
                return {
                    "generated": "2026-10-22T12:00:00+00:00",
                    "genres": [{"key": key, "label": feeds.LABELS[key],
                                "built": "2026-10-22T12:00:00+00:00",
                                "age_seconds": 0, "considered": 1,
                                "cards": [{"title": "T", "url": "https://x.test",
                                           "source": "S", "genre": key,
                                           "published": None, "point": "P",
                                           "facts": [], "method": "article",
                                           "note": ""}]}
                               for key in keys],
                    "failures": [],
                    "counts": {"stories": len(keys), "sources": 1},
                }

        server.Handler.board = StubBoard()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path):
        with urlopen(self.base + path, timeout=10) as response:
            return response.status, response.headers["Content-Type"], response.read()

    def test_page_is_served(self):
        status, content_type, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn(b"mediatop", body)
        self.assertIn(b"/api/top", body)

    def test_page_offers_a_background_switcher(self):
        status, _, body = self.get("/")
        self.assertEqual(status, 200)
        for choice in (b"auto", b"light", b"dark"):
            self.assertIn(b'data-choice="' + choice + b'"', body)
        self.assertIn(b"mediatop.theme", body)
        self.assertIn(b":root[data-theme=\"light\"]", body)

    def test_board_json(self):
        status, content_type, body = self.get("/api/top")
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        payload = json.loads(body)
        self.assertEqual([g["key"] for g in payload["genres"]], list(feeds.GENRES))
        self.assertEqual(payload["counts"]["stories"], 4)

    def test_single_genre(self):
        payload = json.loads(self.get("/api/top?genre=tech")[2])
        self.assertEqual([g["key"] for g in payload["genres"]], ["tech"])

    def test_unknown_genre_is_a_400(self):
        with self.assertRaises(HTTPError) as caught:
            self.get("/api/top?genre=gossip")
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("unknown genre", caught.exception.read().decode())

    def test_unknown_path_is_a_404(self):
        with self.assertRaises(HTTPError) as caught:
            self.get("/nope")
        self.assertEqual(caught.exception.code, 404)

    def test_favicon(self):
        status, content_type, _ = self.get("/favicon.svg")
        self.assertEqual(status, 200)
        self.assertIn("image/svg+xml", content_type)


class Spreading(unittest.TestCase):
    def test_sources_take_turns(self):
        cards = ([{"source": "A", "n": index} for index in range(5)]
                 + [{"source": "B", "n": 10 + index} for index in range(3)])
        order = server.spread(cards, 4)
        self.assertEqual([(card["source"], card["n"]) for card in order],
                         [("A", 0), ("B", 10), ("A", 1), ("B", 11)])

    def test_limit_larger_than_the_board(self):
        cards = [{"source": "A", "n": 1}, {"source": "B", "n": 2}]
        self.assertEqual(len(server.spread(cards, 10)), 2)


class BoardCaching(unittest.TestCase):
    def setUp(self):
        item = feeds.Item(title="A story", url="https://example.test/a",
                          source="S", genre="tech",
                          published=None, description="")
        self.items = ([item], [{"source": "Broken", "error": "HTTP 500"}])
        self.card = {"title": "A story", "url": "https://example.test/a",
                     "source": "S", "genre": "tech", "published": None,
                     "point": "P", "facts": [], "method": "article", "note": ""}

    def test_second_read_is_cached_and_refresh_rebuilds(self):
        with mock.patch.object(feeds, "load_genre", return_value=self.items) as load, \
             mock.patch.object(summary, "card", return_value=self.card):
            board = server.Board(per_genre=2, ttl=600, workers=2)
            try:
                first = board.board(genre="tech")
                second = board.board(genre="tech")
            finally:
                board.close()
        self.assertEqual(load.call_count, 1)
        self.assertEqual(first["counts"]["stories"], 1)
        self.assertEqual(second["counts"]["stories"], 1)
        self.assertEqual(first["failures"][0]["source"], "Broken")

    def test_a_failed_build_reports_instead_of_raising(self):
        with mock.patch.object(feeds, "load_genre",
                              side_effect=RuntimeError("boom")):
            board = server.Board(per_genre=2, ttl=600, workers=2)
            try:
                payload = board.board(genre="tech")
            finally:
                board.close()
        self.assertEqual(payload["genres"][0]["cards"], [])
        self.assertIn("boom", payload["failures"][0]["error"])

    def test_unknown_genre_raises_keyerror(self):
        board = server.Board(per_genre=1, ttl=600, workers=1)
        try:
            with self.assertRaises(KeyError):
                board.board(genre="gossip")
        finally:
            board.close()


if __name__ == "__main__":
    unittest.main()
