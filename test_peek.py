"""The attention viewer, minus the browser.

What can be pinned here is the analysis: that capturing weights does not change
the model's own output, that attention stays causal and normalised once it has
been through the API, that the surprise figures are the model's real
probabilities rather than hindsight, and that the window drops the head of a
long passage rather than quietly reading less than it claims.

A tiny random model with a character tokenizer stands in for a checkpoint: the
properties being checked are structural, and none of them need weights that
learned anything.
"""

import json
import math
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import torch

import peek
from model import GPT


class CharTokenizer:
    """The smallest thing that satisfies the tokenizer contract."""

    def __init__(self, chars="abcdefghijklmnopqrstuvwxyz .\n"):
        self.chars = list(chars)
        self.stoi = {char: index for index, char in enumerate(self.chars)}
        self.itos = {index: char for index, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.stoi[char] for char in text if char in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


BLOCK = 12


def tiny_model(seed=0):
    torch.manual_seed(seed)
    tokenizer = CharTokenizer()
    model = GPT(vocab_size=tokenizer.vocab_size, n_embd=16, n_head=2,
                n_layer=2, block_size=BLOCK, dropout=0.0)
    model.eval()
    return model, tokenizer


class LookCase(unittest.TestCase):
    def setUp(self):
        self.model, self.tokenzier = tiny_model()
        # Short enough to sit inside the 12-token window: the long-passage case
        # is its own test, where dropping the head is the point.
        self.text = "the cat sat"

    def look(self, text=None, **kwargs):
        return peek.look(self.model, self.tokenzier,
                         self.text if text is None else text, **kwargs)


class TestCapture(LookCase):
    def test_capturing_weights_does_not_change_the_logits(self):
        # A viewer that altered the thing it observes would be worse than none.
        idx = torch.tensor([self.tokenzier.encode(self.text)])
        with torch.no_grad():
            plain, _ = self.model(idx)
            captured = []
            watched, _ = self.model(idx, capture=captured)
        self.assertTrue(torch.equal(plain, watched))
        self.assertEqual(len(captured), len(self.model.blocks))

    def test_each_entry_is_one_layers_heads_by_tokens(self):
        idx = torch.tensor([self.tokenzier.encode("abcd")])
        capture = []
        with torch.no_grad():
            self.model(idx, capture=capture)
        for layer in capture:
            self.assertEqual(tuple(layer.shape), (1, 2, 4, 4))


class TestAnalysis(LookCase):
    def test_attention_never_looks_ahead(self):
        data = self.look()
        for layer in data["attention"]:
            for head in layer:
                for query, row in enumerate(head):
                    for key in range(query + 1, len(row)):
                        self.assertEqual(row[key], 0,
                                         f"query {query} saw key {key}")

    def test_every_row_still_sums_to_one_after_rounding(self):
        data = self.look()
        for layer in data["attention"]:
            for head in layer:
                for row in head:
                    self.assertAlmostEqual(sum(row), 1000, delta=10)

    def test_one_matrix_per_layer_and_head(self):
        data = self.look()
        self.assertEqual(data["layers"], len(self.model.blocks))
        self.assertEqual(data["heads"], 2)
        self.assertEqual(len(data["attention"]), data["layers"])
        for layer in data["attention"]:
            self.assertEqual(len(layer), data["heads"])

    def test_tokens_rebuild_the_text(self):
        data = self.look()
        self.assertEqual("".join(token["text"] for token in data["tokens"]),
                         self.text)
        self.assertEqual(data["decoded"], self.text)

    def test_surprise_is_the_probability_the_model_actually_gave(self):
        # Recomputed from a plain forward pass, so this checks the reported bits
        # against the model rather than against peek's own arithmetic.
        ids = self.tokenzier.encode(self.text)
        with torch.no_grad():
            logits, _ = self.model(torch.tensor([ids]))
        probs = torch.softmax(logits[0], dim=-1)
        data = self.look()
        self.assertIsNone(data["tokens"][0]["surprise"])
        for index in range(1, len(ids)):
            expected = -math.log2(float(probs[index - 1, ids[index]]))
            self.assertAlmostEqual(data["tokens"][index]["surprise"], expected,
                                   delta=0.02)

    def test_the_window_is_the_tail_and_says_what_it_dropped(self):
        long_text = "abcde " * 10          # 60 characters, a 12-token window
        data = self.look(long_text)
        self.assertEqual(len(data["tokens"]), BLOCK)
        self.assertEqual(data["dropped"], len(long_text) - BLOCK)
        self.assertEqual(data["window"], BLOCK)
        self.assertTrue(data["decoded"].endswith("abcde "[-6:]))

    def test_next_token_bars_descend_and_are_a_distribution(self):
        data = self.look(top_k=5)
        probabilities = [entry["p"] for entry in data["next"]]
        self.assertEqual(len(probabilities), 5)
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        self.assertLessEqual(sum(probabilities), 1.0001)
        for entry in data["next"]:
            self.assertEqual(len(self.tokenzier.decode([entry["id"]])), 1)

    def test_entropy_matches_the_distribution_shape(self):
        # Two equally likely outcomes are one bit undecided; a spread across the
        # whole vocabulary is more. A randomly initialised model is the latter.
        data = self.look()
        self.assertGreater(data["entropy_bits"], 1.0)
        self.assertLessEqual(data["entropy_bits"], math.log2(self.tokenzier.vocab_size))

    def test_nothing_to_read_is_answered_rather_than_refused(self):
        data = self.look("")
        self.assertEqual(data["tokens"], [])
        self.assertEqual(data["attention"], [])
        self.assertEqual(data["next"], [])


class TestServer(LookCase):
    def setUp(self):
        super().setUp()
        self.server = peek.serve(self.model, self.tokenzier, "runs/test/model.pt",
                                 "127.0.0.1", 0, open_browser=False,
                                 sample="the cat sat")
        port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, path, payload):
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base + path, data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=20) as reply:
                return reply.status, json.loads(reply.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=20) as reply:
                return reply.status, reply.headers.get("Content-Type", ""), reply.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers.get("Content-Type", ""), exc.read()

    def test_the_page_and_its_assets_are_served(self):
        status, content_type, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn(b"What the model is reading", body)
        for path, expected in (("/peek.js", "javascript"), ("/peek.css", "text/css")):
            status, content_type, _ = self.get(path)
            self.assertEqual((status, expected in content_type), (200, True), path)

    def test_the_page_ships_the_head_gallery_and_no_head_chips(self):
        # The gallery draws every head from the one attention payload, tile by
        # tile. It replaced the chips that showed one head at a time, so the
        # container the script fills has to be in the page — and the chips, if
        # they ever come back, would be a page the gallery no longer describes.
        _, _, page = self.get("/")
        self.assertIn(b'id="gallery"', page)
        self.assertNotIn(b'id="heads"', page)
        _, _, script = self.get("/peek.js")
        self.assertIn(b'getElementById("gallery")', script)

    def test_info_describes_the_checkpoint_the_page_is_reading(self):
        status, _, body = self.get("/api/info")
        self.assertEqual(status, 200)
        info = json.loads(body)
        self.assertEqual(info["checkpoint"], "runs/test/model.pt")
        self.assertEqual((info["layers"], info["heads"]), (2, 2))
        self.assertEqual(info["block_size"], BLOCK)
        self.assertEqual(info["sample"], "the cat sat")
        self.assertEqual(info["params"], self.model.num_params())

    def test_a_posted_passage_comes_back_analysed(self):
        status, data = self.post("/api/look", {"text": "the cat"})
        self.assertEqual(status, 200)
        self.assertEqual("".join(token["text"] for token in data["tokens"]), "the cat")
        self.assertEqual(len(data["attention"]), 2)

    def test_a_malformed_body_is_a_bad_request_not_a_crash(self):
        request = urllib.request.Request(
            self.base + "/api/look", data=b"not json",
            headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=20)
        self.assertEqual(caught.exception.code, 400)

    def test_an_overlong_passage_is_refused_before_it_is_tokenised(self):
        status, data = self.post("/api/look", {"text": "a" * (peek.MAX_TEXT_CHARS + 1)})
        self.assertEqual(status, 413)
        self.assertIn("longer than", data["error"])


class TestSample(unittest.TestCase):
    def test_the_runs_own_sample_is_offered_when_it_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "sample.txt").write_text("TIZELUS:\nYes, Yer-What\n", encoding="utf-8")
            self.assertTrue(peek.sample_beside(run / "model.pt").startswith("TIZELUS"))

    def test_a_run_without_a_sample_offers_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(peek.sample_beside(Path(tmp) / "model.pt"), "")

    def test_a_long_sample_is_cut_to_a_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "sample.txt").write_text("x" * 5000, encoding="utf-8")
            self.assertEqual(len(peek.sample_beside(run / "model.pt")), peek.SAMPLE_CHARS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
