#!/usr/bin/env python3
"""
The repo's test suite: the invariants this project actually depends on.

These are not smoke tests. Each one pins a property that has either broken
before or that the numbers in the README would become wrong without:

- a BPE round-trip is lossless, deterministic, and leaves no merge unapplied
- a checkpoint's tokenizer rebuilds the exact vocabulary the weights were
  trained against, including checkpoints written before tokenizers were pluggable
- `sample_next` behaves as documented (top-k, top-p, temperature, top_k <= 0)
  and `generate` is exactly the loop around it
- the bigram floor is measured on the same validation split as the model, so the
  two can honestly be plotted on one axis
- resuming writes no duplicate point at the seam step
- the web server streams NDJSON chunks over chunked encoding, rejects bad
  settings with a reason, and stops sampling when the client walks away
- the page's token view cannot be squeezed out of its own height: the layout
  contract that keeps it visible is resolved from the stylesheet the server
  actually sends, and the check is shown to catch the layout that once
  collapsed it
- the edge of the model's window is a real character offset, not an
  approximation: it is cross-checked two ways, works for both tokenizers, and
  goes before the character when a byte-level token cuts one in half
- the window can be deliberately shortened but never stretched past what the
  checkpoint was trained for, and the crop is pinned where sampling noise cannot
  mask it — on the width of the tensors the model is actually handed — alongside
  a check that the preview the control asks for is exactly what a reply of the
  same window reports

Standard library only, on purpose: this project has one dependency (torch) and
the tests should not add a second.

    .venv/bin/python test_all.py               # everything
    .venv/bin/python test_all.py TestBPE       # one class
    .venv/bin/python test_all.py TestWebServer # just the server
"""

import html.parser
import http.client
import io
import json
import math
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

import torch

from bpe import BYTE_VOCAB, BPETokenizer
from bpe import train as train_bpe
import chat
import serve
from model import GPT, sample_next
from train import CharTokenizer, tokenizer_from_state

CORPUS = "data/input.txt"
HERE = os.path.dirname(os.path.abspath(__file__))


def read_corpus(limit=None):
    with open(os.path.join(HERE, CORPUS), "r", encoding="utf-8") as f:
        text = f.read()
    return text[:limit] if limit else text


# ------------------------------------------------------------------- layout
# The page's geometry, read out of the page itself.
#
# The token view exists to answer "what does the model actually read", and it
# once collapsed to 12px — every chip present, none of them visible — because a
# shrinkable flex item yields its space to whatever above it is taller. That is
# a layout bug, and the suite is standard library only, so rather than growing a
# browser dependency these helpers resolve the declarations the served page
# really carries.
#
# What follows can prove the *contract* the layout rests on ("this panel cannot
# shrink, and the pane above it is the one that gives space up"). What it cannot
# do is measure a box; where a number depends on font metrics or on an
# element's content, it is not guessed at here. A pixel measurement stays a
# browser check.


VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}

# At-rules that declare no box of their own, so there is nothing for a layout
# check to read inside them. Anything else raises rather than being skipped: a
# rule the resolver cannot see is a rule that escapes the checks below, and
# quietly passing is the failure mode worth avoiding.
IGNORED_AT_RULES = ("@keyframes",)


class ElementPaths(html.parser.HTMLParser):
    """Where each identified element sits in the document.

    Flex factors only mean anything between siblings: whether the transcript
    pane and the token view really do share one flex column is a question about
    the tree, not about the stylesheet.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.paths = {}          # id -> tuple of (tag, id, classes) from the root

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        node = (tag, attrs.get("id"), tuple((attrs.get("class") or "").split()))
        if tag in VOID_ELEMENTS:
            return
        self.stack.append(node)
        if node[1]:
            self.paths[node[1]] = tuple(self.stack)

    def handle_endtag(self, tag):
        if tag not in VOID_ELEMENTS and self.stack:
            self.stack.pop()


def element_paths(html):
    parser = ElementPaths()
    parser.feed(html)
    return parser.paths


def page_stylesheet(html):
    """The contents of the page's inline <style> block."""
    start = html.index("<style>") + len("<style>")
    end = html.index("</style>", start)
    return html[start:end]


def strip_comments(css):
    """CSS with its comments removed, so a comment cannot read as a rule."""
    out, i = [], 0
    while i < len(css):
        if css.startswith("/*", i):
            i = css.index("*/", i) + 2
        else:
            out.append(css[i])
            i += 1
    return "".join(out)


def brace_blocks(text):
    """Split one stylesheet level into (prelude, body) pairs."""
    pairs, depth, prelude, body = [], 0, "", ""
    for ch in text:
        if ch == "{" and depth == 0:
            depth, body = 1, ""
        elif ch == "{":
            depth += 1
            body += ch
        elif ch == "}" and depth == 1:
            depth = 0
            pairs.append((prelude, body))
            prelude = ""
        elif ch == "}":
            depth -= 1
            body += ch
        elif depth:
            body += ch
        else:
            prelude += ch
    return pairs


def stylesheet_rules(html):
    """(media, selector, declarations) for every rule, in source order.

    `declarations` is an ordered list of (property, value) pairs, because the
    cascade settles `flex: 0 0 auto` against a later `flex-shrink` by source
    order rather than by name. `media` is the condition a rule sits inside, or
    None at the top level.
    """
    rules = []

    def parse(text, media):
        for prelude, body in brace_blocks(text):
            prelude = prelude.strip()
            if prelude.startswith("@"):
                if prelude.startswith("@media"):
                    parse(body, prelude[len("@media"):].strip())
                elif not prelude.startswith(IGNORED_AT_RULES):
                    raise ValueError(f"unsupported at-rule in the page: {prelude}")
                continue
            declarations = []
            for chunk in body.split(";"):
                if not chunk.strip():
                    continue
                prop, _, value = chunk.partition(":")
                declarations.append((prop.strip().lower(), value.strip()))
            for selector in prelude.split(","):
                rules.append((media, selector.strip(), declarations))

    parse(strip_comments(page_stylesheet(html)), None)
    return rules


def media_matches(condition, width):
    """Would a media block apply at viewport width `width`?

    Only the features this page uses are understood, and anything else raises:
    a condition that silently evaluated to False would hide rules from the
    layout checks instead of failing them.
    """
    if condition is None:
        return True
    matched = True
    for part in condition.split(" and "):
        prop, _, raw = part.strip().strip("()").partition(":")
        prop, raw = prop.strip().lower(), raw.strip()
        if prop not in ("max-width", "min-width"):
            raise ValueError(f"unsupported media feature: {prop}")
        bound = length_px(raw)
        matched = matched and (width <= bound if prop == "max-width"
                               else width >= bound)
    return matched


def declarations_for(rules, selector, width):
    """The declarations `selector` ends up with at viewport width `width`,
    in cascade order (later rules win, so the order is the resolution)."""
    out = []
    for media, sel, declarations in rules:
        if sel == selector and media_matches(media, width):
            out.extend(declarations)
    return out


def length_px(value, viewport_height=None):
    """A CSS length in pixels.

    `vh` needs a viewport height to mean anything; a length whose pixel size
    depends on a font or on a percentage raises, because guessing it would turn
    a layout check into a check of the guess.
    """
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*(px|vh)", value.strip().lower())
    if not match:
        raise ValueError(f"this is not a length the checks can resolve: {value!r}")
    amount, unit = float(match.group(1)), match.group(2)
    if unit == "vh":
        if viewport_height is None:
            raise ValueError("a vh length needs a viewport height")
        return amount / 100.0 * viewport_height
    return amount


def expand_flex(value):
    """The `flex` shorthand as its three longhands (grow, shrink, basis)."""
    if value == "none":
        return 0.0, 0.0, "auto"
    if value == "initial":
        return 0.0, 1.0, "auto"
    if value == "auto":
        return 1.0, 1.0, "auto"
    parts = value.split()
    if len(parts) != 3:
        # Every `flex` on this page is written in the three-value form. A
        # shorthand whose defaults differ per form is not worth resolving by
        # guesswork: write it out and let the checks read it.
        raise ValueError(f"write flex in its three-value form: {value!r}")
    return float(parts[0]), float(parts[1]), parts[2]


def flex_factors(declarations):
    """(grow, shrink, basis) as the cascade resolves them for one element.

    The initial values are the whole point of this helper: an element with no
    `flex` declaration at all is `0 1 auto`, and that default shrink of 1 is
    what let a tall pane above the token view collapse it.
    """
    grow, shrink, basis = 0.0, 1.0, "auto"
    for prop, value in declarations:
        value = value.strip().lower()
        if prop == "flex":
            grow, shrink, basis = expand_flex(value)
        elif prop == "flex-grow":
            grow = float(value)
        elif prop == "flex-shrink":
            shrink = float(value)
        elif prop == "flex-basis":
            basis = value
    return grow, shrink, basis


class TestBPE(unittest.TestCase):
    """Byte-level BPE: the properties the tokenizer work rests on."""

    VOCAB = 384          # 128 merges over the 256 bytes
    SLICE = 30_000       # enough repetition to learn real merges, and fast

    @classmethod
    def setUpClass(cls):
        cls.text = read_corpus(cls.SLICE)
        cls.merges = train_bpe(cls.text, cls.VOCAB)
        cls.tok = BPETokenizer(cls.merges)

    def test_round_trip_is_lossless(self):
        """Every corpus byte survives encode then decode."""
        self.assertEqual(self.tok.decode(self.tok.encode(self.text)), self.text)

    def test_round_trip_survives_non_ascii(self):
        """Bytes, not characters, are the unit — so non-ASCII must round-trip."""
        for sample in ("héllo wörld", "世界 and back", "emoji 🙂 inside",
                       "mixed — dashes… and 'quotes'", ""):
            with self.subTest(sample=sample):
                self.assertEqual(
                    self.tok.decode(self.tok.encode(sample)), sample
                )

    def test_vocab_size_is_exactly_what_was_asked_for(self):
        self.assertEqual(self.tok.vocab_size, self.VOCAB)
        self.assertEqual(len(self.merges), self.VOCAB - BYTE_VOCAB)

    def test_vocab_size_at_or_below_bytes_is_rejected(self):
        for size in (0, 100, BYTE_VOCAB):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    train_bpe(self.text, size)

    def test_training_is_deterministic(self):
        """Same corpus, same vocabulary — no reliance on dict/set ordering."""
        self.assertEqual(train_bpe(self.text, 300), train_bpe(self.text, 300))

    def test_encoding_leaves_no_mergeable_pair(self):
        """The greedy encoder must not stop with a known pair still adjacent."""
        for sample in (self.text[:500], "the king shall speak", "thou art here"):
            with self.subTest(sample=sample[:30]):
                ids = self.tok.encode(sample)
                for a, b in zip(ids, ids[1:]):
                    self.assertNotIn(
                        (a, b), self.tok.ranks,
                        f"pair {(a, b)} was mergeable but left in place",
                    )

    def test_tokens_expand_to_the_bytes_they_stand_for(self):
        """A merged id decodes to exactly the concatenation of its two halves."""
        for i in range(BYTE_VOCAB, self.tok.vocab_size):
            a, b = self.tok.merges[i - BYTE_VOCAB]
            with self.subTest(token=i):
                self.assertEqual(
                    self.tok.decode([i]),
                    self.tok.decode([a]) + self.tok.decode([b]),
                )

    def test_longest_tokens_are_whole_words(self):
        """If this fails, the pretokeniser or the merge loop is broken."""
        longest = sorted(
            (self.tok.decode([i]) for i in range(BYTE_VOCAB, self.tok.vocab_size)),
            key=len,
        )
        self.assertGreaterEqual(len(longest[-1]), 5)
        self.assertIn(longest[-1], self.text)

    def test_compression_beats_one_token_per_character(self):
        ratio = len(self.text) / len(self.tok.encode(self.text))
        self.assertGreater(ratio, 1.5,
                           f"only {ratio:.2f} characters per token")

    def test_save_and_load_reproduce_the_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tokenizer.json")
            self.tok.save(path)
            reloaded = BPETokenizer.load(path)
            self.assertEqual(reloaded.merges, self.tok.merges)
            self.assertEqual(reloaded.encode(self.text), self.tok.encode(self.text))

    def test_state_rebuilds_through_tokenizer_from_state(self):
        """This is the path a checkpoint takes; it must be lossless."""
        rebuilt = tokenizer_from_state(self.tok.state())
        self.assertEqual(rebuilt.name, "bpe")
        self.assertEqual(rebuilt.vocab_size, self.tok.vocab_size)
        self.assertEqual(rebuilt.encode(self.text), self.tok.encode(self.text))

    def test_show_agrees_with_encode(self):
        """`/tokens` in chat.py is built on this, so it must match exactly."""
        for sample in ("the king", "GLOUCESTER: I am here", ""):
            with self.subTest(sample=sample):
                pieces = self.tok.show(sample)
                self.assertEqual([i for i, _ in pieces], self.tok.encode(sample))
                self.assertEqual("".join(p for _, p in pieces), sample)


class TestCharTokenizerState(unittest.TestCase):
    """The character tokenizer and the legacy checkpoint format."""

    def test_round_trip_and_state(self):
        tok = CharTokenizer(text="the quick brown fox")
        self.assertEqual(tok.decode(tok.encode("the quick")), "the quick")
        rebuilt = tokenizer_from_state(tok.state())
        self.assertEqual(rebuilt.vocab_size, tok.vocab_size)
        self.assertEqual(rebuilt.encode("the fox"), tok.encode("the fox"))

    def test_legacy_itos_dict_is_accepted(self):
        """checkpoints from before tokenizers were pluggable stored itos as a dict."""
        tok = CharTokenizer(text="abcabc")
        state = {"name": "char", "itos": [tok.itos[i] for i in sorted(tok.itos)]}
        rebuilt = tokenizer_from_state(state)
        self.assertEqual(rebuilt.itos, tok.itos)
        self.assertEqual(rebuilt.vocab_size, tok.vocab_size)


class TestSampling(unittest.TestCase):
    """`sample_next` and `generate` — documented behaviour, pinned."""

    @staticmethod
    def reference_sample(logits, temperature=1.0, top_k=None, top_p=None):
        """An independent implementation of the documented sampling rules.

        Written from the description rather than by calling the code under test,
        so that a change in `sample_next` shows up as a mismatch here.
        """
        logits = logits / max(temperature, 1e-8)
        if top_k is not None and top_k > 0:
            cutoff = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., [-1]]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        if top_p is not None:
            sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            sorted_logits = sorted_logits.masked_fill(
                cumulative - probs > top_p, float("-inf")
            )
            logits = logits.scatter(-1, order, sorted_logits)
        return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(1, 40)

    def test_matches_the_reference_implementation(self):
        """Same seed, same cutoffs, same token — for every documented control."""
        configs = [
            {},
            {"temperature": 0.5},
            {"temperature": 2.0},
            {"top_k": 1},
            {"top_k": 3},
            {"top_k": 0},              # documented as "disables it"
            {"top_k": 40},
            {"top_p": 0.5},
            {"top_p": 0.9},
            {"top_k": 5, "top_p": 0.8, "temperature": 1.3},
        ]
        for kwargs in configs:
            with self.subTest(**kwargs):
                torch.manual_seed(1234)
                got = sample_next(self.logits.clone(), **kwargs)
                torch.manual_seed(1234)
                want = self.reference_sample(self.logits.clone(), **kwargs)
                self.assertTrue(torch.equal(got, want))

    def test_top_k_zero_and_negative_disable_the_cutoff(self):
        """The bug this guards: top_k=0 used to crash inside sample_next."""
        for top_k in (0, -1, -5):
            with self.subTest(top_k=top_k):
                torch.manual_seed(7)
                got = sample_next(self.logits.clone(), top_k=top_k)
                torch.manual_seed(7)
                want = sample_next(self.logits.clone(), top_k=None)
                self.assertTrue(torch.equal(got, want))
                self.assertTrue(0 <= int(got) < self.logits.size(-1))

    def test_top_k_one_is_greedy(self):
        for _ in range(20):
            token = sample_next(self.logits.clone(), top_k=1)
            self.assertEqual(int(token), int(self.logits.argmax(dim=-1)))

    def test_top_p_keeps_the_nucleus(self):
        """With one dominant logit, a small nucleus has to select it."""
        peaked = torch.full((1, 10), -20.0)
        peaked[0, 4] = 20.0
        for _ in range(20):
            self.assertEqual(int(sample_next(peaked.clone(), top_p=0.5)), 4)

    def test_temperature_zero_is_finite_and_greedy(self):
        """0 would divide by zero; the floor in sample_next must hold."""
        token = sample_next(self.logits.clone(), temperature=0.0, top_k=1)
        self.assertEqual(int(token), int(self.logits.argmax(dim=-1)))

    def test_sampling_is_deterministic_under_a_seed(self):
        torch.manual_seed(99)
        first = [int(sample_next(self.logits.clone())) for _ in range(5)]
        torch.manual_seed(99)
        second = [int(sample_next(self.logits.clone())) for _ in range(5)]
        self.assertEqual(first, second)
        torch.manual_seed(100)
        third = [int(sample_next(self.logits.clone())) for _ in range(5)]
        self.assertNotEqual(first, third)

    def test_generate_is_exactly_a_loop_over_sample_next(self):
        """Pins the refactor that let chat.py and train.py share one sampler."""
        torch.manual_seed(0)
        model = GPT(vocab_size=17, n_embd=16, n_head=2, n_layer=1,
                    block_size=8, dropout=0.0)
        model.eval()
        prompt = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)

        torch.manual_seed(42)
        generated = model.generate(prompt.clone(), max_new_tokens=5,
                                   temperature=0.9, top_k=4)

        torch.manual_seed(42)
        idx = prompt.clone()
        with torch.no_grad():
            for _ in range(5):
                logits, _ = model(idx[:, -model.block_size:])
                nxt = sample_next(logits[:, -1, :], 0.9, 4)
                idx = torch.cat((idx, nxt), dim=1)

        self.assertTrue(torch.equal(generated, idx))

    def test_generate_crops_context_to_block_size(self):
        """A prompt longer than the window must not trip the forward assert."""
        torch.manual_seed(0)
        model = GPT(vocab_size=11, n_embd=8, n_head=2, n_layer=1,
                    block_size=4, dropout=0.0)
        model.eval()
        out = model.generate(torch.tensor([[1, 2, 3, 4, 5, 6]]), max_new_tokens=3)
        self.assertEqual(out.shape, (1, 9))


class ModelSpy:
    """A model wrapper that records the shape of every input it is handed.

    Sampling from a randomly initialised model is close to RNG-dominated: with
    the same seed, two different windows can produce the same characters by
    coincidence, so comparing *outputs* is a weak way to show that the crop
    happened. The input tensor is not subject to that — it either has the
    window's width or it does not.
    """

    def __init__(self, model):
        self.model = model
        self.block_size = model.block_size
        self.shapes = []

    def __call__(self, idx):
        self.shapes.append(tuple(idx.shape))
        return self.model(idx)


class TestContextWindow(unittest.TestCase):
    """chat.py: how much of the text the model is actually given.

    `block_size` is a ceiling, not a requirement — that is what the positional
    embeddings were trained for — so a caller can hand the model less than it
    can hold. Everything measured against the window (what was dropped, where
    the line falls) has to move with it, which is what these pin.
    """

    @classmethod
    def setUpClass(cls):
        # One token per character, so a window of n is exactly n characters.
        cls.tokenizer = CharTokenizer(text="hello there abcdefghijklmnopqrstuvwxyz")
        torch.manual_seed(0)
        cls.model = GPT(vocab_size=cls.tokenizer.vocab_size, n_embd=16, n_head=2,
                        n_layer=1, block_size=8, dropout=0.0)
        cls.model.eval()
        cls.ids = cls.tokenizer.encode("hello there")     # longer than the window

    def args(self, max_new_tokens=6):
        return SimpleNamespace(temperature=0.8, top_k=200, top_p=None,
                               repetition_penalty=1.0, penalty_window=64,
                               max_new_tokens=max_new_tokens, stop=None)

    def reply(self, context=None, max_new_tokens=6, seed=3, model=None):
        torch.manual_seed(seed)
        return chat.stream_reply(model or self.model, self.tokenizer, self.ids,
                                 self.args(max_new_tokens), io.StringIO(),
                                 context=context)[0]

    def test_the_default_is_what_the_checkpoint_was_trained_for(self):
        self.assertEqual(chat.context_window(self.model), 8)
        self.assertEqual(chat.context_window(self.model, None), 8)

    def test_a_longer_window_is_clamped_to_the_ceiling(self):
        """More tokens than the positional embeddings address is not something
        the model can honour, so it is not honoured."""
        self.assertEqual(chat.context_window(self.model, 4096), 8)

    def test_a_shorter_window_is_allowed(self):
        self.assertEqual(chat.context_window(self.model, 3), 3)

    def test_a_window_of_zero_is_refused(self):
        with self.assertRaises(ValueError):
            chat.context_window(self.model, 0)

    def test_the_model_is_handed_the_window_and_nothing_older(self):
        """The crop is the whole feature, and a randomly initialised model makes
        sampled text a poor witness for it, so it is measured where it cannot be
        masked: on the width of the tensors the model is actually handed."""
        for asked, expected in ((1, 1), (2, 2), (5, 5), (8, 8), (99, 8), (None, 8)):
            spy = ModelSpy(self.model)
            self.reply(context=asked, max_new_tokens=4, model=spy)
            self.assertEqual([shape[1] for shape in spy.shapes], [expected] * 4,
                             f"asked for {asked}, the model got {spy.shapes}")

    def test_the_reply_is_measured_from_the_cropped_prompt(self):
        """What this pins is the bookkeeping rather than the crop: the text
        returned excludes the part of the prompt the model never saw. Written out
        by hand, since `stream_reply` is the only place the two meet."""
        window, steps = 2, 6
        text = self.reply(context=window, max_new_tokens=steps)

        torch.manual_seed(3)
        idx = torch.tensor(self.ids[-window:], dtype=torch.long)[None, ...]
        base = self.tokenizer.decode(idx[0].tolist())
        with torch.no_grad():
            for _ in range(steps):
                logits, _ = self.model(idx[:, -window:])
                idx = torch.cat((idx, sample_next(logits[:, -1, :], 0.8, 200)),
                                dim=1)
        expected = self.tokenizer.decode(idx[0].tolist())[len(base):]
        self.assertEqual(text, expected)
        self.assertTrue(text)
        self.assertEqual(len(base), window)          # one token per character


class CheckpointFixture(unittest.TestCase):
    """Shared helper: write a checkpoint the way train.py would."""

    def write_checkpoint(self, path, legacy=False, bpe=None):
        tok = bpe if bpe is not None else CharTokenizer(text="abcdef")
        cfg = {"n_embd": 16, "n_head": 2, "n_layer": 1, "block_size": 8,
               "dropout": 0.0}
        torch.manual_seed(0)
        model = GPT(vocab_size=tok.vocab_size, **cfg)
        payload = {"model_state": model.state_dict(), "config": cfg, "step": 12}
        if legacy:
            # As written before the tokenizer became pluggable.
            payload["stoi"] = dict(tok.stoi)
            payload["itos"] = dict(tok.itos)
        else:
            payload["tokenizer"] = tok.state()
        torch.save(payload, path)
        return tok


class TestChatLoading(CheckpointFixture):
    """chat.py rebuilds a model from what the checkpoint actually carries."""

    def test_modern_char_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pt")
            tok = self.write_checkpoint(path)
            model, rebuilt, ckpt = chat.load_model(path)
            self.assertEqual(rebuilt.name, "char")
            self.assertEqual(rebuilt.vocab_size, tok.vocab_size)
            self.assertEqual(rebuilt.encode("abc"), tok.encode("abc"))
            self.assertEqual(model.block_size, 8)
            self.assertEqual(ckpt["step"], 12)

    def test_bpe_checkpoint_carries_its_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = BPETokenizer(train_bpe(read_corpus(20_000), 300))
            path = os.path.join(tmp, "model.pt")
            self.write_checkpoint(path, bpe=tok)
            _, rebuilt, _ = chat.load_model(path)
            self.assertEqual(rebuilt.name, "bpe")
            self.assertEqual(rebuilt.vocab_size, tok.vocab_size)
            self.assertEqual(rebuilt.encode("the king"), tok.encode("the king"))

    def test_legacy_checkpoint_without_a_tokenizer_block(self):
        """runs_v3/model.pt predates the tokenizer work and must still load."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pt")
            tok = self.write_checkpoint(path, legacy=True)
            _, rebuilt, _ = chat.load_model(path)
            self.assertEqual(rebuilt.name, "char")
            self.assertEqual(rebuilt.vocab_size, tok.vocab_size)

    def test_missing_checkpoint_exits_with_a_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx:
                chat.load_model(os.path.join(tmp, "nope.pt"))
            self.assertIn("no checkpoint at", str(ctx.exception))

    def test_checkpoint_without_any_tokenizer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pt")
            torch.save({"model_state": {}, "config": {}}, path)
            with self.assertRaises(SystemExit) as ctx:
                chat.load_model(path)
            self.assertIn("no tokenizer", str(ctx.exception))


class TestWebServer(CheckpointFixture):
    """serve.py: the browser's contract with the model.

    A real server on a real socket, because the point of the module is the
    streaming, and a mocked handler would prove nothing about it.
    """

    HOST = "127.0.0.1"

    @classmethod
    def setUpClass(cls):
        import serve

        cls.tmp = tempfile.TemporaryDirectory()
        cls.checkpoint = os.path.join(cls.tmp.name, "model.pt")
        fixture = CheckpointFixture("setUpClass")
        cls.tokenizer = fixture.write_checkpoint(
            cls.checkpoint, bpe=BPETokenizer(train_bpe(read_corpus(20_000), 300))
        )
        cls.server = serve.build_server(cls.checkpoint, cls.HOST, 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    # ------------------------------------------------------------- helpers
    def connect(self, timeout=60):
        return http.client.HTTPConnection(self.HOST, self.port, timeout=timeout)

    def stream(self, body, stop_after=None):
        conn = self.connect()
        conn.request("POST", "/api/generate", json.dumps(body),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        headers = dict(resp.getheaders())
        events = []
        while True:
            line = resp.readline()
            if not line:
                break
            events.append(json.loads(line.decode()))
            if stop_after and len(events) >= stop_after:
                break
        conn.close()
        return events, headers

    def post(self, path, payload=None, raw=None):
        conn = self.connect()
        conn.request("POST", path,
                     raw if raw is not None else json.dumps(payload),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        body = resp.read().decode()
        status = resp.status
        conn.close()
        return status, body

    def get(self, path):
        conn = self.connect()
        conn.request("GET", path)
        resp = conn.getresponse()
        body, headers = resp.read().decode(), dict(resp.getheaders())
        status = resp.status
        conn.close()
        return status, body, headers

    # ----------------------------------------------------------------- page
    def test_page_is_served_with_its_controls(self):
        status, body, headers = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        for control in ("temperature", "top_k", "top_p", "rep",
                        "max_new_tokens", "stop", "seed"):
            self.assertIn(f'id="{control}"', body, f"{control} missing from the page")

    def test_page_is_honest_about_what_the_model_is(self):
        _, body, _ = self.get("/")
        self.assertIn("not an assistant", body)

    def test_page_is_self_contained(self):
        """No build step: styles and script are in the document itself."""
        _, body, _ = self.get("/")
        self.assertIn("<style>", body)
        self.assertIn("<script>", body)
        self.assertNotIn("http://cdn", body)
        self.assertNotIn("https://cdn", body)

    def test_the_served_page_is_the_document_the_layout_checks_read(self):
        """TestPageLayout resolves the stylesheet of `serve.PAGE`; this is what
        makes that the page a browser is actually given."""
        _, body, _ = self.get("/")
        self.assertEqual(body, serve.PAGE)

    def test_info_describes_the_loaded_checkpoint(self):
        status, body, _ = self.get("/api/info")
        self.assertEqual(status, 200)
        info = json.loads(body)
        # Counts, not values: tests must not depend on the order they run in.
        self.assertIn("abandoned_replies", info)
        self.assertIn("last_abandoned_chunks", info)
        self.assertEqual(info["checkpoint"], self.checkpoint)
        self.assertEqual(info["tokenizer"], "bpe")
        self.assertEqual(info["vocab_size"], self.tokenizer.vocab_size)
        self.assertGreater(info["num_params"], 0)
        self.assertEqual(info["block_size"], 8)

    # ------------------------------------------------------------- generate
    def test_generate_streams_ndjson_chunks(self):
        events, headers = self.stream(
            {"prompt": "KING:\n", "max_new_tokens": 12, "seed": 1}
        )
        kinds = [e["t"] for e in events]
        chunks = [e["text"] for e in events if e["t"] == "chunk"]
        done = [e for e in events if e["t"] == "done"]

        self.assertEqual(headers["Content-Type"], "application/x-ndjson")
        self.assertEqual(headers["Transfer-Encoding"].lower(), "chunked")
        self.assertEqual(kinds[0], "meta")
        self.assertEqual(kinds[-1], "done")
        self.assertNotIn("error", kinds)
        self.assertEqual(len(done), 1)
        self.assertEqual("".join(chunks), done[0]["text"])
        self.assertEqual(done[0]["steps"], len(chunks))
        self.assertEqual(done[0]["block_size"], 8)
        self.assertGreater(done[0]["in_context"], done[0]["prompt_tokens"])

    def test_done_reports_what_the_window_dropped(self):
        """The model reads the last block_size tokens and nothing older, so a
        reply has to say how much was written but never seen — the page cannot
        be honest about the window without these counts."""
        # A prompt that fits: nothing is dropped on the prompt's account.
        events, _ = self.stream({"prompt": "a\n", "max_new_tokens": 4, "seed": 3})
        done = [e for e in events if e["t"] == "done"][0]
        self.assertLessEqual(done["prompt_tokens"], done["block_size"])
        self.assertEqual(done["dropped_prompt_tokens"], 0)
        self.assertEqual(done["dropped_tokens"],
                         max(0, done["in_context"] - done["block_size"]))

        # A prompt longer than the window: its opening tokens never reach it.
        events, _ = self.stream(
            {"prompt": "KING HENRY: We are in the forest now, and it is cold.\n",
             "max_new_tokens": 4, "seed": 3}
        )
        done = [e for e in events if e["t"] == "done"][0]
        self.assertGreater(done["prompt_tokens"], done["block_size"])
        self.assertEqual(done["dropped_prompt_tokens"],
                         done["prompt_tokens"] - done["block_size"])
        self.assertEqual(done["dropped_tokens"],
                         max(0, done["in_context"] - done["block_size"]))
        self.assertGreaterEqual(done["dropped_tokens"],
                                done["dropped_prompt_tokens"])

    def test_done_says_where_the_window_begins_not_only_how_much_it_dropped(self):
        """The counts say how much fell out of the window; this says where it
        opens, so the page can draw the edge of the model's memory in the place
        it actually falls rather than at an approximation of it."""
        prompt = "KING HENRY:\n" + read_corpus(600)
        events, _ = self.stream({"prompt": prompt, "max_new_tokens": 8,
                                 "seed": 11})
        done = [e for e in events if e["t"] == "done"][0]
        full = prompt + done["text"]
        ids = self.tokenizer.encode(full)
        block = done["block_size"]
        self.assertEqual(done["in_context"], len(ids))
        self.assertEqual(done["dropped_tokens"], max(0, len(ids) - block))

        offset = done["boundary_chars"]
        self.assertIsNotNone(offset)
        self.assertGreater(offset, 0)          # something is above the line
        self.assertLess(offset, len(full))
        # Pinned against a second computation of the same definition, so the
        # payload cannot drift from the arithmetic it claims to report.
        self.assertEqual(offset,
                         len(full) - len(self.tokenizer.decode(ids[-block:])))
        # And it is a real place in the text: below it is the window's text.
        self.assertEqual(full[offset:], self.tokenizer.decode(ids[-block:]))
        # (The offset is relative to the transcript as *characters*, which is what
        # the page has: the JSON round trip preserves the same code points, so
        # the line lands where the browser's own text puts it.)

    def test_done_reports_the_window_it_actually_used(self):
        """The control can hand the model less of the text than it can hold, so
        every count has to be measured against the window that was used rather
        than against what the model could have been given."""
        prompt = "KING HENRY:\n" + read_corpus(600)
        events, _ = self.stream({"prompt": prompt, "max_new_tokens": 4,
                                 "context": 3, "seed": 5})
        done = [e for e in events if e["t"] == "done"][0]
        self.assertEqual(done["context"], 3)
        self.assertEqual(done["block_size"], 8)      # the ceiling is unchanged
        full = prompt + done["text"]
        ids = self.tokenizer.encode(full)
        self.assertEqual(done["dropped_tokens"], max(0, len(ids) - 3))
        self.assertEqual(full[done["boundary_chars"]:],
                         self.tokenizer.decode(ids[-3:]))

        # With the model's own window the same text keeps far more of itself.
        events, _ = self.stream({"prompt": prompt, "max_new_tokens": 4,
                                 "seed": 5})
        wide = [e for e in events if e["t"] == "done"][0]
        self.assertEqual(wide["context"], wide["block_size"])
        self.assertLess(wide["dropped_tokens"], done["dropped_tokens"])
        # A wider window leaves less text above the line, so its boundary is
        # earlier in the transcript, not later.
        self.assertLess(wide["boundary_chars"], done["boundary_chars"])

    def test_a_window_larger_than_the_model_is_clamped_not_refused(self):
        events, _ = self.stream({"prompt": "a\n", "max_new_tokens": 2,
                                 "context": 4000, "seed": 6})
        done = [e for e in events if e["t"] == "done"][0]
        self.assertEqual(done["context"], done["block_size"])

    def test_a_window_of_zero_is_rejected(self):
        status, body = self.post("/api/generate", {"prompt": "a\n", "context": 0})
        self.assertEqual(status, 400)
        self.assertIn("context", body)
        status, _ = self.post("/api/generate", {"prompt": "a\n", "context": 2.5})
        self.assertEqual(status, 400)

    def test_the_window_endpoint_says_where_the_line_would_fall(self):
        """The control asks this before generating anything, so the line moves
        as the slider does rather than after a reply."""
        text = read_corpus(600)
        ids = self.tokenizer.encode(text)
        for asked, expected in ((8, 8), (3, 3), (4000, 8)):
            status, body = self.post("/api/window",
                                     {"text": text, "context": asked})
            self.assertEqual(status, 200)
            info = json.loads(body)
            self.assertEqual(info["context"], expected)
            self.assertEqual(info["block_size"], 8)
            self.assertEqual(info["tokens"], len(ids))
            self.assertEqual(info["dropped_tokens"], max(0, len(ids) - expected))
            self.assertEqual(text[info["boundary_chars"]:],
                             self.tokenizer.decode(ids[-expected:]))
        # No context given means the model's own window.
        _, body = self.post("/api/window", {"text": text})
        self.assertEqual(json.loads(body)["context"], 8)

    def test_the_window_endpoint_agrees_with_what_a_reply_reports(self):
        """The preview drawn while dragging has to be the line the next reply
        would draw, or the page contradicts itself."""
        prompt = "KING HENRY:\n" + read_corpus(600)
        events, _ = self.stream({"prompt": prompt, "max_new_tokens": 4,
                                 "context": 3, "seed": 5})
        done = [e for e in events if e["t"] == "done"][0]
        _, body = self.post("/api/window",
                            {"text": prompt + done["text"], "context": 3})
        info = json.loads(body)
        self.assertEqual(info["boundary_chars"], done["boundary_chars"])
        self.assertEqual(info["dropped_tokens"], done["dropped_tokens"])
        self.assertEqual(info["tokens"], done["in_context"])
        self.assertEqual(info["context"], done["context"])

    def test_the_window_endpoint_rejects_bad_input(self):
        for payload, why in (({"text": 42}, "text must be a string"),
                             ({"text": "a", "context": 0}, "zero window"),
                             ({"text": "a", "context": "big"}, "not a number"),
                             ({"text": "a", "context": 2.5}, "not whole")):
            status, body = self.post("/api/window", payload)
            self.assertEqual(status, 400, why)
            self.assertIn("error", body)

    def test_page_offers_the_window_as_a_control(self):
        """The control has to be there, be sent with the request, and be what
        the readouts measure against."""
        _, body, _ = self.get("/")
        self.assertIn('id="context"', body)
        self.assertIn('id="context-n"', body)
        self.assertIn("context: windowSize()", body)
        self.assertIn('fetch("/api/window"', body)
        self.assertIn("inWindow", body)

    def test_a_window_that_drops_nothing_has_no_boundary_to_draw(self):
        events, _ = self.stream({"prompt": "a\n", "max_new_tokens": 2,
                                 "seed": 12})
        done = [e for e in events if e["t"] == "done"][0]
        self.assertEqual(done["dropped_tokens"], 0)
        self.assertEqual(done["boundary_chars"], 0)

    def test_page_accounts_for_the_context_it_drops(self):
        """A window that quietly forgets the beginning is the failure mode; the
        page has somewhere to say so and marks what fell out."""
        _, body, _ = self.get("/")
        self.assertIn('id="window"', body)
        self.assertIn("dropped_tokens", body)
        self.assertIn("dropped_prompt_tokens", body)
        self.assertIn("out of view", body)
        self.assertIn('classList.add("dropped")', body)

    def test_same_seed_reproduces_the_reply(self):
        body = {"prompt": "KING:\n", "max_new_tokens": 8, "seed": 5}
        first, _ = self.stream(body)
        second, _ = self.stream(body)
        self.assertEqual([e["text"] for e in first if e["t"] == "chunk"],
                         [e["text"] for e in second if e["t"] == "chunk"])

    def test_different_seed_gives_a_different_reply(self):
        text = []
        for seed in (6, 7):
            events, _ = self.stream(
                {"prompt": "KING:\n", "max_new_tokens": 8, "seed": seed}
            )
            text.append("".join(e["text"] for e in events if e["t"] == "chunk"))
        self.assertNotEqual(text[0], text[1])

    def test_stop_string_never_reaches_the_stream(self):
        events, _ = self.stream(
            {"prompt": "KING:\n", "max_new_tokens": 40, "seed": 2, "stop": "\n"}
        )
        chunks = "".join(e["text"] for e in events if e["t"] == "chunk")
        done = [e for e in events if e["t"] == "done"][0]
        self.assertNotIn("\n", chunks)
        self.assertEqual(done["text"], chunks)

    def test_documented_controls_are_accepted(self):
        cases = [
            {"top_k": 0},                    # documented as “disables it”
            {"top_k": 1},
            {"top_p": 0.5},
            {"top_p": 0},                    # 0 means off, not an error
            {"temperature": 0},
            {"repetition_penalty": 1.3},
            {"penalty_window": 4},
        ]
        for extra in cases:
            with self.subTest(**extra):
                body = {"prompt": "KING:\n", "max_new_tokens": 6, "seed": 1}
                body.update(extra)
                events, _ = self.stream(body)
                self.assertEqual(events[-1]["t"], "done", events[-1])

    # ----------------------------------------------------------- validation
    def test_bad_requests_are_rejected_with_a_reason(self):
        cases = [
            {"prompt": ""},
            {},
            {"prompt": 5},
            {"prompt": "a", "top_k": 2.5},
            {"prompt": "a", "temperature": 9},
            {"prompt": "a", "max_new_tokens": 0},
            {"prompt": "a", "max_new_tokens": 99999},
            {"prompt": "a", "top_p": 1.5},
            {"prompt": "a", "seed": "abc"},
            {"prompt": "a", "seed": -3},
            [1, 2, 3],
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                status, body = self.post("/api/generate", payload)
                self.assertEqual(status, 400, body[:80])
                self.assertIn("error", json.loads(body))

    def test_malformed_json_is_rejected(self):
        status, body = self.post("/api/generate", None, raw=b"{not json")
        self.assertEqual(status, 400)
        self.assertIn("not valid JSON", body)

    def test_unknown_routes_are_404(self):
        status, _, _ = self.get("/nope")
        self.assertEqual(status, 404)
        status, _ = self.post("/nope", {})
        self.assertEqual(status, 404)

    # --------------------------------------------------------------- tokens
    def test_tokens_endpoint_matches_the_tokenizer(self):
        status, body = self.post("/api/tokens", {"text": "the king"})
        self.assertEqual(status, 200)
        info = json.loads(body)
        self.assertEqual(info["count"], len(info["tokens"]))
        self.assertEqual(info["chars"], len("the king"))
        self.assertEqual("".join(t["text"] for t in info["tokens"]), "the king")
        self.assertEqual([t["id"] for t in info["tokens"]],
                         self.tokenizer.encode("the king"))

    def test_non_string_tokens_input_is_rejected(self):
        status, body = self.post("/api/tokens", {"text": 42})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    # ------------------------------------------------------- stopping work
    def test_server_survives_a_client_walking_away(self):
        """Abandoning a reply must not take the server down with it.

        Whether the *sampling* stops promptly is not decidable at this level: on
        loopback with a small model the kernel can buffer a whole reply, so the
        write never fails and there is nothing to notice. That path is tested
        directly in TestAbandonedStream; this one is about survival.
        """
        conn = self.connect()
        conn.request("POST", "/api/generate",
                     json.dumps({"prompt": "a\n", "max_new_tokens": 40,
                                 "seed": 4}),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.readline()
        resp.readline()                      # walk away mid-reply
        conn.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                             struct.pack("ii", 1, 0))
        conn.sock.close()

        events, _ = self.stream({"prompt": "b\n", "max_new_tokens": 4, "seed": 5})
        self.assertEqual(events[-1]["t"], "done")

    def test_info_survives_a_page_reload(self):
        first = json.loads(self.get("/api/info")[1])
        second = json.loads(self.get("/api/info")[1])
        self.assertEqual(first, second)
        self.assertEqual(first["checkpoint"], self.checkpoint)

    def test_concurrent_requests_all_complete(self):
        """The lock serialises generation without deadlocking or dropping one."""
        results = []

        def worker(seed):
            events, _ = self.stream(
                {"prompt": f"line {seed}\n", "max_new_tokens": 4, "seed": seed}
            )
            results.append(events[-1]["t"])

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, ["done"] * 3)

    # ------------------------------------------------------------ validate()
    def test_validate_applies_defaults(self):
        import serve

        params = serve.validate({"prompt": "hi"})
        self.assertEqual(params["temperature"], 0.8)
        self.assertEqual(params["top_k"], 200)
        self.assertIsNone(params["top_p"])       # nucleus off by default
        self.assertEqual(params["repetition_penalty"], 1.0)
        self.assertEqual(params["max_new_tokens"], 160)
        self.assertIsNone(params["context"])     # unset means the model's own window
        self.assertIsNone(params["stop"])
        self.assertIsNone(params["seed"])        # no seed means a fresh draw

    def test_validate_treats_zero_nucleus_and_empty_stop_as_off(self):
        import serve

        params = serve.validate({"prompt": "hi", "top_p": 0, "stop": ""})
        self.assertIsNone(params["top_p"])
        self.assertIsNone(params["stop"])


class TestWindowBoundary(unittest.TestCase):
    """serve.py: where the model's window opens, as a place in the text.

    The page draws the edge of the model's memory into the transcript, so this
    offset has to be a real character position rather than a close one. Two
    things make that delicate: it is measured from the visible tail, because
    slicing from the other end counts differently when a token cuts a character
    in half — and a byte-level token can, since a token is a run of bytes.
    """

    BLOCK = 40

    @classmethod
    def setUpClass(cls):
        text = read_corpus(20_000)
        cls.text = text[:400]
        cls.tokenizer = BPETokenizer(train_bpe(text, 300))

    def boundary(self, text, block, tokenizer=None):
        tok = tokenizer or self.tokenizer
        return serve.window_boundary_chars(tok, text, tok.encode(text), block)

    def test_a_window_that_holds_everything_drops_nothing(self):
        self.assertEqual(self.boundary(self.text[:20], self.BLOCK), 0)

    def test_the_offset_is_measured_in_characters_not_bytes(self):
        """The page places the line by character, so a multi-byte corpus must
        not be counted in the bytes under it."""
        text = "héllo wörld, héllo"
        offset = self.boundary(text, 4)
        self.assertEqual(text[offset:],
                         self.tokenizer.decode(self.tokenizer.encode(text)[-4:]))
        self.assertLess(offset, len(text.encode("utf-8")))

    def test_the_offset_is_where_the_window_actually_opens(self):
        ids = self.tokenizer.encode(self.text)
        offset = self.boundary(self.text, self.BLOCK)
        self.assertEqual(self.text[offset:],
                         self.tokenizer.decode(ids[-self.BLOCK:]))

    def test_two_ways_of_slicing_the_same_text_agree(self):
        """Sliced from the visible side and from the dropped side, the boundary
        is the same place.

        Only for text that decodes cleanly, which is the point of keeping this
        separate from the test above: a byte-level tokenisation can hold a run of
        bytes that is not valid UTF-8 — an untrained checkpoint emits them — and
        decoding such a run one token at a time splits it into more replacement
        characters than decoding it whole. That is a property of the display,
        not of the boundary, so the boundary is measured from the visible side.
        """
        text = self.text.encode("ascii", "ignore").decode()[:200]
        ids = self.tokenizer.encode(text)
        block = 30
        offset = serve.window_boundary_chars(self.tokenizer, text, ids, block)
        pieces = self.tokenizer.show(text)
        self.assertEqual(len(pieces), len(ids))
        self.assertEqual(sum(len(p) for _, p in pieces[:len(ids) - block]),
                         offset)

    def test_a_character_tokenizer_needs_no_special_case(self):
        """One character per token: the arithmetic has to work for the other
        tokenizer too, not just the one with bytes under it."""
        char = CharTokenizer(text=self.text)
        offset = self.boundary(self.text, self.BLOCK, tokenizer=char)
        self.assertEqual(offset, len(self.text) - self.BLOCK)
        self.assertEqual(self.text[offset:], self.text[-self.BLOCK:])

    def test_a_boundary_inside_a_character_goes_before_it(self):
        """With one token per byte, a window can open in the middle of a
        character. The line then goes before that character: the model read
        part of it, and nothing above the line is claimed as readable."""
        byte_only = BPETokenizer([])          # no merges: every token is a byte
        text = "é" * 3
        ids = byte_only.encode(text)
        self.assertEqual(len(ids), 6)
        self.assertEqual(byte_only.decode(ids[-1:]), "\ufffd")
        offset = serve.window_boundary_chars(byte_only, text, ids, 1)
        # The window holds one byte: the second half of the last character, so
        # the line goes before it. Two whole é above the line, and the third one
        # split — which is the honest thing to show, since the model has read
        # half of it and the page cannot draw half a character.
        self.assertEqual(offset, 2)
        self.assertEqual(text[offset:], "é")

    def test_an_offset_that_cannot_be_placed_is_refused(self):
        """The fallback is None and not a guess: the page then draws nothing,
        since a line in the wrong place is worse than no line at all."""
        class Liar:
            def decode(self, ids):
                return "not the tail of this text"

        self.assertIsNone(
            serve.window_boundary_chars(Liar(), "some text", [1, 2, 3, 4], 2)
        )


class TestPageLayout(unittest.TestCase):
    """serve.py: the page's geometry, resolved without a browser.

    The token view is the answer to "what does the model actually read", and it
    was once invisible: as a shrinkable flex item it gave its height up to the
    transcript pane above it and collapsed to 12px, with all 350 chips in the
    DOM and 12px of them on screen. Nothing in the suite noticed, because a
    passing HTTP test says nothing about a box.

    So this reads the stylesheet the server sends and resolves it: who shrinks,
    who caps, who yields. It is deliberately a contract rather than a
    measurement — a browser would measure, and the suite has no browser — and
    the one test that matters most is the one that removes the current
    declaration and shows the check would have caught the old layout.
    """

    WIDE = 1200      # two columns
    NARROW = 800     # stacked, where the page scrolls instead of squeezing

    @classmethod
    def setUpClass(cls):
        cls.html = serve.PAGE
        cls.rules = stylesheet_rules(cls.html)
        cls.paths = element_paths(cls.html)

    def declarations(self, selector, width):
        return declarations_for(self.rules, selector, width)

    def props(self, selector, width):
        return dict(self.declarations(selector, width))

    # ------------------------------------------------------- the parse itself
    def test_the_stylesheet_was_actually_parsed(self):
        """Guard the guard: a resolver that returned nothing would make the
        tests below fail loudly, but a resolver that dropped one rule would not."""
        selectors = {sel for _, sel, _ in self.rules}
        for needed in ("body", "main", ".doc-scroll", ".tokens", "aside"):
            self.assertIn(needed, selectors)
        self.assertGreater(len(self.rules), 20)

    def test_every_media_query_is_understood(self):
        """A condition the resolver cannot evaluate would hide rules from these
        checks rather than fail them, so an unknown one has to be an error."""
        conditions = {media for media, _, _ in self.rules if media}
        self.assertTrue(conditions, "no media queries were parsed at all")
        for condition in conditions:
            for width in (self.WIDE, self.NARROW):
                media_matches(condition, width)

    def test_the_rules_this_file_asserts_on_really_switch_at_the_breakpoint(self):
        """Otherwise the two viewports below would be the same viewport, and the
        stacked-layout test would be asserting nothing about a stacked layout."""
        self.assertNotEqual(self.props("main", self.WIDE),
                            self.props("main", self.NARROW))

    # ------------------------------------------------- the token view's height
    def test_the_token_view_cannot_be_shrunk(self):
        """`flex-shrink: 0` is the whole defence, and it is not the initial
        value: it is what turns "squeezed by a taller sibling" into "keeps its
        height, and the sibling scrolls instead"."""
        grow, shrink, _ = flex_factors(self.declarations(".tokens", self.WIDE))
        self.assertEqual(shrink, 0.0, "the token view can be squeezed again")
        self.assertEqual(grow, 0.0, "the token view should not grow either")

    def test_the_check_catches_the_layout_that_squeezed_it(self):
        """The check above is only worth having if it would have failed before
        the fix. The old rule carried no `flex` declaration at all, so it
        resolved to the initial `0 1 auto` — shrink 1 — which is the bug."""
        without_flex = [
            (media, sel, [d for d in decls if d[0] != "flex"])
            for media, sel, decls in self.rules
        ]
        _, shrink, _ = flex_factors(declarations_for(without_flex, ".tokens", self.WIDE))
        self.assertEqual(shrink, 1.0)
        self.assertNotEqual(shrink, 0.0, "this check would pass a shrinkable panel")

    def test_the_pages_own_line_cannot_become_part_of_the_transcript(self):
        """The window's edge carries a label, and the document's text is the
        transcript the next prompt is built from and that Copy copies — so the
        label has to come from a pseudo-element, never from a text node."""
        self.assertEqual(self.props(".cut::after", self.WIDE).get("content"),
                         "attr(data-label)")

    def test_the_window_edge_takes_a_line_of_its_own(self):
        """Inline would put it in the middle of a sentence, which is not a
        boundary a reader can see."""
        self.assertEqual(self.props(".cut", self.WIDE).get("display"), "block")

    def test_text_above_the_window_is_dimmed(self):
        self.assertLess(float(self.props(".unread", self.WIDE)["opacity"]), 1.0)

    def test_the_page_consumes_the_boundary_the_server_reports(self):
        """The offset is worth nothing if nothing reads it."""
        self.assertIn("boundary_chars", self.html)
        self.assertIn('marker.className = "cut"', self.html)

    def test_the_token_view_keeps_room_for_chips(self):
        """Unshrinkable is not the same as usable: a cap of its own could still
        starve the panel, so the cap has to leave room for a few rows and the
        rest has to be reachable by scrolling."""
        props = self.props(".tokens", self.WIDE)
        self.assertGreaterEqual(length_px(props["max-height"], 600), 120.0)
        self.assertEqual(props.get("overflow-y"), "auto")

    # ------------------------------------------------------ who yields instead
    def test_the_token_view_shares_a_flex_column_with_the_pane_above_it(self):
        """A flex factor only means something between siblings. If the token
        view moved out of `section.doc`, the tests above would keep passing
        while the panel went back to living inside a different box."""
        section = self.paths["tokens"][-2]
        self.assertEqual(section[0], "section")
        self.assertIn("doc", section[2])
        self.assertEqual(self.paths["doc"][-2], section,
                         "the transcript pane is no longer a sibling")
        props = self.props("section.doc", self.WIDE)
        self.assertEqual(props.get("display"), "flex")
        self.assertEqual(props.get("flex-direction"), "column")

    def test_the_pane_above_is_the_one_that_gives_up_space(self):
        """Someone has to give: with the token view fixed, the transcript pane
        must be the item that shrinks, and it must scroll its own content rather
        than pushing the panel off the bottom of the column."""
        declarations = self.declarations(".doc-scroll", self.WIDE)
        grow, shrink, _ = flex_factors(declarations)
        self.assertGreaterEqual(grow, 1.0)
        self.assertEqual(shrink, 1.0, "nothing would absorb the shortfall")
        props = dict(declarations)
        self.assertEqual(props.get("overflow-y"), "auto")
        self.assertGreaterEqual(length_px(props["min-height"], 600), 100.0)

    # -------------------------------------------------------- stacked layout
    def test_stacked_layout_grows_instead_of_squeezing(self):
        """Below the breakpoint there is no second column to move work into, so
        the page itself has to scroll: rows sized to their content, a fixed
        share of the viewport for the transcript, and the panel still unable to
        be shrunk by either."""
        main = self.props("main", self.NARROW)
        self.assertEqual(main.get("grid-template-columns"), "1fr")
        self.assertEqual(main.get("align-content"), "start")
        self.assertEqual(self.props("body", self.NARROW).get("overflow"), "auto")
        _, pane_shrink, pane_basis = flex_factors(
            self.declarations(".doc-scroll", self.NARROW))
        self.assertEqual(pane_shrink, 0.0, "the transcript pane must keep its share")
        self.assertIn("vh", pane_basis)
        _, panel_shrink, _ = flex_factors(self.declarations(".tokens", self.NARROW))
        self.assertEqual(panel_shrink, 0.0)


class TestAbandonedStream(unittest.TestCase):
    """The cancellation path, driven directly.

    A socket is the wrong instrument for this: the kernel buffers, so on a fast
    model a whole reply can be written without a single error and the code under
    test is never reached. A stand-in writer that fails on demand reaches it.
    """

    class Wfile:
        """Stands in for the socket: records writes, then fails like a reset."""

        def __init__(self, fail_after=None):
            self.fail_after = fail_after
            self.writes = []

        def write(self, data):
            if self.fail_after is not None and len(self.writes) >= self.fail_after:
                raise BrokenPipeError("the client went away")
            self.writes.append(data)

        def flush(self):
            pass

    class FakeSession:
        checkpoint = "fake/model.pt"

        def __init__(self):
            self.tokenizer = SimpleNamespace(name="fake", vocab_size=7)
            self.abandoned = []

        def note_abandoned(self, tokens):
            self.abandoned.append(tokens)

        def generate(self, params, emit):
            for i in range(10):
                emit(f"word{i} ")
            return {"text": "word0 ", "steps": 10, "seconds": 0.1,
                    "prompt_tokens": 1, "in_context": 11, "block_size": 8}

    def make_handler(self, wfile):
        import serve

        handler = serve.Handler.__new__(serve.Handler)   # no socket, no parsing
        handler.wfile = wfile
        handler.close_connection = False
        handler.start_stream = lambda: None
        return handler

    def test_events_are_written_as_chunked_frames(self):
        """http.server does not chunk for you, so the framing is verified here."""
        wfile = self.Wfile()
        handler = self.make_handler(wfile)

        event = {"t": "chunk", "text": "hi"}
        handler.stream_event(event)

        payload = (json.dumps(event) + "\n").encode("utf-8")
        expected = b"%x\r\n" % len(payload) + payload + b"\r\n"
        self.assertEqual(wfile.writes, [expected])
        self.assertFalse(handler.close_connection)

    def test_stream_ends_with_a_terminating_chunk(self):
        wfile = self.Wfile()
        handler = self.make_handler(wfile)
        handler.end_stream()
        self.assertEqual(wfile.writes, [b"0\r\n\r\n"])

    def test_a_failed_write_becomes_client_gone(self):
        """A dead socket must surface as ClientGone, not as a socket error."""
        wfile = self.Wfile(fail_after=2)
        handler = self.make_handler(wfile)
        handler.stream_event({"t": "chunk", "text": "one"})
        handler.stream_event({"t": "chunk", "text": "two"})
        with self.assertRaises(serve.ClientGone):
            handler.stream_event({"t": "chunk", "text": "three"})
        self.assertTrue(handler.close_connection,
                        "the connection should be dropped")

    def test_abandoned_reply_is_counted_and_not_terminated(self):
        """The contract: stop emitting, note how far it got, send no terminator.

        The count includes the chunk whose write failed — it was attempted, and
        whether the client's kernel kept it is not knowable from here.
        """
        wfile = self.Wfile(fail_after=3)      # meta + two chunks, then the pipe dies
        handler = self.make_handler(wfile)
        session = self.FakeSession()
        # `session` is a read-only property on the handler, so supply the server.
        handler.server = SimpleNamespace(session=session)

        handler.stream_generate({"prompt": "a"})     # returns quietly

        self.assertEqual(session.abandoned, [3], "should record how far it got")
        self.assertNotIn(b"0\r\n\r\n", wfile.writes,
                         "a half-finished stream must not be terminated")
        self.assertEqual(len(wfile.writes), 3)

    def test_a_complete_reply_is_terminated_and_not_counted(self):
        wfile = self.Wfile()
        handler = self.make_handler(wfile)
        session = self.FakeSession()
        handler.server = SimpleNamespace(session=session)

        handler.stream_generate({"prompt": "a"})

        self.assertEqual(session.abandoned, [])
        self.assertEqual(wfile.writes[-1], b"0\r\n\r\n")

        def payload(frame):
            _, _, body = frame.partition(b"\r\n")
            return json.loads(body[:-2].decode("utf-8"))

        kinds = [payload(w)["t"] for w in wfile.writes[:-1]]
        self.assertEqual(kinds[0], "meta")
        self.assertEqual(kinds[-1], "done")
        self.assertEqual(kinds.count("chunk"), 10)


class TestBigramFloor(unittest.TestCase):
    """The floor has to be measured the same way the model is."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with open(os.path.join(cls.tmp.name, "corpus.txt"), "w",
                  encoding="utf-8") as f:
            f.write(read_corpus(40_000))
        cls.data = os.path.join(cls.tmp.name, "corpus.txt")
        cls.tok = BPETokenizer(train_bpe(read_corpus(40_000), 384))
        cls.tok_path = os.path.join(cls.tmp.name, "tokenizer.json")
        cls.tok.save(cls.tok_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_bigram(self, out_name, extra):
        out = os.path.join(self.tmp.name, out_name)
        proc = subprocess.run(
            [sys.executable, "bigram.py", "--data", self.data, "--out", out]
            + extra,
            cwd=HERE, capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with open(out, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_character_floor_is_per_character(self):
        m = self.run_bigram("char.json", [])
        self.assertEqual(m["tokenizer"], "char")
        self.assertEqual(m["chars_per_token"], 1.0)
        self.assertIsNone(m["val_bits_per_char"])
        self.assertAlmostEqual(m["uniform_loss"], math.log(m["vocab_size"]), places=4)
        self.assertAlmostEqual(m["bigram_val_perplexity"],
                               math.exp(m["val_loss"]), places=2)
        self.assertLess(m["val_loss"], m["uniform_loss"])

    def test_bpe_floor_uses_the_validation_split(self):
        """The bug this guards: chars/token measured on the training split.

        It has to be the validation split, because that is where the model
        measures it — otherwise the two floors are not on the same footing.
        """
        m = self.run_bigram("bpe.json", ["--tokenizer", "bpe",
                                         "--bpe-tokenizer", self.tok_path])
        self.assertEqual(m["tokenizer"], "bpe")
        self.assertEqual(m["vocab_size"], self.tok.vocab_size)

        ids = self.tok.encode(read_corpus(40_000))
        n = int(0.9 * len(ids))
        expected = len(self.tok.decode(ids[n:])) / len(ids[n:])
        self.assertAlmostEqual(m["chars_per_token"], round(expected, 4), places=4)

        # And the bits/char quote is that loss on that axis, nothing else.
        # The file stores loss and chars/token rounded to 4 places, so the
        # recomputed ratio can sit one rounding unit away from the stored one.
        self.assertAlmostEqual(
            m["val_bits_per_char"],
            round((m["val_loss"] / math.log(2)) / m["chars_per_token"], 4),
            delta=2e-4,
        )


class TestResumeSeam(unittest.TestCase):
    """A resumed run continues the curve instead of restarting or doubling it.

    Training is slow on this CPU, so the two runs under test — a fresh one and
    the same model resumed — are trained once for the whole class and shared.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = os.path.join(cls.tmp.name, "corpus.txt")
        with open(cls.data, "w", encoding="utf-8") as f:
            f.write(read_corpus(20_000))
        cls.run1 = os.path.join(cls.tmp.name, "first")
        cls.run2 = os.path.join(cls.tmp.name, "second")
        cls.first = cls.train(cls.run1, [])
        cls.second = cls.train(
            cls.run2, ["--resume", os.path.join(cls.run1, "model.pt")]
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def train(cls, out_dir, extra):
        cmd = [
            sys.executable, "train.py",
            "--data", cls.data, "--out-dir", out_dir,
            "--max-iters", "2", "--eval-interval", "1", "--eval-iters", "2",
            "--log-interval", "1", "--checkpoint-interval", "1",
            "--sample-tokens", "16", "--max-seconds", "300",
            "--warmup-iters", "2",
            "--n-layer", "1", "--n-embd", "32", "--n-head", "2",
            "--block-size", "8", "--batch-size", "4",
            "--dropout", "0.0", "--seed", "0",
        ] + extra
        proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        with open(os.path.join(out_dir, "metrics.json"), "r",
                  encoding="utf-8") as f:
            return json.load(f)

    def test_fresh_run_records_every_evaluated_step(self):
        steps = [h["step"] for h in self.first["history"]]
        self.assertEqual(steps, [0, 1, 2])

    def test_resume_does_not_double_count_the_seam_step(self):
        steps = [h["step"] for h in self.second["history"]]

        # The defect: the step it resumed from appeared twice at one x.
        self.assertEqual(len(steps), len(set(steps)), f"duplicate steps: {steps}")
        self.assertEqual(steps, sorted(steps))
        self.assertEqual(steps, [0, 1, 2, 3, 4])

        # History carried over, not restarted from zero: everything before the
        # seam survives the round-trip through the checkpoint untouched.
        self.assertEqual(self.second["prior_steps"], 2)
        self.assertEqual(self.second["history"][:2], self.first["history"][:2])
        self.assertEqual(self.second["final"]["steps_completed"], 4)

        # The seam step itself is re-measured on resume, so its losses may
        # differ from the first run's; what must hold is that it is one point
        # at one x, on the same schedule position.
        seam, original = self.second["history"][2], self.first["history"][2]
        self.assertEqual(seam["step"], original["step"])
        self.assertEqual(seam["lr"], original["lr"])
        self.assertGreater(seam["val_loss"], 0.0)

    def test_learning_rate_continues_rather_than_restarting(self):
        """A resumed run is already warm; it must not jump back to the peak."""
        peak = max(h["lr"] for h in self.first["history"])
        self.assertLessEqual(self.second["history"][0]["lr"], peak)

    def test_metrics_are_self_consistent(self):
        text = read_corpus(20_000)
        m = self.first
        self.assertEqual(m["corpus_tokens"], len(CharTokenizer(text=text).encode(text)))
        self.assertEqual(m["final"]["chars_per_token"], 1.0)
        self.assertAlmostEqual(
            m["final"]["val_bits_per_char"],
            round((m["final"]["val_loss"] / math.log(2)), 4), places=4,
        )
        self.assertEqual(m["final"]["steps_completed"], 2)
        with open(os.path.join(self.run1, "loss_curve.csv"), "r",
                  encoding="utf-8") as f:
            rows = f.read().strip().splitlines()
        self.assertEqual(rows[0], "step,train_loss,val_loss")
        self.assertEqual(len(rows) - 1, len(m["history"]))


if __name__ == "__main__":
    if not os.path.exists(os.path.join(HERE, CORPUS)):
        sys.exit(f"missing corpus at {CORPUS}; the tests need it")
    unittest.main(verbosity=2)
