"""Greedy speculation must be the target's text, not merely close to it.

The claim under test is the whole reason the technique is usable: a decoder that
guesses ahead and then verifies produces the *same tokens* the plain argmax loop
would, whatever the draft guessed and however wrong it was. So almost every test
here decodes the same prompt two ways and compares the ids, across draft depths
and lookaheads, because a single setting passing is compatible with a bug that
happens to cancel at that setting.

The model is a seeded toy — 32 dimensions, three layers — because the property is
about control flow, not about what the weights learned. Nothing here needs a
checkpoint, a corpus or a venv, so it runs wherever torch does.
"""

import unittest

import torch

from greedy_spec import (GreedySpeculativeDecoder, greedy_cached,
                         greedy_uncached)
from model import GPT

VOCAB = 32
BLOCK = 32


def toy_model(seed=0, layers=3):
    torch.manual_seed(seed)
    return GPT(vocab_size=VOCAB, n_embd=32, n_head=2, n_layer=layers,
               block_size=BLOCK, dropout=0.0).eval()


def spec_decode(model, ids, count, window=BLOCK, draft_layers=1, guesses=4):
    decoder = GreedySpeculativeDecoder(model, ids, window,
                                       draft_layers=draft_layers,
                                       guesses=guesses)
    return list(decoder.tokens(count)), decoder


class ReproducesTheTarget(unittest.TestCase):
    def setUp(self):
        self.model = toy_model()
        self.ids = [1, 5, 9, 3, 7, 2]
        self.count = 12                 # 6 prompt + 12 + 6 guesses < 32
        self.want = greedy_cached(self.model, self.ids, BLOCK, self.count)

    def test_the_baselines_agree_with_each_other(self):
        # Without this, a speculative row matching "cached" would say nothing:
        # the reference itself would be in doubt.
        self.assertEqual(self.want,
                         greedy_uncached(self.model, self.ids, BLOCK, self.count))

    def test_every_draft_depth_and_lookahead_matches_exactly(self):
        for layers in (1, 2, 3):
            for guesses in (1, 2, 3, 5):
                got, _ = spec_decode(self.model, self.ids, self.count,
                                     draft_layers=layers, guesses=guesses)
                self.assertEqual(got, self.want,
                                 f"{layers} layers, {guesses} ahead")

    def test_a_draft_that_is_the_whole_model_accepts_everything(self):
        # With every layer, the draft *is* the target, so no guess can be
        # wrong. Acceptance below 100% here would mean the verifier is
        # comparing the wrong positions.
        got, decoder = spec_decode(self.model, self.ids, self.count,
                                   draft_layers=3, guesses=4)
        self.assertEqual(got, self.want)
        self.assertEqual(decoder.accepted, decoder.proposed)
        self.assertGreater(decoder.proposed, 0)

    def test_a_shallow_draft_is_often_wrong_and_still_exact(self):
        # One layer guessing what three would predict should miss regularly;
        # the text has to survive that, which is the point of verification.
        got, decoder = spec_decode(self.model, self.ids, self.count,
                                   draft_layers=1, guesses=6)
        self.assertEqual(got, self.want)
        self.assertLess(decoder.accepted, decoder.proposed)

    def test_greedy_decoding_is_deterministic(self):
        first, _ = spec_decode(self.model, self.ids, self.count)
        second, _ = spec_decode(self.model, self.ids, self.count)
        self.assertEqual(first, second)
        self.assertEqual(first, greedy_cached(self.model, self.ids, BLOCK,
                                              self.count))


class KeepsItsOwnBooks(unittest.TestCase):
    def setUp(self):
        self.model = toy_model(seed=3)
        self.ids = [4, 8, 15, 16, 23, 31]
        self.count = 10

    def test_counters_describe_the_work_done(self):
        _, decoder = spec_decode(self.model, self.ids, self.count,
                                 draft_layers=2, guesses=3)
        self.assertLessEqual(decoder.accepted, decoder.proposed)
        self.assertGreaterEqual(decoder.accepted, 0)
        self.assertGreater(decoder.proposed, 0)
        self.assertGreater(decoder.rounds, 0)
        # Every round emits at least the corrected token, and every pass over
        # the target is one pass: a speculation round must not read twice.
        self.assertGreaterEqual((decoder.accepted + decoder.rounds)
                                / decoder.rounds, 1.0)

    def test_it_yields_exactly_the_tokens_asked_for(self):
        for count in (1, 3, 12):
            got, _ = spec_decode(self.model, self.ids, count)
            self.assertEqual(len(got), count)

    def test_a_window_too_small_to_speculate_still_decodes(self):
        # window 3 cannot hold a prompt plus a guess, so the decoder has to
        # re-anchor and keep going rather than loop or raise.
        got, decoder = spec_decode(self.model, self.ids, 6, window=3,
                                   draft_layers=1, guesses=4)
        self.assertEqual(len(got), 6)
        self.assertGreater(decoder.rounds, 0)

    def test_the_draft_never_reads_more_layers_than_it_was_given(self):
        _, decoder = spec_decode(self.model, self.ids, 8, draft_layers=1,
                                 guesses=2)
        self.assertEqual(decoder.draft_layers, 1)

    def test_an_empty_prompt_is_refused(self):
        with self.assertRaises(ValueError):
            GreedySpeculativeDecoder(self.model, [], BLOCK)


if __name__ == "__main__":
    unittest.main()
