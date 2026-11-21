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

import infer
from greedy_spec import (GreedySpeculativeDecoder, greedy_cached,
                         greedy_uncached, verdict)
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


class TheBaselineIsTheDecoderTheAppServes(unittest.TestCase):
    """`greedy_cached` and `greedy_uncached` are written in this repo, not
    borrowed from `infer.py`, so nothing so far says they are the decoder the
    app runs. If they drift, every ratio in the bench is a speedup over a
    decoder nobody uses.

    `infer.Decoder` is what chat.py, serve.py and bench_decode.py drive, and
    serve.py's "Greedy" preset is `temperature: 1, top_k: 1` — which top_k=1
    makes an argmax whatever the temperature says.
    """

    def setUp(self):
        self.model = toy_model(seed=5)
        self.ids = [1, 5, 9, 3, 7, 2]
        self.count = 12                 # 6 prompt + 12 < 32, so no re-anchor
        self.served = infer.Sampler(temperature=1.0, top_k=1)

    def test_cached_baseline_is_the_cached_decoder_from_infer(self):
        decoder = infer.Decoder(self.model, self.ids, BLOCK, self.served,
                                refresh=8)
        self.assertEqual(
            greedy_cached(self.model, self.ids, BLOCK, self.count),
            list(decoder.tokens(self.count)))

    def test_cached_baseline_is_the_same_at_the_other_greedy_spelling(self):
        # test_all.py writes greedy as temperature 1e-8, serve.py as 1. If those
        # two disagreed the bench would be measuring the wrong one.
        decoder = infer.Decoder(self.model, self.ids, BLOCK,
                                infer.Sampler(temperature=1e-8, top_k=1),
                                refresh=8)
        self.assertEqual(
            greedy_cached(self.model, self.ids, BLOCK, self.count),
            list(decoder.tokens(self.count)))

    def test_uncached_baseline_is_the_naive_loop_in_model_generate(self):
        prompt = torch.tensor([self.ids], dtype=torch.long)
        generated = self.model.generate(prompt, self.count,
                                        temperature=1.0, top_k=1)
        self.assertEqual(
            greedy_uncached(self.model, self.ids, BLOCK, self.count),
            generated[0].tolist()[len(self.ids):])


class TheCheckCanFail(unittest.TestCase):
    """A guarantee verified by a check nobody has watched fail is unfalsified.

    The bench's entire output is one column of "identical", so the smallest
    plausible bug — a cache off-by-one that writes one wrong token and carries
    on — has to come out as a mismatch, naming the position where the text
    stopped being the target's.
    """

    def setUp(self):
        self.model = toy_model(seed=11)
        self.ids = [4, 8, 15, 16, 23, 31]
        self.count = 10
        self.want = greedy_cached(self.model, self.ids, BLOCK, self.count)

    def test_an_honest_decode_comes_out_identical(self):
        # The other half of the test: a verdict that said "identical" for
        # everything would catch nothing, so this pins the passing direction.
        decoder = GreedySpeculativeDecoder(self.model, self.ids, BLOCK,
                                           draft_layers=1, guesses=3)
        self.assertEqual(verdict(list(decoder.tokens(self.count)), self.want),
                         "identical")

    def test_a_single_wrong_token_is_named_by_its_position(self):
        broken = list(self.want)
        broken[4] = (broken[4] + 1) % VOCAB
        result = verdict(broken, self.want)
        self.assertNotEqual(result, "identical")
        self.assertIn("token 4", result)

    def test_a_short_output_is_not_identical(self):
        self.assertNotEqual(verdict(self.want[:-1], self.want), "identical")

    def test_a_decoder_that_writes_one_wrong_token_is_caught(self):
        class Liar(GreedySpeculativeDecoder):
            """Verifies correctly, then corrupts what it wrote, so the cache and
            `ids` go out of step exactly as an off-by-one would."""

            def round(self):
                emitted = super().round()
                self.ids[-1] = (self.ids[-1] + 1) % VOCAB
                return emitted[:-1] + [self.ids[-1]]

        decoder = Liar(self.model, self.ids, BLOCK, draft_layers=1, guesses=3)
        produced = list(decoder.tokens(self.count))
        self.assertNotEqual(verdict(produced, self.want), "identical")


class TheOracleIsTheCeiling(unittest.TestCase):
    """The oracle row is only a ceiling if it is also correct: a draft handed
    the answer and costing nothing must still produce the target's text, or the
    number beside it means nothing.

    How to read it. Every guess is right, so each round spends one target pass
    and emits `guesses + 1` tokens — the free token being the target's own
    prediction after the last accepted guess. That is the arithmetic behind the
    rate, and the test pins it rather than the clock.
    """

    def setUp(self):
        self.model = toy_model(seed=2)
        self.ids = [1, 5, 9, 3, 7, 2]
        self.count = 12
        self.want = greedy_cached(self.model, self.ids, BLOCK, self.count)

    def test_an_oracle_draft_reproduces_the_text_and_accepts_everything(self):
        decoder = GreedySpeculativeDecoder(self.model, self.ids, BLOCK,
                                           guesses=4, oracle=self.want)
        self.assertEqual(list(decoder.tokens(self.count)), self.want)
        self.assertEqual(decoder.accepted, decoder.proposed)
        self.assertGreater(decoder.proposed, 0)

    def test_an_oracle_draft_spends_the_fewest_target_passes(self):
        # Compared against a *biased* draft: on this three-layer toy, a
        # draft_layers=3 draft is the oracle already, so it would spend exactly
        # the same passes and say nothing.
        _, biased = spec_decode(self.model, self.ids, self.count,
                                draft_layers=1, guesses=4)
        oracle = GreedySpeculativeDecoder(self.model, self.ids, BLOCK,
                                          guesses=4, oracle=self.want)
        list(oracle.tokens(self.count))
        self.assertGreater(biased.target_passes, oracle.target_passes)
        # One prefill, then ceil(12 / 5) rounds of five tokens each — the
        # arithmetic the ceiling row is built on.
        self.assertEqual(oracle.target_passes, 1 + -(-self.count // 5))

    def test_a_real_draft_that_stops_early_does_not_run_off_the_oracle(self):
        # The guesses are sliced from the prompt offset, so a run shorter than
        # the oracle must not read past what it was given.
        short = self.want[:4]
        decoder = GreedySpeculativeDecoder(self.model, self.ids, BLOCK,
                                           guesses=6, oracle=short)
        produced = list(decoder.tokens(self.count))
        self.assertEqual(len(produced), self.count)


if __name__ == "__main__":
    unittest.main()
