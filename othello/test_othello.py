"""The rules against published counts, the search against brute force, the ladder
against arithmetic.

Three kinds of check, for three different ways this engine can lie to you:

*The rules* are checked against the published perft counts for the opening
position — 4, 12, 56, 244, 1396, 8200 — which is the only check here that is not
self-referential. A move generator with a missing flip direction passes its own
tests and disagrees with those numbers at depth three.

*The search* is checked against a deliberately stupid reference: plain negamax,
no pruning, no table, full window. Alpha-beta and a transposition table are
supposed to return the same answer with less work, and the only way to know they
do is to compare. This is the test that would have caught the two bugs this
engine shipped with for an hour: a root window whose beta came from a move
bitboard, and a table keyed without the turn (a pass changes the turn and
nothing else, so the two sides shared an entry).

*The ladder* is arithmetic, so it is checked exactly: equal results give equal
ratings, a clean sweep gives a rating gap in the direction of the winner, and
the scale stays anchored where it claims to be.
"""

import os
import random
import tempfile
import unittest

from . import arena
from .board import BLACK, WHITE, Position, perft
from .evaluate import evaluate, final_score
from .search import best_move, policy

#: Counts for the opening position, depths one through six. Published; the tests
#: below never derive them, which is the point.
PUBLISHED_PERFT = {1: 4, 2: 12, 3: 56, 4: 244, 5: 1396, 6: 8200}

#: A position found by playing seeded random games and stopping at the first
#: one where the side to move has nothing: white is stuck, black is not. Real
#: rather than constructed, because a made-up pass position tends to be one
#: where *both* sides are stuck, which is a different case entirely.
PASS_BLACK = 0x1397DEDD5ADC181
PASS_WHITE = 0xFEC682122A523E5E


def brute(position, depth, counter):
    """The reference: every move, no pruning, no table, no cleverness."""
    counter[0] += 1
    moves = position.moves()
    if not moves:
        if position.other_moves():
            return -brute(position.passed(), depth, counter)
        return final_score(position)
    if depth == 0:
        return evaluate(position)

    best = -10 ** 9
    while moves:
        move = moves & -moves
        moves ^= move
        value = -brute(position.play(move), depth - 1, counter)
        if value > best:
            best = value
    return best


def midgame(plies, seed):  # noqa: D401 - a fixture, not an API
    """A reproducible position a few random plies in."""
    rng = random.Random(seed)
    position = Position()
    for _ in range(plies):
        moves = position.moves()
        if not moves:
            position = position.passed()
            continue
        options = [index for index in range(64) if moves >> index & 1]
        position = position.play(1 << rng.choice(options))
    return position


class RulesTest(unittest.TestCase):
    def test_the_opening_position_is_the_standard_one(self):
        position = Position()
        # White on d4 and e5, black on d5 and e4, black to move.
        self.assertEqual(position.score(), (2, 2))
        self.assertEqual(position.turn, BLACK)
        self.assertEqual({position.square_name(move) for move in _bits(position.moves())},
                         {"d3", "c4", "f5", "e6"})

    def test_perft_matches_the_published_counts(self):
        position = Position()
        for depth, expected in PUBLISHED_PERFT.items():
            self.assertEqual(perft(position, depth), expected,
                             f"perft {depth} disagrees with the published count")

    def test_a_disc_flips_along_the_line_it_closes(self):
        position = Position()
        after = position.play(_square("d3"))
        # Black at d3 closes the line d3-d4-d5, so d4 changes colour.
        self.assertTrue(after.black >> _index("d4") & 1)
        self.assertFalse(after.white >> _index("d4") & 1)
        self.assertEqual(after.score(), (4, 1))

    def test_an_illegal_move_is_refused(self):
        with self.assertRaises(ValueError):
            Position().play(_square("a1"))

    def test_a_player_with_no_move_must_pass(self):
        position = Position(black=PASS_BLACK, white=PASS_WHITE, turn=WHITE)
        self.assertEqual(position.moves(), 0)
        self.assertTrue(position.other_moves())
        self.assertFalse(position.over(), "one side can still play")

        # A pass hands the turn back without touching the board.
        passed = position.passed()
        self.assertEqual((passed.black, passed.white),
                         (position.black, position.white))
        self.assertEqual(passed.turn, BLACK)

    def test_a_finished_game_scores_by_count(self):
        # Two discs, no legal move for either side, black ahead.
        position = Position(black=(1 << 0) | (1 << 1), white=1 << 63,
                            turn=BLACK)
        self.assertTrue(position.over())
        self.assertEqual(position.score(), (2, 1))
        self.assertEqual(position.winner(), BLACK)


class EvaluationTest(unittest.TestCase):
    def test_the_score_is_from_the_side_to_move_and_flips_with_it(self):
        for seed in range(4):
            position = midgame(10, seed).passed()
            self.assertEqual(evaluate(position), -evaluate(position.passed()),
                             "the evaluation is not antisymmetric in the turn")

    def test_a_corner_is_worth_more_than_a_disc(self):
        # Same discs, same number of moves, only the corner differs: black owns
        # a1 in the first, white owns it in the second.
        ours = Position(black=_square("a1") | _square("d4"),
                        white=_square("d5") | _square("e4"), turn=BLACK)
        theirs = Position(black=_square("d4"),
                          white=_square("d5") | _square("e4") | _square("a1"),
                          turn=BLACK)
        self.assertEqual(bin(ours.moves()).count("1"),
                         bin(theirs.moves()).count("1"),
                         "the fixtures differ in mobility as well as the corner")
        self.assertGreater(evaluate(ours), evaluate(theirs) + 100)


class SearchTest(unittest.TestCase):
    def test_the_pruned_search_returns_what_the_plain_one_does(self):
        # The bug this exists for: a wrong window returns a wrong score, and
        # nothing else in the suite noticed because the move was still legal.
        for seed, plies in ((0, 0), (7, 12), (11, 20)):
            position = midgame(plies, seed)
            for depth in (1, 2, 3):
                counter = [0]
                expected = brute(position, depth, counter)
                result = best_move(position, depth)
                self.assertEqual(result.score, expected,
                                 f"depth {depth} on seed {seed}/{plies} plies")
                self.assertEqual(result.depth, depth)
                self.assertLess(result.nodes, counter[0],
                                "pruning should visit fewer nodes than no pruning")

    def test_the_move_returned_is_always_legal(self):
        position = midgame(14, 5)
        result = best_move(position, 3)
        self.assertTrue(result.move & position.moves())

    def test_a_corner_is_taken_when_one_is_on_offer(self):
        # h1 can be taken through g2. A search that misses that is not searching.
        position = Position(black=_square("f3"), white=_square("g2"), turn=BLACK)
        self.assertEqual(best_move(position, 2).move, _square("h1"))

    def test_a_position_with_no_move_returns_no_move(self):
        # Not `None` by accident: the caller must pass, and inventing a move
        # here would be the engine playing for a side that cannot play.
        position = Position(black=PASS_BLACK, white=PASS_WHITE, turn=WHITE)
        self.assertIsNone(best_move(position, 2).move)

    def test_a_time_budget_returns_a_finished_depth(self):
        result = best_move(Position(), depth=12, seconds=0.05)
        self.assertGreaterEqual(result.depth, 1, "not even depth one finished")
        self.assertLessEqual(result.depth, 12)
        self.assertLess(result.seconds, 2.0, "the budget was not respected")


class LadderTest(unittest.TestCase):
    def test_four_hundred_points_is_ten_to_one(self):
        self.assertAlmostEqual(arena.expected(1500, 1100), 10 / 11, places=6)
        self.assertAlmostEqual(arena.expected(1100, 1500), 1 / 11, places=6)
        self.assertAlmostEqual(arena.expected(1500, 1500), 0.5, places=6)

    def test_equal_results_give_equal_ratings(self):
        results = {("a", "b"): arena.Result(wins_a=5, wins_b=5)}
        ratings = arena.fit_ratings(results)
        self.assertEqual(ratings, {"a": arena.ANCHOR, "b": arena.ANCHOR})

    def test_a_clean_sweep_moves_the_ratings_apart(self):
        results = {("a", "b"): arena.Result(wins_a=20)}
        ratings = arena.fit_ratings(results)
        self.assertGreater(ratings["a"], ratings["b"])
        # The scale is anchored: a gap of this size is a large one, not a small one.
        self.assertGreater(ratings["a"] - ratings["b"], 400)

    def test_the_scale_stays_centred(self):
        results = {("a", "b"): arena.Result(wins_a=9, draws=2, wins_b=5),
                   ("b", "c"): arena.Result(wins_a=3, wins_b=11),
                   ("a", "c"): arena.Result(wins_a=8, draws=1, wins_b=7)}
        ratings = arena.fit_ratings(results)
        self.assertAlmostEqual(sum(ratings.values()) / len(ratings),
                               arena.ANCHOR, places=1)

    def test_a_ladder_is_written_and_read_back(self):
        results = {("a", "b"): arena.Result(wins_a=3, draws=1, wins_b=2)}
        ratings = arena.fit_ratings(results)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ladder.json")
            arena.save(path, results, ratings, 6)
            payload = arena.load(path)
        self.assertEqual(payload["games_per_pair"], 6)
        self.assertEqual(set(payload["ratings"]), {"a", "b"})
        self.assertIn("a", arena.render_stored(payload))
        self.assertIn("a", arena.table(results, ratings))


class MatchTest(unittest.TestCase):
    def policies(self):
        return policy(1), policy(2)

    def test_a_game_has_a_result_and_it_is_one_of_the_players(self):
        first, second = self.policies()
        winner = arena.play_game(first, second, opening_plies=2,
                                 rng=random.Random(0))
        self.assertIn(winner, {first.name, second.name, None})

    def test_a_match_plays_the_games_it_was_asked_for(self):
        first, second = self.policies()
        result = arena.match(first, second, games=4, opening_plies=2, seed=0)
        self.assertEqual(result.games, 4)
        self.assertEqual(result.wins_a + result.wins_b + result.draws, 4)

    def test_the_same_seed_is_the_same_match(self):
        first, second = self.policies()
        one = arena.match(first, second, games=2, opening_plies=2, seed=5)
        two = arena.match(first, second, games=2, opening_plies=2, seed=5)
        self.assertEqual((one.wins_a, one.draws, one.wins_b),
                         (two.wins_a, two.draws, two.wins_b))

    def test_a_deeper_search_beats_a_shallower_one(self):
        # Small, but the direction is the whole premise of the ladder: if this
        # ever flips, the engine has a bug, not a bad day.
        shallow, deep = policy(1), policy(3)
        result = arena.match(shallow, deep, games=4, opening_plies=2, seed=2)
        self.assertGreater(result.wins_b, result.wins_a)

    def test_an_engine_that_cannot_play_is_an_error_not_a_silent_loss(self):
        first, second = self.policies()

        def broken(position):
            return 1 << 0

        broken.name = "broken"
        with self.assertRaises(ValueError):
            arena.play_game(broken, second, opening_plies=0,
                            rng=random.Random(0))


def _bits(mask):
    for index in range(64):
        if mask >> index & 1:
            yield 1 << index


def _index(name):
    return (int(name[1]) - 1) * 8 + (ord(name[0]) - ord("a"))


def _square(name):
    return 1 << _index(name)


if __name__ == "__main__":
    unittest.main()
