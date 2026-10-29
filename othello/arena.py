#!/usr/bin/env python3
"""
Matches, and one number for the result.

Two engines that both play "well" are hard to tell apart by watching, easy to
tell apart by counting. This module plays them against each other, fits a rating
to the wins, and keeps the table on disk — so a new idea is judged against the
one it replaces instead of against a feeling.

Three things are deliberate:

*Colours alternate.* Othello is not symmetric: black moves first and gets the
last word in a close game. A match where one engine always plays black measures
the colour as well as the engine.

*Openings vary.* At fixed depth both engines are deterministic, so the same
opening would be the same game every time — forty identical games with a score
of 40-0 that means nothing. A few random plies at the start, from a seeded
generator, give a spread of positions that is still reproducible.

*Draws count as half.* They are rare in Othello and almost never happen between
equal engines; scoring them as half a win is the convention every rating list
uses, and it keeps the arithmetic honest when they do appear.
"""

import datetime
import json
import os
import random

from .board import BLACK, Position

DEFAULT_LADDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "ladder.json")

#: Anchor for the scale. Arbitrary and conventional; the numbers only ever mean
#: something relative to each other.
ANCHOR = 1500.0


class Result:
    """One pairing's tally: games won by each side, and how many were drawn."""

    __slots__ = ("wins_a", "draws", "wins_b")

    def __init__(self, wins_a=0, draws=0, wins_b=0):
        self.wins_a = wins_a
        self.draws = draws
        self.wins_b = wins_b

    @property
    def games(self):
        return self.wins_a + self.draws + self.wins_b

    @property
    def score_a(self):
        """Points for `a`, draws counted half, as every rating list does."""
        return self.wins_a + 0.5 * self.draws

    def add(self, other):
        self.wins_a += other.wins_a
        self.draws += other.draws
        self.wins_b += other.wins_b

    def as_json(self):
        return {"wins_a": self.wins_a, "draws": self.draws, "wins_b": self.wins_b}


def play_game(black, white, opening_plies=2, rng=None):
    """One game. Returns `black`'s name, `white`'s name, or None for a draw.

    The random opening is played before either engine is asked anything, so the
    engines start from a position neither of them would have chosen. Two plies
    is enough to break the symmetry without handing the game to luck.
    """
    rng = rng or random.Random(0)
    position = Position()

    for _ in range(opening_plies):
        moves = position.moves()
        if not moves:
            break
        options = [bit for bit in range(64) if moves >> bit & 1]
        position = position.play(1 << rng.choice(options))

    while not position.over():
        moves = position.moves()
        if not moves:
            position = position.passed()
            continue
        engine = black if position.turn == BLACK else white
        move = engine(position)
        if move is None or not move & moves:
            raise ValueError(f"{engine.name} chose an illegal move")
        position = position.play(move)

    winner = position.winner()
    if winner is None:
        return None
    return black.name if winner == BLACK else white.name


def match(a, b, games=2, opening_plies=2, seed=0):
    """`games` games between two engines, colours alternating."""
    rng = random.Random(seed)
    result = Result()
    for index in range(games):
        a_is_black = index % 2 == 0
        black, white = (a, b) if a_is_black else (b, a)
        winner = play_game(black, white, opening_plies, rng)
        # `play_game` answers with a name, so the colour it was given does not
        # matter here — which is the point of alternating it.
        if winner is None:
            result.draws += 1
        elif winner == a.name:
            result.wins_a += 1
        else:
            result.wins_b += 1
    return result


def round_robin(engines, games=2, opening_plies=2, seed=0, progress=None):
    """Every pairing once, as `{(name_a, name_b): Result}` in name order."""
    pairs = {}
    names = [engine.name for engine in engines]
    for index, first in enumerate(engines):
        for second in engines[index + 1:]:
            key = (first.name, second.name)
            result = match(first, second, games, opening_plies, seed)
            pairs[key] = result
            if progress is not None:
                progress(first.name, second.name, result)
    if len(names) < 2:
        pairs[(names[0] if names else "?", names[0] if names else "?")] = Result()
    return pairs


def expected(rated, opponent):
    """The logistic everyone means by Elo: 400 points is ten-to-one odds."""
    return 1.0 / (1.0 + 10 ** ((opponent - rated) / 400.0))


def fit_ratings(results, anchor=ANCHOR, iterations=600, step=24.0):
    """Ratings that explain the results, by gradient ascent on the likelihood.

    Not the pairwise formula from the FIDE handbook: that one needs a fixed
    reference player, and here there is no champion — every engine is being
    compared with every other at once. Maximising the Bradley-Terry likelihood
    is the same model solved for all of them together, and for a round robin of
    five engines it converges in well under a millisecond.
    """
    names = sorted({name for pair in results for name in pair})
    if not names:
        return {}
    ratings = {name: anchor for name in names}
    if len(names) == 1:
        return ratings

    played = {name: 0 for name in names}
    for (a, b), result in results.items():
        played[a] += result.games
        played[b] += result.games

    for _ in range(iterations):
        shift = {name: 0.0 for name in names}
        for (a, b), result in results.items():
            if not result.games:
                continue
            win_a = expected(ratings[a], ratings[b])
            shift[a] += (result.score_a - result.games * win_a) / result.games
            shift[b] += ((result.games - result.score_a) -
                         result.games * (1.0 - win_a)) / result.games

        largest = 0.0
        for name in names:
            change = step * shift[name] * 4.0 / max(1, len(results))
            ratings[name] += change
            largest = max(largest, abs(change))

        # The scale has no absolute zero, so pinning the mean is what makes two
        # runs comparable: otherwise every rating drifts with the field size.
        mean = sum(ratings.values()) / len(ratings)
        for name in ratings:
            ratings[name] += anchor - mean

        if largest < 0.01:
            break
    return {name: round(value, 1) for name, value in ratings.items()}


def tally(results):
    """Per engine: games, wins, draws, losses, from the pairing results."""
    stats = {}
    for (a, b), result in results.items():
        for name in (a, b):
            stats.setdefault(name, {"games": 0, "wins": 0, "draws": 0,
                                    "losses": 0})
        # The artificial single-engine pairing `round_robin` adds when there is
        # nobody to play must not count as games.
        if a == b:
            stats[a]["games"] += 0
            continue
        stats[a]["games"] += result.games
        stats[a]["wins"] += result.wins_a
        stats[a]["draws"] += result.draws
        stats[a]["losses"] += result.wins_b
        stats[b]["games"] += result.games
        stats[b]["wins"] += result.wins_b
        stats[b]["draws"] += result.draws
        stats[b]["losses"] += result.wins_a
    return stats


def table(results, ratings=None):
    """The ladder as text, strongest first."""
    stats = tally(results)
    ratings = ratings if ratings is not None else fit_ratings(results)
    rows = sorted(stats, key=lambda name: (-ratings.get(name, 0.0), name))
    width = max((len(name) for name in rows), default=4)

    lines = [f"{'engine':<{width}}  {'rating':>7}  {'games':>5}  "
             f"{'W':>3} {'D':>3} {'L':>3}  {'score':>6}"]
    lines.append("-" * len(lines[0]))
    for name in rows:
        row = stats[name]
        games = row["games"]
        points = row["wins"] + 0.5 * row["draws"]
        percent = f"{100.0 * points / games:.1f}%" if games else "-"
        lines.append(f"{name:<{width}}  {ratings.get(name, 0.0):>7.1f}  "
                     f"{games:>5}  {row['wins']:>3} {row['draws']:>3} "
                     f"{row['losses']:>3}  {percent:>6}")

    if len(rows) > 1:
        leader = ratings.get(rows[0], 0.0)
        runner = ratings.get(rows[1], 0.0)
        lines.append("")
        lines.append(f"{rows[0]} is {leader - runner:.0f} points ahead of {rows[1]}")
    return "\n".join(lines)


def save(path, results, ratings, games_per_pair):
    """Write the ladder down, so the next version has something to beat."""
    stats = tally(results)
    payload = {
        "updated": datetime.datetime.now(datetime.timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "games_per_pair": games_per_pair,
        "ratings": {name: {"rating": ratings.get(name, 0.0), **stats[name]}
                    for name in sorted(stats)},
        "pairings": [
            {"a": a, "b": b, **result.as_json()}
            for (a, b), result in sorted(results.items())
        ],
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def load(path=DEFAULT_LADDER):
    """The stored ladder, or None if nothing has been played yet."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


def render_stored(payload):
    """The ladder file as a table, without replaying any games."""
    ratings = payload.get("ratings", {})
    width = max((len(name) for name in ratings), default=4)
    lines = [f"{'engine':<{width}}  {'rating':>7}  {'games':>5}  "
             f"{'W':>3} {'D':>3} {'L':>3}  {'updated':>10}"]
    lines.append("-" * len(lines[0]))
    rows = sorted(ratings, key=lambda name: -ratings[name]["rating"])
    for name in rows:
        row = ratings[name]
        lines.append(f"{name:<{width}}  {row['rating']:>7.1f}  "
                     f"{row['games']:>5}  {row['wins']:>3} {row['draws']:>3} "
                     f"{row['losses']:>3}  {payload.get('updated', '')[:10]:>10}")
    lines.append("")
    lines.append(f"{payload.get('games_per_pair', 0)} games per pairing, "
                 f"{len(rows)} engines")
    return "\n".join(lines)
