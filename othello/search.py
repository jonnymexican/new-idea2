#!/usr/bin/env python3
"""
Looking ahead, and stopping early when the clock says so.

Negamax with alpha-beta pruning, iterative deepening, a small transposition
table, and move ordering that costs a bit of work to save a lot of it. Nothing
here is clever beyond the standard book; the interesting decisions are the ones
listed at the end of this docstring, because they are what makes two versions
comparable on a ladder.

*Passes keep their depth.* A player with no move hands the turn back, and that
is not a ply of thinking — charging it one would make the search shallower in
exactly the positions where Othello gets sharpest.

*Iterative deepening is what makes a time limit safe.* Each depth is searched
from scratch with the previous depth's best move first, so a cutoff returns the
last depth that *finished* rather than a half-built answer. A search that returns
whatever it had at the moment the clock ran out is worse than a shallower one
that finished, and much harder to explain.

*The transposition table is keyed on the discs and the turn.* The obvious
key is the two board integers, since a position cannot normally come up with
the other side to move — but a pass breaks that: passing leaves both boards
identical and changes only whose turn it is. Without the turn in the key, the
score of the position before a pass is served as the score after it, with the
wrong side's sign. That one is invisible until a search deeper than its
sibling starts losing, which is exactly how it was found.
"""

import time

from .evaluate import CORNERS, evaluate, final_score

EXACT, LOWER, UPPER = 0, 1, 2

INFINITY = 10 ** 9


class Timeout(Exception):
    """The deadline arrived mid-search; the caller keeps the last finished depth."""


class SearchResult:
    """What a search decided, and how much work it admitted to."""

    __slots__ = ("move", "score", "depth", "nodes", "seconds")

    def __init__(self, move, score, depth, nodes, seconds):
        self.move = move
        self.score = score
        self.depth = depth
        self.nodes = nodes
        self.seconds = seconds

    def __repr__(self):
        move = None if self.move is None else bin(self.move).count("1")
        return (f"SearchResult(move={move}, score={self.score}, "
                f"depth={self.depth}, nodes={self.nodes})")


def order_moves(position, moves, first=None):
    """Corners first, then whatever leaves the opponent fewest replies.

    The second half is the whole trick: in Othello the value of a move is
    mostly the mobility it denies, so sorting by the opponent's reply count
    finds the cutoffs alpha-beta needs. One ply of that ordering is cheap and
    pays for itself several times over.
    """
    ranked = []
    while moves:
        move = moves & -moves
        moves ^= move
        if move == first:
            rank = -1
        elif move & CORNERS:
            rank = 0
        else:
            rank = 1 + bin(position.play(move).moves()).count("1")
        ranked.append((rank, move))
    ranked.sort()
    return [move for _, move in ranked]


def _negamax(position, depth, alpha, beta, table, deadline, counter):
    counter[0] += 1
    if deadline is not None and not counter[0] & 1023:
        # Checking the clock every node costs more than it saves; every 1024
        # nodes is still far under a millisecond of overshoot.
        if time.perf_counter() > deadline:
            raise Timeout

    moves = position.moves()
    if not moves:
        if position.other_moves():
            return -_negamax(position.passed(), depth, -beta, -alpha,
                             table, deadline, counter)
        return final_score(position)

    if depth == 0:
        return evaluate(position)

    key = (position.black, position.white, position.turn)
    entry = table.get(key)
    first = None
    if entry is not None:
        entry_depth, flag, score, first = entry
        if entry_depth >= depth:
            if flag == EXACT:
                return score
            if flag == LOWER and score >= beta:
                return score
            if flag == UPPER and score <= alpha:
                return score

    best = -INFINITY
    best_move = first
    start = alpha
    for move in order_moves(position, moves, first):
        score = -_negamax(position.play(move), depth - 1, -beta, -alpha,
                          table, deadline, counter)
        if score > best:
            best, best_move = score, move
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    if best <= start:
        flag = UPPER
    elif best >= beta:
        flag = LOWER
    else:
        flag = EXACT
    table[key] = (depth, flag, best, best_move)
    return best


def best_move(position, depth=4, seconds=None):
    """The move this engine would play, having looked `depth` plies ahead.

    `seconds` is a budget rather than a promise: the search stops between
    iterative-deepening passes, so the answer is always one that was finished,
    and the time taken is the time the last *completed* depth needed.
    """
    started = time.perf_counter()
    deadline = None if seconds is None else started + seconds
    moves = position.moves()

    if not moves:
        # Nothing to choose. The caller is expected to pass, and a `None` move
        # says so rather than inventing one.
        return SearchResult(None, 0.0, 0, 0, 0.0)

    table = {}
    counter = [0]
    chosen = moves & -moves
    score = 0.0
    reached = 0

    for current in range(1, depth + 1):
        try:
            # The move and its score are two variables, and keeping them apart is
            # not style: an earlier version used one name for both, which made
            # the window's beta a move bitboard after the first move. Pruning
            # with a nonsense bound returns a nonsense score, and the engine then
            # picks the move that nonsense favours -- depth four lost to depth
            # three until this was split.
            best_move = chosen
            best_score = -INFINITY
            for move in order_moves(position, moves, chosen):
                value = -_negamax(position.play(move), current - 1,
                                  -INFINITY, -best_score,
                                  table, deadline, counter)
                if value > best_score:
                    best_move, best_score = move, value
            if best_score > -INFINITY:
                chosen, score = best_move, best_score
                reached = current
        except Timeout:
            break

    return SearchResult(chosen, score, reached, counter[0],
                        time.perf_counter() - started)


def policy(depth=4, seconds=None):
    """A callable the arena can play: `policy(position) -> move`.

    Named by what it searches, because a ladder of unnamed players is a ladder
    of numbers with no story.
    """
    def choose(position):
        return best_move(position, depth, seconds).move

    choose.name = f"depth{depth}"
    choose.depth = depth
    return choose
