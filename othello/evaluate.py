#!/usr/bin/env python3
"""
What a position is worth, if you cannot see the end of it.

Heuristics, plainly: this is the part that decides how the engine *feels* to
play against, and the part the ladder exists to argue with. Every weight below
is a guess, and the only claim made for any of them is that it is better than no
opinion at all.

Three things matter in Othello, in this order:

* corners, because a corner disc can never be flipped and it is the anchor every
  line ends at;
* mobility, because a player with no moves must pass and passes hand the board
  over;
* disc count — but only late. Taking discs early is how you run out of moves,
  which is why a naive "flip as much as possible" player loses to a beginner.

The score is returned from the side-to-move's point of view, because that is
what a negamax search wants and converting it in one place is one place to get
wrong.
"""

from .board import FULL, MASK_A, MASK_H

#: A corner is worth more than the seven discs around it, because none of them
#: can ever be taken back and it decides whether its edges are safe.
CORNER = 120.0

#: Squares next to a corner. While that corner is empty, playing one hands the
#: corner to the opponent: an X-square does it diagonally, a C-square along an
#: edge. Worth nothing once the corner is taken, which is why they scale.
X_SQUARE = 45.0
C_SQUARE = 30.0

#: Per disc of mobility difference. High enough that the engine will happily
#: give away discs early, which is the one thing a human beginner never does.
MOBILITY = 14.0

#: Discs sitting on the edge of the empty region: they are the ones that give
#: the opponent moves, so they are liabilities rather than assets.
FRONTIER = 6.0

#: Disc count, which only starts to matter when the board is nearly settled.
DISC = 1.0
DISC_LATE = 8.0
LATE_EMPTIES = 16

CORNERS = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)

#: The two diagonals that touch a corner: each is only a liability while its own
#: corner is still empty.
CORNER_NEIGHBOURS = {
    1 << 0: ((1 << 9), (1 << 1), (1 << 8)),
    1 << 7: ((1 << 14), (1 << 6), (1 << 15)),
    1 << 56: ((1 << 49), (1 << 57), (1 << 48)),
    1 << 63: ((1 << 54), (1 << 62), (1 << 55)),
}

WIN = 100000.0


def neighbour_mask(bits):
    """Every square orthogonally or diagonally adjacent to `bits`."""
    result = 0
    # A bit shifted off one side wraps around to the other; masking by the two
    # files before shifting is what keeps a neighbour on the board.
    result |= (bits & ~MASK_H) << 1
    result |= (bits & ~MASK_A) >> 1
    result |= (bits << 8) & FULL
    result |= bits >> 8
    result |= ((bits & ~MASK_H) << 9) & FULL
    result |= ((bits & ~MASK_A) << 7) & FULL
    result |= (bits & ~MASK_H) >> 7
    result |= (bits & ~MASK_A) >> 9
    return result & FULL


def _corner_penalty(occupancy):
    """What the empty corners cost the owner of the squares around them."""
    penalty = 0.0
    for corner, (x_square, c_a, c_b) in CORNER_NEIGHBOURS.items():
        if occupancy & corner:
            continue
        for square, weight in ((x_square, X_SQUARE), (c_a, C_SQUARE), (c_b, C_SQUARE)):
            if occupancy & square:
                penalty += weight
    return penalty


def frontier_count(bits, empty_region):
    """Discs of `bits` that touch an empty square, i.e. offer the opponent moves."""
    return bin(bits & neighbour_mask(empty_region)).count("1")


def evaluate(position):
    """The score from the side to move's point of view, in centi-discs.

    Rounded to a whole number so a transposition table and a tie-break cannot
    disagree about two positions that are the same to within float noise.
    """
    mine = position.discs()
    theirs = position.opponent_discs()
    empty = ~(mine | theirs) & FULL
    empties = 64 - bin(mine | theirs).count("1")

    if not position.moves() and not position.other_moves():
        # Nothing left to estimate: a finished game is worth its result. The
        # score below would call a 33-31 win "a bit better than even".
        return final_score(position)

    score = 0.0
    score += CORNER * (bin(mine & CORNERS).count("1") -
                       bin(theirs & CORNERS).count("1"))
    score -= _corner_penalty(mine)
    score += _corner_penalty(theirs)

    score += MOBILITY * (bin(position.moves()).count("1") -
                         bin(position.other_moves()).count("1"))

    score -= FRONTIER * (frontier_count(mine, empty) -
                         frontier_count(theirs, empty))

    weight = DISC_LATE if empties <= LATE_EMPTIES else DISC
    score += weight * (bin(mine).count("1") - bin(theirs).count("1"))
    return float(round(score))


def final_score(position):
    """The score of a finished game, from the side to move's point of view.

    A win is worth `WIN` plus the margin, so the search prefers winning by more
    when it can afford to — that is what makes the endgame look deliberate
    rather than arbitrary among equal wins.
    """
    diff = (bin(position.discs()).count("1") -
            bin(position.opponent_discs()).count("1"))
    if not diff:
        return 0.0
    margin = diff * 100.0
    return (WIN + margin) if diff > 0 else -(WIN - margin)
