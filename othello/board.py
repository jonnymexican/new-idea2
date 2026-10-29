#!/usr/bin/env python3
"""
Othello, on two 64-bit boards.

The rules are small enough to state in a paragraph and fiddly enough to get
wrong in the corners: a disc flips along every line it closes, a player with no
legal move passes, and the game ends when neither player can move even though
squares are still empty. Those are exactly the cases where a bug hides until it
costs a game, so the rules live here and nowhere else, and the tests check the
move generator against published perft counts rather than against my reading of
the rules.

Board layout: bit `8 * row + col`, row 0 at the top and column 0 at the left, so
`a1` in the usual diagram is bit 0 and `h8` is bit 63. Directions are compass
names with north being up the page, which is why the *southward* shifts are the
ones that grow the integer.

A move is a single bit. `Position` holds the two sides and whose turn it is;
every method that changes the board returns a new `Position` rather than
mutating, because a search holds hundreds of them at once and an aliasing bug
there is invisible until the answer is quietly wrong.
"""

FULL = (1 << 64) - 1

#: Files, for the horizontal shifts. A bit shifted east off the h-file would
#: otherwise wrap into the next row, which is the classic bitboard bug: legal
#: moves appearing on the far side of the board.
MASK_A = 0x0101010101010101
MASK_H = 0x8080808080808080

BLACK = 0
WHITE = 1

EMPTY, B_DISC, W_DISC, MOVE_DOT = ".", "X", "O", "*"

#: The four central discs, in the arrangement every Othello set starts from:
#: white on d4 and e5, black on d5 and e4, black to move.
START_BLACK = (1 << 28) | (1 << 35)
START_WHITE = (1 << 27) | (1 << 36)


def _north(x):
    return x >> 8


def _south(x):
    return (x << 8) & FULL


def _east(x):
    return (x & ~MASK_H) << 1


def _west(x):
    return (x & ~MASK_A) >> 1


def _north_east(x):
    return (x & ~MASK_H) >> 7


def _north_west(x):
    return (x & ~MASK_A) >> 9


def _south_east(x):
    return ((x & ~MASK_H) << 9) & FULL


def _south_west(x):
    return ((x & ~MASK_A) << 7) & FULL


#: The eight lines a flip can travel along. Every rule here is one of these
#: applied until it runs out of board or out of opponent discs.
SHIFTS = (_north, _south, _east, _west,
          _north_east, _north_west, _south_east, _south_west)


def _legal(player, opponent):
    """Every empty square where `player` would close a line of opponent discs."""
    empty = ~(player | opponent) & FULL
    moves = 0
    for shift in SHIFTS:
        run = shift(player) & opponent
        if not run:
            continue
        # Walk the run to its end; the square just past it is a legal move only
        # if it is empty. An own disc there means the line was already closed.
        found = run
        while run:
            run = shift(run) & opponent
            found |= run
        moves |= shift(found) & empty
    return moves


def _flips(player, opponent, move):
    """The discs a move turns over: every run of opponent discs it brackets."""
    flipped = 0
    for shift in SHIFTS:
        run = shift(move) & opponent
        if not run:
            continue
        found = run
        while run:
            run = shift(run) & opponent
            found |= run
        # Only a run that ends in one of the player's own discs flips.
        if shift(found) & player:
            flipped |= found
    return flipped


def squares(bits):
    """The set bits, as `(row, col)` pairs, in board order."""
    return [(index // 8, index % 8) for index in range(64) if bits >> index & 1]


class Position:
    """A board, and whose turn it is. Passes are explicit, never implied."""

    __slots__ = ("black", "white", "turn")

    def __init__(self, black=START_BLACK, white=START_WHITE, turn=BLACK):
        self.black = black
        self.white = white
        self.turn = turn

    def discs(self, side=None):
        side = self.turn if side is None else side
        return self.black if side == BLACK else self.white

    def opponent_discs(self, side=None):
        side = self.turn if side is None else side
        return self.white if side == BLACK else self.black

    def moves(self):
        """Legal moves for the side to move; zero means it has to pass."""
        return _legal(self.discs(), self.opponent_discs())

    def other_moves(self):
        """Legal moves for the side that is *not* to move, for spotting a pass."""
        return _legal(self.opponent_discs(), self.discs())

    def play(self, move):
        """Place a disc, turn over what it brackets, and hand over the turn."""
        if not move & self.moves():
            raise ValueError(f"illegal move at {move:#x}")
        player = self.discs()
        opponent = self.opponent_discs()
        flips = _flips(player, opponent, move)

        player |= move | flips
        opponent &= ~flips
        if self.turn == BLACK:
            return Position(player, opponent, WHITE)
        return Position(opponent, player, BLACK)

    def passed(self):
        """Hand the turn over without a move. A player with none must do this."""
        return Position(self.black, self.white,
                         WHITE if self.turn == BLACK else BLACK)

    def over(self):
        """True when neither side can move: the board is settled or full."""
        if not self.moves():
            return not self.other_moves()
        return False

    def empties(self):
        return 64 - bin(self.black | self.white).count("1")

    def score(self):
        """`(black, white)` discs on the board."""
        return bin(self.black).count("1"), bin(self.white).count("1")

    def winner(self):
        """`BLACK`, `WHITE`, or None for a draw. Only meaningful once over."""
        black, white = self.score()
        if black > white:
            return BLACK
        if white > black:
            return WHITE
        return None

    def square_name(self, move):
        index = move.bit_length() - 1
        return f"{chr(ord('a') + index % 8)}{index // 8 + 1}"

    def render(self, marks=0):
        """The board as text, with `marks` (a move bitboard) drawn as dots."""
        lines = []
        for row in range(8):
            cells = []
            for col in range(8):
                bit = 1 << (row * 8 + col)
                if self.black & bit:
                    cells.append(B_DISC)
                elif self.white & bit:
                    cells.append(W_DISC)
                elif marks & bit:
                    cells.append(MOVE_DOT)
                else:
                    cells.append(EMPTY)
            lines.append(f"{row + 1} {' '.join(cells)}")
        lines.append(f"  {' '.join(chr(ord('a') + col) for col in range(8))}")
        return "\n".join(lines)


def perft(position, depth):
    """The number of move sequences of `depth` plies, passes counted as plies.

    The published counts for the opening position are the only honest way to
    check a move generator: 4, 12, 56, 244, 1396, 8200, 55092, 390216 at depths
    one through eight. A generator that misses a flip direction or wraps a
    shift disagrees at depth three, not at depth one.

    A pass is a ply here, because that is what the published numbers count. So a
    player with no move extends the sequence by passing rather than ending it.
    """
    if depth == 0:
        return 1

    moves = position.moves()
    if not moves:
        # No move and the opponent has one: the sequence continues with a pass.
        # Neither side can move: the game ended at this node, which is one
        # sequence of the length asked for, and it cannot be extended.
        if position.other_moves():
            return perft(position.passed(), depth - 1)
        return 1

    total = 0
    while moves:
        move = moves & -moves
        moves ^= move
        total += perft(position.play(move), depth - 1)
    return total
