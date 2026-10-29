# Othello engine

A from-scratch Othello player, and a ladder that rates the versions of itself.
No pretrained weights, no engine libraries, no GPU: the rules, the evaluation
and the search are the four files in this directory.

The question it asks is the one the other benches cannot: **is it getting
better?** There is a number here that moves, and it moves because the engine
plays other engines, not because a test passed.

## Run it

```bash
# you are black; the engine searches four plies
python -m othello.cli play

# engine moves first, and thinks harder
python -m othello.cli play --depth 6 --white

# replay a few moves and look at the result, without a terminal
python -m othello.cli play --moves "d3 e3" --depth 4

# re-measure the ladder: 20 games per pairing, depths 1 to 4
python -m othello.cli arena --games 20 --engines 1,2,3,4

# what the last run wrote
python -m othello.cli ladder

# the move generator, counted
python -m othello.cli perft --depth 7
```

Standard library only. Nothing here imports `torch`, so it runs on any Python
3.12 without the root `.venv` at all — from a plain `python3`, if you like.

## The ladder

Measured, not estimated: 20 games per pairing at fixed depth, colours
alternating, two random plies of opening so the engines are not replaying one
game forty times. Written to [`ladder.json`](ladder.json).

| engine | rating | games | W | D | L | score |
| --- | --- | --- | --- | --- | --- | --- |
| depth4 | 2161.3 | 60 | 59 | 0 | 1 | 98.3% |
| depth3 | 1634.0 | 60 | 38 | 0 | 22 | 63.3% |
| depth2 | 1306.0 | 60 | 21 | 0 | 39 | 35.0% |
| depth1 | 898.8 | 60 | 2 | 0 | 58 | 3.3% |

Two things to read out of that. Deeper is monotonically stronger, which is the
sanity check the ladder exists to make possible — and **each extra ply is worth
about 400–500 points**, i.e. roughly ten-to-one odds against the version below
it. Depth 1 is nearly free (it plays the highest-scoring single move, corners
first) and is almost useless; the entire strength of this engine is in the
search, not the evaluation.

## How it is checked

`python ../ci_tests.py othello.test_othello` — 23 tests, about three seconds.

- **The rules, against published perft counts.** 4, 12, 56, 244, 1396, 8200 move
  sequences at depths one through six for the opening position. These numbers
  are not derived from the implementation; a move generator with a missing flip
  direction agrees with itself and disagrees with these at depth three.
- **The search, against brute force.** A plain negamax with no pruning and no
  table is the reference; alpha-beta with a transposition table must return the
  same score, having visited far fewer nodes. On a midgame position at depth 4
  that is 7,477 nodes against 368.
- **The ladder, against arithmetic.** Equal results give equal ratings, a clean
  sweep moves the ratings apart, and the field stays centred on its anchor.

## Two bugs the tests caught

Both were found within an hour of each other, and both produced *legal moves*
that lost games, which is the kind of bug that survives a suite that only checks
legality.

**The root window took its beta from a move.** `best` held the best move's
bitboard and `best_score` its score; the search window for the next child was
built from `-best`, so from the second move on, alpha-beta was pruning against a
number like `-134217728`. Depth 4 lost 13–27 to depth 3. Splitting the variable
fixed it, and the ladder above is the result: depth 4 now goes 59–1.

**The transposition table was keyed without the turn.** The natural key is the
two board integers, because a position cannot normally be reached with the other
side to move — but a *pass* leaves both boards untouched and changes only whose
turn it is. Scores were being served across that boundary with the wrong sign.
The key now includes the turn, and a test pins the pass itself.

## What is not here yet

- **No endgame solver.** With the board nearly full the search should solve
  exactly rather than evaluate; that is worth a lot of the remaining strength.
- **One evaluation.** Corners, mobility, frontier discs and a late disc count,
  with weights I chose by reasoning rather than by tuning. The ladder is what
  will tell whether a change is an improvement, which is the point of building
  it first.
- **Depth is the only axis.** Once there are two evaluations to compare, the
  ladder says which is better instead of which one looks tidier.
