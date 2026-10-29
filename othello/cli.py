#!/usr/bin/env python3
"""
Four things to do with an Othello engine: play it, rate it, look at the ratings,
count positions.

    python -m othello.cli play                      # you are black
    python -m othello.cli play --depth 6 --white    # engine moves first
    python -m othello.cli play --moves "d3 c4 f5"   # scripted, for a quick look
    python -m othello.cli arena --games 8           # the ladder, re-measured
    python -m othello.cli ladder                    # the ladder as last written
    python -m othello.cli perft --depth 7           # the move generator, counted

`play` without `--moves` wants a terminal; with `--moves` it plays the sequence
and prints the board, which is how the engine is checked by hand and how the
README's example stays honest.
"""

import argparse
import sys
import time

from . import arena
from .board import BLACK, Position, perft
from .search import best_move, policy


def _parse_square(text):
    """`d3` -> the move bit, or None if it is not a square."""
    text = text.strip().lower()
    if len(text) != 2 or not "a" <= text[0] <= "h" or not "1" <= text[1] <= "8":
        return None
    column = ord(text[0]) - ord("a")
    row = int(text[1]) - 1
    return 1 << (row * 8 + column)


def _report(position, result, depth):
    """One line about what the engine just did."""
    if result.move is None:
        return "no legal move — passing"
    return (f"engine plays {position.square_name(result.move)}  "
            f"(depth {result.depth}, {result.nodes} nodes, "
            f"{result.seconds:.2f}s, score {result.score:+.0f})")


def play(args):
    position = Position()
    human = None if args.white else BLACK

    # A scripted game is for looking at, not for playing: when the moves run
    # out the game stops there. Falling through to `input()` would sit waiting
    # for a keypress in a run that was never interactive.
    scripted_mode = args.moves is not None
    scripted = [token for token in (args.moves or "").replace(",", " ").split()]

    print(position.render(position.moves()))
    print(f"you are {'white' if args.white else 'black'}; "
          f"the engine searches {args.depth} plies\n")

    while not position.over():
        moves = position.moves()
        if not moves:
            print("no move available — passing\n")
            position = position.passed()
            continue

        if position.turn != human:
            result = best_move(position, args.depth)
            print(_report(position, result, args.depth))
            position = position.play(result.move)
            print(position.render(position.moves() if not position.over() else 0))
            print()
            continue

        # Only when it is the human's turn: the engine's replies are the point
        # of playing a scripted opening out.
        if scripted_mode and not scripted:
            print("(scripted moves finished)")
            break

        if scripted:
            token = scripted.pop(0)
            move = _parse_square(token)
            if move is None or not move & moves:
                print(f"not a legal move: {token}", file=sys.stderr)
                return 1
            print(f"you play {position.square_name(move)}")
        else:
            offered = [position.square_name(bit) for bit in _bits(moves)]
            print(f"your move ({', '.join(offered)}): ", end="", flush=True)
            try:
                line = input()
            except EOFError:
                print()
                return 0
            if line.strip().lower() in ("q", "quit", "exit"):
                return 0
            move = _parse_square(line)
            if move is None or not move & moves:
                print(f"not a legal move: {line.strip()!r}")
                continue

        position = position.play(move)
        print(position.render(position.moves() if not position.over() else 0))
        print()

    black, white = position.score()
    winner = position.winner()
    if scripted_mode:
        # Not "final": the script stopped where it stopped, and calling a
        # midgame count a result would be a small lie.
        print(f"stopped at black {black}, white {white}")
    else:
        print(f"final: black {black}, white {white} — "
              f"{'draw' if winner is None else ('you' if winner == human else 'the engine')}")
    return 0


def _bits(mask):
    for index in range(64):
        if mask >> index & 1:
            yield 1 << index


def arena_run(args):
    depths = [int(piece) for piece in args.engines.split(",") if piece.strip()]
    engines = [policy(depth) for depth in depths]

    def progress(first, second, result):
        print(f"  {first:<8} vs {second:<8} "
              f"{result.wins_a}-{result.draws}-{result.wins_b}")

    started = time.time()
    print(f"{args.games} games per pairing, {len(engines)} engines, "
          f"openings of {args.opening} random plies\n")
    results = arena.round_robin(engines, args.games, args.opening, args.seed,
                                progress=progress)
    ratings = arena.fit_ratings(results)

    print()
    print(arena.table(results, ratings))
    print(f"\n{time.time() - started:.1f}s")

    if args.out:
        payload = arena.save(args.out, results, ratings, args.games)
        print(f"wrote {args.out} ({payload['updated']})")
    return 0


def ladder(args):
    payload = arena.load(args.path)
    if payload is None:
        print(f"no ladder at {args.path} — run `arena` first", file=sys.stderr)
        return 1
    print(arena.render_stored(payload))
    return 0


def perft_run(args):
    position = Position()
    print(position.render(position.moves()))
    for depth in range(1, args.depth + 1):
        started = time.time()
        count = perft(position, depth)
        print(f"perft {depth}: {count:>10}  {time.time() - started:.2f}s")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m othello.cli",
        description="An Othello engine, played or measured.")
    sub = parser.add_subparsers(dest="command")

    play_parser = sub.add_parser("play", help="play against the engine")
    play_parser.add_argument("--depth", type=int, default=4,
                             help="plies the engine searches (default: 4)")
    play_parser.add_argument("--white", action="store_true",
                             help="engine plays black and moves first")
    play_parser.add_argument("--moves", help="play these moves and stop reading input")
    play_parser.set_defaults(handler=play)

    arena_parser = sub.add_parser("arena", help="run the ladder's matches")
    arena_parser.add_argument("--games", type=int, default=8,
                              help="games per pairing (default: 8)")
    arena_parser.add_argument("--engines", default="1,2,3,4",
                              help="depths to compare, comma separated "
                                   "(default: 1,2,3,4)")
    arena_parser.add_argument("--opening", type=int, default=2,
                              help="random plies before the engines start "
                                   "(default: 2)")
    arena_parser.add_argument("--seed", type=int, default=0)
    arena_parser.add_argument("--out", default=arena.DEFAULT_LADDER,
                              help="where to write the ladder (default: the "
                                   "engine's own ladder.json)")
    arena_parser.set_defaults(handler=arena_run)

    ladder_parser = sub.add_parser("ladder", help="print the stored ladder")
    ladder_parser.add_argument("--path", default=arena.DEFAULT_LADDER)
    ladder_parser.set_defaults(handler=ladder)

    perft_parser = sub.add_parser("perft", help="count move sequences")
    perft_parser.add_argument("--depth", type=int, default=6)
    perft_parser.set_defaults(handler=perft_run)

    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 1
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
