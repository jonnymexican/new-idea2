#!/usr/bin/env python3
"""
Build today's board from the live feeds and check that the colour rule holds.

The unit tests in `test_mediatop.py` prove the rule against fixtures. This
proves it against whatever the twelve feeds are publishing now, which is a
different claim: the sentences that reached it on 23 October were the ones no
fixture had, and a selection rule can be right in isolation while a code path
that never calls it puts the same sentence on the card.

So the check is on the finished board, not on the function. Every point and
fact a reader would see is offered back to `summary.is_colour`, and the run
fails if any of them is scene-setting. That is deliberately the same predicate
the card filter uses: what is being tested is that nothing bypasses it — the
feed-summary fallback, the cache, a genre built by a different path.

    python -m mediatop.livecheck
    python -m mediatop.livecheck --per-genre 4 --timeout 20

Exit status is 0 when nothing leaked, 1 when something did or when no story
could be read at all. A publisher refusing one page is reported and not
counted as a failure: a 403 is their morning, not our bug.
"""

import argparse
import sys

from . import extract, summary
from .server import Board


def inspect(payload):
    """Problems found on a built board, and how many facts were examined.

    Each problem is `(where, what)`: the genre and headline it was found on,
    and the rule it broke.
    """
    problems = []
    facts = 0
    for genre in payload["genres"]:
        for card in genre["cards"]:
            where = f"{genre['key']}/{card['source']}: {card['title'][:70]}"
            if not card["point"]:
                problems.append((where, "the point is empty"))
            if extract.is_boilerplate(card["point"]):
                problems.append((where, f"furniture as the point: {card['point'][:90]}"))
            for fact in card["facts"]:
                facts += 1
                if extract.is_boilerplate(fact):
                    problems.append((where, f"furniture as a fact: {fact[:90]}"))
                if summary.is_colour(fact):
                    problems.append((where, f"colour as a fact: {fact[:90]}"))
    return problems, facts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-genre", type=int, default=3,
                        help="stories per genre to check (default 3)")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)

    board = Board(per_genre=args.per_genre, timeout=args.timeout)
    try:
        payload = board.board(refresh=True)
    finally:
        board.close()

    counts = payload["counts"]
    problems, facts = inspect(payload)
    print(f"{counts['stories']} stories from {counts['sources']} sources, "
          f"{facts} facts, {sum(len(g['cards']) for g in payload['genres'])} cards")

    # Reported, never failed on: a paywall or a 403 is the publisher's morning,
    # and the board already says so on the card and in its failures list.
    for failure in payload["failures"]:
        print(f"  ! {failure['genre']}/{failure['source']}: {failure['error'][:100]}")

    if not counts["stories"]:
        print("no story could be read, so nothing was checked", file=sys.stderr)
        return 1

    for where, what in problems:
        print(f"  ✗ {where}\n      {what}", file=sys.stderr)
    if problems:
        print(f"{len(problems)} problem(s) on today's board", file=sys.stderr)
        return 1

    print("no colour and no page furniture reached a point or a fact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
