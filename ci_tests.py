#!/usr/bin/env python3
"""
Run a unittest suite the way CI wants it: a test that did not run is a failure.

`python -m unittest` reports a suite that skipped itself as a pass. A module
raises SkipTest because a dependency is missing, unittest records one skip,
nothing runs, and the exit status is 0 — so the step is green and its log is the
only place the difference shows. That has happened here: the disk tests were
green through a run in which the SDK they import was never installed.

The reason a suite skips is nearly always an install that did not do what the
workflow said it did, not a machine that cannot support the test. So this runs
the suite the way the unittest CLI would, and then fails if anything skipped, or
if nothing ran at all.

    python ci_tests.py test_all test_peek
    python ci_tests.py --discover .        # every test module under the cwd

The skips are named in a GitHub `::error::` annotation, so the reason is on the
run summary instead of half a log down — the log needs admin rights to download,
the summary does not.
"""

import argparse
import sys
import unittest

#: GitHub reads this prefix on a line of output as an annotation on the run.
ANNOTATION = "::error::"


def build(loader, names, discover, pattern):
    """The suite for the given names, or everything found under `discover`."""
    if discover is not None:
        return loader.discover(discover, pattern=pattern)
    return unittest.TestSuite(loader.loadTestsFromNames(names))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a unittest suite and fail if any test skipped.")
    parser.add_argument(
        "names", nargs="*", metavar="TEST",
        help="dotted module or test name, as `python -m unittest` takes it")
    parser.add_argument(
        "--discover", metavar="DIR",
        help="discover test modules under DIR instead of naming them")
    parser.add_argument(
        "--pattern", default="test*.py",
        help="discovery pattern (default: %(default)s)")
    args = parser.parse_args(argv)

    if args.discover is not None and args.names:
        parser.error("give test names or --discover, not both")
    if args.discover is None and not args.names:
        parser.error("give at least one test name, or --discover")

    suite = build(unittest.defaultTestLoader, args.names, args.discover,
                  args.pattern)
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    skipped = [test.id() for test, _ in result.skipped]
    if skipped:
        print(f"{ANNOTATION}{len(skipped)} skipped, so this suite did not run: "
              f"{', '.join(skipped)}", file=sys.stderr)
    # An empty suite is successful by unittest's reckoning, and a discovery
    # pattern that matches nothing looks exactly like a pass.
    if not result.testsRun:
        print(f"{ANNOTATION}no tests ran", file=sys.stderr)

    return 0 if result.wasSuccessful() and not skipped and result.testsRun else 1


if __name__ == "__main__":
    sys.exit(main())
