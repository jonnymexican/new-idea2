"""demo.py: the one command that measures, builds the page and serves it.

The serving half is a launcher around `http.server`, so what is pinned here is
the half that produces files: the local arm really runs, and the page really
comes out of the report it wrote. Both use a temporary directory, so the demo's
own `out/` is never touched and a test run leaves no report behind.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import demo


class TestDemoBuild(unittest.TestCase):

    def test_one_command_measures_the_arm_and_builds_the_page(self):
        """The whole path: sign, serve locally, verify, report, render."""
        with tempfile.TemporaryDirectory() as tmp:
            out, site = Path(tmp) / "demo", Path(tmp) / "site"
            # The arm prints its Markdown report; a test does not need it.
            with contextlib.redirect_stdout(io.StringIO()):
                code = demo.main(["--out", str(out), "--site", str(site),
                                  "--no-serve"])

            self.assertEqual(code, 0)
            self.assertTrue((out / "platforms.json").exists(),
                            "the arm wrote no report")
            page = (site / "index.html").read_text(encoding="utf-8")
            self.assertIn("<code>local</code>", page)
            history = json.loads((site / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(history["runs"]), 1)
            # `copy` is the control whose bytes are identical, so this is the
            # column the page must record as intact.
            self.assertEqual(
                history["runs"][0]["platforms"]["local"]["outcome"], "valid")

    def test_each_measurement_adds_a_column(self):
        """The page is a series, so building it twice is two columns rather
        than one replaced — which is what makes the demo worth running again."""
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site"
            report = Path(tmp) / "platforms.json"
            for when in ("2026-01-01T00:00:00+00:00", "2026-01-08T00:00:00+00:00"):
                report.write_text(json.dumps({
                    "meta": {"generated": when},
                    "rows": [{"platform": "local", "variant": "copy",
                              "verdict": "valid", "served_format": "jpeg",
                              "served_bytes": 100, "sha256_match": True}],
                }), encoding="utf-8")
                demo.build_page(report, site)

            history = json.loads((site / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(history["runs"]), 2)
            page = (site / "index.html").read_text(encoding="utf-8")
            self.assertIn("01 Jan", page)
            self.assertIn("08 Jan", page)
            self.assertIn("credential intact", page)


if __name__ == "__main__":
    unittest.main(verbosity=2)
