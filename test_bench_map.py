"""The bench map page, checked against the repo it describes.

The page's whole value is that it is not a second source of truth: it is the
README table, rendered. So most of what is worth testing is that agreement --
a bench in the table has a tile, a tile has a bench, and the count on a tile is
the count in that bench's folder rather than a number typed once -- plus the two
things a generated page gets wrong silently: an unescaped cell, and a claim of
CI coverage that no job actually runs.

The rest is the parser's edges, because the README is edited by hand and a cell
will eventually be written differently: a status of "Parked 28 Oct" is parked, a
status nobody has used yet is ink rather than a crash, and a missing workflow
file means no coverage rather than an exception.

No network, no torch, no venv: this reads three text files and renders strings.
"""

import html
import re
import unittest
from pathlib import Path

import bench_map

ROOT = Path(__file__).resolve().parent


class ParsesTheReadme(unittest.TestCase):
    def setUp(self):
        self.benches = bench_map.gather()

    def test_finds_every_bench_in_the_table(self):
        names = [bench.name for bench in self.benches]
        self.assertEqual(len(names), 5, names)
        self.assertIn("othello", names)
        self.assertIn("mediatop", names)

    def test_keeps_the_order_the_readme_uses(self):
        # The page is a map, so its order should be the one the reader has
        # already seen in the README rather than alphabetical.
        self.assertEqual([bench.name for bench in self.benches][0], "the model")

    def test_separates_name_from_subtitle(self):
        othello = next(b for b in self.benches if b.name == "othello")
        self.assertEqual(othello.subtitle, "the engine")
        self.assertEqual(othello.link, "othello/README.md")
        self.assertEqual(othello.directory, "othello")

    def test_the_root_bench_has_no_directory(self):
        model = next(b for b in self.benches if b.name == "the model")
        self.assertEqual(model.directory, "")
        # Its first link is a source file, not a README: the bench is the repo.
        self.assertNotIn("/", model.link)

    def test_a_cell_of_filenames_is_not_a_subtitle(self):
        # That cell lists model.py, train.py and so on; a tile is not a table
        # of contents, so the line is dropped rather than printed as prose.
        model = next(b for b in self.benches if b.name == "the model")
        self.assertEqual(model.subtitle, "")
        # The empty line stays in the markup so the fact rows of every tile sit
        # at the same height; it is blank, not missing.
        self.assertIn('<p class="where"></p>', bench_map.render([model]))

    def test_every_bench_has_a_question_and_a_status(self):
        for bench in self.benches:
            self.assertTrue(bench.question.endswith("?"), bench.name)
            self.assertTrue(bench.status, bench.name)

    def test_markdown_is_not_left_in_the_text(self):
        for bench in self.benches:
            for value in (bench.name, bench.subtitle, bench.question, bench.status):
                self.assertNotIn("**", value)
                self.assertNotIn("](", value)

    def test_a_missing_table_is_a_message_not_a_traceback(self):
        with self.assertRaises(SystemExit) as raised:
            bench_map.readme_benches("# Nothing here\n")
        self.assertIn("benches", str(raised.exception))


class CountsWhatIsThere(unittest.TestCase):
    def test_othello_tests_match_its_file(self):
        othello = next(b for b in bench_map.gather() if b.name == "othello")
        found = (ROOT / "othello" / "test_othello.py").read_text(
            encoding="utf-8").count("def test_")
        self.assertEqual(othello.tests, found)
        self.assertGreater(othello.tests, 15)

    def test_root_bench_counts_only_its_named_suite(self):
        # The root holds several suites; counting them all as the model's
        # would inflate the tile with tests the model does not have.
        model = next(b for b in bench_map.gather() if b.name == "the model")
        expected = (ROOT / "test_all.py").read_text(encoding="utf-8").count(
            "def test_")
        self.assertEqual(model.tests, expected)
        self.assertLess(model.tests, expected + 1)

    def test_a_bench_with_no_tests_says_zero(self):
        self.assertEqual(bench_map.count_tests("does_not_exist"), 0)

    def test_every_bench_reports_some_tests(self):
        for bench in bench_map.gather():
            self.assertGreater(bench.tests, 0, bench.name)


class KnowsItsCi(unittest.TestCase):
    def test_jobs_are_parsed_from_the_workflow(self):
        jobs = bench_map.workflow_jobs()
        self.assertIn("othello", jobs)
        self.assertIn("mediatop", jobs)

    def test_a_named_module_gives_the_job(self):
        bench = next(b for b in bench_map.gather() if b.name == "othello")
        self.assertEqual(bench.job, "othello")

    def test_a_module_no_job_runs_is_not_covered(self):
        bench = bench_map.Bench(name="ghost", subtitle="", link="", directory="",
                                question="?", status="Active", tests=1)
        bench = bench_map.Bench(**{**bench.__dict__,
                                   "directory": "ghost_bench"})
        self.assertEqual(bench.job, "")
        self.assertEqual(bench_map.ci_job(bench, bench_map.workflow_jobs()), "")

    def test_a_missing_workflow_is_no_coverage(self):
        self.assertEqual(bench_map.workflow_jobs(Path("no-such-workflow.yml")), {})

    def test_coverage_needs_the_exact_module_name(self):
        jobs = {"fuzzy": "run: python ci_tests.py othello.test_othello_extra"}
        bench = next(b for b in bench_map.gather() if b.name == "othello")
        self.assertEqual(bench_map.ci_job(bench, jobs), "")

    def test_a_discovery_step_counts_for_the_folder_it_runs_in(self):
        # How the lab is covered: it names no module, only `--discover .` from
        # its own directory. Missing that said "no CI job" about a bench that
        # has one, which is the same kind of wrong as inventing coverage.
        lab = next(b for b in bench_map.gather() if b.name == "provenance_lab")
        self.assertEqual(lab.job, "provenance-lab")

    def test_discovery_in_another_folder_does_not_cover_this_bench(self):
        jobs = {"other": "working-directory: somewhere_else\n  run: python ../ci_tests.py --discover ."}
        lab = next(b for b in bench_map.gather() if b.name == "provenance_lab")
        self.assertEqual(bench_map.ci_job(lab, jobs), "")


class RendersAPage(unittest.TestCase):
    def setUp(self):
        self.benches = bench_map.gather()
        self.page = bench_map.render(self.benches)

    def test_one_tile_per_bench(self):
        self.assertEqual(self.page.count('class="tile"'), len(self.benches))

    def test_each_tile_links_to_its_readme(self):
        for bench in self.benches:
            self.assertIn(f'href="{bench.target}"', self.page)
            self.assertTrue(bench.target.endswith("README.md"), bench.name)

    def test_the_root_bench_links_to_the_readme_that_documents_it(self):
        # Its table cell links to model.py first; a map should not send a
        # reader into a source file when the prose is right there.
        model = next(b for b in self.benches if b.name == "the model")
        self.assertEqual(model.link, "model.py")
        self.assertEqual(model.target, "README.md")
        self.assertNotIn('href="model.py"', self.page)

    def test_carries_the_question_it_was_given(self):
        for bench in self.benches:
            # Escaped the way the page escapes it, apostrophes included.
            self.assertIn(html.escape(bench.question), self.page)

    def test_is_self_contained(self):
        # No stylesheet, font, script, image or link may reach off the machine:
        # the page has to open from a checkout with nothing behind it.
        self.assertNotIn("<script", self.page)
        self.assertNotIn("<link", self.page)
        self.assertNotIn("<img", self.page)
        self.assertIsNone(re.search(r'(src|href)="https?:', self.page))

    def test_a_new_status_word_does_not_crash(self):
        odd = bench_map.Bench(name="n", subtitle="s", link="l", directory="",
                              question="q?", status="Snoozing", tests=3)
        page = bench_map.render([odd])
        self.assertIn("Snoozing", page)
        self.assertIn(bench_map.UNKNOWN_COLOUR, page)

    def test_status_colour_ignores_a_date_after_the_word(self):
        self.assertEqual(bench_map.status_colour("Parked 28 Oct"),
                         bench_map.COLOURS["parked"])
        self.assertEqual(bench_map.status_colour("Live — Actions daily"),
                         bench_map.COLOURS["live"])

    def test_an_empty_status_uses_ink(self):
        self.assertEqual(bench_map.status_colour(""), bench_map.UNKNOWN_COLOUR)

    def test_counts_in_the_header_match_the_tiles(self):
        parked = sum(1 for bench in self.benches
                     if bench.status.split()[0].lower() == "parked")
        self.assertIn(f"{parked} parked", self.page)


class EscapesUntrustedCells(unittest.TestCase):
    """README text reaches HTML, and a README is edited by hand."""

    def test_angle_brackets_and_quotes_are_escaped(self):
        nasty = bench_map.Bench(
            name='<script>alert("x")</script>', subtitle="a & b",
            link="ok/README.md", directory="", question="<b>why?</b>",
            status="Active", tests=1)
        page = bench_map.render([nasty])
        self.assertNotIn("<script>alert", page)
        self.assertNotIn("<b>why?</b>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("a &amp; b", page)

    def test_a_quote_in_a_link_cannot_break_out_of_the_attribute(self):
        nasty = bench_map.Bench(
            name="n", subtitle="", link='x" onmouseover="alert(1)',
            directory="", question="q?", status="Active", tests=1)
        page = bench_map.render([nasty])
        self.assertNotIn('onmouseover="alert(1)"', page)


class WritesAFile(unittest.TestCase):
    def test_main_writes_the_page_and_reports_the_benches(self):
        import contextlib
        import io
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "map.html"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(bench_map.main(["--out", str(target)]), 0)
            self.assertTrue(target.is_file())
            self.assertIn("<!doctype html>", target.read_text(encoding="utf-8"))
            self.assertIn("othello", out.getvalue())

    def test_json_arm_is_machine_readable(self):
        import contextlib
        import io
        import json

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(bench_map.main(["--json"]), 0)
        rows = json.loads(out.getvalue())
        self.assertEqual(len(rows), 5)
        self.assertIn("question", rows[0])


if __name__ == "__main__":
    unittest.main()
