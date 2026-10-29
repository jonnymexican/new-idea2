"""The environment check, against environments built out of nothing.

The check exists because a venv can be a directory shaped exactly like a working
one — `pyvenv.cfg`, a `bin/`, a `python` in it — while nothing in it can be
executed. So these tests build that shape in a temporary directory, break one
thing at a time, and assert the check names the thing that is broken.

No real venv is created, and no real interpreter is asked anything: the probe is
replaced with a function returning what an interpreter would have said. Building
a venv per test would mean running `venv` a dozen times to test a check whose
whole job is to be runnable when that is exactly what is not working.

The one thing not faked is the filesystem: dangling links, missing
`pyvenv.cfg`, empty site-packages are the failure modes, so they are real files
in a temporary directory.
"""

import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import venv_check


class Case(unittest.TestCase):
    """A temporary directory, and a helper that builds a venv shape in it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def make_venv(self, name=".venv", home=None, interpreter=True,
                  packages=0):
        """A directory that looks like a venv, with what the arguments allow.

        `interpreters` are plain files rather than links to a real Python: the
        probe is faked, and what is under test is whether the check notices the
        file is absent, dangling, or the wrong interpreter once asked.
        """
        venv = os.path.join(self.tmp.name, name)
        bin_dir = os.path.join(venv, "bin")
        os.makedirs(bin_dir)

        if home is None:
            # Shared between the venvs in one test, so it may already be there.
            home = os.path.join(self.tmp.name, "python-home", "bin")
            os.makedirs(home, exist_ok=True)
        with open(os.path.join(venv, "pyvenv.cfg"), "w", encoding="utf-8") as handle:
            handle.write(
                f"home = {home}\n"
                "implementation = CPython\n"
                "version_info = 3.12\n"
                "include-system-site-packages = false\n"
            )

        if interpreter:
            open(os.path.join(bin_dir, "python"), "w").close()

        site = os.path.join(venv, "lib", "python3.12", "site-packages")
        os.makedirs(site)
        for index in range(packages):
            os.makedirs(os.path.join(site, f"package{index}-1.0.dist-info"))

        return venv

    def as_itself(self, version="3.12.16"):
        """A probe that answers as an interpreter inside the venv it is given."""
        def probe(python):
            venv = os.path.dirname(os.path.dirname(python))
            return venv, version, None
        return probe


class HealthyTest(Case):
    def test_a_working_environment_is_ok_and_reports_its_version(self):
        sha = self.make_venv(packages=11)
        report = venv_check.check(sha, probe=self.as_itself("3.12.16"))

        self.assertTrue(report.ok)
        self.assertEqual(report.version, "3.12.16")
        # Counted from metadata directories, which is what an interpreter would
        # say too, and is the cheap way to spot an install that never finished.
        self.assertEqual(report.packages, 11)

    def test_the_probe_is_asked_the_interpreter_in_that_directory(self):
        sha = self.make_venv()
        asked = []

        def probe(python):
            asked.append(python)
            return sha, "3.12.16", None

        venv_check.check(sha, probe=probe)
        self.assertEqual(asked, [os.path.join(sha, "bin", "python")])

    def test_a_directory_can_be_reached_through_a_symlink(self):
        sha = self.make_venv()
        # `uv` and `pwd` disagree about /mnt/c paths; a prefix that resolves to
        # the same place is the same environment.
        link = os.path.join(self.tmp.name, "elsewhere")
        os.symlink(sha, link)

        def probe(python):
            return link, "3.12.16", None

        self.assertTrue(venv_check.check(os.path.join(link, ""), probe=probe).ok)


class BrokenTest(Case):
    def test_a_directory_that_is_not_there(self):
        report = venv_check.check(os.path.join(self.tmp.name, "absent"),
                                  probe=self.as_itself())
        self.assertFalse(report.ok)
        self.assertIn("no environment here", report.problem)

    def test_a_directory_that_is_not_an_environment(self):
        venv = os.path.join(self.tmp.name, "not-a-venv")
        os.makedirs(venv)
        report = venv_check.check(venv, probe=self.as_itself())
        self.assertFalse(report.ok)
        self.assertIn("pyvenv.cfg", report.problem)

    def test_the_home_it_was_built_against_has_gone(self):
        # The observed failure: the recorder of a home directory that no longer
        # exists, because the cache was rebuilt under another one.
        gone = os.path.join(self.tmp.name, "home", "jvoigt", ".local", "share",
                            "uv", "python", "cpython-3.12-linux-x86_64-gnu", "bin")
        sha = self.make_venv(home=gone)
        report = venv_check.check(sha, probe=self.as_itself())

        self.assertFalse(report.ok)
        self.assertIn(gone, report.problem)
        self.assertIn("not on this machine", report.problem)
        # Rebuilding is the fix, so the check is allowed to say so.
        self.assertTrue(report.rebuild)

    def test_no_interpreter_at_all(self):
        # `uv venv` can leave the config behind and then fail to place the link.
        sha = self.make_venv(interpreter=False)
        report = venv_check.check(sha, probe=self.as_itself())
        self.assertFalse(report.ok)
        self.assertIn("no interpreter at", report.problem)

    def test_a_link_whose_interpreter_was_removed(self):
        sha = self.make_venv(interpreter=False)
        target = os.path.join(self.tmp.name, "python-home", "bin", "python3.12")
        os.symlink(target, os.path.join(sha, "bin", "python"))
        report = venv_check.check(sha, probe=self.as_itself())

        self.assertFalse(report.ok)
        # Naming the target matters: it is the path to go and look at.
        self.assertIn(target, report.problem)

    def test_an_interpreter_that_will_not_start(self):
        sha = self.make_venv()

        def probe(python):
            return None, None, "Exec format error"

        report = venv_check.check(sha, probe=probe)
        self.assertFalse(report.ok)
        self.assertIn("would not start", report.problem)
        self.assertIn("Exec format error", report.problem)

    def test_an_interpreter_that_is_not_using_the_environment(self):
        # Everything is present and the thing answering is the system Python:
        # nothing can be imported, and the error reads like a missing package.
        sha = self.make_venv()
        report = venv_check.check(sha, probe=lambda python: ("/usr", "3.12.3", None))

        self.assertFalse(report.ok)
        self.assertIn("/usr", report.problem)
        self.assertIn("not using the environment", report.problem)

    def test_a_linux_environment_read_from_windows(self):
        # True from Windows and not a broken venv: the fix is a different shell,
        # so the check must not offer to rebuild over a working environment.
        sha = self.make_venv(home="/home/kali/.local/share/uv/python/bin")
        report = venv_check.check(sha, probe=self.as_itself(), platform="nt")

        self.assertFalse(report.ok)
        self.assertIn("inside WSL", report.problem)
        self.assertFalse(report.rebuild)


class PackagesTest(Case):
    def test_an_environment_with_nothing_installed_still_counts(self):
        sha = self.make_venv(packages=0)
        report = venv_check.check(sha, probe=self.as_itself())
        self.assertTrue(report.ok)
        self.assertEqual(report.packages, 0)

    def test_counts_distributions_across_both_layouts(self):
        sha = self.make_venv(packages=3)
        # A Windows-shaped tree in the same directory: counted, not confused
        # with the POSIX one, and not counted twice.
        other = os.path.join(sha, "Lib", "site-packages")
        os.makedirs(other)
        for name in ("package0-1.0.dist-info", "extra-2.0.dist-info"):
            os.makedirs(os.path.join(other, name))
        self.assertEqual(venv_check.count_packages(sha), 4)


class HintTest(unittest.TestCase):
    def test_the_root_environment_installs_against_the_cpu_index(self):
        lines = venv_check.rebuild_hint(".venv")
        self.assertEqual(len(lines), 2)
        self.assertIn("uv venv --clear --python 3.12 .venv", lines[0])
        self.assertIn("-r requirements.txt", lines[1])
        # The root file only resolves against the CPU index; PyPI's Linux torch
        # is the CUDA build.
        self.assertIn("download.pytorch.org/whl/cpu", lines[1])

    def test_the_lab_environment_installs_from_its_own_pins(self):
        lines = venv_check.rebuild_hint(os.path.join("provenance_lab", ".venv"))
        self.assertIn("-r provenance_lab/requirements.txt", lines[1])
        # Its pins are on PyPI, and the 403ing index would fail the install.
        self.assertNotIn("download.pytorch.org", lines[1])


class MainTest(Case):
    def run_main(self, argv, probe=None):
        """`main` as the shell runs it, with the interpreter's answers faked.

        Nothing here is executable, so without the fake every environment would
        fail at the same step and the exit codes would prove nothing.
        """
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(venv_check, "probe_interpreter",
                               probe or self.as_itself()):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = venv_check.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_zero_when_every_named_environment_is_usable(self):
        report = self.make_venv()
        code, out, err = self.run_main([report])
        self.assertEqual(code, 0, err)
        self.assertIn("ok", out)
        self.assertIn("1 environment ok", out)

    def test_one_when_a_named_environment_is_broken(self):
        report = self.make_venv(interpreter=False)
        code, out, err = self.run_main([report])
        self.assertEqual(code, 1)
        self.assertIn("BROKEN", err)
        # The remedy is on stderr with the failure, not somewhere else.
        self.assertIn("rebuild it", err)
        self.assertIn("uv venv --clear --python 3.12", err)

    def test_matches_by_how_many_are_broken(self):
        good = self.make_venv(name="good")
        bad = self.make_venv(name="bad", interpreter=False)
        code, out, err = self.run_main([good, bad])
        self.assertEqual(code, 1)
        self.assertIn("1 of 2 environments", err)

    def test_a_named_environment_that_does_not_exist_is_broken(self):
        code, out, err = self.run_main([os.path.join(self.tmp.name, "absent")])
        self.assertEqual(code, 1)
        self.assertIn("no environment here", err)

    def test_no_environments_anywhere_is_a_failure_that_names_them(self):
        # The default is "whichever of the documented ones exist"; with none,
        # there is nothing to check and nothing to run, which is the answer.
        with mock.patch.object(venv_check, "STANDARD_VENVS",
                               (".venv", "provenance_lab/.venv")):
            os.chdir(self.tmp.name)
            self.addCleanup(os.chdir, os.getcwd())
            code, out, err = self.run_main([])

        self.assertEqual(code, 1)
        self.assertIn(".venv", err)
        self.assertIn("README.md", err)

    def test_the_default_skips_the_environment_that_was_never_made(self):
        self.make_venv(name=".venv", interpreter=False)
        with mock.patch.object(venv_check, "STANDARD_VENVS",
                               (".venv", os.path.join(self.tmp.name, "absent"))):
            os.chdir(self.tmp.name)
            self.addCleanup(os.chdir, os.getcwd())
            code, out, err = self.run_main([])

        # Checked because it is there — and reported, since it has no
        # interpreter — while the one that was never made is not mentioned.
        self.assertEqual(code, 1)
        self.assertIn(".venv", err)
        self.assertNotIn("absent", err)


if __name__ == "__main__":
    unittest.main()
