#!/usr/bin/env python3
"""
Is this environment still an interpreter, or a directory that used to hold one?

A venv fails in a way that reads like something else. It is created with a link
to an interpreter under the uv cache, and `pyvenv.cfg` records where that
interpreter lived. When the cache is rebuilt under a different home — a
different `$HOME`, a different user, a container that came back with another
uid — the link has nothing at the end of it, `.venv/bin/python` is gone, and the
first thing that says so is whatever command came next. On this machine that was
a shell answering `No such file or directory` ten minutes into a test run, after
the venv had been reported as created.

The other form is quieter: everything is in place and the interpreter is the
wrong one, so it runs as the system Python and reports a missing package rather
than a missing environment.

So: check the link, check that the interpreter it points at exists, and ask the
interpreter itself where it thinks it is. That last one is the only way to tell
a venv from a directory shaped like one.

    python3 venv_check.py                     # the environments that exist
    python3 venv_check.py .venv               # one of them, named
    python3 venv_check.py provenance_lab/.venv

Standard library only, and it has to be runnable by the system `python3` rather
than by a venv: the failure it is looking for is one where the venv's own
interpreter cannot be executed at all. It exits 0 when every environment it
checked is usable, 1 otherwise, naming the commands that rebuild what is not.

On Windows this reports a Linux venv as unusable, which is true from there —
`.venv/bin/python` is a link to a path only WSL can follow. Run it from inside
WSL, where the environments were made.
"""

import argparse
import glob
import os
import subprocess
import sys

#: The environments the READMEs document: the root one for torch and the
#: training scripts, the lab's for c2pa and its own pins. Different files, so
#: different environments, and either can rot on its own.
STANDARD_VENVS = (".venv", os.path.join("provenance_lab", ".venv"))

#: Printed by the interpreter under test: where it thinks it is, and what it is.
#: `sys.prefix` is the answer to "is this venv actually being used", because a
#: Python started from a directory with a `bin/` and no `pyvenv.cfg` behind it
#: reports the prefix it was built with instead.
PROBE = "import sys; print(sys.prefix); print(sys.version.split()[0])"

#: A venv that cannot answer in twenty seconds is not slow, it is broken.
TIMEOUT = 20


class Report:
    """What one environment turned out to be."""

    def __init__(self, path):
        self.path = path
        self.ok = False
        self.problem = None
        self.version = None
        self.packages = 0
        # False only when rebuilding this path would be the wrong advice — a
        # Linux venv looked at from Windows, where the fix is to change the
        # shell rather than to rewrite the environment.
        self.rebuild = True


def read_pyvenv_cfg(venv):
    """The `key = value` lines of a venv's `pyvenv.cfg`, or None if there is none.

    None is not an error to hide: a directory without one is not an
    environment, and saying so is more useful than probing it.
    """
    try:
        with open(os.path.join(venv, "pyvenv.cfg"), encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None

    settings = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            settings[key.strip().lower()] = value.strip()
    return settings


def python_executable(venv):
    """Where this environment keeps its interpreter, POSIX layout first."""
    posix = os.path.join(venv, "bin", "python")
    if os.path.lexists(posix):
        return posix
    windows = os.path.join(venv, "Scripts", "python.exe")
    if os.path.lexists(windows):
        return windows
    # Neither is there; return the one the message should name.
    return posix


def count_packages(venv):
    """How many distributions are installed, by metadata directory.

    Not a substitute for the version check — it answers the other question, "was
    this environment ever filled in", which is what an interrupted install
    leaves behind.
    """
    found = set()
    for pattern in ("lib/*/site-packages/*.dist-info",
                    "Lib/site-packages/*.dist-info"):
        for entry in glob.glob(os.path.join(venv, pattern)):
            found.add(os.path.basename(entry))
    return len(found)


def probe_interpreter(python):
    """Ask the interpreter what it is: `(prefix, version, error)`.

    Any of the three can be None, and exactly one of `version`/`error` is set.
    """
    try:
        done = subprocess.run(
            [python, "-c", PROBE], capture_output=True, text=True,
            timeout=TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        # A dangling link raises FileNotFoundError here rather than at the
        # caller's first command, which is the whole reason this exists.
        return None, None, str(exc)

    if done.returncode != 0:
        lines = done.stderr.strip().splitlines()
        return None, None, lines[0] if lines else f"exit {done.returncode}"

    lines = done.stdout.strip().splitlines()
    if len(lines) < 2:
        return None, None, f"it printed {done.stdout.strip()!r}"
    return lines[0], lines[1], None


def check(venv, probe=None, platform=None):
    """Look at one environment and say whether anything can be run from it.

    The probe is resolved here rather than in the signature, so a test can
    replace the module attribute and have `main` pick the replacement up too.
    `platform` is `os.name`, as a parameter because the one case that needs to
    be recognised by name is a Linux venv looked at from Windows, and a test for
    it should not have to claim to be Windows to the whole standard library.
    """
    if probe is None:
        probe = probe_interpreter
    if platform is None:
        platform = os.name
    report = Report(venv)

    if not os.path.isdir(venv):
        report.problem = "there is no environment here"
        return report

    settings = read_pyvenv_cfg(venv)
    if settings is None:
        report.problem = "there is no pyvenv.cfg, so this is not an environment"
        return report

    home = settings.get("home", "")
    if home and not os.path.isdir(home):
        report.problem = f"it was built against {home}, which is not on this machine"
        if platform == "nt" and home.startswith("/"):
            # Not stale, just read from the wrong side: rebuilding it here would
            # replace a working Linux venv with a Windows one.
            report.problem += "; it is a Linux environment, so run this from inside WSL"
            report.rebuild = False
        return report

    python = python_executable(venv)
    if not os.path.lexists(python):
        report.problem = f"there is no interpreter at {python}"
        return report
    if not os.path.exists(python):
        target = os.readlink(python) if os.path.islink(python) else python
        report.problem = (f"{python} is a link to {target}, "
                          "and that interpreter is not there")
        return report

    prefix, version, error = probe(python)
    if error is not None:
        report.problem = f"its interpreter would not start: {error}"
        return report
    if os.path.realpath(prefix) != os.path.realpath(os.path.abspath(venv)):
        report.problem = (f"it runs as {prefix} rather than "
                          f"{os.path.abspath(venv)}: this interpreter is not "
                          "using the environment")
        return report

    report.ok = True
    report.version = version
    report.packages = count_packages(venv)
    return report


def rebuild_hint(venv):
    """The two commands that would replace this environment, as the docs give them.

    Written out rather than pointed at, because the reader has just been told
    something they did not want to hear and should not have to go looking for
    the incantation. The lab installs from its own pin set and from PyPI; the
    root one resolves against the CPU torch index, which is why the URL is
    there and why the two commands are not quite the same.
    """
    parts = venv.replace(os.sep, "/").split("/")
    requirements = ("provenance_lab/requirements.txt" if "provenance_lab" in parts
                    else "requirements.txt")
    install = f"uv pip install --python {venv}/bin/python -r {requirements}"
    if "provenance_lab" not in parts:
        install += " --index-url https://download.pytorch.org/whl/cpu"
    return [f"uv venv --clear --python 3.12 {venv}", install]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check that this repo's environments can still run anything.")
    parser.add_argument(
        "venvs", nargs="*", metavar="VENV",
        help="environment directories to check (default: whichever of "
             + ", ".join(STANDARD_VENVS) + " exist)")
    args = parser.parse_args(argv)

    targets = list(args.venvs)
    if not targets:
        # Only the ones that exist, so a checkout that never needed the lab's
        # environment is not red for not having made one. A path named on the
        # command line is checked whether or not it is there.
        targets = [path for path in STANDARD_VENVS if os.path.isdir(path)]
        if not targets:
            print(f"no environment here; looked in {', '.join(STANDARD_VENVS)}",
                  file=sys.stderr)
            print("see README.md for the commands that create one", file=sys.stderr)
            return 1

    reports = [check(venv) for venv in targets]
    width = max(len(report.path) for report in reports)

    for report in reports:
        if report.ok:
            packages = (f"{report.packages} packages" if report.packages
                        else "no packages installed")
            print(f"{report.path:<{width}}  ok      {report.version}, {packages}")

    broken = [report for report in reports if not report.ok]
    for report in broken:
        print(f"{report.path:<{width}}  BROKEN  {report.problem}", file=sys.stderr)
        if report.rebuild:
            print("  rebuild it:", file=sys.stderr)
            for line in rebuild_hint(report.path):
                print(f"    {line}", file=sys.stderr)

    if broken:
        print(f"\n{len(broken)} of {len(reports)} environments cannot run anything",
              file=sys.stderr)
        return 1

    print(f"\n{len(reports)} environment{'' if len(reports) == 1 else 's'} ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
