#!/usr/bin/env python3
"""
The bench map, made bright: one page that says what this repo is.

The README answers "what is this repo?" with a five-row table, and for a month
the answer kept being rediscovered from scratch. A table is the right shape for
the answer and the wrong shape for noticing it, so this writes the same five
benches as a single page you can look at: the question each one asks, printed
large, under a colour that says whether the thing is alive.

It is generated *from* the README table rather than written beside it. A page
kept by hand drifts the moment a bench is parked, and then there are two
answers to the same question and no way to tell which is stale. Here the README
is the source, the page is the rendering, and a test compares them: a bench in
the table with no tile fails, and so does a tile with no row.

Every number on a tile comes from the repo, not from prose:

- the test count is `def test_` counted in that bench's own test files, so a
  suite that was deleted cannot keep its number;
- "on every push" is claimed only where the tests workflow names that bench's
  module in a job, so a job that was renumbered or removed turns the claim off
  by itself. A bench with no job says "no CI job" instead of leaving a gap that
  reads like one.

The page is one file with no external stylesheet, font, script or image, so it
opens from a checkout, survives being emailed, and cannot rot when a CDN moves.

    python bench_map.py                    # writes bench-map.html
    python bench_map.py --out /tmp/map.html
"""

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"

#: The model bench lives at the repo root, which also holds other suites, so
#: its tests are named rather than globbed -- otherwise every root suite would
#: be counted as the model's.
MODEL_SUITE = "test_all.py"

#: Status word -> accent. The first word of the "where it stands" cell decides
#: the colour, so "Parked 28 Oct" and "Parked" mean the same thing to the page.
COLOURS = {
    "active": "#1f9d55",
    "live": "#0b74c4",
    "new": "#7c3aed",
    "quiet": "#c28800",
    "parked": "#8a8580",
}
UNKNOWN_COLOUR = "#14110f"


@dataclass(frozen=True)
class Bench:
    """One row of the README table, plus what the repo says about it."""

    name: str
    subtitle: str
    link: str
    directory: str  # "" for the root bench
    question: str
    status: str
    tests: int = 0
    job: str = ""

    @property
    def colour(self):
        return status_colour(self.status)

    @property
    def target(self):
        """Where the tile goes: a bench's own README, or the one that is it.

        The model bench is documented by the root README rather than by a file
        of its own, so the cell's first link is `model.py`. Sending a reader to
        a source file from a map is not what the map is for.
        """
        return self.link if self.link.endswith("README.md") else "README.md"

    @property
    def modules(self):
        """Dotted test module names this bench's suite is reached by."""
        if not self.directory:
            return [MODEL_SUITE.removesuffix(".py")]
        return [f"{self.directory}.{path.stem}"
                for path in sorted((ROOT / self.directory).glob("test_*.py"))]


def status_colour(status):
    """The accent for a status cell: its first word, or ink for an odd one."""
    word = re.sub(r"[^a-z]", "", status.split()[0].lower()) if status.strip() else ""
    return COLOURS.get(word, UNKNOWN_COLOUR)


def plain(text):
    """Markdown emphasis, links and code out of a cell, for measuring it."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return text.replace("**", "").replace("`", "").strip()


def links(cell):
    """Every link target in a cell, in order."""
    return re.findall(r"\[[^\]]*\]\(([^)]+)\)", cell)


def readme_benches(text):
    """Parse the benches table out of the README's markdown.

    Deliberately narrow: the table under "## The benches" and nothing else.
    A parser that understands all of markdown would be a dependency; a parser
    that stops at the first blank line needs no more than the file gives it.
    """
    lines = [line.rstrip("\r") for line in text.splitlines()]
    try:
        start = lines.index("## The benches")
    except ValueError:
        raise SystemExit("README.md has no '## The benches' heading to read")

    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in ("Bench",) or set(cells[0]) <= set("-: "):
            continue
        rows.append(cells)

    benches = []
    for title_cell, question, status in rows:
        head = re.split(r"\s+—\s+", title_cell, maxsplit=1)
        name = plain(head[0])
        subtitle = plain(head[1]) if len(head) > 1 else ""
        if ".py" in subtitle:
            # The model bench's cell lists its files instead of describing
            # itself; a tile is not a table of contents, so it gets no line.
            subtitle = ""
        targets = links(title_cell)
        link = next((target for target in targets if target.endswith("README.md")),
                    targets[0] if targets else "")
        directory = ""
        if link.endswith("/README.md"):
            directory = link[: -len("/README.md")]
        benches.append(Bench(
            name=name, subtitle=subtitle, link=link, directory=directory,
            question=plain(question), status=plain(status),
        ))
    return benches


def count_tests(directory):
    """`def test_` in the bench's own test files. Zero when it has none."""
    if not directory:
        paths = [ROOT / MODEL_SUITE]
    else:
        paths = sorted((ROOT / directory).glob("test_*.py"))
    total = 0
    for path in paths:
        if path.is_file():
            total += path.read_text(encoding="utf-8").count("def test_")
    return total


def workflow_jobs(path=WORKFLOW):
    """Job name -> that job's text, for the tests workflow."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    jobs = {}
    name = None
    for line in text.splitlines():
        found = re.match(r"^  ([A-Za-z][\w-]*):\s*$", line)
        if found:
            name = found.group(1)
            jobs[name] = []
        elif name is not None:
            jobs[name].append(line)
    return {name: "\n".join(body) for name, body in jobs.items()}


def ci_job(bench, jobs):
    """The job that runs this bench's suite. "" when none does.

    Two ways a job can cover a bench, because the workflow uses both. Most
    steps name their module (`ci_tests.py mediatop.test_mediatop`), and that is
    the claim the page makes: a bench is covered if the exact dotted name
    appears in a job, so a near-miss does not count.

    The lab step does not name anything -- it runs `--discover .` from the lab
    as its working directory -- and matching only names described a bench with
    a job as having none, which is the same kind of lie as the reverse. So a
    discovery step counts when its working directory is the bench's own folder
    (the repo root for the bench that lives there).
    """
    for name, body in jobs.items():
        if any(re.search(rf"\b{re.escape(module)}\b", body)
               for module in bench.modules):
            return name
    for name, body in jobs.items():
        if not re.search(r"--discover\s+\.", body):
            continue
        found = re.search(r"working-directory:\s*(\S+)", body)
        where = found.group(1).strip("\"'") if found else "."
        if where.rstrip("/") == (bench.directory or "."):
            return name
    return ""


def gather(readme_path=README, workflow_path=WORKFLOW):
    """The benches, with their counted tests and their CI job filled in."""
    bench_list = readme_benches(readme_path.read_text(encoding="utf-8"))
    jobs = workflow_jobs(workflow_path)
    return [
        Bench(**{**bench.__dict__, "tests": count_tests(bench.directory),
                 "job": ci_job(bench, jobs)})
        for bench in bench_list
    ]


CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 48px 28px 64px;
  background: #fbf8f3; color: #14110f;
  font: 16px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}
header, main, footer { max-width: 1080px; margin: 0 auto; }
h1 {
  margin: 0 0 6px; font: 600 2.6rem/1.1 Georgia, "Times New Roman", serif;
  letter-spacing: -0.02em;
}
.lede { margin: 0 0 4px; font-size: 1.05rem; color: #4a443c; }
.counts {
  margin: 14px 0 0; font: 500 0.78rem/1.4 ui-monospace, Menlo, monospace;
  letter-spacing: 0.08em; text-transform: uppercase; color: #6b655c;
}
main {
  display: grid; gap: 18px; margin-top: 34px;
  grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
}
.tile {
  position: relative; display: flex; flex-direction: column;
  background: #fff; border: 1px solid #e6ddd0; border-radius: 10px;
  padding: 20px 20px 18px; text-decoration: none; color: inherit;
  box-shadow: 0 1px 0 rgba(20, 17, 15, 0.04);
}
.tile::before {
  content: ""; position: absolute; inset: 0 0 auto; height: 5px; border-radius:
  10px 10px 0 0; background: var(--accent);
}
.tile:hover { border-color: var(--accent); box-shadow: 0 6px 18px rgba(20, 17, 15, 0.09); }
.tile:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
.status {
  margin: 6px 0 12px; font: 600 0.7rem/1 ui-monospace, Menlo, monospace;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent);
}
.question {
  margin: 0 0 14px; font: 500 1.3rem/1.28 Georgia, "Times New Roman", serif;
  letter-spacing: -0.01em;
}
.name { margin: 0 0 2px; font-weight: 650; }
.where {
  margin: 0; font-size: 0.86rem; color: #6b655c;
  min-height: 1.5em;  /* one line's height even when empty, so rows line up */
}
.facts {
  display: flex; flex-wrap: wrap; gap: 6px 8px; margin: 16px 0 0;
  padding: 0; list-style: none; border-top: 1px solid #f0e9dd; padding-top: 12px;
  font: 500 0.72rem/1 ui-monospace, Menlo, monospace; letter-spacing: 0.04em;
  text-transform: uppercase; color: #57514a;
}
.facts li { background: #f6f1e8; border-radius: 4px; padding: 5px 7px; }
.facts .quiet { background: transparent; padding-left: 0; color: #a09a90; }
footer {
  margin-top: 40px; font-size: 0.82rem; color: #6b655c;
  border-top: 1px solid #e6ddd0; padding-top: 16px;
}
"""


def href(bench, base=""):
    """Where a tile points: relative in a checkout, absolute on a site.

    Served from Pages the page sits under a project path, and the READMEs are
    not part of that artifact -- a relative `othello/README.md` would 404. A
    `--base` naming the repository's rendered tree turns each tile into a link
    that works from anywhere, which is also the only way a reader of the
    published page can act on it.
    """
    if not base:
        return bench.target
    return f"{base.rstrip('/')}/{bench.target.lstrip('/')}"


def tile(bench, base=""):
    """One card. Everything user-visible is escaped; the accent comes from us."""
    tests = f"{bench.tests} tests" if bench.tests else "no tests"
    # A missing job is dimmed rather than hidden: "no CI job" is information
    # about the bench, and a blank space would read as an oversight.
    job = (f"<li>{html.escape('CI: ' + bench.job)}</li>" if bench.job
           else '<li class="quiet">no CI job</li>')
    link = href(bench, base)
    return f"""    <a class="tile" href="{html.escape(link)}" style="--accent: {bench.colour}">
      <p class="status">{html.escape(bench.status.split()[0] if bench.status.split() else "bench")}</p>
      <p class="question">{html.escape(bench.question)}</p>
      <p class="name">{html.escape(bench.name)}</p>
      <p class="where">{html.escape(bench.subtitle)}</p>
      <ul class="facts"><li>{html.escape(tests)}</li>{job}</ul>
    </a>"""


def render(benches, base=""):
    """The whole page: a header, the tiles, and where they came from."""
    total = len(benches)
    words = {}
    for bench in benches:
        word = bench.status.split()[0].lower() if bench.status.split() else "other"
        words[word] = words.get(word, 0) + 1
    summary = " · ".join(f"{count} {word}" for word, count in sorted(words.items()))
    covered = sum(1 for bench in benches if bench.job)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The benches — new-idea2</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>{total} benches, {total} questions</h1>
  <p class="lede">Each one asks something the others cannot, and none of them
  depends on another. Colour says whether it is alive.</p>
  <p class="counts">{html.escape(summary)} · {covered} with a CI job · generated from README.md</p>
</header>
<main>
{chr(10).join(tile(bench, base) for bench in benches)}
</main>
<footer>
  Written by <code>bench_map.py</code> from the table in <code>README.md</code>,
  so this page cannot name a bench the repository does not have. Test counts are
  <code>def test_</code> counted in each bench's own files; a CI job is shown
  only where the tests workflow names that bench's suite. Regenerate with
  <code>python bench_map.py</code>.
</footer>
</body>
</html>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render the README bench table as one bright page.")
    parser.add_argument("--out", default="bench-map.html",
                        help="where to write the page (default: %(default)s)")
    parser.add_argument("--base", default="",
                        help="absolute tree URL to link tiles at, e.g. "
                             "https://github.com/o/r/blob/main (default: relative)")
    parser.add_argument("--json", action="store_true",
                        help="print the gathered rows as json instead")
    args = parser.parse_args(argv)

    benches = gather()
    if args.json:
        print(json.dumps([bench.__dict__ for bench in benches], indent=2))
        return 0

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(benches, base=args.base), encoding="utf-8")
    print(f"{len(benches)} benches -> {output}")
    for bench in benches:
        job = f"ci:{bench.job}" if bench.job else "no CI job"
        print(f"  {bench.name:<14} {bench.tests:>3} tests  {job}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
