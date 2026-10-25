"""Turn the weekly platform report into a page that remembers.

``platforms.py run`` answers one question at one hour: what did each service
serve back for the signed fixture? That answer is only interesting over time --
a platform quietly starting to preserve the credential, or quietly stopping
being measured because a token expired, looks like a single unremarkable row
until you can see it against the weeks before it.

So each run is appended to ``history.json`` and the whole series is rendered as
a matrix: one row per adapter, one column per run, colour for what came back.

Deliberately stdlib only. The page is built here rather than in the browser, so
it works with JavaScript disabled, loads nothing from a CDN, and can be rebuilt
from the history alone without the signing stack installed.

Usage:
    python pages.py build --report out/live/platforms.json --site out/site
    python pages.py build --report out/live/platforms.json --site out/site \\
        --expect discord,telegram --run-url https://github.com/o/r/actions/runs/1
    python pages.py publish --site out/site --branch platform-history
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import verifier

VERSION = 1

# Two years of weekly runs. The page shows a window of this; nothing older is
# worth a column, but keeping it costs a few kilobytes and lets the window
# grow later without having lost the evidence.
KEEP_RUNS = 104
PAGE_RUNS = 26

DEFAULT_TITLE = "Platform survival"

# A run that exits before writing a report still deserves a column, so the
# page does not quietly skip the week something broke.
UNMEASURED = "unmeasured"

_LABELS = {
    "valid": "credential intact",
    "invalid": "credential present but invalid",
    "gone": "stripped",
    "unavailable": "served, could not be read",
    "unconfigured": "needs config",
    "failed": "run failed",
    UNMEASURED: "not measured",
}

_SHORT = {
    "valid": "intact",
    "invalid": "invalid",
    "gone": "stripped",
    "unavailable": "unreadable",
    "unconfigured": "no config",
    "failed": "failed",
    UNMEASURED: "no data",
}

# Order matters: a platform where one rendition kept the credential and another
# dropped it is reported as ``valid``, because "somebody can still show the
# credential" is the finding worth surfacing.
_VERDICTS = (
    ("valid", "valid"),
    ("invalid", "invalid"),
    ("absent", "gone"),
    ("unavailable", "unavailable"),
)

# Upload errors can quote the request URL, and a Telegram file URL embeds the
# bot token. Before any of that reaches a public page it is scrubbed here, so
# the same text is safe in history.json and in the HTML.
_SCRUB = (
    (re.compile(r"(bot)\d+:[A-Za-z0-9_-]+"), r"\1<redacted>"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1<redacted>"),
    (
        re.compile(r"(?i)\b(access_token|api_key|token|secret|password|key)=([^&\s]+)"),
        r"\1=<redacted>",
    ),
    # Any remaining credential-in-authority form: scheme://user:pass@host
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@"), r"\1<redacted>@"),
)


def scrub(text: str, limit: int = 240) -> str:
    """Make an adapter's free text fit to publish."""
    cleaned = "".join(ch if ch.isprintable() else " " for ch in str(text))
    for pattern, replacement in _SCRUB:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned[:limit]


_STATUS = re.compile(r"\bHTTP (\d{3})\b")
_NETWORK = re.compile(r"(?i)timed out|timeout|connection|max retries|network|ssl|dns")


def failure_reason(reason: str) -> str:
    """Reduce an adapter's error to something safe to publish.

    A raw exception message is not fit for a permanent public page: requests
    quotes the failing URL, a Discord webhook URL carries its token as a path
    segment, and a Telegram file URL carries one after ``bot<id>:``. Scrubbing
    every shape of credential in free text is a losing game, so the page keeps
    only the part an operator acts on -- which class of failure it was.
    """
    text = scrub(reason)
    if text.startswith("needs config:") or text == "unknown platform":
        # Both are built from our own identifiers, never from a reply or a URL.
        return text
    status = _STATUS.search(text)
    if status:
        return f"upload failed: HTTP {status.group(1)}"
    if _NETWORK.search(text):
        return "upload failed: network error"
    return "upload failed"


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------


def classify(result: dict | None, expected: Iterable[str] = ()) -> dict[str, dict]:
    """Reduce one run's report to an outcome per platform.

    ``platforms.json`` describes renditions; the page only needs to know
    whether the credential survived, was damaged, or was never measured, plus
    enough variant detail to explain the colour when someone asks.
    """
    found: dict[str, dict] = {}

    if result:
        for item in result.get("skipped") or []:
            name = str(item.get("platform") or "?")
            reason = failure_reason(str(item.get("reason") or ""))
            found[name] = {
                # ``needs config`` is not a failure: it is the expected state
                # before anyone has set the secret, and it should not read as
                # news on the chart.
                "outcome": "unconfigured" if reason.startswith("needs config:") else "failed",
                "reason": reason,
                "variants": [],
            }

        for row in result.get("rows") or []:
            name = str(row.get("platform") or "?")
            record = found.setdefault(name, {"variants": []})
            record["variants"].append(
                {
                    "name": scrub(str(row.get("variant") or "?"), 60),
                    "verdict": str(row.get("verdict") or "unavailable"),
                    "format": row.get("served_format"),
                    "bytes": row.get("served_bytes"),
                    "identical": bool(row.get("sha256_match")),
                }
            )

        for record in found.values():
            if "outcome" in record:
                continue
            verdicts = {variant["verdict"] for variant in record["variants"]}
            record["outcome"] = next(
                (outcome for verdict, outcome in _VERDICTS if verdict in verdicts),
                "unavailable",
            )
            if len(verdicts) > 1:
                # The renditions disagreed, which is the interesting case and
                # the one a single word would flatten.
                record["mixed"] = True
            record["variants"].sort(key=lambda variant: variant["name"])

    for name in expected:
        found.setdefault(str(name), {"outcome": UNMEASURED, "variants": []})
    return found


def run_record(
    result: dict | None,
    *,
    expected: Iterable[str] = (),
    run_url: str | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
) -> dict:
    """One column: when it ran, where its log is, what every adapter did."""
    now = now or datetime.now(timezone.utc)
    meta = (result or {}).get("meta") or {}
    generated = str(meta.get("generated") or now.isoformat(timespec="seconds"))
    record: dict[str, Any] = {
        # The workflow run id, so a re-run replaces its own column instead of
        # adding a second reading of the same hour.
        "id": str(run_id or generated),
        "generated": generated,
        "platforms": classify(result, expected),
    }
    if run_url:
        record["run_url"] = str(run_url)
    for key in ("fixture", "signed_bytes"):
        if meta.get(key) not in (None, ""):
            record[key] = meta[key]
    if result is None:
        record["note"] = "the measurement step produced no report"
    return record


def _when(value: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def load_history(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"version": VERSION, "runs": []}
    try:
        history = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        # Starting over would erase the series this file exists to keep, and a
        # fresh page would look like a first run rather than a lost history.
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(history, dict) or not isinstance(history.get("runs"), list):
        raise ValueError(f"{path} has no runs list")
    return history


def append(history: dict, record: dict, *, keep: int = KEEP_RUNS) -> dict:
    """Add or replace one run, oldest first, trimmed to the retained window."""
    runs = [run for run in history.get("runs", []) if run.get("id") != record["id"]]
    runs.append(record)
    runs.sort(key=lambda run: _when(run.get("generated", "")))
    history["version"] = VERSION
    history["runs"] = runs[-keep:]
    return history


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

_CSS = """
:root{
  color-scheme: light dark;
  --bg:#fbfbfa; --fg:#18181b; --muted:#5f6368; --line:#e4e4e7;
  --valid:#b4690e; --invalid:#a3282d; --gone:#2f6f6a; --unavailable:#8a8a93;
  --unconfigured:#d6d6da; --failed:#7f1d1d; --none:#f1f1f3;
  --card:#ffffff;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#121214; --fg:#e8e8ea; --muted:#a1a1aa; --line:#2a2a2f;
    --valid:#e0a04a; --invalid:#e06c72; --gone:#5fb3a8; --unavailable:#85858e;
    --unconfigured:#3f3f46; --failed:#b4545a; --none:#1b1b1f;
    --card:#18181b;
  }
}
*{box-sizing:border-box}
body{margin:0;padding:2.5rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:60rem;margin:0 auto}
h1{font-size:1.5rem;line-height:1.2;margin:0 0 .4rem;letter-spacing:-.01em}
h2{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  margin:2.5rem 0 .75rem;font-weight:600}
p{margin:0 0 .75rem}
.lede{color:var(--muted);max-width:44rem}
.scroll{overflow-x:auto;padding-bottom:.5rem}
table{border-collapse:separate;border-spacing:3px}
.matrix th{font-weight:500;font-size:.72rem;color:var(--muted)}
.matrix thead th{writing-mode:vertical-rl;transform:rotate(180deg);text-align:left;
  height:4.2rem;vertical-align:bottom;padding:0 0 .2rem;white-space:nowrap}
.matrix thead th.latest{color:var(--fg);font-weight:600}
.matrix thead th.nowhead{writing-mode:horizontal-tb;transform:none;height:auto;
  vertical-align:bottom;padding-left:.5rem;font-size:.72rem}
.matrix tbody th{text-align:left;padding-right:.75rem;white-space:nowrap;font-size:.85rem}
.matrix tbody th code{font-size:.82rem}
.cell{width:1.15rem;height:1.15rem;border-radius:3px;background:var(--none)}
td.cell{min-width:1.15rem}
.o-valid{background:var(--valid)} .o-invalid{background:var(--invalid)}
.o-gone{background:var(--gone)} .o-unavailable{background:var(--unavailable)}
.o-unconfigured{background:var(--unconfigured)}
.o-failed{background:var(--failed)} .o-unmeasured{background:var(--none);
  box-shadow:inset 0 0 0 1px var(--line)}
.now{font-size:.78rem;color:var(--muted);padding-left:.5rem;white-space:nowrap}
.legend{display:flex;flex-wrap:wrap;gap:.35rem 1rem;list-style:none;margin:1rem 0 0;
  padding:0;font-size:.78rem;color:var(--muted)}
.legend li{display:flex;align-items:center;gap:.4rem}
.legend .swatch{width:.8rem;height:.8rem;border-radius:2px;display:inline-block}
.summary{width:100%;font-size:.85rem}
.summary th,.summary td{text-align:left;padding:.4rem .6rem;border-bottom:1px solid var(--line)}
.summary th{font-weight:500;color:var(--muted);font-size:.72rem;text-transform:uppercase;
  letter-spacing:.06em}
.summary td:first-child{white-space:nowrap}
details{margin-top:1rem}
summary{cursor:pointer;color:var(--muted);font-size:.85rem}
.caveats{color:var(--muted);font-size:.82rem;margin-top:2.5rem;border-top:1px solid var(--line);
  padding-top:1rem}
footer{color:var(--muted);font-size:.75rem;margin-top:1.5rem}
a{color:inherit}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
  clip:rect(0 0 0 0);white-space:nowrap;border:0}
.empty{border:1px dashed var(--line);border-radius:8px;padding:2rem;text-align:center;
  color:var(--muted)}
"""


def _fmt(value: str, pattern: str = "%d %b") -> str:
    stamp = _when(value)
    return "?" if stamp == datetime.min.replace(tzinfo=timezone.utc) else stamp.strftime(pattern)


def _names(history: dict) -> list[str]:
    names: set[str] = set()
    for run in history.get("runs", []):
        names.update((run.get("platforms") or {}).keys())
    return sorted(names)


def _streak(runs: list[dict], name: str) -> tuple[str | None, str | None, int]:
    """Current outcome, when it started, and how many times it changed."""
    outcomes: list[str | None] = [
        (run.get("platforms") or {}).get(name, {}).get("outcome") for run in runs
    ]
    present = [outcome for outcome in outcomes if outcome]
    changes = sum(1 for a, b in zip(present, present[1:]) if a != b)
    if not present:
        return None, None, 0
    last = present[-1]
    index = max(i for i, outcome in enumerate(outcomes) if outcome == last)
    start = index
    while start > 0 and outcomes[start - 1] == last:
        start -= 1
    return last, runs[start].get("generated") if start < len(runs) else None, changes


def render_page(
    history: dict,
    *,
    title: str = "Platform survival",
    max_runs: int = PAGE_RUNS,
) -> str:
    runs = sorted(history.get("runs", []), key=lambda run: _when(run.get("generated", "")))
    shown = runs[-max_runs:]
    names = _names(history)

    out: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
        f"<style>{_CSS}</style>",
        '<link rel="alternate" type="application/json" href="history.json">',
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{html.escape(title)}</h1>",
        '<p class="lede">',
        "Each column is one run of the live adapters: a signed fixture uploaded to a "
        "real service, then the bytes the service actually served back, verified. "
        "A cell says whether the C2PA credential survived the platform on that day.",
        "</p>",
        '<p class="lede">',
        '<a href="verifier/">Verify a rendition in your browser</a> &mdash; the same '
        "run's downloads, read by the CAI verifier a viewer would embed, or a file "
        "of your own.",
        "</p>",
    ]

    if not runs:
        out.append(
            '<p class="empty">No runs recorded yet. The workflow appends one '
            "column per measurement.</p>"
        )
    else:
        out.append(f'<div class="scroll"><table class="matrix">')
        out.append(
            f'<caption class="sr-only">Credential outcome per platform over the last '
            f"{len(shown)} of {len(runs)} runs, oldest first</caption>"
        )
        out.append("<thead><tr><th scope=\"col\"><span>platform</span></th>")
        for index, run in enumerate(shown):
            latest = index == len(shown) - 1
            stamp = _fmt(run.get("generated", ""))
            link = run.get("run_url")
            label = (
                f'<a href="{html.escape(str(link))}">{html.escape(stamp)}</a>'
                if link
                else html.escape(stamp)
            )
            marker = ' class="latest"' if latest else ""
            out.append(f'<th scope="col"{marker}><span>{label}</span></th>')
        out.append('<th scope="col" class="nowhead">now</th></tr></thead><tbody>')

        for name in names:
            out.append(f"<tr><th scope=\"row\"><code>{html.escape(name)}</code></th>")
            for run in shown:
                record = (run.get("platforms") or {}).get(name)
                outcome = (record or {}).get("outcome")
                if not outcome:
                    out.append(
                        '<td class="cell o-unmeasured" title="'
                        f'{html.escape(name)}: not part of this run"></td>'
                    )
                    continue
                title_text = f"{name} · {_fmt(run.get('generated', ''), '%d %b %Y')} · {_LABELS[outcome]}"
                reason = (record or {}).get("reason")
                if reason:
                    title_text += f": {reason}"
                if (record or {}).get("mixed"):
                    title_text += " (variants disagree)"
                out.append(
                    f'<td class="cell o-{outcome}" title="{html.escape(title_text)}">'
                    f'<span class="sr-only">{html.escape(title_text)}</span></td>'
                )
            outcome, since, _ = _streak(runs, name)
            label = _LABELS.get(outcome or UNMEASURED, "no data")
            suffix = f" since {_fmt(since, '%d %b %Y')}" if since and outcome else ""
            out.append(f'<td class="now">{html.escape(label + suffix)}</td>')
            out.append("</tr>")

        out.append("</tbody></table></div>")
        out.append("<ul class=\"legend\">")
        for outcome in ("gone", "valid", "invalid", "unavailable", "unconfigured", "failed", UNMEASURED):
            out.append(
                f'<li><span class="swatch o-{outcome}"></span>{html.escape(_LABELS[outcome])}</li>'
            )
        out.append("</ul>")
        if len(runs) > len(shown):
            out.append(
                f'<p class="now">Showing the last {len(shown)} of {len(runs)} runs; '
                f"the rest are in the history file.</p>"
            )

    if runs:
        out.append("<h2>Where each adapter stands</h2>")
        out.append('<table class="summary">')
        out.append(
            "<thead><tr><th scope=\"col\">platform</th><th scope=\"col\">now</th>"
            "<th scope=\"col\">since</th><th scope=\"col\">changes</th>"
            "<th scope=\"col\">runs measured</th></tr></thead><tbody>"
        )
        for name in names:
            outcome, since, changes = _streak(runs, name)
            measured = sum(
                1
                for run in runs
                if ((run.get("platforms") or {}).get(name) or {}).get("outcome")
                not in (None, UNMEASURED)
            )
            out.append(
                f"<tr><td><code>{html.escape(name)}</code></td>"
                f"<td>{html.escape(_LABELS.get(outcome or UNMEASURED, 'no data'))}</td>"
                f"<td>{html.escape(_fmt(since, '%d %b %Y') if since else '-')}</td>"
                f"<td>{changes}</td><td>{measured} of {len(runs)}</td></tr>"
            )
        out.append("</tbody></table>")

        latest = runs[-1]
        variants = [
            (name, variant)
            for name in names
            for variant in ((latest.get("platforms") or {}).get(name) or {}).get("variants", [])
        ]
        if variants:
            out.append("<h2>The latest run, rendition by rendition</h2>")
            out.append('<table class="summary">')
            out.append(
                "<thead><tr><th scope=\"col\">platform</th><th scope=\"col\">variant</th>"
                "<th scope=\"col\">credential</th><th scope=\"col\">served</th>"
                "<th scope=\"col\">bytes</th><th scope=\"col\">identical</th></tr></thead><tbody>"
            )
            for name, variant in variants:
                verdict = str(variant.get("verdict", "unavailable"))
                out.append(
                    f"<tr><td>{html.escape(name)}</td>"
                    f"<td>{html.escape(str(variant.get('name', '?')))}</td>"
                    f"<td>{html.escape(verdict)}</td>"
                    f"<td>{html.escape(str(variant.get('format') or '-'))}</td>"
                    f"<td>{variant.get('bytes') or '-'}</td>"
                    f"<td>{'yes' if variant.get('identical') else 'no'}</td></tr>"
                )
            out.append("</tbody></table>")

    out.extend(
        [
            '<p class="caveats">',
            "The signing certificate is a throwaway test anchor created by the lab. "
            "&ldquo;Credential intact&rdquo; means the signature verifies against that "
            "anchor and the hash still binds the image; it says nothing about a public "
            "trust list. &ldquo;Stripped&rdquo; is the ordinary result and the one the "
            "lab exists to measure. A grey cell means that adapter was skipped — usually "
            "no secret is configured — not that it passed.",
            "</p>",
            "<footer>",
            f"Built from {len(runs)} recorded run{'s' if len(runs) != 1 else ''} · "
            '<a href="history.json">history.json</a>',
            "</footer>",
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )
    return "\n".join(out)


# --------------------------------------------------------------------------
# build and publish
# --------------------------------------------------------------------------


def build(
    report: Path | None,
    site_dir: Path,
    *,
    expected: Iterable[str] = (),
    run_url: str | None = None,
    run_id: str | None = None,
    title: str = DEFAULT_TITLE,
    max_runs: int = PAGE_RUNS,
    keep: int = KEEP_RUNS,
    now: datetime | None = None,
    verifier_roots: Iterable[str | Path] = (),
) -> dict:
    site_dir = Path(site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)
    history_path = site_dir / "history.json"
    history = load_history(history_path)

    result = None
    if report and Path(report).exists():
        result = json.loads(Path(report).read_text())
    if result is None:
        # A run that died before writing a report still gets its column, so the
        # gap is visible rather than absent.
        expected = [name.strip() for name in expected if str(name).strip()]

    record = run_record(result, expected=expected, run_url=run_url, run_id=run_id, now=now)
    append(history, record, keep=keep)
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (site_dir / "index.html").write_text(
        render_page(history, title=title, max_runs=max_runs), encoding="utf-8"
    )
    # The site is both pages. Staging the verifier here rather than in a step of
    # its own keeps the link from index.html from pointing at nothing: every
    # build produces it, and `verifier_roots` only decides whether it has
    # renditions to list alongside the drop zone.
    verifier.build_site(verifier_roots, site_dir)
    return record


class PublishError(RuntimeError):
    pass


def _git(args: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            env=env,
            # A stalled network must fail the step, not hold the job open for
            # its whole timeout.
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        raise PublishError(f"git {' '.join(args)} timed out") from None
    if check and proc.returncode != 0:
        # git quotes remote URLs in its errors, and the push URL carries the
        # workflow token. Never let that reach a log.
        message = scrub(proc.stderr or proc.stdout or "", 2000)
        raise PublishError(f"git {' '.join(args)} failed: {message.strip()}")
    return proc


def publish(
    site_dir: Path,
    *,
    branch: str = "platform-history",
    remote: str = "origin",
    push_url: str | None = None,
    message: str | None = None,
    repo: Path | None = None,
    title: str = DEFAULT_TITLE,
    max_runs: int = PAGE_RUNS,
    keep: int = KEEP_RUNS,
) -> int:
    """Merge the freshly built page into the history branch and push.

    The branch is cloned rather than force-pushed over, so its commit history
    survives and a hand edit on the branch is not silently discarded. If the
    branch does not exist yet the repository is initialised empty, which is the
    first-run case.

    Its ``history.json`` is merged rather than overwritten: on a CI runner the
    build starts from an empty directory and knows only the run that just
    finished, so copying that file over the branch would trade the whole series
    for one column. Runs are merged by id, so a re-run of the same id still
    replaces its own column, and the page is re-rendered from the merged series
    and written back to ``site_dir``, since that directory is what a deployment
    stages.
    """
    site_dir = Path(site_dir)
    repo = Path(repo or ".")
    if not (site_dir / "index.html").exists():
        raise PublishError(f"{site_dir} has no index.html to publish")

    url = push_url
    if not url:
        proc = _git(["remote", "get-url", remote], repo, check=False)
        if proc.returncode != 0:
            raise PublishError(f"no {remote} remote and no --push-url given")
        url = proc.stdout.strip()

    message = message or f"platform report {datetime.now(timezone.utc):%Y-%m-%d}"

    with tempfile.TemporaryDirectory(prefix="pages-") as tmp:
        work = Path(tmp) / "pages"
        exists = _git(["ls-remote", "--exit-code", "--heads", url, branch], repo, check=False)
        if exists.returncode == 0:
            _git(["clone", "--quiet", "--depth", "1", "--branch", branch, url, str(work)], repo)
        elif exists.returncode == 2:
            # `ls-remote --exit-code` uses 2 for "no matching refs", which is
            # exactly the empty-branch case; anything else is a real error.
            _git(["init", "--quiet", str(work)], repo)
            _git(["remote", "add", "origin", url], work)
            _git(["checkout", "--quiet", "--orphan", branch], work)
        else:
            raise PublishError(
                f"cannot read branch {branch}: {scrub(exists.stderr or 'unknown error', 400)}"
            )

        # The branch is the page's memory, so the series is what it already
        # holds plus whatever this build added -- never just the latter.
        history = load_history(work / "history.json")
        for record in load_history(site_dir / "history.json")["runs"]:
            append(history, record, keep=keep)
        merged = json.dumps(history, indent=2) + "\n"
        page = render_page(history, title=title, max_runs=max_runs)
        (work / "history.json").write_text(merged, encoding="utf-8")
        (work / "index.html").write_text(page, encoding="utf-8")
        # Written back as well as pushed: whoever stages the deployment uploads
        # this directory, and an artifact of the single-column build would put
        # the snapshot back on the page the merge was meant to fix.
        (site_dir / "history.json").write_text(merged, encoding="utf-8")
        (site_dir / "index.html").write_text(page, encoding="utf-8")

        for item in site_dir.iterdir():
            # Those two files come from the merge above; copying them would put
            # the single-column build back in charge of the series.
            if item.name in (".git", "history.json", "index.html"):
                continue
            target = work / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

        _git(["add", "--all"], work)
        dirty = _git(["status", "--porcelain"], work).stdout.strip()
        if not dirty:
            print(f"nothing to publish: {branch} already matches {site_dir}")
            return 0
        _git(
            [
                "-c", "user.name=provenance-lab",
                "-c", "user.email=provenance-lab@users.noreply.github.com",
                "commit", "--quiet", "--message", message,
            ],
            work,
        )
        _git(["push", "--quiet", "origin", f"HEAD:{branch}"], work)
        head = _git(["rev-parse", "--short", "HEAD"], work).stdout.strip()
        print(f"published {head} to {branch}")
    return 0


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd")

    build_parser = sub.add_parser("build", help="append a run and render the page")
    build_parser.add_argument("--report", help="platforms.json from the measurement run")
    build_parser.add_argument("--site", default="out/site", help="output directory")
    build_parser.add_argument(
        "--expect",
        default="",
        help="comma-separated adapters, recorded as not measured if the report is missing",
    )
    build_parser.add_argument("--run-url", help="link back to the workflow run")
    build_parser.add_argument("--run-id", help="stable id; a re-run replaces its column")
    build_parser.add_argument("--title", default="Platform survival")
    build_parser.add_argument("--max-runs", type=int, default=PAGE_RUNS)
    build_parser.add_argument("--keep", type=int, default=KEEP_RUNS)
    build_parser.add_argument(
        "--verifier-out",
        action="append",
        default=[],
        help="run directory whose served/ renditions the verifier page lists "
             "(repeatable; without it the page only verifies dropped files)",
    )

    publish_parser = sub.add_parser("publish", help="push the page to a branch")
    publish_parser.add_argument("--site", default="out/site")
    publish_parser.add_argument("--branch", default="platform-history")
    publish_parser.add_argument("--remote", default="origin")
    publish_parser.add_argument("--push-url", help="clone/push URL, e.g. one carrying a token")
    publish_parser.add_argument("--message")

    args = parser.parse_args(argv)

    if args.cmd == "build":
        expected = [name.strip() for name in args.expect.split(",") if name.strip()]
        record = build(
            Path(args.report) if args.report else None,
            Path(args.site),
            expected=expected,
            run_url=args.run_url,
            run_id=args.run_id,
            title=args.title,
            max_runs=args.max_runs,
            keep=args.keep,
            verifier_roots=args.verifier_out,
        )
        outcomes = {
            name: data["outcome"] for name, data in sorted(record["platforms"].items())
        }
        print(f"appended {record['generated']}")
        for name, outcome in outcomes.items():
            print(f"  {name}: {outcome}")
        return 0

    if args.cmd == "publish":
        return publish(
            Path(args.site),
            branch=args.branch,
            remote=args.remote,
            push_url=args.push_url,
            message=args.message,
        )

    parser.print_help()
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PublishError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
