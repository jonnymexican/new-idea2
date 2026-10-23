"""The board as a dated Markdown file, for reading without a browser.

Same promise as the page, in a form that survives outside it: every point and
every fact is a sentence from the article. The file is named for the day it was
written, so a morning run leaves one file per morning rather than a growing
list.
"""

import re
from datetime import datetime
from pathlib import Path

STAMP = "%Y-%m-%d"


def _clean(text):
    """One line, and no brackets that would break a Markdown link's text."""
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.replace("[", "\\[").replace("]", "\\]")


def _long_date(when):
    # %-d is not portable to Windows, so the day number is spliced in.
    return f"{when.strftime('%A')} {when.day} {when.strftime('%B %Y')}"


def dated_path(target=None, when=None):
    """Where a digest for `when` belongs.

    No target: `mediatop-2026-10-23.md` in the current directory. A directory
    (one that exists, or that is written with a trailing separator): the dated
    name inside it. Anything else is taken as the file name, exactly as given.
    """
    when = when or datetime.now()
    name = f"mediatop-{when.strftime(STAMP)}.md"
    if target is None:
        return Path(name)
    text = str(target)
    path = Path(text).expanduser()
    if path.is_dir() or text.endswith(("/", "\\")):
        return path / name
    return path


def render(payload, when=None):
    """The whole board as one Markdown document."""
    when = when or datetime.now()
    genres = payload.get("genres") or []
    counts = payload.get("counts") or {}
    stories = counts.get("stories")
    if stories is None:
        stories = sum(len(genre.get("cards") or []) for genre in genres)
    sources = counts.get("sources")
    if sources is None:
        sources = len({card.get("source") for genre in genres
                       for card in genre.get("cards") or []})

    lines = [
        f"# mediatop — {_long_date(when)}",
        "",
        f"{stories} stories from {sources} sources, "
        f"assembled {when.strftime('%H:%M')}.",
        "",
        "*Every point and fact below is a sentence from the article, "
        "quoted rather than written.*",
    ]

    for genre in genres:
        cards = genre.get("cards") or []
        lines += ["", f"## {genre.get('label') or genre.get('key')}", ""]
        if not cards:
            lines.append("*Nothing readable in this genre today.*")
            continue
        for card in cards:
            title = _clean(card.get("title"))
            url = card.get("url") or ""
            heading = f"### [{title}]({url})" if url else f"### {title}"
            lines.append(heading)
            meta = _clean(card.get("source"))
            if card.get("note"):
                meta += f" — summary only: {_clean(card['note'])}"
            lines += [f"*{meta}*", "", f"**Point.** {_clean(card.get('point'))}"]
            facts = [fact for fact in card.get("facts") or [] if fact]
            if facts:
                lines.append("")
                lines += [f"- {_clean(fact)}" for fact in facts]
            lines.append("")

    failures = payload.get("failures") or []
    if failures:
        lines += ["## Feeds that failed", ""]
        lines += [f"- {_clean(failure.get('genre'))}/"
                  f"{_clean(failure.get('source'))}: "
                  f"{_clean(failure.get('error'))}" for failure in failures]
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write(payload, target=None, when=None):
    """Write the digest and return where it landed."""
    path = dated_path(target, when)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(payload, when), encoding="utf-8", newline="\n")
    return path
