"""Tests for the Pages trend page: outcome reduction, history, rendering, publish."""

import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pages


def report(rows=None, skipped=None, generated="2026-10-19T06:00:00+00:00"):
    return {
        "meta": {
            "generated": generated,
            "fixture": "1024x768",
            "signed_bytes": 4096,
        },
        "rows": rows or [],
        "skipped": skipped or [],
    }


def row(platform, variant, verdict, fmt="jpeg", size=1000, identical=False):
    return {
        "platform": platform,
        "variant": variant,
        "verdict": verdict,
        "served_format": fmt,
        "served_bytes": size,
        "sha256_match": identical,
        "url": "https://example.test/<redacted>",
    }


class TestClassify(unittest.TestCase):
    def test_all_gone_is_gone(self):
        outcomes = pages.classify(report(rows=[
            row("discord", "photo", "absent"),
            row("discord", "file", "absent"),
        ]))
        self.assertEqual(outcomes["discord"]["outcome"], "gone")
        self.assertNotIn("mixed", outcomes["discord"])

    def test_one_survivor_wins_and_marks_the_disagreement(self):
        outcomes = pages.classify(report(rows=[
            row("telegram", "photo", "absent"),
            row("telegram", "file", "valid"),
        ]))
        self.assertEqual(outcomes["telegram"]["outcome"], "valid")
        self.assertTrue(outcomes["telegram"]["mixed"])

    def test_invalid_beats_gone(self):
        outcomes = pages.classify(report(rows=[
            row("mastodon", "photo", "invalid"),
            row("mastodon", "file", "absent"),
        ]))
        self.assertEqual(outcomes["mastodon"]["outcome"], "invalid")

    def test_unreadable_rendition(self):
        outcomes = pages.classify(report(rows=[row("facebook", "photo", "unavailable")]))
        self.assertEqual(outcomes["facebook"]["outcome"], "unavailable")

    def test_missing_credentials_are_not_failures(self):
        outcomes = pages.classify(
            report(skipped=[{"platform": "bluesky", "reason": "needs config: bluesky_handle"}])
        )
        self.assertEqual(outcomes["bluesky"]["outcome"], "unconfigured")

    def test_an_upload_that_failed_is_a_failure(self):
        outcomes = pages.classify(
            report(skipped=[{"platform": "discord", "reason": "upload failed: HTTP 401"}])
        )
        self.assertEqual(outcomes["discord"]["outcome"], "failed")
        self.assertEqual(outcomes["discord"]["reason"], "upload failed: HTTP 401")

    def test_variants_are_sorted_for_a_stable_page(self):
        outcomes = pages.classify(report(rows=[
            row("p", "photo", "absent"),
            row("p", "file", "absent"),
        ]))
        names = [variant["name"] for variant in outcomes["p"]["variants"]]
        self.assertEqual(names, ["file", "photo"])

    def test_expected_platforms_are_recorded_as_unmeasured(self):
        outcomes = pages.classify(None, ["discord", "telegram"])
        self.assertEqual(
            {name: data["outcome"] for name, data in outcomes.items()},
            {"discord": "unmeasured", "telegram": "unmeasured"},
        )

    def test_a_token_in_an_error_is_never_published(self):
        # requests puts the failing URL in its connection error, and Telegram's
        # file URLs carry the bot token. The page is public and permanent.
        reason = (
            "upload failed: HTTPSConnectionPool(host='api.telegram.org', port=443): "
            "Max retries exceeded with url: /bot99:SECRET_TOKEN/sendPhoto"
        )
        outcomes = pages.classify(report(skipped=[{"platform": "telegram", "reason": reason}]))
        self.assertNotIn("SECRET_TOKEN", json.dumps(outcomes))
        self.assertEqual(outcomes["telegram"]["reason"], "upload failed: network error")

    def test_a_webhook_token_in_a_path_is_never_published(self):
        # Discord's webhook token is a path segment, so no URL-shape rule saves
        # us here: the whole message has to stay out of the page.
        reason = (
            "upload failed: HTTP 404 {\"message\": \"Unknown Webhook\", \"url\": "
            "https://discord.com/api/webhooks/12345:TOK/sendPhoto}"
        )
        outcomes = pages.classify(report(skipped=[{"platform": "discord", "reason": reason}]))
        self.assertNotIn("TOK", json.dumps(outcomes))
        self.assertEqual(outcomes["discord"]["reason"], "upload failed: HTTP 404")

    def test_an_unrecognised_error_keeps_its_message_out(self):
        outcomes = pages.classify(
            report(skipped=[{"platform": "x", "reason": "ValueError: bad key sk-live-1234"}])
        )
        self.assertNotIn("sk-live", json.dumps(outcomes))
        self.assertEqual(outcomes["x"]["reason"], "upload failed")

    def test_other_credential_shapes_are_scrubbed(self):
        text = (
            "GET https://user:pa55@example.test/x "
            "Authorization: Bearer abc.def-ghi "
            "?access_token=deadbeef&page=1"
        )
        cleaned = pages.scrub(text)
        for secret in ("pa55", "abc.def-ghi", "deadbeef"):
            self.assertNotIn(secret, cleaned)

    def test_scrub_keeps_ordinary_text(self):
        text = "needs config: telegram_bot_token, telegram_chat_id"
        self.assertEqual(pages.scrub(text), text)


class TestHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "history.json"

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, run_id="r1", generated="2026-10-19T06:00:00+00:00"):
        return pages.run_record(report(generated=generated), run_id=run_id)

    def test_missing_file_starts_empty(self):
        self.assertEqual(pages.load_history(self.path), {"version": 1, "runs": []})

    def test_corrupt_history_is_an_error_not_a_reset(self):
        # Silently starting over would erase the series the file exists for.
        self.path.write_text("{not json")
        with self.assertRaises(ValueError):
            pages.load_history(self.path)

    def test_append_is_oldest_first(self):
        history = {"version": 1, "runs": []}
        pages.append(history, self.record("b", "2026-10-26T06:00:00+00:00"))
        pages.append(history, self.record("a", "2026-10-19T06:00:00+00:00"))
        self.assertEqual([run["id"] for run in history["runs"]], ["a", "b"])

    def test_same_id_replaces_its_column(self):
        history = {"version": 1, "runs": []}
        pages.append(history, self.record("r1"))
        pages.append(history, pages.run_record(report(generated="2026-10-19T07:00:00+00:00"), run_id="r1"))
        self.assertEqual(len(history["runs"]), 1)
        self.assertEqual(history["runs"][0]["generated"], "2026-10-19T07:00:00+00:00")

    def test_old_runs_are_trimmed(self):
        history = {"version": 1, "runs": []}
        for week in range(6):
            pages.append(
                history,
                self.record(f"r{week}", f"2026-0{week + 1}-01T06:00:00+00:00"),
                keep=3,
            )
        self.assertEqual([run["id"] for run in history["runs"]], ["r3", "r4", "r5"])


class TestRender(unittest.TestCase):
    def history(self, count=2):
        history = {"version": 1, "runs": []}
        for week in range(count):
            pages.append(
                history,
                pages.run_record(
                    report(
                        generated=f"2026-10-{19 + week * 7:02d}T06:00:00+00:00",
                        rows=[
                            row("discord", "photo", "absent"),
                            row("telegram", "photo", "valid"),
                        ],
                        skipped=[{"platform": "bluesky", "reason": "needs config: bluesky_handle"}],
                    ),
                    run_id=f"r{week}",
                ),
            )
        return history

    def test_empty_history_renders_a_placeholder(self):
        page = pages.render_page({"version": 1, "runs": []})
        self.assertIn("No runs recorded yet", page)
        self.assertNotIn("<table", page)

    def test_outcomes_become_cell_classes(self):
        page = pages.render_page(self.history())
        self.assertIn("o-gone", page)
        self.assertIn("o-valid", page)
        self.assertIn("o-unconfigured", page)
        self.assertEqual(page.count('class="cell o-gone"'), 2)

    def test_every_platform_gets_a_row(self):
        page = pages.render_page(self.history())
        for name in ("discord", "telegram", "bluesky"):
            self.assertIn(f"<code>{name}</code>", page)

    def test_platform_names_are_escaped(self):
        history = {"version": 1, "runs": []}
        pages.append(
            history,
            pages.run_record(report(rows=[row("<script>alert(1)</script>", "photo", "absent")])),
        )
        page = pages.render_page(history)
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;", page)

    def test_title_is_escaped(self):
        page = pages.render_page({"version": 1, "runs": []}, title="<b>x</b>")
        self.assertNotIn("<b>x</b>", page)

    def test_streak_reports_a_change(self):
        history = {"version": 1, "runs": []}
        for week, verdict in enumerate(["absent", "valid"]):
            pages.append(
                history,
                pages.run_record(
                    report(
                        generated=f"2026-10-{19 + week * 7:02d}T06:00:00+00:00",
                        rows=[row("discord", "photo", verdict)],
                    ),
                    run_id=f"r{week}",
                ),
            )
        page = pages.render_page(history)
        self.assertIn("since 26 Oct 2026", page)
        self.assertIn(">1</td>", page)  # one change

    def test_an_adapter_added_later_shows_gaps_not_false_results(self):
        # A blank cell must read as "that adapter was not in this run", never as
        # a result it did not produce.
        history = {"version": 1, "runs": []}
        pages.append(history, pages.run_record(
            report(generated="2026-10-19T06:00:00+00:00", rows=[row("discord", "photo", "absent")]),
            run_id="r1",
        ))
        pages.append(history, pages.run_record(
            report(generated="2026-10-26T06:00:00+00:00", rows=[
                row("discord", "photo", "absent"),
                row("telegram", "photo", "valid"),
            ]),
            run_id="r2",
        ))
        page = pages.render_page(history)
        self.assertIn("telegram: not part of this run", page)
        self.assertEqual(page.count('class="cell o-unmeasured"'), 1)

    def test_the_window_caps_columns(self):
        history = self.history(count=4)
        page = pages.render_page(history, max_runs=2)
        self.assertIn("Showing the last 2 of 4 runs", page)

    def test_latest_variants_are_listed(self):
        page = pages.render_page(self.history())
        self.assertIn("The latest run, rendition by rendition", page)

    def test_the_verifier_page_is_linked_from_the_page(self):
        for history in ({"version": 1, "runs": []}, self.history()):
            page = pages.render_page(history)
            self.assertIn('href="verifier/"', page)

    def test_no_javascript(self):
        page = pages.render_page(self.history())
        self.assertNotIn("<script", page)
        self.assertNotIn("cdn.", page)


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.site = self.root / "site"

    def tearDown(self):
        self.tmp.cleanup()

    def write_report(self, name, generated, rows):
        path = self.root / name
        path.write_text(json.dumps(report(generated=generated, rows=rows)))
        return path

    def test_two_runs_become_two_columns(self):
        first = self.write_report("a.json", "2026-10-19T06:00:00+00:00", [row("discord", "photo", "absent")])
        second = self.write_report("b.json", "2026-10-26T06:00:00+00:00", [row("discord", "photo", "valid")])
        pages.build(first, self.site, run_id="1")
        pages.build(second, self.site, run_id="2")
        history = json.loads((self.site / "history.json").read_text())
        self.assertEqual(len(history["runs"]), 2)
        page = (self.site / "index.html").read_text()
        self.assertEqual(page.count('class="cell o-gone"'), 1)
        self.assertEqual(page.count('class="cell o-valid"'), 1)

    def test_missing_report_still_records_the_week(self):
        pages.build(
            self.root / "absent.json",
            self.site,
            expected=["discord", "telegram"],
            run_id="42",
        )
        history = json.loads((self.site / "history.json").read_text())
        outcomes = {name: data["outcome"] for name, data in history["runs"][0]["platforms"].items()}
        self.assertEqual(outcomes, {"discord": "unmeasured", "telegram": "unmeasured"})
        self.assertIn("the measurement step produced no report", history["runs"][0]["note"])

    def test_history_carries_no_urls_or_tokens(self):
        result = report(rows=[row("telegram", "photo", "absent")])
        result["rows"][0]["url"] = "https://api.telegram.org/file/bot99:LEAK/x.jpg"
        result["skipped"] = [{"platform": "x", "reason": "upload failed: /bot99:LEAK/sendPhoto"}]
        path = self.root / "leaky.json"
        path.write_text(json.dumps(result))
        pages.build(path, self.site, run_id="1")
        written = (self.site / "history.json").read_text()
        self.assertNotIn("LEAK", written)
        # The page is public, so served URLs are not carried at all.
        self.assertNotIn("http", written)

    def test_build_stages_the_verifier_page_beside_the_trend_page(self):
        report_path = self.write_report(
            "a.json", "2026-10-19T06:00:00+00:00", [row("discord", "photo", "absent")]
        )
        pages.build(report_path, self.site, run_id="1")
        self.assertTrue((self.site / "verifier" / "index.html").is_file())
        self.assertTrue((self.site / "verifier" / "app.js").is_file())

    def test_the_trend_page_links_to_the_bench_map(self):
        # The map is built into this same directory by the workflow, and this
        # page is the only door in: a link that is missing (or spelled
        # differently from the file) leaves the map at its URL with nothing
        # pointing at it.
        report_path = self.write_report(
            "a.json", "2026-10-19T06:00:00+00:00", [row("discord", "photo", "absent")]
        )
        pages.build(report_path, self.site, run_id="1")
        page = (self.site / "index.html").read_text()
        self.assertIn('href="bench-map.html"', page)
        # Relative, like the verifier link: the site is served under a project
        # path, so a leading slash would resolve against the host root.
        self.assertNotIn('href="/bench-map.html"', page)

    def test_the_staged_verifier_links_back_to_the_trend_page(self):
        # The trend page links forward; the verifier page has to lead back, and
        # to this page's own title rather than a second name for it.
        report_path = self.write_report(
            "a.json", "2026-10-19T06:00:00+00:00", [row("discord", "photo", "absent")]
        )
        pages.build(report_path, self.site, run_id="1", title="Platform survival")
        payload = json.loads((self.site / "verifier" / "samples.json").read_text())
        self.assertEqual(payload["home"], {"href": "../", "label": "Platform survival"})

        pages.build(report_path, self.site, run_id="2", title="What platforms keep")
        payload = json.loads((self.site / "verifier" / "samples.json").read_text())
        self.assertEqual(payload["home"]["label"], "What platforms keep")

    def test_verifier_roots_put_the_runs_renditions_on_the_staged_page(self):
        report_path = self.write_report(
            "a.json", "2026-10-19T06:00:00+00:00", [row("discord", "photo", "absent")]
        )
        served = self.root / "live" / "served"
        served.mkdir(parents=True)
        (served / "discord__attachment.jpeg").write_bytes(b"JPEG")
        pages.build(report_path, self.site, run_id="1", verifier_roots=[self.root / "live"])
        payload = json.loads((self.site / "verifier" / "samples.json").read_text())
        self.assertEqual(payload["runs"][0]["name"], "live")
        self.assertEqual(payload["runs"][0]["files"][0]["url"],
                         "samples/live/discord__attachment.jpeg")

    def test_build_is_idempotent_for_the_same_run(self):
        report_path = self.write_report(
            "a.json", "2026-10-19T06:00:00+00:00", [row("discord", "photo", "absent")]
        )
        pages.build(report_path, self.site, run_id="7")
        pages.build(report_path, self.site, run_id="7")
        history = json.loads((self.site / "history.json").read_text())
        self.assertEqual(len(history["runs"]), 1)


@unittest.skipUnless(shutil.which("git"), "git is required to test publishing")
class TestPublish(unittest.TestCase):
    BRANCH = "platform-history"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bare = self.root / "published.git"
        subprocess.run(["git", "init", "--quiet", "--bare", str(self.bare)], check=True)
        self.url = f"file://{self.bare}"
        self.site = self.root / "site"
        self.site.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, run_id, generated, verdict):
        report_path = self.root / f"{run_id}.json"
        report_path.write_text(json.dumps(report(
            generated=generated, rows=[row("discord", "photo", verdict)]
        )))
        pages.build(report_path, self.site, run_id=run_id)

    def published(self, path):
        proc = subprocess.run(
            ["git", "--git-dir", str(self.bare), "show", f"{self.BRANCH}:{path}"],
            capture_output=True, text=True,
        )
        return proc.stdout

    def commits(self):
        proc = subprocess.run(
            ["git", "--git-dir", str(self.bare), "rev-list", "--count", self.BRANCH],
            capture_output=True, text=True,
        )
        return int(proc.stdout.strip())

    def test_first_publish_creates_the_branch(self):
        self.build("1", "2026-10-19T06:00:00+00:00", "absent")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(pages.publish(self.site, push_url=self.url), 0)
        self.assertIn("published", out.getvalue())
        self.assertIn("Platform survival", self.published("index.html"))
        self.assertEqual(self.commits(), 1)

    def test_the_branch_carries_the_verifier_page_as_well(self):
        # The branch is the readable copy of the site, so the page the trend
        # page links to has to be on it, with the bytes it lists.
        served = self.root / "live" / "served"
        served.mkdir(parents=True)
        (served / "discord__attachment.jpeg").write_bytes(b"JPEG")
        report_path = self.root / "r.json"
        report_path.write_text(json.dumps(report(rows=[row("discord", "photo", "absent")])))
        pages.build(report_path, self.site, run_id="1",
                    verifier_roots=[self.root / "live"])
        with contextlib.redirect_stdout(io.StringIO()):
            pages.publish(self.site, push_url=self.url)

        self.assertIn("Verify a content credential", self.published("verifier/index.html"))
        payload = json.loads(self.published("verifier/samples.json"))
        self.assertEqual(payload["runs"][0]["files"][0]["url"],
                         "samples/live/discord__attachment.jpeg")
        tree = subprocess.run(
            ["git", "--git-dir", str(self.bare), "ls-tree", "-r", "--name-only", self.BRANCH],
            capture_output=True, text=True,
        ).stdout
        self.assertIn("verifier/samples/live/discord__attachment.jpeg", tree)

    def test_later_publishes_append_and_keep_the_series(self):
        self.build("1", "2026-10-19T06:00:00+00:00", "absent")
        with contextlib.redirect_stdout(io.StringIO()):
            pages.publish(self.site, push_url=self.url)
        self.build("2", "2026-10-26T06:00:00+00:00", "valid")
        with contextlib.redirect_stdout(io.StringIO()):
            pages.publish(self.site, push_url=self.url)
        self.assertEqual(self.commits(), 2)
        history = json.loads(self.published("history.json"))
        self.assertEqual([run["id"] for run in history["runs"]], ["1", "2"])

    def test_a_fresh_site_dir_does_not_erase_published_columns(self):
        # CI's case: there is no `out/site` from last week on the runner, so the
        # build knows only the run that just finished. Publishing must add that
        # column to the branch's series rather than trade the series for it.
        self.build("1", "2026-10-19T06:00:00+00:00", "absent")
        with contextlib.redirect_stdout(io.StringIO()):
            pages.publish(self.site, push_url=self.url)

        fresh = self.root / "fresh"
        fresh.mkdir()
        report_path = self.root / "second.json"
        report_path.write_text(json.dumps(report(
            generated="2026-10-26T06:00:00+00:00",
            rows=[row("discord", "photo", "valid")],
        )))
        pages.build(report_path, fresh, run_id="2")
        with contextlib.redirect_stdout(io.StringIO()):
            pages.publish(fresh, push_url=self.url)

        history = json.loads(self.published("history.json"))
        self.assertEqual([run["id"] for run in history["runs"]], ["1", "2"])
        page = self.published("index.html")
        self.assertIn("19 Oct", page)
        self.assertIn("26 Oct", page)
        self.assertEqual(page.count('class="cell o-gone"'), 1)
        self.assertEqual(page.count('class="cell o-valid"'), 1)
        # The freshly built directory is what a deployment uploads, so it must
        # carry the merged series too rather than the single column it was built
        # with; otherwise the published page stays a snapshot.
        local = json.loads((fresh / "history.json").read_text())
        self.assertEqual([run["id"] for run in local["runs"]], ["1", "2"])
        self.assertIn("19 Oct", (fresh / "index.html").read_text())

    def test_a_re_run_replaces_its_column_without_dropping_the_others(self):
        self.build("1", "2026-10-19T06:00:00+00:00", "absent")
        with contextlib.redirect_stdout(io.StringIO()):
            pages.publish(self.site, push_url=self.url)
        self.build("2", "2026-10-26T06:00:00+00:00", "absent")
        with contextlib.redirect_stdout(io.StringIO()):
            pages.publish(self.site, push_url=self.url)

        # Same run id as the first column, re-measured, from an empty site dir.
        fresh = self.root / "fresh"
        fresh.mkdir()
        report_path = self.root / "rerun.json"
        report_path.write_text(json.dumps(report(
            generated="2026-10-19T06:05:00+00:00",
            rows=[row("discord", "photo", "valid")],
        )))
        pages.build(report_path, fresh, run_id="1")
        with contextlib.redirect_stdout(io.StringIO()):
            pages.publish(fresh, push_url=self.url)

        history = json.loads(self.published("history.json"))
        self.assertEqual([run["id"] for run in history["runs"]], ["1", "2"])
        first = history["runs"][0]["platforms"]["discord"]["outcome"]
        self.assertEqual(first, "valid")

    def test_republishing_an_unchanged_page_commits_nothing(self):
        self.build("1", "2026-10-19T06:00:00+00:00", "absent")
        with contextlib.redirect_stdout(io.StringIO()):
            pages.publish(self.site, push_url=self.url)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(pages.publish(self.site, push_url=self.url), 0)
        self.assertIn("nothing to publish", out.getvalue())
        self.assertEqual(self.commits(), 1)

    def test_missing_page_is_refused_before_touching_git(self):
        with self.assertRaises(pages.PublishError):
            pages.publish(self.site, push_url=self.url)

    def test_a_token_in_the_push_url_is_not_echoed_on_failure(self):
        # git quotes the remote URL in its errors, and in CI that URL carries
        # the workflow token. No network here: the failure is fed straight to
        # the error path so the assertion is deterministic.
        from unittest import mock

        failure = subprocess.CompletedProcess(
            ["git"], 128, "",
            "fatal: unable to access 'https://x-access-token:TOKEN_SECRET@github.com/o/r.git/': "
            "Could not resolve host: github.com",
        )
        with mock.patch("pages.subprocess.run", return_value=failure):
            with self.assertRaises(pages.PublishError) as caught:
                pages._git(["ls-remote", "origin"], self.root)
        self.assertNotIn("TOKEN_SECRET", str(caught.exception))
        self.assertIn("<redacted>@github.com", str(caught.exception))

    def test_an_unreadable_remote_is_reported(self):
        self.build("1", "2026-10-19T06:00:00+00:00", "absent")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(pages.PublishError):
            pages.publish(self.site, push_url=f"file://{self.root / 'absent.git'}")


if __name__ == "__main__":
    unittest.main()
