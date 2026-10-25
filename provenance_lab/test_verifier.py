"""The verifier server: which renditions it lists, and what it will serve.

The page itself is JavaScript and is exercised in a browser, not here; what is
pinned here is the half this repo can test — that a run's `served/` files are
found, labelled the way the page expects, and served by exact URL only, so a
request cannot climb out of the run directory.
"""

import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import verifier


class ServeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.server = None

    def tearDown(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        self.tmp.cleanup()

    def make_run(self, run="demo", name="discord__attachment.jpeg", data=b"JPEG"):
        served = self.root / run / "served"
        served.mkdir(parents=True, exist_ok=True)
        (served / name).write_bytes(data)
        return served / name

    def start(self, roots=None):
        self.server = verifier.serve(roots if roots is not None else [self.root],
                                     "127.0.0.1", 0, open_browser=False)
        port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{port}"

    def get(self, url):
        try:
            with urllib.request.urlopen(url, timeout=10) as reply:
                return reply.status, reply.headers.get("Content-Type", ""), reply.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers.get("Content-Type", ""), exc.read()


class TestStaticBuild(ServeCase):
    """The published copy: same files, plus a catalogue and its bytes."""

    def build(self, roots=None):
        return verifier.build_site(
            roots if roots is not None else [self.root], self.root / "site"
        )

    def test_the_page_its_bytes_and_a_catalogue_are_staged_as_one_directory(self):
        self.make_run(name="discord__attachment.jpeg", data=b"\xff\xd8JPEG")
        target = self.build()
        self.assertEqual(target, self.root / "site" / "verifier")
        for name in verifier.PAGE_FILES:
            self.assertTrue((target / name).is_file(), name)
        staged = target / "samples" / "demo" / "discord__attachment.jpeg"
        self.assertEqual(staged.read_bytes(), b"\xff\xd8JPEG")

    def test_the_staged_catalogue_matches_what_the_server_would_list(self):
        self.make_run(name="discord__attachment.jpeg")
        self.make_run(name="discord__proxy_webp.webp")
        payload = json.loads((self.build() / "samples.json").read_text(encoding="utf-8"))
        self.assertEqual(payload, verifier.listing([self.root]))

    def test_every_url_is_relative_so_a_project_path_still_resolves(self):
        # Pages serves the site from /<repo>/verifier/, not from the host root,
        # so a leading slash would send the browser somewhere that 404s.
        self.make_run(name="discord__attachment.jpeg")
        payload = json.loads((self.build() / "samples.json").read_text(encoding="utf-8"))
        url = payload["runs"][0]["files"][0]["url"]
        self.assertEqual(url, "samples/demo/discord__attachment.jpeg")
        self.assertFalse(url.startswith("/"))

    def test_a_root_that_was_never_written_still_publishes_the_page(self):
        # The trend page links to this one unconditionally, so a build that has
        # no renditions to offer must still produce the page rather than fail.
        target = self.build([self.root / "never-written"])
        self.assertTrue((target / "index.html").is_file())
        payload = json.loads((target / "samples.json").read_text(encoding="utf-8"))
        self.assertEqual(payload, {"runs": []})

    def test_rebuilding_replaces_the_catalogue_instead_of_appending(self):
        self.make_run(run="live", name="discord__attachment.jpeg")
        self.build()
        payload = json.loads((self.build() / "samples.json").read_text(encoding="utf-8"))
        self.assertEqual([run["name"] for run in payload["runs"]], ["live"])


class TestDiscovery(ServeCase):
    def test_a_run_lists_its_renditions_with_platform_and_variant(self):
        self.make_run(name="discord__attachment.jpeg")
        self.make_run(name="discord__proxy_webp.webp")
        runs = verifier.listing([self.root])["runs"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["name"], "demo")
        labels = [file["label"] for file in runs[0]["files"]]
        self.assertEqual(labels, ["discord · attachment", "discord · proxy webp"])

    def test_files_that_are_not_images_are_left_out(self):
        self.make_run()
        (self.root / "demo" / "served" / "notes.txt").write_text("not a rendition")
        files = verifier.listing([self.root])["runs"][0]["files"]
        self.assertEqual([file["label"] for file in files], ["discord · attachment"])

    def test_a_run_with_no_served_files_is_not_listed(self):
        (self.root / "empty" / "served").mkdir(parents=True)
        self.assertEqual(verifier.listing([self.root])["runs"], [])

    def test_the_newest_run_comes_first(self):
        old = self.make_run(run="older")
        new = self.make_run(run="newer")
        for path, when in ((old, 1_000_000), (new, 2_000_000)):
            os.utime(path, (when, when))
        names = [run["name"] for run in verifier.listing([self.root])["runs"]]
        self.assertEqual(names, ["newer", "older"])

    def test_duplicate_run_names_do_not_collide(self):
        # Two --out roots can each hold a run called `live`, and neither root
        # may contain the other, or the same files would be found twice.
        first, second = self.root / "a", self.root / "b"
        for root in (first, second):
            (root / "live" / "served").mkdir(parents=True)
            (root / "live" / "served" / "local__copy.jpeg").write_bytes(b"JPEG")
        names = [run["name"] for run in verifier.listing([first, second])["runs"]]
        self.assertEqual(sorted(names), ["live", "live (2)"])

    def test_a_single_file_root_is_listed_under_its_directory(self):
        path = self.make_run(run="photo", name="local__copy.jpeg")
        runs = verifier.listing([path])["runs"]
        self.assertEqual(runs[0]["name"], "photo")
        self.assertEqual(runs[0]["files"][0]["label"], "local · copy")


class TestRoutes(ServeCase):
    def test_the_page_and_its_scripts_are_served(self):
        base = self.start([self.root])
        status, content_type, body = self.get(base + "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn(b"Verify a content credential", body)
        for path, expected in (("/app.js", "javascript"), ("/style.css", "text/css")):
            status, content_type, _ = self.get(base + path)
            self.assertEqual((status, expected in content_type), (200, True), path)

    def test_samples_json_names_every_rendition_without_a_path(self):
        self.make_run(name="discord__attachment.jpeg")
        base = self.start([self.root])
        status, content_type, body = self.get(base + "/samples.json")
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        payload = json.loads(body)
        self.assertEqual(payload["runs"][0]["files"][0]["url"],
                         "samples/demo/discord__attachment.jpeg")
        self.assertNotIn(str(self.root).replace("\\", "\\\\"), body.decode())

    def test_a_listed_rendition_is_served_as_an_image(self):
        self.make_run(name="discord__attachment.jpeg", data=b"\xff\xd8JPEGBODY")
        base = self.start([self.root])
        status, content_type, body = self.get(
            base + "/samples/demo/discord__attachment.jpeg")
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "image/jpeg")
        self.assertEqual(body, b"\xff\xd8JPEGBODY")

    def test_a_path_outside_the_run_is_refused(self):
        # `/samples/../platforms.json` must not become the credentials file.
        (self.root / "platforms.json").write_text("{}")
        self.make_run()
        base = self.start([self.root])
        for attack in (
            "/samples/../platforms.json",
            "/samples/demo/../../platforms.json",
            "/samples/demo/%2e%2e/%2e%2e/platforms.json",
            "/samples/other/discord__attachment.jpeg",
        ):
            status, _, _ = self.get(base + attack)
            self.assertEqual(status, 404, attack)

    def test_an_unknown_name_is_refused_even_inside_a_real_run(self):
        self.make_run()
        base = self.start([self.root])
        status, _, _ = self.get(base + "/samples/demo/missing.jpeg")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
