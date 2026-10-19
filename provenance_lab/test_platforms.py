"""Tests for the platform arm.

The third-party adapters are exercised with a stub transport (no network), and
the whole upload -> fetch -> verify path is exercised for real against the local
CDN harness, which speaks HTTP and re-encodes with Pillow.

Run with:  .venv/bin/python -m unittest -v
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import lab
import platforms
from platforms import (
    Bluesky,
    Catbox,
    DiscordWebhook,
    FacebookPage,
    LocalCdn,
    Mastodon,
    TelegramBot,
    Variant,
)


class Reply:
    def __init__(self, status=200, content=b"", headers=None, payload=None):
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload
        if payload is None:
            self.content = content
            self.text = content.decode("utf-8", "replace")
        elif isinstance(payload, str):
            self.content = payload.encode()
            self.text = payload
        else:
            self.content = json.dumps(payload).encode()
            self.text = self.content.decode()

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        if isinstance(self._payload, (dict, list)):
            return self._payload
        return json.loads(self._payload)


class StubHttp:
    """Minimal stand-in for platforms.Http; routes on (method, URL substring)."""

    def __init__(self, routes):
        self.routes = [
            (m, sub, list(r) if isinstance(r, list) else [r]) for m, sub, r in routes
        ]
        self.calls = []

    def _answer(self, method, url, kw):
        self.calls.append((method, url, kw))
        for want_method, sub, queue in self.routes:
            if want_method == method and sub in url and queue:
                item = queue.pop(0) if len(queue) > 1 else queue[0]
                return item(url, kw) if callable(item) else item
        raise AssertionError(f"unexpected request: {method} {url}")

    def get(self, url, headers=None, **kw):
        return self._answer("GET", url, {**kw, "headers": headers})

    def post(self, url, **kw):
        return self._answer("POST", url, kw)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.certs = lab.make_certs(cls.root / "certs")
        cls.signer = lab.load_signer(cls.certs["chain"], cls.certs["key"])
        cls.ca = (cls.root / "certs" / "ca.cert.pem").read_text()
        cls.fixture = lab.make_fixture(cls.root / "source.jpg")
        cls.signed = cls.root / "signed.jpg"
        lab.sign_created(cls.fixture, cls.signed, cls.signer)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def upload_urls(self, adapter):
        upload = adapter.upload(self.signed)
        return upload, {v.name: v for v in upload.variants}


class TestSniff(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(platforms.sniff_format(b"\xff\xd8\xff\xe0rest"), "jpeg")
        self.assertEqual(platforms.sniff_format(b"\x89PNG\r\n\x1a\nrest"), "png")
        self.assertEqual(platforms.sniff_format(b"RIFF\x00\x00\x00\x00WEBPVP8 "), "webp")
        self.assertEqual(platforms.sniff_format(b"GIF89a...."), "gif")
        self.assertEqual(platforms.sniff_format(b"not an image at all"), "unknown")

    def test_dimensions_of_a_real_jpeg(self):
        buf = io.BytesIO()
        Image.new("RGB", (33, 17), (10, 20, 30)).save(buf, format="JPEG")
        self.assertEqual(platforms.dimensions(buf.getvalue()), (33, 17))

    def test_dimensions_of_garbage(self):
        self.assertIsNone(platforms.dimensions(b"nope"))


class TestRedaction(unittest.TestCase):
    def test_telegram_bot_token_in_path(self):
        url = "https://api.telegram.org/file/bot123456:AAH-secret/file_1.jpg"
        self.assertNotIn("AAH-secret", platforms.redact_url(url))
        self.assertIn("<redacted>", platforms.redact_url(url))

    def test_discord_webhook_token(self):
        url = "https://discord.com/api/webhooks/998877/tOkEn-Value?wait=true"
        out = platforms.redact_url(url)
        self.assertNotIn("tOkEn-Value", out)
        self.assertIn("998877", out)

    def test_query_secrets(self):
        out = platforms.redact_url("https://cdn.example/img.jpg?token=abc123&w=800")
        self.assertNotIn("abc123", out)
        self.assertIn("w=800", out)

    def test_access_token_query_is_redacted(self):
        out = platforms.redact_url(
            "https://graph.facebook.com/v26.0/9/photos?access_token=EAAB-secret&x=1"
        )
        self.assertNotIn("EAAB-secret", out)

    def test_facebook_cdn_signature_is_redacted_whole(self):
        """Meta signs the query string; a per-parameter list would leak `ccb`."""
        url = (
            "https://scontent-lhr8-1.xx.fbcdn.net/v/t39.30808-6/abc.jpg"
            "?stp=dst-jpg&_nc_cat=1&ccb=1-7&_nc_sid=9&oh=SECRET&oe=DEAD"
        )
        out = platforms.redact_url(url)
        self.assertNotIn("SECRET", out)
        self.assertNotIn("DEAD", out)
        self.assertIn("scontent-lhr8-1.xx.fbcdn.net/v/t39.30808-6/abc.jpg", out)
        self.assertIn("<redacted>", out)

    def test_error_messages(self):
        msg = platforms.redact_error(
            "POST https://api.telegram.org/bot42:ZZZ/sendPhoto failed: "
            "Authorization: Bearer eyJhbGciOiJI"
        )
        self.assertNotIn("ZZZ", msg)
        self.assertNotIn("eyJhbGciOiJI", msg)

    def test_empty_url(self):
        self.assertEqual(platforms.redact_url(""), "")


class TestConfig(Base):
    def test_every_required_key_is_known(self):
        keys = platforms.config_keys()
        for cls in platforms.registry().values():
            for key in (*cls.requires, *cls.optional):
                self.assertIn(key, keys)

    def test_optional_keys_survive_the_config_filter(self):
        path = self.root / "cfg3.json"
        path.write_text(json.dumps({"facebook_graph_version": "v19.0"}))
        self.assertEqual(platforms.load_config(path)["facebook_graph_version"], "v19.0")

    def test_no_paid_platform_is_registered(self):
        """X's API is credit-based, so its adapter was removed on purpose."""
        self.assertNotIn("x", platforms.registry())
        self.assertNotIn("x_access_token", platforms.config_keys())

    def test_env_fills_a_blank_config(self):
        with mock.patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "9",
        }):
            config = platforms.load_config(None)
        self.assertEqual(config["telegram_bot_token"], "tok")
        self.assertEqual(config["telegram_chat_id"], "9")

    def test_prefixed_env_also_works(self):
        with mock.patch.dict(os.environ, {"PROVENANCE_MASTODON_TOKEN": "m"}):
            self.assertEqual(platforms.load_config(None)["mastodon_token"], "m")

    def test_file_beats_environment(self):
        path = self.root / "cfg.json"
        path.write_text(json.dumps({"discord_webhook": "from-file"}))
        with mock.patch.dict(os.environ, {"DISCORD_WEBHOOK": "from-env"}):
            config = platforms.load_config(path)
        self.assertEqual(config["discord_webhook"], "from-file")

    def test_unknown_json_keys_are_ignored(self):
        path = self.root / "cfg2.json"
        path.write_text(json.dumps({"_comment": "hi"}))
        self.assertNotIn("_comment", platforms.load_config(path))

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(platforms.load_config(self.root / "nope.json"), {})


class TestRequirements(Base):
    def test_missing_keys_are_reported_not_raised(self):
        adapter = TelegramBot({}, http=StubHttp([]))
        ok, why = adapter.available()
        self.assertFalse(ok)
        self.assertIn("telegram_bot_token", why)
        self.assertIn("telegram_chat_id", why)

    def test_present_keys_are_ready(self):
        adapter = Mastodon(
            {"mastodon_instance": "https://example.invalid", "mastodon_token": "t"},
            http=StubHttp([]),
        )
        self.assertEqual(adapter.available(), (True, ""))

    def test_anonymous_hosts_need_nothing(self):
        self.assertTrue(Catbox({}).available()[0])
        self.assertTrue(platforms.ZeroXZero({}).available()[0])


class TestDiscord(Base):
    def test_attachment_and_proxy_variants(self):
        hook = "https://discord.com/api/webhooks/1/abc"
        stub = StubHttp([
            ("POST", "/webhooks/1/abc", Reply(payload={
                "attachments": [{
                    "id": "55",
                    "url": "https://cdn.discordapp.com/attachments/1/2/signed.jpg",
                    "proxy_url": "https://media.discordapp.net/attachments/1/2/signed.jpg?ex=1&is=2&hm=3",
                    "width": 960, "height": 640, "size": 60633, "content_type": "image/jpeg",
                }]
            })),
        ])
        upload, by_name = self.upload_urls(DiscordWebhook({"discord_webhook": hook}, http=stub))
        self.assertEqual(upload.platform, "discord")
        self.assertEqual(set(by_name), {"attachment", "proxy", "proxy_webp"})
        self.assertTrue(by_name["proxy_webp"].url.endswith("&format=webp"))
        self.assertEqual(by_name["proxy_webp"].headers["Accept"], "image/webp")
        self.assertIn("960x640", upload.notes[0])
        # The webhook URL carries the token; the record must not.
        self.assertNotIn("abc", json.dumps(upload.to_dict()))

    def test_no_attachment_is_a_platform_error(self):
        stub = StubHttp([("POST", "/webhooks/", Reply(payload={"attachments": []}))])
        with self.assertRaises(platforms.PlatformError):
            DiscordWebhook({"discord_webhook": "https://discord.com/api/webhooks/9/x"},
                           http=stub).upload(self.signed)


class TestTelegram(Base):
    def test_photo_and_document_pipelines(self):
        token = "12345:SECRET"
        photo = Reply(payload={"ok": True, "result": {"photo": [
            {"file_id": "small", "file_size": 900},
            {"file_id": "big", "file_size": 41000},
        ]}})
        doc = Reply(payload={"ok": True, "result": {"document": {
            "file_id": "doc1", "file_size": 60633,
        }}})
        getfile = Reply(payload={"ok": True, "result": {"file_path": "photos/file_7.jpg"}})
        stub = StubHttp([
            ("POST", "sendPhoto", photo),
            ("POST", "sendDocument", doc),
            ("GET", "getFile", getfile),
        ])
        upload, by_name = self.upload_urls(
            TelegramBot({"telegram_bot_token": token, "telegram_chat_id": "-100"}, http=stub)
        )
        self.assertEqual(set(by_name), {"photo", "document"})
        self.assertIn("file_7.jpg", by_name["photo"].url)
        self.assertIn("largest of 2", by_name["photo"].note)
        self.assertNotIn(token, json.dumps(upload.to_dict()))

    def test_largest_photo_size_is_chosen(self):
        photo = Reply(payload={"ok": True, "result": {"photo": [
            {"file_id": "a", "file_size": 10},
            {"file_id": "c", "file_size": 3000},
            {"file_id": "b", "file_size": 2000},
        ]}})
        seen = []

        def getfile(url, kw):
            seen.append(kw["params"]["file_id"])
            return Reply(payload={"ok": True, "result": {"file_path": "p.jpg"}})

        stub = StubHttp([
            ("POST", "sendPhoto", photo),
            ("POST", "sendDocument", Reply(payload={"ok": False})),
            ("GET", "getFile", getfile),
        ])
        upload = TelegramBot(
            {"telegram_bot_token": "t", "telegram_chat_id": "1"}, http=stub
        ).upload(self.signed)
        self.assertEqual(seen, ["c"])
        # A missing document is recorded, not fatal: the photo arm still ran.
        self.assertEqual([v.name for v in upload.variants], ["photo"])
        self.assertIn("sendDocument returned no document", upload.notes)


class TestTelegramChats(Base):
    UPDATES = {"ok": True, "result": [
        {"update_id": 1, "message": {"chat": {
            "id": 7, "type": "private", "first_name": "Jo", "last_name": "Ng",
        }}},
        {"update_id": 2, "message": {"chat": {
            "id": 7, "type": "private", "first_name": "Jo", "last_name": "Ng",
        }}},
        {"update_id": 3, "my_chat_member": {"chat": {
            "id": -100123, "type": "supergroup", "title": "Provenance lab",
        }}},
        {"update_id": 4, "channel_post": {"chat": {
            "id": -100999, "type": "channel", "title": "Feed", "username": "feed",
        }}},
    ]}

    def stub(self, updates=None, status=200):
        return StubHttp([
            ("GET", "getMe", Reply(payload={"ok": True, "result": {
                "id": 555, "username": "lab_bot", "is_bot": True,
            }})),
            ("GET", "getUpdates", Reply(
                status=status,
                payload=updates if updates is not None else self.UPDATES,
            )),
        ])

    def test_every_chat_is_reported_once(self):
        chats = TelegramBot(
            {"telegram_bot_token": "t"}, http=self.stub()
        ).chats()
        self.assertEqual([c["id"] for c in chats], ["-100999", "7", "-100123"])
        self.assertEqual([c["type"] for c in chats], ["channel", "private", "supergroup"])
        self.assertEqual(chats[1]["title"], "Jo Ng")
        self.assertEqual(chats[2]["title"], "Provenance lab")

    def test_discovery_needs_only_the_token(self):
        """No chat id in config -- that is the value this call exists to find."""
        adapter = TelegramBot({"telegram_bot_token": "t"}, http=self.stub())
        self.assertIn("telegram_chat_id", adapter.missing())
        self.assertEqual(len(adapter.chats()), 3)

    def test_no_offset_is_sent_so_updates_are_not_consumed(self):
        stub = self.stub()
        TelegramBot({"telegram_bot_token": "t"}, http=stub).chats(limit=25)
        params = stub.calls[-1][2]["params"]
        self.assertNotIn("offset", params)
        self.assertEqual(params, {"limit": 25, "timeout": 0})

    def test_webhook_conflict_is_explained(self):
        stub = StubHttp([("GET", "getUpdates", Reply(status=409, payload={
            "ok": False, "error_code": 409,
            "description": "Conflict: can't use getUpdates method while webhook is active",
        }))])
        with self.assertRaises(platforms.PlatformError) as ctx:
            TelegramBot({"telegram_bot_token": "t"}, http=stub).chats()
        self.assertIn("webhook", str(ctx.exception))
        self.assertIn("deleteWebhook", str(ctx.exception))

    def test_bad_token_surfaces_telegrams_description(self):
        stub = StubHttp([("GET", "getUpdates", Reply(status=401, payload={
            "ok": False, "error_code": 401, "description": "Unauthorized",
        }))])
        with self.assertRaises(platforms.PlatformError) as ctx:
            TelegramBot({"telegram_bot_token": "t"}, http=stub).chats()
        self.assertIn("Unauthorized", str(ctx.exception))

    def test_missing_token_sends_nothing(self):
        stub = StubHttp([])
        with self.assertRaises(platforms.PlatformError) as ctx:
            TelegramBot({}, http=stub).chats()
        self.assertIn("telegram_bot_token", str(ctx.exception))
        self.assertEqual(stub.calls, [])

    def test_identity_returns_the_bot(self):
        who = TelegramBot({"telegram_bot_token": "t"}, http=self.stub()).identity()
        self.assertEqual(who["username"], "lab_bot")
        self.assertEqual(who["id"], 555)

    def test_cli_chats_prints_the_ids_and_keeps_the_token_out(self):
        cfg = self.root / "chats-cfg.json"
        cfg.write_text(json.dumps({"telegram_bot_token": "SEGRET-TOKEN"}))
        with mock.patch.object(platforms, "Http", lambda *a, **k: self.stub()), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            code = platforms.main(["chats", "--config", str(cfg)])
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("-100123", text)
        self.assertIn("Provenance lab", text)
        self.assertIn("telegram_chat_id", text)
        self.assertNotIn("SEGRET-TOKEN", text)

    def test_cli_chats_fails_without_a_token(self):
        cfg = self.root / "chats-empty.json"
        cfg.write_text(json.dumps({}))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = platforms.main(["chats", "--config", str(cfg)])
        self.assertEqual(code, 1)
        self.assertIn("telegram_bot_token", out.getvalue())

    def test_cli_chats_rejects_adapters_without_discovery(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = platforms.main(["chats", "--platforms", "catbox", "--config", "none.json"])
        self.assertEqual(code, 1)
        self.assertIn("no chat discovery", out.getvalue())


class TestMastodon(Base):
    def test_polls_until_processed(self):
        base = "https://mastodon.invalid"
        stub = StubHttp([
            ("POST", "/api/v2/media", Reply(status=202, payload={"id": "77", "url": None})),
            ("GET", "/api/v1/media/77", [
                Reply(status=206, payload={"id": "77", "url": None}),
                Reply(payload={
                    "id": "77",
                    "url": "https://files.mastodon.invalid/media/77.jpg",
                    "preview_url": "https://files.mastodon.invalid/media/small/77.jpg",
                    "meta": {"original": {"width": 960, "height": 640, "size": "60633"}},
                }),
            ]),
        ])
        upload, by_name = self.upload_urls(
            Mastodon({"mastodon_instance": base + "/", "mastodon_token": "t"}, http=stub)
        )
        self.assertEqual(set(by_name), {"original", "preview"})
        self.assertEqual(upload.remote_id, "77")
        self.assertIn("60633", upload.notes[0])

    def test_bearer_header_is_sent(self):
        base = "https://mastodon.invalid"
        stub = StubHttp([
            ("POST", "/api/v2/media", Reply(payload={
                "id": "1", "url": "https://x/1.jpg", "preview_url": "https://x/1.jpg",
            })),
        ])
        Mastodon({"mastodon_instance": base, "mastodon_token": "tok"}, http=stub).upload(
            self.signed
        )
        self.assertEqual(stub.calls[0][2]["headers"]["Authorization"], "Bearer tok")


class TestBluesky(Base):
    def test_cdn_variants_carry_the_cid(self):
        stub = StubHttp([
            ("POST", "createSession", Reply(payload={
                "accessJwt": "jwt", "did": "did:plc:abc123",
            })),
            ("POST", "uploadBlob", Reply(payload={"blob": {"ref": {"$link": "bafyrei"}}})),
        ])
        upload, by_name = self.upload_urls(
            Bluesky({"bluesky_handle": "h.invalid", "bluesky_app_password": "pw"}, http=stub)
        )
        self.assertEqual(set(by_name), {"feed_fullsize", "feed_fullsize_webp", "feed_thumbnail"})
        for variant in by_name.values():
            self.assertIn("did:plc:abc123/bafyrei", variant.url)
        self.assertTrue(by_name["feed_fullsize"].url.endswith("@jpeg"))
        self.assertTrue(by_name["feed_fullsize_webp"].url.endswith("@webp"))
        self.assertTrue(by_name["feed_thumbnail"].url.endswith("@jpeg"))
        self.assertIn("feed_thumbnail", by_name["feed_thumbnail"].url)
        self.assertEqual(upload.notes, ["blob is addressed by CID, not filename"])

    def test_app_password_is_not_echoed(self):
        stub = StubHttp([
            ("POST", "createSession", Reply(payload={"accessJwt": "j", "did": "d"})),
            ("POST", "uploadBlob", Reply(payload={"blob": {"ref": {"$link": "c"}}})),
        ])
        upload, _ = self.upload_urls(
            Bluesky({"bluesky_handle": "h", "bluesky_app_password": "hunter2"}, http=stub)
        )
        self.assertNotIn("hunter2", json.dumps(upload.to_dict()))


class TestFacebook(Base):
    CONFIG = {"facebook_page_id": "42", "facebook_token": "EAAB-SECRET"}

    def stub(self, images=None, webp=None):
        return StubHttp([
            ("POST", "/42/photos", Reply(payload={"id": "900"})),
            ("GET", "/900", Reply(payload={
                "id": "900", "width": 1200, "height": 800,
                "images": images if images is not None else [
                    {"source": "https://scontent.x.fbcdn.net/a.jpg?oh=1", "width": 480, "height": 320},
                    {"source": "https://scontent.x.fbcdn.net/c.jpg?oh=3", "width": 1200, "height": 800},
                    {"source": "https://scontent.x.fbcdn.net/b.jpg?oh=2", "width": 960, "height": 640},
                ],
                "webp_images": webp if webp is not None else [
                    {"source": "https://scontent.x.fbcdn.net/w.webp?oh=4", "width": 1200, "height": 800},
                ],
            })),
        ])

    def test_every_rendition_sorted_largest_first(self):
        upload, by_name = self.upload_urls(FacebookPage(self.CONFIG, http=self.stub()))
        self.assertEqual(upload.remote_id, "900")
        self.assertEqual(
            [v.name for v in upload.variants],
            ["image_1200x800", "image_960x640", "image_480x320", "webp_1200x800"],
        )
        self.assertEqual(by_name["image_1200x800"].url, "https://scontent.x.fbcdn.net/c.jpg?oh=3")

    def test_upload_is_unpublished_and_bearer_authenticated(self):
        stub = self.stub()
        FacebookPage(self.CONFIG, http=stub).upload(self.signed)
        method, url, kw = stub.calls[0]
        self.assertEqual(method, "POST")
        self.assertIn("/v26.0/42/photos", url)
        self.assertEqual(kw["data"]["published"], "false")
        self.assertEqual(kw["headers"]["Authorization"], "Bearer EAAB-SECRET")
        self.assertEqual(list(kw["files"]), ["source"])
        # The token goes in a header, never the query string.
        self.assertNotIn("access_token", url)

    def test_graph_version_override(self):
        stub = self.stub()
        FacebookPage({**self.CONFIG, "facebook_graph_version": "19.0"}, http=stub).upload(
            self.signed
        )
        self.assertIn("/v19.0/42/photos", stub.calls[0][1])

    def test_no_renditions_is_a_platform_error(self):
        with self.assertRaises(platforms.PlatformError) as ctx:
            FacebookPage(self.CONFIG, http=self.stub(images=[], webp=[])).upload(self.signed)
        self.assertIn("no image renditions", str(ctx.exception))

    def test_duplicate_sources_become_one_variant(self):
        same = {"source": "https://scontent.x.fbcdn.net/a.jpg?oh=1", "width": 640, "height": 480}
        _, by_name = self.upload_urls(
            FacebookPage(self.CONFIG, http=self.stub(images=[same, dict(same)], webp=[]))
        )
        self.assertEqual(list(by_name), ["image_640x480"])

    def test_two_qualities_at_one_size_keep_distinct_names(self):
        a = {"source": "https://scontent.x.fbcdn.net/a.jpg?oh=1", "width": 640, "height": 480}
        b = {"source": "https://scontent.x.fbcdn.net/b.jpg?oh=2", "width": 640, "height": 480}
        _, by_name = self.upload_urls(
            FacebookPage(self.CONFIG, http=self.stub(images=[a, b], webp=[]))
        )
        self.assertEqual(list(by_name), ["image_640x480", "image_640x480_2"])
        self.assertNotEqual(by_name["image_640x480"].url, by_name["image_640x480_2"].url)

    def test_token_is_not_in_the_record(self):
        upload, _ = self.upload_urls(FacebookPage(self.CONFIG, http=self.stub()))
        self.assertNotIn("SECRET", json.dumps(upload.to_dict()))


class TestCredentialCheck(Base):
    def test_facebook_page_token_is_recognised(self):
        stub = StubHttp([("GET", "/me", Reply(payload={"id": "42", "name": "Lab Page"}))])
        ok, detail = FacebookPage(
            {"facebook_page_id": "42", "facebook_token": "t"}, http=stub
        ).check()
        self.assertTrue(ok)
        self.assertIn("Lab Page", detail)

    def test_facebook_user_token_is_diagnosed(self):
        stub = StubHttp([("GET", "/me", Reply(payload={"id": "7", "name": "A Person"}))])
        ok, detail = FacebookPage(
            {"facebook_page_id": "42", "facebook_token": "t"}, http=stub
        ).check()
        self.assertFalse(ok)
        self.assertIn("user token", detail)
        self.assertIn("/me/accounts", detail)

    def test_facebook_error_body_is_surfaced(self):
        stub = StubHttp([("GET", "/me", Reply(status=400, payload={"error": {
            "message": "An active access token must be used", "code": 2500,
        }}))])
        ok, detail = FacebookPage(
            {"facebook_page_id": "42", "facebook_token": "t"}, http=stub
        ).check()
        self.assertFalse(ok)
        self.assertIn("active access token", detail)

    def test_cli_check_exit_status(self):
        cfg = self.root / "check-cfg.json"
        cfg.write_text(json.dumps({"facebook_page_id": "42", "facebook_token": "t"}))
        good = StubHttp([("GET", "/me", Reply(payload={"id": "42", "name": "Lab"}))])
        with mock.patch.object(platforms, "Http", lambda *a, **k: good), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            codes = [platforms.main(["check", "--platforms", "facebook", "--config", str(cfg)])]
        self.assertEqual(codes, [0])
        self.assertIn("ok", out.getvalue())

        bad = StubHttp([("GET", "/me", Reply(status=400, payload={
            "error": {"message": "An active access token must be used", "code": 2500},
        }))])
        with mock.patch.object(platforms, "Http", lambda *a, **k: bad), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                platforms.main(["check", "--platforms", "facebook", "--config", str(cfg)]), 1
            )

    def test_cli_check_skips_adapters_without_a_preflight(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = platforms.main(["check", "--platforms", "local", "--config", "nope.json"])
        self.assertEqual(code, 0)
        self.assertIn("no read-only preflight", out.getvalue())


class TestCatbox(Base):
    def test_reads_the_url_from_plain_text(self):
        stub = StubHttp([
            ("POST", "catbox.moe", Reply(payload="https://files.catbox.moe/abc.jpg")),
        ])
        upload, by_name = self.upload_urls(Catbox({}, http=stub))
        self.assertEqual(upload.page_url, "https://files.catbox.moe/abc.jpg")
        self.assertEqual(upload.remote_id, "abc.jpg")
        self.assertEqual(list(by_name), ["direct"])

    def test_error_body_is_surfaceable(self):
        stub = StubHttp([("POST", "catbox.moe", Reply(status=503, payload="nope"))])
        with self.assertRaises(platforms.PlatformError) as ctx:
            Catbox({}, http=stub).upload(self.signed)
        self.assertIn("503", str(ctx.exception))


class TestMeasureVariant(Base):
    def test_404_is_unavailable_not_a_crash(self):
        stub = StubHttp([("GET", "cdn.invalid", Reply(status=404, payload="gone"))])
        row = platforms.measure_variant(
            "x", Variant("v", "https://cdn.invalid/a.jpg"), self.signed,
            self.root / "mv", self.ca, stub,
        )
        self.assertEqual(row["status"], 404)
        self.assertEqual(row["verdict"], "unavailable")
        self.assertEqual(row["failures"], ["http_404"])

    def test_fetch_error_is_unavailable(self):
        def boom(url, kw):
            raise ConnectionResetError("peer reset")

        stub = StubHttp([("GET", "cdn.invalid", boom)])
        row = platforms.measure_variant(
            "x", Variant("v", "https://cdn.invalid/a.jpg"), self.signed,
            self.root / "mv2", self.ca, stub,
        )
        self.assertIsNone(row["status"])
        self.assertEqual(row["verdict"], "unavailable")
        self.assertEqual(row["failures"], ["fetch_error:ConnectionResetError"])

    def test_empty_body_is_unavailable(self):
        stub = StubHttp([("GET", "cdn.invalid", Reply(status=200, content=b""))])
        row = platforms.measure_variant(
            "x", Variant("v", "https://cdn.invalid/a.jpg"), self.signed,
            self.root / "mv3", self.ca, stub,
        )
        self.assertEqual(row["verdict"], "unavailable")


class TestLocalCdnEndToEnd(Base):
    """The whole arm, over real HTTP, against a Pillow re-encoder."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.out = cls.root / "platforms"
        cls.result = platforms.run(
            cls.out,
            ["local"],
            {},
            cls.ca,
            {"ca": cls.certs["ca"], "chain": cls.certs["chain"], "key": cls.certs["key"]},
        )
        cls.rows = {r["variant"]: r for r in cls.result["rows"]}

    def test_all_four_policies_were_served(self):
        self.assertEqual(set(self.rows), set(platforms.LOCAL_POLICIES))
        for row in self.rows.values():
            self.assertEqual(row["status"], 200)

    def test_copy_policy_is_byte_identical_and_valid(self):
        row = self.rows["copy"]
        self.assertTrue(row["sha256_match"])
        self.assertEqual(row["verdict"], "valid")
        self.assertEqual(row["state"], "Trusted")
        self.assertTrue(row["data_hash_ok"])
        self.assertTrue(row["trusted"])

    def test_strip_policy_loses_the_credential(self):
        row = self.rows["strip"]
        self.assertFalse(row["present"])
        self.assertFalse(row["sha256_match"])
        self.assertEqual(row["verdict"], "absent")
        self.assertLess(row["served_bytes"], self.rows["copy"]["served_bytes"])

    def test_segment_policy_is_present_but_invalid(self):
        row = self.rows["segments"]
        self.assertTrue(row["present"])
        self.assertTrue(row["claim_readable"])
        self.assertTrue(row["ai_claim"])
        self.assertFalse(row["data_hash_ok"])
        self.assertEqual(row["verdict"], "invalid")
        self.assertEqual(row["failures"], ["assertion.dataHash.mismatch"])

    def test_webp_policy_changes_the_format_and_loses_it(self):
        row = self.rows["webp"]
        self.assertEqual(row["served_format"], "webp")
        self.assertEqual(row["content_type"], "image/webp")
        self.assertEqual(row["verdict"], "absent")

    def test_report_files_and_summary(self):
        self.assertTrue((self.out / "report.md").exists())
        self.assertTrue((self.out / "platforms.json").exists())
        self.assertTrue((self.out / "refetch.json").exists())
        self.assertTrue((self.out / "uploads" / "local.json").exists())
        stored = json.loads((self.out / "platforms.json").read_text())
        stats = stored["summary"]["platforms"]["local"]
        self.assertEqual(stats["variants"], 4)
        self.assertEqual(stats["valid"], 1)
        self.assertEqual(stats["invalid"], 1)
        self.assertEqual(stats["byte_identical"], 1)
        self.assertEqual(sorted(stats["formats"]), ["jpeg", "webp"])
        text = (self.out / "report.md").read_text()
        self.assertIn("| `local` | segments |", text)
        self.assertIn("INVALID", text)

    def test_served_files_are_written(self):
        served = self.out / "served"
        self.assertTrue((served / "local__copy.jpeg").exists())
        self.assertTrue((served / "local__webp.webp").exists())

    def test_upload_record_keeps_the_variant_queries(self):
        record = json.loads((self.out / "uploads" / "local.json").read_text())
        self.assertEqual(
            [v["name"] for v in record["upload"]["variants"]],
            list(platforms.LOCAL_POLICIES),
        )
        for variant in record["upload"]["variants"]:
            self.assertIn("policy=", variant["url"])


class TestSummarise(unittest.TestCase):
    def test_counts_and_ratios(self):
        rows = [
            {"platform": "p", "variant": "a", "verdict": "valid", "sha256_match": True,
             "served_format": "jpeg", "served_bytes": 100, "uploaded_bytes": 100,
             "claim_readable": True},
            {"platform": "p", "variant": "b", "verdict": "invalid", "sha256_match": False,
             "served_format": "jpeg", "served_bytes": 60, "uploaded_bytes": 100,
             "claim_readable": True},
            {"platform": "p", "variant": "c", "verdict": "absent", "sha256_match": False,
             "served_format": "webp", "served_bytes": 40, "uploaded_bytes": 100,
             "claim_readable": False},
            {"platform": "p", "variant": "d", "verdict": "unavailable", "sha256_match": False,
             "served_format": None, "served_bytes": 0, "uploaded_bytes": 100,
             "claim_readable": False},
        ]
        stats = platforms.summarise(rows)["platforms"]["p"]
        self.assertEqual(stats["variants"], 4)
        self.assertEqual(stats["valid"], 1)
        self.assertEqual(stats["invalid"], 1)
        self.assertEqual(stats["absent"], 1)
        self.assertEqual(stats["unavailable"], 1)
        self.assertEqual(stats["byte_identical"], 1)
        self.assertEqual(stats["formats"], ["jpeg", "webp"])
        self.assertEqual(stats["max_ratio"], 1.0)


class TestReportRedaction(Base):
    def test_tokens_never_reach_the_report(self):
        result = {
            "meta": {
                "generated": "now", "c2pa_python": "0", "c2pa_sdk": "0",
                "fixture": "1x1", "signed_bytes": 10, "elapsed_s": 0,
                "signing_ca": "test",
            },
            "uploads": [{
                "platform": "telegram",
                "remote_id": "x",
                "page_url": "",
                "variants": [{
                    "name": "photo",
                    "url": platforms.redact_url(
                        "https://api.telegram.org/file/bot99:LEAK/file.jpg"),
                    "note": "",
                }],
                "notes": [],
            }],
            "skipped": [],
            "rows": [{
                "platform": "telegram", "variant": "photo", "status": 200,
                "url": platforms.redact_url(
                    "https://api.telegram.org/file/bot99:LEAK/file.jpg"),
                "served_format": "jpeg", "served_dims": [1, 1], "served_bytes": 1,
                "sha256_match": False, "verdict": "absent", "state": None,
                "failures": [],
            }],
        }
        result["summary"] = {"rows": 1, "platforms": {
            "telegram": {"variants": 1, "absent": 1, "invalid": 0, "valid": 0,
                         "byte_identical": 0, "formats": ["jpeg"]},
        }}
        text = platforms.render_markdown(result)
        self.assertNotIn("LEAK", text)
        # The row URL lives in the JSON, where it must already be redacted.
        serialised = json.dumps(result)
        self.assertNotIn("LEAK", serialised)
        self.assertIn("<redacted>", serialised)
        self.assertIn("<redacted>", result["uploads"][0]["variants"][0]["url"])


class TokenAdapter(platforms.Adapter):
    """An adapter whose served URL embeds a credential, like Telegram's."""

    name = "fake"
    label = "fake"

    def upload(self, path):
        return platforms.Upload(
            self.name,
            "id",
            "",
            [Variant("v", "http://127.0.0.1:1/bot1:SECRET/file.jpg")],
        )


class TestRefetchSplit(Base):
    """Shareable artifacts carry redacted URLs; refetch.json carries the real ones."""

    def test_upload_record_is_redacted_but_refetch_is_not(self):
        out = self.root / "split"
        record = platforms.measure_platform(
            TokenAdapter({}, http=platforms.Http(timeout=1)),
            self.signed,
            out,
            self.ca,
        )
        self.assertNotIn("SECRET", record["upload"]["variants"][0]["url"])
        self.assertIn("SECRET", record["refetch"][0]["url"])
        self.assertNotIn("SECRET", (out / "uploads" / "fake.json").read_text())

    def test_write_refetch_is_owner_only(self):
        out = self.root / "split2"
        out.mkdir(parents=True, exist_ok=True)
        path = out / "refetch.json"
        used, warning = platforms.write_refetch(
            path, [{"platform": "fake", "name": "v", "url": "https://x/y", "headers": {}}]
        )
        self.assertEqual(used, path)
        self.assertIsNone(warning)
        self.assertEqual(used.stat().st_mode & 0o777, 0o600)

    def test_refetch_moves_out_when_chmod_is_ignored(self):
        """9p/drvfs silently ignores chmod; the file must not stay world-readable."""
        import os as _os
        from unittest import mock

        out = self.root / "split4"
        out.mkdir(parents=True, exist_ok=True)
        state = self.root / "state"
        real_chmod = _os.chmod

        def drvfs_chmod(target, mode, *args, **kwargs):
            if Path(target).is_relative_to(out):
                return  # accepted and ignored, exactly like drvfs
            return real_chmod(target, mode, *args, **kwargs)

        with mock.patch.dict(_os.environ, {"XDG_STATE_HOME": str(state)}), \
                mock.patch("platforms.os.chmod", side_effect=drvfs_chmod):
            used, warning = platforms.write_refetch(
                out / "refetch.json",
                [{"platform": "fake", "name": "v", "url": "https://x/SECRET", "headers": {}}],
            )
        self.assertFalse((out / "refetch.json").exists(), "the copy must be removed")
        self.assertIn(str(state / "provenance-lab"), str(used))
        self.assertIn("SECRET", used.read_text())
        self.assertIsNone(warning)
        self.assertEqual(used.stat().st_mode & 0o777, 0o600)

    def test_refetch_warns_when_no_location_is_safe(self):
        import os as _os
        from unittest import mock

        out = self.root / "split5"
        out.mkdir(parents=True, exist_ok=True)
        with mock.patch.dict(_os.environ, {"XDG_STATE_HOME": str(out / "state")}), \
                mock.patch("platforms.os.chmod", side_effect=lambda *a, **k: None):
            used, warning = platforms.write_refetch(out / "refetch.json", [])
        self.assertIsNotNone(warning)
        self.assertEqual(used, out / "refetch.json")

    def test_refetch_targets_prefers_the_sibling_file(self):
        out = self.root / "split3"
        out.mkdir(parents=True, exist_ok=True)
        platforms.write_refetch(out / "refetch.json", [
            {"platform": "fake", "name": "v", "url": "https://real/SECRET", "headers": {}},
        ])
        record = {"uploads": [{
            "platform": "fake",
            "variants": [{"name": "v", "url": "https://real/<redacted>"}],
        }]}
        targets = platforms.refetch_targets(out / "platforms.json", record)
        self.assertEqual(targets[0]["url"], "https://real/SECRET")

    def test_refetch_targets_falls_back_to_the_record(self):
        record = {"uploads": [{
            "platform": "fake",
            "variants": [{"name": "v", "url": "https://cdn/a.jpg"}],
        }]}
        targets = platforms.refetch_targets(self.root / "nothing.json", record)
        self.assertEqual(targets, [{"platform": "fake", "name": "v",
                                    "url": "https://cdn/a.jpg", "headers": {}}])

    def test_targets_file_can_be_passed_directly(self):
        payload = {"targets": [{"platform": "p", "name": "n", "url": "u"}]}
        self.assertEqual(
            platforms.refetch_targets(self.root / "x.json", payload), payload["targets"]
        )


class TestLocalCdnServer(Base):
    def test_unknown_policy_is_a_500(self):
        import urllib.error
        import urllib.request

        with LocalCdn() as server:
            data = self.signed.read_bytes()
            req = urllib.request.Request(
                server.origin + "/upload", data=data, method="POST",
                headers={"Content-Type": "image/jpeg"},
            )
            key = json.loads(urllib.request.urlopen(req).read())["id"]
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"{server.origin}/media/{key}?policy=nope")
            self.assertEqual(ctx.exception.code, 500)

    def test_missing_key_is_a_404(self):
        import urllib.error
        import urllib.request

        with LocalCdn() as server:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"{server.origin}/media/nosuchkey")
            self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
