"""Platform arm: does the rendition a real service serves still carry the credential?

The local matrix (``lab.py``) measures what an editing tool does to a file. This
measures what a *platform* does: upload a signed fixture, then fetch the bytes
the platform actually serves and verify those.

Every platform is reduced to the same thing -- an upload that yields one or more
served URLs -- so the measurement is identical everywhere:

    upload  ->  fetch each served URL  ->  verify the fetched bytes

Two variants matter per platform and are recorded separately when they exist:
the pipeline that recompresses (a chat "photo", a feed thumbnail) and the one
that stores bytes verbatim (a chat "document", a raw attachment). They usually
disagree, and that disagreement is the result.

Credentials are read from a JSON file that is never committed and never printed;
reports carry redacted URLs only.

Usage:
    python platforms.py list
    python platforms.py check --platforms facebook
    python platforms.py chats --platforms telegram
    python platforms.py run --platforms local
    python platforms.py run --platforms 0x0
    python platforms.py run --platforms telegram --config platforms.json
    python platforms.py reuse --from out/platforms/platforms.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

import requests
from PIL import Image

import lab

VERSION = "0.1.0"
USER_AGENT = f"provenance-lab/{VERSION}"
DEFAULT_TIMEOUT = 30
CONFIG_NAME = "platforms.json"


class PlatformError(RuntimeError):
    """An upload step failed; the run records it and moves on."""


# --------------------------------------------------------------------------
# redaction
# --------------------------------------------------------------------------

_TELEGRAM_PATH = re.compile(r"/bot[^/]+/")
_DISCORD_HOOK = re.compile(r"(https://(?:discord|discordapp)\.com/api/webhooks/[^/]+/)[^/?#]+")
_QUERY_SECRET = re.compile(
    r"([?&](?:token|key|sig|signature|hm|X-Amz-Signature|access_token)=)[^&#]*", re.I
)
# Facebook's CDN signs a whole query string rather than one named parameter
# (`oh`, `oe`, `_nc_sid`, `_nc_ohc`, `ccb`, `dl`), and Meta documents neither
# which ones matter nor how long they last, so the entire query goes.
_FBCDN_QUERY = re.compile(r"(https?://[^/?#]+(?:fbcdn\.net|cdninstagram\.com)/[^?#]*)\?[^#]*")


def redact_url(url: str) -> str:
    """Strip credentials out of a URL so it can go in a report."""
    if not url:
        return url
    out = _TELEGRAM_PATH.sub("/bot<redacted>/", url)
    out = _DISCORD_HOOK.sub(r"\1<redacted>", out)
    out = _FBCDN_QUERY.sub(r"\1?<redacted>", out)
    out = _QUERY_SECRET.sub(r"\1<redacted>", out)
    return out


def redact_error(exc: BaseException | str) -> str:
    text = str(exc)
    text = _TELEGRAM_PATH.sub("/bot<redacted>/", text)
    text = _DISCORD_HOOK.sub(r"\1<redacted>", text)
    text = _FBCDN_QUERY.sub(r"\1?<redacted>", text)
    text = re.sub(r"(?i)(bearer|token|password|secret)[\"'=: ]+[^\s\"',}]+", r"\1 <redacted>", text)
    return text[:300]


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------


class Http:
    """Thin requests wrapper. Tests substitute an object with the same methods."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.session = requests.Session()

    def get(self, url: str, headers: dict[str, str] | None = None, **kw) -> Any:
        return self.session.get(
            url,
            headers={"User-Agent": USER_AGENT, **(headers or {})},
            timeout=self.timeout,
            allow_redirects=True,
            **kw,
        )

    def post(self, url: str, **kw) -> Any:
        headers = {"User-Agent": USER_AGENT, **kw.pop("headers", {})}
        return self.session.post(url, headers=headers, timeout=self.timeout, **kw)



# --------------------------------------------------------------------------
# bytes inspection
# --------------------------------------------------------------------------


def sniff_format(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        return {b"avif": "avif", b"avis": "avif"}.get(brand, "heif")
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return "unknown"


def dimensions(data: bytes) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.size
    except Exception:
        return None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mime_for(path: Path, data: bytes | None = None) -> str:
    fmt = sniff_format(data) if data is not None else None
    return {
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(fmt or "", "application/octet-stream")


# --------------------------------------------------------------------------
# what an upload produces
# --------------------------------------------------------------------------


@dataclass
class Variant:
    """One URL a platform serves the uploaded bytes from."""

    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    note: str = ""


@dataclass
class Upload:
    platform: str
    remote_id: str
    page_url: str
    variants: list[Variant]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "remote_id": self.remote_id,
            "page_url": redact_url(self.page_url),
            "variants": [
                {"name": v.name, "url": redact_url(v.url), "note": v.note}
                for v in self.variants
            ],
            "notes": self.notes,
        }


class Adapter:
    """Upload one file, return the URLs that serve it back."""

    name: str = ""
    label: str = ""
    requires: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()  # tuning keys, not credentials
    public: bool = False  # upload lands somewhere reachable by anyone

    def __init__(self, config: dict, http: Http | None = None):
        self.config = config
        self.http = http or Http()

    def missing(self) -> list[str]:
        return [k for k in self.requires if not self.config.get(k)]

    def available(self) -> tuple[bool, str]:
        missing = self.missing()
        if missing:
            return False, f"needs config: {', '.join(missing)}"
        return True, ""

    def upload(self, path: Path) -> Upload:  # pragma: no cover - interface
        raise NotImplementedError



def _raise_for(reply: Any, what: str) -> None:
    if reply.status_code >= 400:
        body = reply.text[:200] if hasattr(reply, "text") else ""
        raise PlatformError(f"{what} failed: HTTP {reply.status_code} {body}".strip())


# --------------------------------------------------------------------------
# platform: 0x0.st -- anonymous host, stores bytes verbatim (control)
# --------------------------------------------------------------------------


class ZeroXZero(Adapter):
    name = "0x0"
    label = "0x0.st anonymous file host (currently rejecting new uploads)"
    public = True
    BASE = "https://0x0.st"

    def upload(self, path: Path) -> Upload:
        data = path.read_bytes()
        reply = self.http.post(
            self.BASE + "/",
            data={"expires": "1"},
            files={"file": (path.name, data, mime_for(path, data))},
        )
        _raise_for(reply, "0x0.st upload")
        url = reply.text.strip().splitlines()[0]
        if not url.startswith("http"):
            raise PlatformError("0x0.st returned no URL")
        return Upload(
            self.name,
            url.rsplit("/", 1)[-1],
            url,
            [Variant("direct", url)],
            ["anonymous host; policy is to serve the stored file unchanged"],
        )


# --------------------------------------------------------------------------
# platform: catbox.moe -- anonymous host, stores bytes verbatim (control)
# --------------------------------------------------------------------------


class Catbox(Adapter):
    """A real third party that stores and serves the bytes unchanged.

    Without a control like this, a credential vanishing at a real service could
    just mean this lab's transport is broken rather than the service re-encoding.
    """

    name = "catbox"
    label = "catbox.moe anonymous host (expected: byte-identical, credential intact)"
    public = True
    BASE = "https://catbox.moe/user/api.php"

    def upload(self, path: Path) -> Upload:
        data = path.read_bytes()
        reply = self.http.post(
            self.BASE,
            data={"reqtype": "fileupload"},
            files={"fileToUpload": (path.name, data, mime_for(path, data))},
        )
        _raise_for(reply, "catbox upload")
        url = reply.text.strip().splitlines()[0]
        if not url.startswith("http"):
            raise PlatformError(f"catbox returned no URL: {url[:120]}")
        return Upload(
            self.name,
            url.rsplit("/", 1)[-1],
            url,
            [Variant("direct", url)],
            ["permanent public URL; serves the stored file unchanged"],
        )


# --------------------------------------------------------------------------
# platform: Discord webhook
# --------------------------------------------------------------------------


class DiscordWebhook(Adapter):
    name = "discord"
    label = "Discord attachment + media proxy"
    requires = ("discord_webhook",)
    public = True

    def upload(self, path: Path) -> Upload:
        data = path.read_bytes()
        hook = self.config["discord_webhook"]
        reply = self.http.post(
            hook + ("&" if "?" in hook else "?") + "wait=true",
            data={"payload_json": json.dumps({"content": "provenance-lab test fixture"})},
            files={"files[0]": (path.name, data, mime_for(path, data))},
        )
        _raise_for(reply, "discord upload")
        body = reply.json()
        attachments = body.get("attachments") or []
        if not attachments:
            raise PlatformError("discord returned no attachment")
        att = attachments[0]
        direct = att.get("url") or ""
        proxy = att.get("proxy_url") or ""
        variants = [Variant("attachment", direct)] if direct else []
        if proxy:
            variants.append(Variant("proxy", proxy))
            # media.discordapp.net will re-encode on demand; ask for WebP.
            variants.append(
                Variant("proxy_webp", proxy + "&format=webp", {"Accept": "image/webp"})
            )
        if not variants:
            raise PlatformError("discord attachment had no URL")
        notes = [
            f"reported {att.get('width')}x{att.get('height')} "
            f"{att.get('content_type')} {att.get('size')} bytes"
        ]
        return Upload(self.name, str(att.get("id", "")), direct, variants, notes)


# --------------------------------------------------------------------------
# platform: Telegram bot (photo pipeline vs document pipeline)
# --------------------------------------------------------------------------

# Every update kind that carries a chat object, so a bot that has only ever
# been added to a group (my_chat_member) is still discoverable.
_CHAT_UPDATE_KINDS = (
    "message",
    "edited_message",
    "channel_post",
    "edited_channel_post",
    "my_chat_member",
    "chat_member",
    "chat_join_request",
)


def _chat_summary(chat: dict) -> dict:
    title = chat.get("title") or " ".join(
        part for part in (chat.get("first_name"), chat.get("last_name")) if part
    )
    if not title and chat.get("username"):
        title = f"@{chat['username']}"
    return {
        "id": str(chat["id"]),
        "type": chat.get("type") or "unknown",
        "title": title,
    }


def _as_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class TelegramBot(Adapter):
    name = "telegram"
    label = "Telegram photo (recompressed) vs document (stored)"
    requires = ("telegram_bot_token", "telegram_chat_id")
    BASE = "https://api.telegram.org"

    def _token(self) -> str:
        token = self.config.get("telegram_bot_token")
        if not token:
            raise PlatformError("needs config: telegram_bot_token")
        return token

    def identity(self) -> dict:
        """Who the token belongs to, so a report can name the bot."""
        reply = self.http.get(f"{self.BASE}/bot{self._token()}/getMe")
        _raise_for(reply, "telegram getMe")
        body = reply.json()
        if not body.get("ok"):
            raise PlatformError(f"telegram getMe rejected: {body.get('description', '')}")
        return body.get("result") or {}

    def chats(self, limit: int = 100) -> list[dict]:
        """Chats this bot has already seen, read from getUpdates.

        No `offset` is sent, so pending updates are neither confirmed nor
        consumed: the bot's backlog is left exactly as it was and the call can
        be repeated. Telegram retains updates for about 24 hours, so a chat the
        bot has not been messaged in since then will not appear.

        Needs only the token -- this is how you find a chat id to put in the
        config in the first place.
        """
        reply = self.http.get(
            f"{self.BASE}/bot{self._token()}/getUpdates",
            params={"limit": limit, "timeout": 0},
        )
        if reply.status_code == 409:
            raise PlatformError(
                "getUpdates is refused while a webhook is set; delete it with "
                "deleteWebhook (or read the chat the webhook feeds) and try again"
            )
        _raise_for(reply, "telegram getUpdates")
        body = reply.json()
        if not body.get("ok"):
            raise PlatformError(
                f"telegram getUpdates rejected: {body.get('description', '')}"
            )
        seen: dict[str, dict] = {}
        for update in body.get("result") or []:
            for key in _CHAT_UPDATE_KINDS:
                chat = (update.get(key) or {}).get("chat") or {}
                if "id" in chat and str(chat["id"]) not in seen:
                    seen[str(chat["id"])] = _chat_summary(chat)
        return sorted(seen.values(), key=lambda c: (c["type"], _as_int(c["id"])))

    def _file_url(self, file_id: str) -> str:
        token = self.config["telegram_bot_token"]
        reply = self.http.get(f"{self.BASE}/bot{token}/getFile", params={"file_id": file_id})
        _raise_for(reply, "telegram getFile")
        body = reply.json()
        if not body.get("ok"):
            raise PlatformError("telegram getFile rejected")
        info = body["result"]
        return f"{self.BASE}/file/bot{token}/{info['file_path']}"

    def upload(self, path: Path) -> Upload:
        data = path.read_bytes()
        token = self.config["telegram_bot_token"]
        chat = str(self.config["telegram_chat_id"])
        notes: list[str] = []
        variants: list[Variant] = []

        photo = self.http.post(
            f"{self.BASE}/bot{token}/sendPhoto",
            data={"chat_id": chat, "disable_notification": "true"},
            files={"photo": (path.name, data, "image/jpeg")},
        )
        _raise_for(photo, "telegram sendPhoto")
        sizes = (photo.json().get("result") or {}).get("photo") or []
        if not sizes:
            raise PlatformError("telegram returned no photo sizes")
        best = max(sizes, key=lambda s: s.get("file_size") or 0)
        variants.append(
            Variant(
                "photo",
                self._file_url(best["file_id"]),
                note=f"largest of {len(sizes)} sizes, reported {best.get('file_size')} bytes",
            )
        )
        notes.append("photo pipeline recompresses and strips metadata")

        doc = self.http.post(
            f"{self.BASE}/bot{token}/sendDocument",
            data={"chat_id": chat, "disable_notification": "true"},
            files={"document": (path.name, data, "image/jpeg")},
        )
        _raise_for(doc, "telegram sendDocument")
        document = (doc.json().get("result") or {}).get("document") or {}
        if document.get("file_id"):
            variants.append(
                Variant(
                    "document",
                    self._file_url(document["file_id"]),
                    note=f"reported {document.get('file_size')} bytes",
                )
            )
            notes.append("document pipeline is expected to store bytes verbatim")
        else:
            notes.append("sendDocument returned no document")

        return Upload(self.name, str(document.get("file_id", "")), "", variants, notes)


# --------------------------------------------------------------------------
# platform: Mastodon
# --------------------------------------------------------------------------


class Mastodon(Adapter):
    name = "mastodon"
    label = "Mastodon media attachment (server-side re-encode)"
    requires = ("mastodon_instance", "mastodon_token")

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config['mastodon_token']}"}

    def upload(self, path: Path) -> Upload:
        data = path.read_bytes()
        base = self.config["mastodon_instance"].rstrip("/")
        reply = self.http.post(
            f"{base}/api/v2/media",
            headers=self._auth(),
            files={"file": (path.name, data, mime_for(path, data))},
        )
        _raise_for(reply, "mastodon upload")
        media = reply.json()
        # v2 answers 202 while it is still processing; poll for the final URL.
        media_id = media.get("id")
        for _ in range(40):
            if media.get("url"):
                break
            time.sleep(0.25)
            poll = self.http.get(f"{base}/api/v1/media/{media_id}", headers=self._auth())
            _raise_for(poll, "mastodon poll")
            media = poll.json()
        if not media.get("url"):
            raise PlatformError("mastodon never produced a media URL")

        variants = [Variant("original", media["url"])]
        if media.get("preview_url") and media["preview_url"] != media["url"]:
            variants.append(Variant("preview", media["preview_url"]))
        meta = (media.get("meta") or {}).get("original") or {}
        notes = [
            f"reported {meta.get('width')}x{meta.get('height')} {meta.get('size')}",
        ]
        return Upload(self.name, str(media_id), media.get("text_url") or "", variants, notes)


# --------------------------------------------------------------------------
# platform: Bluesky (AT Protocol blob + CDN image endpoint)
# --------------------------------------------------------------------------


class Bluesky(Adapter):
    name = "bluesky"
    label = "Bluesky blob on cdn.bsky.app (feed_fullsize / feed_thumbnail)"
    requires = ("bluesky_handle", "bluesky_app_password")
    BASE = "https://bsky.social"
    CDN = "https://cdn.bsky.app/img"

    def upload(self, path: Path) -> Upload:
        data = path.read_bytes()
        session = self.http.post(
            f"{self.BASE}/xrpc/com.atproto.server.createSession",
            json={
                "identifier": self.config["bluesky_handle"],
                "password": self.config["bluesky_app_password"],
            },
        )
        _raise_for(session, "bluesky createSession")
        auth = session.json()
        token, did = auth["accessJwt"], auth["did"]

        blob = self.http.post(
            f"{self.BASE}/xrpc/com.atproto.repo.uploadBlob",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "image/jpeg"},
            data=data,
        )
        _raise_for(blob, "bluesky uploadBlob")
        cid = blob.json()["blob"]["ref"]["$link"]

        variants = [
            Variant("feed_fullsize", f"{self.CDN}/feed_fullsize/plain/{did}/{cid}@jpeg"),
            Variant("feed_fullsize_webp", f"{self.CDN}/feed_fullsize/plain/{did}/{cid}@webp"),
            Variant("feed_thumbnail", f"{self.CDN}/feed_thumbnail/plain/{did}/{cid}@jpeg"),
        ]
        return Upload(self.name, cid, "", variants, ["blob is addressed by CID, not filename"])


# --------------------------------------------------------------------------
# platform: Facebook Page photo (every fbcdn rendition in one read)
# --------------------------------------------------------------------------


def fb_variants(entries: Iterable[dict], prefix: str) -> list[Variant]:
    """One variant per rendition Facebook stored, largest first.

    Meta does not document an order for ``images``, so sort rather than trust it.
    """
    ordered = sorted(
        entries,
        key=lambda e: (e.get("width") or 0) * (e.get("height") or 0),
        reverse=True,
    )
    out: list[Variant] = []
    seen: set[str] = set()
    used: dict[str, int] = {}
    for entry in ordered:
        source = entry.get("source") or ""
        if not source or source in seen:
            continue
        seen.add(source)
        w, h = entry.get("width"), entry.get("height")
        name = f"{prefix}_{w}x{h}"
        if name in used:  # two qualities at one size still get their own row
            used[name] += 1
            name = f"{name}_{used[name]}"
        else:
            used[name] = 1
        out.append(Variant(name, source))
    return out


class FacebookPage(Adapter):
    name = "facebook"
    label = "Facebook Page photo (all fbcdn renditions)"
    requires = ("facebook_page_id", "facebook_token")
    optional = ("facebook_graph_version",)
    public = True
    GRAPH = "https://graph.facebook.com"

    def _graph(self) -> str:
        version = str(self.config.get("facebook_graph_version") or "v26.0")
        return f"{self.GRAPH}/{version if version.startswith('v') else 'v' + version}"

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config['facebook_token']}"}

    def check(self) -> tuple[bool, str]:
        """Confirm this is a Page token for the configured Page. Reads only."""
        missing = self.missing()
        if missing:
            return False, f"needs config: {', '.join(missing)}"
        reply = self.http.get(
            f"{self._graph()}/me", headers=self._auth(), params={"fields": "id,name"}
        )
        body = reply.json() if reply.content else {}
        if reply.status_code >= 400 or (body.get("error") if isinstance(body, dict) else None):
            message = ""
            if isinstance(body, dict):
                message = (body.get("error") or {}).get("message") or ""
            return False, redact_error(
                f"rejected: HTTP {reply.status_code} {message or reply.text[:160]}"
            )
        page_id = str(self.config["facebook_page_id"])
        if str(body.get("id")) != page_id:
            return False, (
                f"token belongs to {body.get('id')} ({body.get('name')}), not Page "
                f"{page_id}; this looks like a user token -- take the Page token "
                "from /me/accounts"
            )
        return True, f"Page token for {body.get('name')} ({page_id})"

    def upload(self, path: Path) -> Upload:
        data = path.read_bytes()
        # published=false keeps the fixture off the Page timeline: the renditions
        # are still readable, and the upload expires on its own.
        reply = self.http.post(
            f"{self._graph()}/{self.config['facebook_page_id']}/photos",
            headers=self._auth(),
            data={"published": "false"},
            files={"source": (path.name, data, mime_for(path, data))},
        )
        _raise_for(reply, "facebook photo upload")
        photo_id = (reply.json() or {}).get("id")
        if not photo_id:
            raise PlatformError("facebook upload returned no photo id")

        info = self.http.get(
            f"{self._graph()}/{photo_id}",
            headers=self._auth(),
            params={"fields": "images,webp_images,width,height"},
        )
        _raise_for(info, "facebook photo read")
        body = info.json() or {}
        variants = fb_variants(body.get("images") or [], "image")
        variants += fb_variants(body.get("webp_images") or [], "webp")
        if not variants:
            raise PlatformError("facebook returned no image renditions")

        notes = [
            f"{len(body.get('images') or [])} jpeg renditions returned",
            f"reported {body.get('width')}x{body.get('height')}",
            "uploaded with published=false, so the fixture is not on the timeline",
            "Meta documents stripping location metadata and resizing; it does not "
            "promise anything about C2PA, which is what this measures",
        ]
        return Upload(self.name, str(photo_id), "", variants, notes)


# --------------------------------------------------------------------------
# local CDN: the arm, exercised without an account
# --------------------------------------------------------------------------


LOCAL_POLICIES = ("copy", "strip", "segments", "webp")


class _CdnHandler(BaseHTTPRequestHandler):
    server_version = f"provenance-lab/{VERSION}"

    def log_message(self, *args):  # keep the test output readable
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        if self.path != "/upload":
            self._send(404, b"no such route", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(length)
        key = uuid.uuid4().hex
        self.server.store[key] = data  # type: ignore[attr-defined]
        body = json.dumps({"id": key, "bytes": len(data)}).encode()
        self._send(200, body, "application/json")

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/media/"):
            self._send(404, b"no such route", "text/plain")
            return
        key = parsed.path.rsplit("/", 1)[-1]
        stored = self.server.store.get(key)  # type: ignore[attr-defined]
        if stored is None:
            self._send(404, b"gone", "text/plain")
            return
        policy = (parse_qs(parsed.query).get("policy") or ["copy"])[0]
        try:
            body, ctype = self.server.transform(stored, policy)
        except Exception as exc:  # a bad policy is a 500, not a crash
            self._send(500, str(exc).encode(), "text/plain")
            return
        self._send(200, body, ctype)


class LocalCdn:
    """A local HTTP re-encoder that behaves like the arms we care about.

    ``copy``      serves the stored bytes (a raw-attachment pipeline)
    ``strip``     decodes and re-encodes, dropping metadata (a chat photo)
    ``segments``  re-encodes the pixels but carries the original APP11 segments
                  through (a metadata-preserving tool that forgot to re-hash)
    ``webp``      transcodes to WebP (a CDN that ignores the source format)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._server = ThreadingHTTPServer((host, port), _CdnHandler)
        self._server.store = {}  # type: ignore[attr-defined]
        self._server.transform = self._transform  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def origin(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "LocalCdn":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _transform(self, stored: bytes, policy: str) -> tuple[bytes, str]:
        if policy == "copy":
            return stored, "image/jpeg"
        with Image.open(io.BytesIO(stored)) as img:
            frame = img.convert("RGB")
        if policy == "webp":
            buf = io.BytesIO()
            frame.save(buf, format="WEBP", quality=80)
            return buf.getvalue(), "image/webp"
        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=80, subsampling=2)
        encoded = buf.getvalue()
        if policy == "strip":
            return encoded, "image/jpeg"
        if policy == "segments":
            return lab.inject_segments(encoded, lab.c2pa_segments(stored)), "image/jpeg"
        raise ValueError(f"unknown policy {policy!r}")


class LocalCdnAdapter(Adapter):
    name = "local"
    label = "local HTTP CDN (copy / strip / segments / webp policies)"

    def __init__(self, config: dict, http: Http | None = None, server: LocalCdn | None = None):
        super().__init__(config, http)
        self.server = server or LocalCdn()

    def upload(self, path: Path) -> Upload:
        data = path.read_bytes()
        reply = self.http.post(
            self.server.origin + "/upload",
            data=data,
            headers={"Content-Type": mime_for(path, data)},
        )
        _raise_for(reply, "local upload")
        key = reply.json()["id"]
        variants = [
            Variant(policy, f"{self.server.origin}/media/{key}?policy={policy}")
            for policy in LOCAL_POLICIES
        ]
        return Upload(self.name, key, "", variants, ["local harness, no third party involved"])


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def registry() -> dict[str, type[Adapter]]:
    return {
        a.name: a
        for a in (
            LocalCdnAdapter,
            Catbox,
            ZeroXZero,
            DiscordWebhook,
            TelegramBot,
            Mastodon,
            Bluesky,
            FacebookPage,
        )
    }


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def measure_variant(
    platform: str,
    variant: Variant,
    signed: Path,
    out_dir: Path,
    ca_pem: str | None,
    http: Http,
) -> dict:
    """Fetch one served URL and verify the bytes that came back."""
    started = time.perf_counter()
    row: dict[str, Any] = {
        "platform": platform,
        "variant": variant.name,
        "url": redact_url(variant.url),
        "note": variant.note,
        "status": None,
        "content_type": None,
        "served_bytes": 0,
        "served_format": None,
        "served_dims": None,
        "sha256_match": False,
        "present": False,
        "claim_readable": False,
        "state": None,
        "signature_ok": False,
        "data_hash_ok": False,
        "trusted": False,
        "ai_claim": False,
        "failures": [],
        "verdict": "unavailable",
        "download_ms": 0.0,
    }
    try:
        reply = http.get(variant.url, headers=variant.headers)
        row["status"] = reply.status_code
        row["content_type"] = reply.headers.get("Content-Type", "").split(";")[0] or None
        row["download_ms"] = round((time.perf_counter() - started) * 1000, 1)
        if reply.status_code != 200 or not reply.content:
            row["failures"] = [f"http_{reply.status_code}"]
            return row
    except Exception as exc:
        row["download_ms"] = round((time.perf_counter() - started) * 1000, 1)
        row["failures"] = [f"fetch_error:{type(exc).__name__}"]
        return row

    data = reply.content
    served = out_dir / "served" / f"{platform}__{variant.name}.{sniff_format(data)}"
    served.parent.mkdir(parents=True, exist_ok=True)
    served.write_bytes(data)
    row["served_bytes"] = len(data)
    row["served_format"] = sniff_format(data)
    row["served_dims"] = list(dimensions(data) or []) or None
    row["sha256_match"] = sha256(data) == sha256(signed.read_bytes())

    check = lab.verify(served, ca_pem)
    for key in ("present", "claim_readable", "state", "signature_ok", "data_hash_ok",
                "trusted", "ai_claim", "failures"):
        row[key] = check[key]
    row["verdict"] = check["verdict"] if "verdict" in check else lab.verdict(check)
    return row


def measure_platform(
    adapter: Adapter,
    signed: Path,
    out_dir: Path,
    ca_pem: str | None,
    only_variants: Iterable[str] | None = None,
) -> dict:
    out_dir = Path(out_dir)
    upload = adapter.upload(signed)
    wanted = set(only_variants or [])
    if wanted:
        upload.variants = [v for v in upload.variants if v.name in wanted]
    record = {"upload": upload.to_dict(), "rows": []}
    # The real URLs live here, separately from anything shareable, because a
    # Telegram file URL embeds the bot token and a Discord proxy URL is signed.
    record["refetch"] = [
        {"platform": adapter.name, "name": v.name, "url": v.url, "headers": v.headers}
        for v in upload.variants
    ]
    for variant in upload.variants:
        record["rows"].append(
            measure_variant(adapter.name, variant, signed, out_dir, ca_pem, adapter.http)
        )
    safe = dict(record)
    safe.pop("refetch")
    (out_dir / "uploads").mkdir(parents=True, exist_ok=True)
    (out_dir / "uploads" / f"{adapter.name}.json").write_text(
        json.dumps(safe, indent=2) + "\n"
    )
    return record


def summarise(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    platforms: dict[str, dict] = {}
    for row in rows:
        stats = platforms.setdefault(
            row["platform"],
            {
                "variants": 0,
                "absent": 0,
                "invalid": 0,
                "valid": 0,
                "unavailable": 0,
                "byte_identical": 0,
                "formats": [],
                "max_ratio": 0.0,
                "stripped_but_still_claims_ai": 0,
            },
        )
        stats["variants"] += 1
        stats[row["verdict"]] += 1
        if row["sha256_match"]:
            stats["byte_identical"] += 1
        if row["served_format"] and row["served_format"] not in stats["formats"]:
            stats["formats"].append(row["served_format"])
        if row["verdict"] == "absent" and row["claim_readable"]:
            stats["stripped_but_still_claims_ai"] += 1
    for row in rows:
        base = row.get("uploaded_bytes") or 0
        if base:
            ratio = row["served_bytes"] / base
            stats = platforms[row["platform"]]
            stats["max_ratio"] = round(max(stats["max_ratio"], ratio), 3)
    for stats in platforms.values():
        stats["max_ratio"] = round(stats["max_ratio"], 3) if stats["max_ratio"] else None
    return {"rows": len(rows), "platforms": platforms}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

_SYMBOL = {"absent": "gone", "invalid": "INVALID", "valid": "valid", "unavailable": "n/a"}


def render_markdown(result: dict) -> str:
    meta = result["meta"]
    lines = ["# Platform survival report", ""]
    lines.append(
        f"Generated {meta['generated']} · c2pa-python {meta['c2pa_python']} "
        f"(SDK {meta['c2pa_sdk']}) · fixture {meta['fixture']} · "
        f"{meta['signed_bytes']:,} bytes signed"
    )
    lines.append("")
    if result.get("skipped"):
        lines.append("## Not measured")
        lines.append("")
        lines.append("| platform | reason |")
        lines.append("|---|---|")
        for item in result["skipped"]:
            lines.append(f"| `{item['platform']}` | {item['reason']} |")
        lines.append("")
    lines.append("## What each service served back")
    lines.append("")
    lines.append("| platform | variant | status | served | format | dims | bytes | identical | credential | failures |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in result["rows"]:
        dims = "x".join(str(v) for v in row["served_dims"]) if row["served_dims"] else "-"
        failures = ", ".join(row["failures"]) or "-"
        lines.append(
            f"| `{row['platform']}` | {row['variant']} | {row['status']} "
            f"| {_SYMBOL[row['verdict']]} | {row['served_format'] or '-'} | {dims} "
            f"| {row['served_bytes']:,} | {'yes' if row['sha256_match'] else 'no'} "
            f"| {row['state'] or '-'} | {failures} |"
        )
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("| platform | variants | gone | invalid | valid | byte-identical | formats served |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, stats in result["summary"]["platforms"].items():
        lines.append(
            f"| `{name}` | {stats['variants']} | {stats['absent']} | {stats['invalid']} "
            f"| {stats['valid']} | {stats['byte_identical']} "
            f"| {', '.join(stats['formats']) or '-'} |"
        )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- The signing CA is a throwaway test anchor. `trusted` means trusted "
        "against that anchor; signature and binding checks are real."
    )
    lines.append(
        "- A platform that serves `gone` may still be showing the credential to "
        "some clients through an internal path this arm cannot see."
    )
    lines.append(
        "- `platforms.json` and `uploads/*.json` carry redacted URLs by design. "
        "`reuse` reads the owner-only refetch file recorded in `platforms.json`; "
        "a run copied without that file can only re-fetch anonymous-host URLs."
    )
    if result.get("refetch_warning"):
        lines.append("")
        lines.append(f"> reuse warning: {result['refetch_warning']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def config_keys() -> list[str]:
    """Every key any adapter can ask for, credentials and tuning alike."""
    keys: list[str] = []
    for cls in registry().values():
        for key in (*cls.requires, *cls.optional):
            if key not in keys:
                keys.append(key)
    return keys


def load_config(path: Path | None) -> dict:
    """Read credentials from JSON, then fill blanks from the environment.

    The environment path exists so a token never has to be pasted into a chat
    or committed: `TELEGRAM_BOT_TOKEN=... python platforms.py run ...` works.
    A value in the file always wins over the environment.
    """
    known = config_keys()
    config: dict[str, Any] = {}
    if path:
        path = Path(path)
        if path.exists():
            loaded = json.loads(path.read_text())
            config = {k: v for k, v in loaded.items() if k in known and v}
    for key in known:
        if config.get(key):
            continue
        value = os.environ.get(key.upper()) or os.environ.get(
            "PROVENANCE_" + key.upper()
        )
        if value:
            config[key] = value
    return config


def run(
    out_dir: Path,
    platforms: Iterable[str],
    config: dict,
    ca_pem: str | None,
    certs: dict[str, Path],
    variants: Iterable[str] | None = None,
    image: Path | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    signer = lab.load_signer(certs["chain"], certs["key"])
    # A photo passed with --image replaces the generated fixture, through the
    # same signing path: the arm measures what a platform does to a credential,
    # not what the fixture looks like, so the pixels are the caller's business.
    if image is not None:
        fixture, fixture_size = lab.prepare_image(image, out_dir / "assets" / "source.jpg")
    else:
        fixture = lab.make_fixture(out_dir / "assets" / "source.jpg")
        fixture_size = lab.FIXTURE_SIZE
    signed = out_dir / "assets" / "signed.jpg"
    lab.sign_created(fixture, signed, signer)

    rows: list[dict] = []
    uploads: list[dict] = []
    refetch: list[dict] = []
    skipped: list[dict] = []
    started = time.perf_counter()

    table = registry()
    for name in platforms:
        cls = table.get(name)
        if cls is None:
            skipped.append({"platform": name, "reason": "unknown platform"})
            continue
        if cls is LocalCdnAdapter:
            with LocalCdn() as server:
                rows, uploads, refetch, skipped = _measure_one(
                    cls, name, config, signed, out_dir, ca_pem, variants,
                    rows, uploads, refetch, skipped, server=server,
                )
            continue
        rows, uploads, refetch, skipped = _measure_one(
            cls, name, config, signed, out_dir, ca_pem, variants,
            rows, uploads, refetch, skipped,
        )

    result = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "c2pa_python": getattr(lab.c2pa, "__version__", "?"),
            "c2pa_sdk": lab.c2pa.sdk_version(),
            "fixture": f"{fixture_size[0]}x{fixture_size[1]}",
            "signed_bytes": signed.stat().st_size,
            "elapsed_s": round(time.perf_counter() - started, 2),
            "signing_ca": "self-issued test CA (not a public trust anchor)",
        },
        "uploads": uploads,
        "rows": rows,
        "skipped": skipped,
    }
    if image is not None:
        # Enough to tell a report made from a photo apart from one made from
        # the generated fixture; the manifest title carries it too.
        result["meta"]["source_image"] = Path(image).name
    result["summary"] = summarise(rows)
    refetch_file, warning = write_refetch(out_dir / "refetch.json", refetch)
    result["meta"]["refetch_file"] = str(refetch_file)
    result["refetch_warning"] = warning
    (out_dir / "platforms.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "report.md").write_text(render_markdown(result) + "\n")
    return result


def state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "provenance-lab"


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def write_refetch(path: Path, targets: list[dict]) -> tuple[Path, str | None]:
    """Write the real served URLs owner-only, so `reuse` can fetch them again.

    Kept apart from `platforms.json` because a Telegram file URL embeds the bot
    token and a Discord proxy URL is signed: those must not sit in the artifact
    that is safe to share.

    Windows-mounted workspaces (9p/drvfs) ignore chmod, so if the mode does not
    actually tighten the file is moved out to the user's state directory rather
    than left world-readable inside the project.

    Returns the path actually used and a warning when no safe location existed.
    """
    path = Path(path)
    payload = json.dumps({"targets": targets}, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    if _mode(path) & 0o077:
        alt = state_dir() / f"refetch-{uuid.uuid4().hex[:12]}.json"
        alt.parent.mkdir(parents=True, exist_ok=True)
        alt.write_text(payload)
        try:
            os.chmod(alt, 0o600)
        except OSError:
            pass
        if _mode(alt) & 0o077:
            return path, (
                "this filesystem ignores file modes; the real served URLs are "
                f"readable by other local users in {path}"
            )
        path.unlink(missing_ok=True)
        return alt, None
    return path, None


def _measure_one(
    cls: type[Adapter],
    name: str,
    config: dict,
    signed: Path,
    out_dir: Path,
    ca_pem: str | None,
    variants: Iterable[str] | None,
    rows: list[dict],
    uploads: list[dict],
    refetch: list[dict],
    skipped: list[dict],
    server: LocalCdn | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Run one platform, folding its rows into the run-level lists."""
    adapter = cls(config, server=server) if cls is LocalCdnAdapter else cls(config)
    ok, why = adapter.available()
    if not ok:
        skipped.append({"platform": name, "reason": why})
        return rows, uploads, refetch, skipped
    try:
        record = measure_platform(adapter, signed, out_dir, ca_pem, variants)
    except Exception as exc:
        skipped.append(
            {"platform": name, "reason": f"upload failed: {redact_error(exc)}"}
        )
        return rows, uploads, refetch, skipped
    uploads.append(record["upload"])
    refetch.extend(record["refetch"])
    base = signed.stat().st_size
    for row in record["rows"]:
        row["uploaded_bytes"] = base
        rows.append(row)
    return rows, uploads, refetch, skipped


def refetch_file_for(record_path: Path, previous: dict) -> Path | None:
    """Where the real URLs live: declared path first, then a sibling file."""
    declared = (previous.get("meta") or {}).get("refetch_file")
    candidates = [Path(declared)] if declared else []
    candidates.append(Path(record_path).parent / "refetch.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def refetch_targets(record_path: Path, previous: dict) -> list[dict]:
    """Prefer the owner-only refetch file; fall back to the redacted record."""
    if "targets" in previous:  # pointed straight at a refetch.json
        return previous["targets"]
    source = refetch_file_for(record_path, previous)
    if source is not None:
        return json.loads(source.read_text()).get("targets", [])
    return [
        {"platform": upload["platform"], "name": variant["name"],
         "url": variant["url"], "headers": {}}
        for upload in previous.get("uploads", [])
        for variant in upload.get("variants", [])
    ]


def reuse(record_path: Path, out_dir: Path, ca_pem: str | None, signed: Path) -> dict:
    """Re-fetch the URLs from a previous run without uploading again.

    Useful for the question the first run cannot answer: does the credential
    survive until the platform's own cache or transcode budget expires?
    """
    previous = json.loads(Path(record_path).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    base = signed.stat().st_size
    http = Http()
    started = time.perf_counter()
    for target in refetch_targets(Path(record_path), previous):
        row = measure_variant(
            target["platform"],
            Variant(target["name"], target["url"], target.get("headers") or {}),
            signed,
            out_dir,
            ca_pem,
            http,
        )
        row["uploaded_bytes"] = base
        rows.append(row)
    result = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "c2pa_python": getattr(lab.c2pa, "__version__", "?"),
            "c2pa_sdk": lab.c2pa.sdk_version(),
            "fixture": f"{lab.FIXTURE_SIZE[0]}x{lab.FIXTURE_SIZE[1]}",
            "signed_bytes": base,
            "elapsed_s": round(time.perf_counter() - started, 2),
            "signing_ca": previous.get("meta", {}).get("signing_ca", "unknown"),
        },
        "uploads": previous.get("uploads", []),
        "rows": rows,
        "skipped": [],
        "reused_from": str(record_path),
        "refetch_warning": None
        if refetch_file_for(Path(record_path), previous)
        else "no refetch file for this run; URLs came from the redacted "
             "record, so token-bearing URLs will fail to fetch",
    }
    result["summary"] = summarise(rows)
    (out_dir / "platforms.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "report.md").write_text(render_markdown(result) + "\n")
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="show platforms and what each needs")
    p_list.add_argument("--config", default=CONFIG_NAME)

    p_check = sub.add_parser(
        "check", help="verify credentials with a read-only call, uploading nothing"
    )
    p_check.add_argument("--platforms", default="",
                         help="comma-separated names; default is every adapter with a preflight")
    p_check.add_argument("--config", default=CONFIG_NAME)

    p_chats = sub.add_parser(
        "chats", help="list chats a bot has already seen; sends nothing"
    )
    p_chats.add_argument("--platforms", default="telegram",
                         help="comma-separated names; only adapters with discovery")
    p_chats.add_argument("--config", default=CONFIG_NAME)
    p_chats.add_argument("--limit", type=int, default=100,
                         help="how many pending updates to read (default 100)")

    p_run = sub.add_parser("run", help="upload the signed fixture and measure the served rendition")
    p_run.add_argument("--out", default="out/platforms")
    p_run.add_argument("--certs", default=None, help="defaults to <out>/certs")
    p_run.add_argument("--platforms", default="local",
                       help="comma-separated platform names, or 'all'")
    p_run.add_argument("--config", default=CONFIG_NAME)
    p_run.add_argument("--variants", default=None,
                       help="comma-separated variant names to keep")
    p_run.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    p_run.add_argument(
        "--strict", action="store_true",
        help="exit 1 if a platform that had credentials could not be measured",
    )
    p_run.add_argument(
        "--image", default=None,
        help="photo to sign and upload instead of the generated fixture",
    )

    p_reuse = sub.add_parser("reuse", help="re-fetch the URLs from a previous run")
    p_reuse.add_argument("--from", dest="source", default="out/platforms/platforms.json")
    p_reuse.add_argument("--out", default="out/platforms/reuse")
    p_reuse.add_argument("--certs", default=None, help="defaults to the source run's certs")

    args = parser.parse_args(argv)
    table = registry()

    if args.cmd == "list":
        config = load_config(Path(args.config) if getattr(args, "config", None) else None)
        for name, cls in table.items():
            adapter = cls(config)
            ok, why = adapter.available()
            print(f"{name:10s} {cls.label}")
            if ok:
                print(f"{'':10s} ready")
            else:
                print(f"{'':10s} {why}")
                if adapter.missing():
                    print(f"{'':10s} set {', '.join(k.upper() for k in adapter.missing())}"
                          f" or put them in {CONFIG_NAME}")
        return 0

    if args.cmd == "check":
        config = load_config(Path(args.config))
        names = [n.strip() for n in args.platforms.split(",") if n.strip()] or list(table)
        failed = 0
        for name in names:
            cls = table.get(name)
            if cls is None:
                print(f"{name:10s} unknown platform")
                failed += 1
                continue
            if not hasattr(cls, "check"):
                print(f"{name:10s} no read-only preflight; the first run is the test")
                continue
            ok, detail = cls(config).check()
            print(f"{name:10s} {'ok' if ok else 'FAILED'}: {detail}")
            failed += 0 if ok else 1
        return 1 if failed else 0

    if args.cmd == "chats":
        config = load_config(Path(args.config))
        names = [n.strip() for n in args.platforms.split(",") if n.strip()]
        failed = 0
        for name in names:
            cls = table.get(name)
            if cls is None:
                print(f"{name:10s} unknown platform")
                failed += 1
                continue
            if not hasattr(cls, "chats"):
                print(f"{name:10s} no chat discovery")
                failed += 1
                continue
            adapter = cls(config)
            try:
                who = adapter.identity()
                found = adapter.chats(args.limit)
            except PlatformError as exc:
                print(f"{name:10s} FAILED: {redact_error(exc)}")
                failed += 1
                continue
            who_label = f"@{who['username']}" if who.get("username") else str(who.get("id"))
            print(f"{name:10s} {who_label} has seen {len(found)} chat(s)")
            if not found:
                print(f"{'':10s} message the bot or add it to a group, then run again")
                continue
            for chat in found:
                print(f"{'':10s} {chat['id']:>16}  {chat['type']:<10} {chat['title']}")
            print(f"{'':10s} put one of those ids in {CONFIG_NAME} as telegram_chat_id")
        return 1 if failed else 0

    if args.cmd == "reuse":
        out = Path(args.out)
        certs = Path(args.certs) if args.certs else Path(args.source).parent / "certs"
        ca = certs / "ca.cert.pem"
        signed = out / "assets" / "signed.jpg"
        if not signed.exists():
            signed = Path(args.source).parent / "assets" / "signed.jpg"
        result = reuse(
            Path(args.source),
            out,
            ca.read_text() if ca.exists() else None,
            signed,
        )
        print(render_markdown(result))
        return 0

    if args.cmd in (None, "run"):
        out = Path(args.out)
        certs = Path(args.certs) if args.certs else out / "certs"
        config = load_config(Path(args.config) if args.config else None)
        if not all((certs / n).exists() for n in ("ca.cert.pem", "chain.pem", "leaf.key.pem")):
            lab.make_certs(certs)
        names = [n.strip() for n in args.platforms.split(",") if n.strip()]
        if names == ["all"]:
            names = list(table)
        variants = args.variants.split(",") if args.variants else None

        try:
            result = run(
                out,
                names,
                config,
                (certs / "ca.cert.pem").read_text(),
                {
                    "ca": certs / "ca.cert.pem",
                    "chain": certs / "chain.pem",
                    "key": certs / "leaf.key.pem",
                },
                variants,
                image=Path(args.image).expanduser() if args.image else None,
            )
        except ValueError as exc:
            # `--image` pointing at something unreadable is a command-line
            # mistake, and the message says which file and why.
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result["summary"], indent=2) if args.json else render_markdown(result))
        if args.strict:
            # An adapter with no credentials is expected to be skipped; one that
            # was configured and then failed is the thing a scheduled run needs
            # to shout about, since the exit status is otherwise 0 either way.
            failed = [
                item for item in result.get("skipped", [])
                if not str(item.get("reason", "")).startswith("needs config: ")
            ]
            for item in failed:
                print(f"{item.get('platform', '?'):10s} not measured: "
                      f"{item.get('reason', '')}")
            if failed:
                return 1
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
