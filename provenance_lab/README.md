# Provenance survival lab

What happens to a C2PA credential when a file is edited and re-shared?

Two arms, one verifier:

- **local** ([lab.py](lab.py)) — sign a real manifest, push the image through
  ten transformations in three editing modes, verify every output. No network,
  about three seconds.
- **platform** ([platforms.py](platforms.py)) — upload the signed fixture to a
  real service, fetch back exactly what it serves, and verify *those* bytes.

Both verify with the same C2PA SDK and the same test CA, so rows from the two
arms are directly comparable.

## Why measure this now

- **EU AI Act Article 50 applies from 2 August 2026** and requires
  machine-readable marking of synthetic media. California's SB 942 (as amended
  by AB 853) became operative the same day.
- **C2PA 2.4** (April 2026) added an AI Disclosure Assertion and crJSON;
  `c2pa-python` 0.38 ships the 0.91 SDK, and `c2patool` 0.28.2 is current.
- Camera and generator support is real (Leica M11-P, Nikon Z6III, Pixel 10,
  OpenAI images carry C2PA + SynthID), but platforms commonly resize and
  re-encode uploads.
- There is **no public, reproducible, cross-platform measurement** of what
  survives. Vendor notes admit credentials can be stripped or invalidated;
  nobody publishes the rate.

## The three arms

| arm | what the editing tool does |
|---|---|
| `as_is` | re-encodes and drops metadata (Pillow's default behaviour) |
| `preserve` | carries the original manifest through unchanged — the interesting failure: present, readable, stale |
| `rebind` | a C2PA-aware editor signs a new manifest with the original as a parent ingredient |

`preserve` is implemented by moving the original APP11 segments into the new
JPEG byte-for-byte, which is exactly what a metadata-preserving editor does
when it forgets to re-hash.

## Run it

```bash
cd provenance_lab
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python -m unittest -v          # 151 tests
.venv/bin/python lab.py run --out out/full
```

Outputs `out/full/report.md` (readable) and `out/full/report.json` (raw rows),
plus every generated file under `out/full/assets/`.

Other commands:

```bash
.venv/bin/python lab.py run --transforms jpeg_q75,crop_10pct   # subset
.venv/bin/python lab.py run --modes as_is,rebind               # skip an arm
.venv/bin/python lab.py verify out/full/assets/signed.jpg --ca out/full/certs/ca.cert.pem
```

### One command: measure, build the page, look at it

```bash
.venv/bin/python demo.py                 # then open http://127.0.0.1:8788/
```

That runs the `local` adapter against its four-policy HTTP harness, writes the
report to `out/demo/`, appends it as one column of the trend page in
`out/site/`, and serves that page until Ctrl-C. Nothing leaves the machine —
the live adapters are not involved — so it is the platform arm end to end
without an account, and the page's matrix is visible before any credential
exists.

```bash
.venv/bin/python demo.py --no-serve      # files only; no server
.venv/bin/python demo.py --port 9000 --no-open
.venv/bin/python demo.py --out out/elsewhere --site out/page
```

Each run adds a column, which is the page doing its job rather than a bug; both
output directories are gitignored. `--port 0` picks a free port and prints it.

## Measured result

960×640 JPEG fixture, signed as `TrainedAlgorithmicMedia`, c2pa-python 0.38.0
(SDK 0.91.0), Pillow 13.0.0, whole matrix 2.2–2.8 s across runs. File sizes move by 1–2 bytes
between runs: ECDSA signatures are variable length, so the manifest is not a
fixed size.

| transform | pixels unchanged | as_is | preserve | rebind |
|---|---|---|---|---|
| `jpeg_q95` | no | gone | INVALID | valid |
| `jpeg_q75` | no | gone | INVALID | valid |
| `jpeg_q50` | no | gone | INVALID | valid |
| `resize_50` | n/a | gone | INVALID | valid |
| `crop_10pct` | n/a | gone | INVALID | valid |
| `rotate_90` | n/a | gone | INVALID | valid |
| `screenshot_1080p` | n/a | gone | INVALID | valid |
| `cdn_w800_q70` | n/a | gone | INVALID | valid |
| `double_share` | no | gone | INVALID | valid |
| `strip_rewrite` | yes | gone | n/a | valid |

Counts: **10/10 credentials gone** under a normal re-encode, **9/9 present but
invalid** when metadata is preserved, **10/10 valid** when re-signed with a
parent ingredient.

### The failure mode worth naming

Every `preserve` output failed on exactly one code: `assertion.dataHash.mismatch`.
The claim still parsed. The `c2pa.actions.v2` assertion still said the image was
`TrainedAlgorithmicMedia`. So a reader that displays the claim without checking
the binding would still call this file AI-generated — and a reader that checks
would call the credential broken. Both are "present", neither is trustworthy.

Even `jpeg_q95` breaks the binding: a JPEG round trip is lossy at any quality,
so the hash never matches. `strip_rewrite` is the control — every pixel
identical, credential gone — which is why "no credential found" carries no
information at all.

### What provenance costs

| file | bytes | vs. unsigned |
|---|---:|---:|
| unsigned fixture | 25,049 | — |
| signed once | 60,635 | +35,586 (+142%) |
| `jpeg_q75`, metadata dropped | 18,709 | −25% |
| `jpeg_q75`, manifest preserved | 54,295 | +117% |
| `jpeg_q75`, re-signed with parent | 125,261 | 5.0× |

The only arm that validates is also the only one that **grows** the file: the
parent credential is embedded as a full ingredient, thumbnail included, so a
single re-share goes from 60,635 to 125,261 bytes — 2.07× its own signed input,
5.0× the unsigned image. Any pipeline that re-signs on every edit inherits that
growth per generation.

## The platform arm

`platforms.py` measures the half the local matrix cannot: what a *service* does
to the file between upload and the bytes a viewer actually downloads.

```bash
.venv/bin/python platforms.py list                       # what each adapter needs
.venv/bin/python platforms.py chats                      # chat ids the bot has seen
.venv/bin/python platforms.py run --platforms local      # no account, no network
.venv/bin/python platforms.py run --platforms catbox     # real third party
.venv/bin/python platforms.py reuse --from out/platforms/platforms.json
```

### Signing your own photo

```bash
.venv/bin/python platforms.py run --platforms discord --image ~/Pictures/test.jpg
```

`--image` replaces the generated fixture with a photo, through the same signing
path. EXIF orientation is applied to the pixels first (rotation usually lives
in that tag, and the re-encode drops it), transparency is flattened onto white,
and the result is re-encoded to JPEG — which also drops EXIF, so the location
does not travel with the upload. It is deliberately not resized: a platform's
own rescale is the thing being measured, so a photo above a platform's size
limit is reported as that platform's HTTP failure rather than shrunk to fit.
The manifest still claims
`TrainedAlgorithmicMedia`, so use a photo you are happy to label that way, and
remember that the run publishes it.

Every adapter is reduced to the same shape — an upload that yields served
URLs — so the measurement is identical everywhere:

    upload  ->  fetch each served URL  ->  verify the fetched bytes

### Adapters

| name | what it reaches | needs |
|---|---|---|
| `local` | local HTTP re-encoder, four policies | nothing |
| `catbox` | anonymous host, stores bytes verbatim | nothing |
| `0x0` | anonymous host, currently rejecting uploads | nothing |
| `discord` | attachment URL plus the media proxy | webhook URL |
| `telegram` | `sendPhoto` (recompressed) vs `sendDocument` (stored) | bot token, chat id |
| `mastodon` | `/api/v2/media`, polls until processed | instance URL, token |
| `bluesky` | `uploadBlob` + `cdn.bsky.app` feed_fullsize/thumbnail | handle, app password |
| `facebook` | Page photo, every `images` and `webp_images` rendition | page id, page token |

X was removed deliberately: its API is credit-based (about $0.015 per post, no
free tier), so the arm could not be run without spending money. Every adapter
left is free to call — anonymous hosts, a Discord webhook, a Telegram bot, a
Mastodon instance, Bluesky, and the Facebook Graph API for your own Page.

Credentials come from `platforms.json`, copied from
[platforms.example.json](platforms.example.json), or from equivalently named
environment variables — so a token never has to be pasted into a chat:

```bash
TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... \
  .venv/bin/python platforms.py run --platforms telegram
```

`platforms.py list` prints the variable name for each missing key, and
`platforms.py check` validates credentials with a read-only call before anything
is published (exit status 1 if any of them fails). The config file is gitignored
and the run never prints either source. Reports and `uploads/*.json` carry redacted URLs only, so a
Telegram bot token or Discord webhook secret cannot leak into an artifact you
might share — but `reuse` still needs the real URLs, so they are written once
to a separate owner-only file and its path is recorded in `platforms.json`.

That second file is `refetch.json`, and creating it with mode 0600 can silently
fail: Windows-mounted workspaces (9p/drvfs, which is what `/mnt/c` is) accept
`chmod` and then ignore it. The lab checks the resulting mode, and if it did not
tighten it moves the file to `~/.local/state/provenance-lab/` rather than leave
token-bearing URLs world-readable inside the project. If no location is safe it
says so in the report instead of pretending.

A platform can expose several renditions of one upload, and the arm measures
each. That is the point of the `telegram` and `discord` shapes: the same bytes
uploaded twice land on two different pipelines with different outcomes.

### Getting credentials

#### Facebook

1. Create a Meta app at developers.facebook.com and leave it in Development
   mode with your account holding an app role. Standard Access covers your own
   Page, so no App Review is needed.
2. In the Graph API Explorer, select the app and request a user token with
   `pages_show_list`, `pages_read_engagement`, `pages_manage_posts`.
3. Call `GET /me/accounts`, then copy your Page's `id` and `access_token`.
4. Put them in `platforms.json` as `facebook_page_id` and `facebook_token`.
5. For a token that does not expire on a timer, exchange the user token for a
   long-lived one (`?grant_type=fb_exchange_token` with the app id and secret)
   and take the Page token from `/me/accounts` again.

The upload uses `published=false`, so the fixture never reaches the Page
timeline, and Meta discards the unpublished photo after about 24 hours.

#### Telegram chat ids

`platforms.py chats` asks the bot which chats it has already seen, so the id can
be discovered instead of hunted for in a client. It needs only the token:

```bash
TELEGRAM_BOT_TOKEN=... .venv/bin/python platforms.py chats
```

It reads `getUpdates` without an `offset`, so pending updates are neither
confirmed nor consumed — the bot's backlog is left exactly as it was and the
command can be repeated. Two limits are reported rather than hidden: a 409 means
a webhook is set, which makes `getUpdates` unavailable until `deleteWebhook`, and
a chat the bot has not been messaged in for about 24 hours is no longer
returned. Exit status is 1 when discovery fails.

### Facebook: one upload, several renditions

`facebook` needs no post: a Page photo uploaded with `published=false` stays off
the timeline and is readable through `GET /{photo-id}?fields=images,webp_images`,
which returns every rendition the CDN stores. Meta does not document an order
for that list, so the arm sorts by pixel area and measures all of them — the
answer can differ between the largest and the smallest.

It has not been run live: that needs a Page token, and it publishes a photo
(unpublished, so off the timeline). What *was* checked is that the endpoint
is where the code thinks it is — `GET graph.facebook.com/v26.0/me` answers an
OAuth error rather than an unknown-version error.

### The local harness is not decoration

The `local` adapter runs a real HTTP server that re-encodes with Pillow under
four policies, which lets the whole arm — upload, fetch, verify, report — be
tested without posting anything anywhere:

| policy | what it serves | result |
|---|---|---|
| `copy` | the stored bytes | **valid**, Trusted, byte-identical |
| `strip` | decoded, re-encoded q80 | **gone** |
| `segments` | re-encoded, original APP11 carried over | **INVALID** — `assertion.dataHash.mismatch` |
| `webp` | transcoded to WebP | **gone**, and the format changed |

That is the same three-way split the local matrix found, reproduced across a
network boundary. Without it, a third-party row saying "gone" would be
indistinguishable from a bug in this lab's transport.

### Measured so far

| platform | variant | served | credential |
|---|---|---|---|
| `catbox` | direct | byte-identical, 60,635 B | **valid**, Trusted |

`catbox` is the control that makes the rest interpretable: a real third party
can store and serve a C2PA-signed JPEG unchanged, so a credential vanishing at
a social platform is that platform's transcode, not this code's.

`0x0.st` now answers HTTP 503 — "uploads disabled because it's been almost
nothing but AI botnet spam" — and the run records it under **Not measured**
rather than dropping it silently.

### Re-fetching instead of re-uploading

`reuse` resolves the recorded refetch file, fetches the same URLs again and
measures them. That answers the question a single run cannot: does the
credential survive until the platform's cache or transcode budget expires?

```bash
.venv/bin/python platforms.py reuse --from out/platforms/platforms.json
```

If the refetch file is missing it falls back to the redacted record and warns
in the report — an anonymous-host URL still works, a token-bearing one will not.

### The scheduled live run

[../.github/workflows/live-platforms.yml](../.github/workflows/live-platforms.yml)
measures the five credential-backed adapters every Monday at 06:00 UTC, and on
demand with an optional comma-separated list. It is kept out of the push
workflow on purpose: a run publishes the fixture to real services, and a fork's
pull request must not be able to ask for that.

Credentials come from repository secrets under the same names the adapters read
— `DISCORD_WEBHOOK`, `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`,
`MASTODON_INSTANCE` and `MASTODON_TOKEN`, `BLUESKY_HANDLE` and
`BLUESKY_APP_PASSWORD`, `FACEBOOK_PAGE_ID` and `FACEBOOK_TOKEN`. Missing ones
are reported as "needs config" while the rest still run, so the workflow is
useful before any account exists.

`--strict` is what makes the job status mean something. Without it `run` exits 0
even when every upload failed, because an adapter with no credentials is a skip
rather than an error. With it, a platform that had credentials and then failed
returns 1 — and an unconfigured one still does not.

The artifact holds `report.md` and `platforms.json`, the redacted pair. It
omits `refetch.json` deliberately: that file carries the real served URLs, and
a Telegram file URL embeds the bot token.

The anonymous hosts are not in the scheduled list. Every run leaves a permanent
public file there, and the `catbox` control only needs re-measuring when
transport is in doubt (`platforms.py run --platforms catbox`).

### The page that remembers

```bash
.venv/bin/python pages.py build --report out/live/platforms.json --site out/site
.venv/bin/python pages.py publish --site out/site --branch platform-history
```

A number nobody re-reads is not evidence of drift. Each run is appended to
`history.json` and rendered as a matrix — one row per adapter, one column per
run, a colour for what came back — so a platform that quietly starts preserving
the credential, or quietly stops being measured because a token expired, shows
up as a change of colour rather than as a paragraph someone has to compare by
memory.

`pages.py` is deliberately stdlib only: the page is built here, not in the
browser, so it works with JavaScript disabled, fetches nothing from a CDN, and
can be rebuilt from the history without the signing stack installed.
`publish` clones the branch and commits onto it rather than force-pushing, so
the branch keeps its history and a hand edit there is not discarded.

Publishing and remembering are two different mechanisms, on purpose. Publishing
is `actions/deploy-pages` and needs one setting: **Settings → Pages → Source:
GitHub Actions** — and a public repository, because Pages refuses private ones
on a personal plan (`422 Your current plan does not support GitHub Pages for
this repository`), which is why the page waited for the repo to be made public.
Remembering is the `platform-history` branch, which the workflow writes *before*
deploying so a failed deployment still leaves the run recorded. `publish`
merges that branch's `history.json` with the run that just finished instead of
copying over it: the runner builds from an empty directory, so a copy would
leave the page with one column every week. A branch Pages
source would not work here: GitHub documents that commits pushed by a workflow
using `GITHUB_TOKEN` do not trigger a branch build, so the branch would fill
with history that is never served. The branch also holds the rendered page,
which is readable in the repository without a deployment.

Four verdicts collapse to one colour per platform: `valid` if any rendition kept
a verifiable credential — the noteworthy result, not the normal one — `invalid`
if one kept it in a form that fails its own hash, `gone` if none did,
`unavailable` if the bytes could not be read. A platform with no secret is
`needs config`; a week whose measurement produced no report is `not measured`,
because an unmeasured week is not the same as a clean one. A re-run reuses its
workflow run id, so it replaces its own column instead of adding a second
reading of the same hour.

Adapter error text is never published. A Discord webhook URL carries its token
as a path segment and a Telegram file URL carries one after `bot<id>:`, so the
page keeps only the class of failure (`HTTP 401`, `network error`) instead of
trying to scrub arbitrary credentials out of free text.

### Reading a rendition the way a browser does

```bash
.venv/bin/python verifier.py --out out/photo-discord
```

`platforms.py` inspects the bytes it downloaded with c2pa-python. This serves
`verifier/`, a static page that runs the Content Authenticity Initiative's own
browser SDK, and lists every `served/` rendition under the run directories you
give it, so a rendition can be read the way a viewer application would read it
rather than through this repo's verifier.

Four renditions from the local arm, as the browser reports them:

| rendition | what a client sees |
|---|---|
| `copy` | `Valid`, `signingCredential.untrusted`, claims `trainedAlgorithmicMedia` |
| `segments` | **`Invalid`** — `assertion.dataHash.mismatch`, though the manifest still parses |
| `strip` | no credential |
| `webp` | no credential |

The `segments` row is the argument for verifying rather than sniffing: a
checker that asks only "is there a manifest?" finds one here. The page is also
honest about whose decision trust is — `Trusted` needs a trust list, this page
has none, so the lab's self-issued certificate reads Valid with an `untrusted`
finding.

The verifier itself, about 9 MB of WebAssembly, comes from a pinned CDN URL on
first use. That is the opposite trade from the trend page, which loads nothing
from anywhere; the difference is that this page is answering a client's
question, and a client has the SDK.

### The verifier on the published site

The same page is deployed at `/verifier/` next to the trend page, and the trend
page links to it. `pages.py build` stages it — the three files, a
`samples.json` and the renditions it names — so one command produces the whole
site and the link can never point at nothing:

```bash
.venv/bin/python pages.py build --report out/live/platforms.json --site out/site \
    --verifier-out out/live
```

`--verifier-out` is what fills the chips; the weekly workflow passes
`out/live`, so the published page lists the renditions that run downloaded and
anyone can open one and see the client's verdict for themselves. Only the run
named on the command line is published — the `out/` runs measured with a
personal photo are never part of a build, and the page would have nothing of
theirs to show anyway. Every URL written into the site is relative, because
Pages serves it from `/<repo>/` rather than a host root.

The two pages link to each other. The trend page's paragraph points forward;
`pages.py` writes this page's title into `samples.json` as `home` and the
verifier renders the way back from it. That is data rather than markup because
only the published site has a trend page above it — `verifier.py` serves this
page alone, sends no `home`, and the link stays hidden instead of pointing at
the page you are already reading.

Each chip is also an address. Opening a rendition puts its catalogue URL in the
fragment — `#samples/live/discord__attachment.jpeg` — so one verdict can be sent
to someone and is already open when they follow it, and a link to a name the
catalogue does not have says so rather than doing nothing. A second chip
replaces the card for the file it opens, so clicking around does not pile up
copies of the same verdict.

## What this does not prove

- The signing CA is generated locally, so `trusted` means trusted against that
  throwaway anchor, not against a public trust list. Signature and binding
  checks are real; the trust decision is ours.
- `screenshot_1080p` re-renders onto a white canvas. It models the metadata
  behaviour of a screenshot, not a real compositor or display pipeline.
- One transformation per file; `double_share` is the only chained case.
- JPEG and PNG only — no video, PDF, HEIF, or HTML embedding.
- The five social adapters (Discord, Telegram, Mastodon, Bluesky, Facebook) are
  implemented and unit-tested against a stub transport, but have **not** been
  run against the live services: each needs an account, and most publish the
  fixture. Their rows appear only when a credential is supplied; nothing in the
  report claims otherwise.

## Files

- `lab.py` — signing, segment surgery, transforms, verification, report
- `platforms.py` — platform adapters, served-rendition fetch, local CDN
  harness, redaction, platform report
- `tests.py` — 16 tests: credential validity and trust, APP11 round-trip,
  each arm, verdict classification, summary counts, matrix integration and
  mode filtering
- `test_platforms.py` — 82 tests: format sniffing, URL/token redaction, config
  and environment handling, each adapter's response parsing against a stub,
  Facebook rendition ordering and duplication, the read-only credential checks,
  Telegram chat discovery and its non-consuming update read, refetch handling
  including the ignored-`chmod` fallback, summarisation, HTTP failures, and the
  full upload → fetch → verify path against the local harness
- `pages.py` — the published trend page: outcome reduction, run history, HTML
  rendering, staging the verifier beside it, and the branch publish (stdlib
  only)
- `test_pages.py` — 45 tests: outcome classification and precedence, credential
  scrubbing of error text, history ordering and idempotence, HTML escaping,
  streak reporting, build idempotence, the staged verifier's link back to this
  page, and publish against a throwaway bare repository, including a fresh
  build directory keeping the published columns
- `verifier.py` — serves the browser verifier and the renditions a run's
  `served/` directory already holds, and stages the static copy that ships
  with the trend page (stdlib only)
- `verifier/` — the page itself: `c2pa-web` 0.15.4 from a pinned CDN URL, no
  build step, no framework
- `test_verifier.py` — 19 tests: rendition discovery and labels, run ordering,
  single-file roots, duplicate run names, the sample routes including the
  traversal attempts that must 404, and the static build's catalogue, relative
  URLs, addressable renditions, the link back to the trend page, and the fact
  that the local server sends no such link, plus empty-root behaviour
- `demo.py` — one command from the local arm to a served page (stdlib only
  itself; the arm it calls needs the signing stack)
- `test_demo.py` — 2 tests: the arm → report → page path end to end, and one
  column per measurement
- `platforms.example.json` — credential template (the real file is gitignored)
- `requirements.txt` — pinned c2pa-python, Pillow, cryptography, requests
