#!/usr/bin/env python3
"""
A local web prompt for a trained checkpoint, streaming tokens into the browser.

One standard-library process serves a single page and a small JSON API. There is
no framework, no build step and no new dependency: what is new here is the
plumbing, not the model. Sampling is *not* reimplemented — the server calls the
same `stream_reply` that `chat.py` uses, so the browser, the terminal prompt and
training-time sampling all share one code path.

    .venv/bin/python serve.py
    .venv/bin/python serve.py --checkpoint runs_v3/model.pt --port 8765

Then open http://127.0.0.1:8765/ .

Endpoints:

    GET  /            the page (self-contained: inline CSS and JS)
    GET  /api/info    checkpoint, tokenizer and model description
    POST /api/tokens  {"text": ...} -> how the tokenizer splits it
    POST /api/generate  JSON settings -> newline-delimited JSON token stream

`/api/generate` streams, so the shape matters. Each line is one JSON object:

    {"t": "meta",  ...}                     what the reply is being made with
    {"t": "chunk", "text": "..."}           text to append, as it is sampled
    {"t": "done",  ...}                     the reply, its statistics, the window
    {"t": "error", "message": "..."}        after the headers, mid-stream

`done` carries `text`, `steps`, `seconds`, `prompt_tokens`, `in_context` (every
token written), `block_size`, and the two counts that make the window honest:
`dropped_tokens` (written, but outside the window by the end) and
`dropped_prompt_tokens` (tokens of the prompt alone that never reached the
model, because `stream_reply` crops to the last `block_size` tokens). A prompt
longer than the window is the quiet case: without that count, the page would
look as though the model had read all of it.

`boundary_chars` is where that window begins, as a character offset into the
transcript the model was given (`prompt + text`), so the page can draw the edge
of its memory in the place it actually falls. It is null when the offset cannot
be placed in the text — see `window_boundary_chars`.

It is deliberately an NDJSON stream over chunked transfer encoding rather than
Server-Sent Events: EventSource cannot POST, and 40 lines of fetch plus a
TextDecoder is a smaller cost than marshalling the request through a query
string.

This binds to loopback by default. It runs whatever checkpoint you point it at
for anyone who can reach the port, so keep it on 127.0.0.1 unless you mean it.
"""

import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from chat import load_model, stream_reply

DEFAULT_CHECKPOINT = "runs_tok_bpe/model.pt"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# Generous enough for real use, small enough that a runaway client cannot ask
# the server to spend the afternoon tokenising a novel.
MAX_BODY_BYTES = 1 << 20
MAX_PROMPT_CHARS = 100_000
MAX_NEW_TOKENS = 2000


class ClientGone(Exception):
    """Raised inside the sampling loop when the browser has disconnected."""


# --------------------------------------------------------------------- model


def window_boundary_chars(tokenizer, text, ids, block):
    """Where the model's window begins in `text`, as a character offset.

    The window holds the last `block` tokens of `ids`, so what it cannot read is
    precisely what the text of those tokens does not cover. Measured from the
    visible tail rather than from the dropped prefix, because those two count
    differently when a tokenisation cuts a character in half — and a byte-level
    one can, since a token is a run of bytes and not a run of characters.

    Returns None rather than a guess if the boundary cannot be placed in the
    text: a line drawn in the wrong place is worse than no line.
    """
    if len(ids) <= block:
        return 0                      # the whole thing is inside the window
    visible = tokenizer.decode(ids[-block:])
    if text.endswith(visible):
        return len(text) - len(visible)
    if visible.startswith("\ufffd") and text.endswith(visible[1:]):
        # The window opens inside a multi-byte character: `visible` is the
        # dropped bytes of that character as one replacement character, followed
        # by the rest. Counting that replacement character as the single
        # character it belongs to is what puts the line *before* it, which is
        # the honest side to err on — the model read part of one character, and
        # nothing above the line can claim to have been read.
        return len(text) - len(visible)
    return None


class Session:
    """A loaded checkpoint, shared by every request.

    `sample_next` and `tokenizer.encode` are called from request threads, and a
    BPE tokenizer keeps a merge cache as it goes, so generation is serialised
    with a lock: one model on one CPU means one reply at a time anyway, and this
    makes the sharing obviously safe rather than subtly so.
    """

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model, self.tokenizer, self.ckpt = load_model(checkpoint)
        self.lock = threading.Lock()
        # Replies the browser walked away from, and how far each got. Worth
        # counting: an abandoned reply stops at whatever the socket buffering
        # allowed, and that is otherwise invisible.
        self.abandoned_replies = 0
        self.last_abandoned_chunks = 0

    def note_abandoned(self, chunks):
        """Record an abandoned reply: `chunks` includes the write that failed."""
        self.abandoned_replies += 1
        self.last_abandoned_chunks = chunks

    def info(self):
        cfg = self.ckpt.get("config", {})
        return {
            "abandoned_replies": self.abandoned_replies,
            "last_abandoned_chunks": self.last_abandoned_chunks,
            "checkpoint": self.checkpoint,
            "tokenizer": self.tokenizer.name,
            "vocab_size": self.tokenizer.vocab_size,
            "num_params": self.model.num_params(),
            "block_size": self.model.block_size,
            "step": self.ckpt.get("step"),
            "config": {k: cfg.get(k) for k in
                       ("n_layer", "n_head", "n_embd", "block_size", "dropout")},
        }

    def tokens(self, text):
        """How the tokenizer splits `text` — the `/tokens` command, for the page."""
        with self.lock:
            if hasattr(self.tokenizer, "show"):
                pieces = self.tokenizer.show(text)   # BPE: ids and their bytes
            else:
                pieces = [(i, self.tokenizer.decode([i]))
                          for i in self.tokenizer.encode(text)]
        return {
            "count": len(pieces),
            "chars": len(text),
            "tokens": [{"id": i, "text": p} for i, p in pieces],
        }

    def generate(self, params, emit):
        """Stream a continuation of `params['prompt']`, calling emit(text).

        Everything below this line is chat.py's code: the settings are handed to
        `stream_reply` in the small namespace it reads, and `emit` is a
        file-like object that writes each flushed chunk to the browser.
        """
        out = _EmitterStream(emit)
        block = self.model.block_size
        args = argparse.Namespace(
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
            penalty_window=params["penalty_window"],
            max_new_tokens=params["max_new_tokens"],
            stop=params["stop"],
        )

        with self.lock:
            if params["seed"] is not None:
                # Seeding per request, so the same text and seed reproduce the
                # same continuation — which is what makes this testable.
                torch.manual_seed(params["seed"])
            ids = self.tokenizer.encode(params["prompt"])
            prompt_tokens = len(ids)
            text, steps, seconds = stream_reply(self.model, self.tokenizer,
                                                ids, args, out)
            # The transcript the model was given, and the tokens it made of it.
            # `in_context` is that whole sequence: every token written.
            full = params["prompt"] + text
            full_ids = self.tokenizer.encode(full)
            in_context = len(full_ids)
            # Inside the lock as well: decoding walks the tokenizer's cache.
            boundary_chars = window_boundary_chars(self.tokenizer, full,
                                                   full_ids, block)

        # `stream_reply` keeps the last block_size tokens, so everything before
        # them is invisible to the model. Counted and reported rather than left
        # for the reader to infer from a running total — and located, as an
        # offset into the transcript, so the page can draw the edge of the
        # window where it actually falls instead of at some approximation.
        return {
            "text": text,
            "steps": steps,
            "seconds": seconds,
            "prompt_tokens": prompt_tokens,
            "dropped_prompt_tokens": max(0, prompt_tokens - block),
            "in_context": in_context,
            "dropped_tokens": max(0, in_context - block),
            "boundary_chars": boundary_chars,
            "block_size": block,
        }


class _EmitterStream:
    """A minimal write/flush sink that turns text into streamed chunks.

    `stream_reply` already flushes exactly the text it wants shown, so no
    buffering logic is duplicated here — a write *is* a chunk.
    """

    def __init__(self, emit):
        self.emit = emit
        self.chunks = 0

    def write(self, text):
        if text:
            self.emit(text)
            self.chunks += 1

    def flush(self):
        pass


# ---------------------------------------------------------------- validation


def validate(body):
    """Turn a request body into sampling settings, or raise ValueError."""
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")

    prompt = body.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt is longer than {MAX_PROMPT_CHARS} characters")
    if not prompt:
        raise ValueError("prompt is empty; there is nothing to continue")

    def number(key, default, low, high):
        value = body.get(key, default)
        if value is None:
            value = default
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")
        if not (low <= value <= high):
            raise ValueError(f"{key} must be between {low} and {high}")
        return float(value)

    def integer(key, default, low, high):
        value = number(key, default, low, high)
        if int(value) != value:
            raise ValueError(f"{key} must be a whole number")
        return int(value)

    temperature = number("temperature", 0.8, 0.0, 5.0)
    top_k = integer("top_k", 200, 0, 100_000)      # 0 disables the cutoff
    top_p = body.get("top_p", None)
    if top_p is not None:
        top_p = number("top_p", 0.95, 0.0, 1.0)
        if top_p <= 0.0:
            top_p = None
    repetition_penalty = number("repetition_penalty", 1.0, 0.01, 10.0)
    penalty_window = integer("penalty_window", 64, 1, 4096)
    max_new_tokens = integer("max_new_tokens", 160, 1, MAX_NEW_TOKENS)

    stop = body.get("stop", None)
    if stop is not None:
        if not isinstance(stop, str):
            raise ValueError("stop must be a string")
        if not stop:
            stop = None

    seed = body.get("seed", None)
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be a whole number")
        if not (0 <= seed < 2 ** 31):
            raise ValueError("seed must be between 0 and 2147483647")

    return {
        "prompt": prompt,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "penalty_window": penalty_window,
        "max_new_tokens": max_new_tokens,
        "stop": stop,
        "seed": seed,
    }


# ------------------------------------------------------------------- server


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "local-prompt/1.0"

    def handle_one_request(self):
        """Swallow the reset a browser causes by closing an idle keep-alive.

        socketserver would otherwise print a full traceback for it, which looks
        like a server fault when it is only a tab being closed.
        """
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    # Quiet: one line per request is useful when watching the log, but the
    # default is noisy enough to bury the streaming.
    def log_message(self, fmt, *args):
        if self.path.startswith("/api/generate"):
            return
        super().log_message(fmt, *args)

    # ------------------------------------------------------------- utilities
    @property
    def session(self):
        return self.server.session

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status, message):
        self.send_json({"error": message}, status=status)

    def read_body(self):
        length = self.headers.get("Content-Length")
        if length is None:
            raise ValueError("expected a JSON body")
        try:
            length = int(length)
        except ValueError:
            raise ValueError("Content-Length is not a number")
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError(f"body is larger than {MAX_BODY_BYTES} bytes")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"body is not valid JSON: {exc}")

    # ------------------------------------------------------- streamed writes
    def start_stream(self):
        """Begin a chunked NDJSON response.

        Written by hand because http.server does not do chunked encoding for
        you, and with a streamed reply there is no length to declare up front.
        """
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self._streaming = True

    def stream_event(self, event):
        """Write one JSON object as a chunk. Raises ClientGone if the client left."""
        data = (json.dumps(event) + "\n").encode("utf-8")
        try:
            self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            # The browser went away (tab closed, Stop pressed). Stop sampling.
            self.close_connection = True
            raise ClientGone() from exc

    def end_stream(self):
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True

    # ---------------------------------------------------------------- routes
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/info":
            self.send_json(self.session.info())
        elif path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_error_json(404, f"no route for {path}")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            body = self.read_body()
        except ValueError as exc:
            self.send_error_json(400, str(exc))
            return

        if path == "/api/tokens":
            text = body.get("text", "")
            if not isinstance(text, str):
                self.send_error_json(400, "text must be a string")
                return
            self.send_json(self.session.tokens(text))

        elif path == "/api/generate":
            try:
                params = validate(body)
            except ValueError as exc:
                self.send_error_json(400, str(exc))
                return
            self.stream_generate(params)

        else:
            self.send_error_json(404, f"no route for {path}")

    def stream_generate(self, params):
        self.start_stream()
        started = time.time()
        emitted = 0

        def emit(text):
            nonlocal emitted
            emitted += 1          # counted before the write, so a failed one counts
            self.stream_event({"t": "chunk", "text": text})

        try:
            self.stream_event({"t": "meta", "checkpoint": self.session.checkpoint,
                               "tokenizer": self.session.tokenizer.name})
            result = self.session.generate(params, emit)
            result["t"] = "done"
            result["seconds"] = round(result.get("seconds", 0.0), 2)
            result["total_seconds"] = round(time.time() - started, 2)
            self.stream_event(result)
        except ClientGone:
            self.session.note_abandoned(emitted)
            return                      # the client left; nothing to send
        except Exception as exc:        # noqa: BLE001 - report it, do not die
            self.stream_event({"t": "error", "message": f"{type(exc).__name__}: {exc}"})
        self.end_stream()


def build_server(checkpoint=DEFAULT_CHECKPOINT, host=DEFAULT_HOST, port=DEFAULT_PORT):
    """Create the server without starting it. Used by main() and by the tests."""
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.session = Session(checkpoint)
    return server


# --------------------------------------------------------------------- page

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local prompt</title>
<style>
  :root {
    --bg: #0f1115; --panel: #161a21; --panel-2: #1b202a; --line: #262c37;
    --text: #e7e7e4; --dim: #8b93a3; --accent: #d8a657; --accent-2: #7fb3d5;
    --danger: #d98282;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    /* A fixed-height column so the output pane can fill what is left instead of
       guessing at it; the page itself never scrolls on a wide screen. */
    display: flex; flex-direction: column; overflow: hidden;
  }
  header {
    padding: 18px 22px 14px; border-bottom: 1px solid var(--line);
    display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: baseline;
  }
  h1 { font-size: 17px; margin: 0; letter-spacing: .2px; font-weight: 600; }
  h1 span { color: var(--accent); }
  .meta { color: var(--dim); font-size: 12.5px; font-variant-numeric: tabular-nums; }
  /* The window line takes a row of its own under the title, and turns amber
     once the transcript is longer than the model can see. */
  .window { width: 100%; }
  .window.warn { color: var(--accent); }
  .note {
    margin: 10px 22px 0; padding: 9px 12px; border-left: 2px solid var(--accent);
    background: #171a20; color: #c9cdd6; font-size: 12.5px; border-radius: 0 5px 5px 0;
  }
  main {
    flex: 1 1 auto; min-height: 0; display: grid;
    grid-template-columns: minmax(0, 1fr) 322px; gap: 0;
  }
  section.doc {
    padding: 16px 22px 18px; min-width: 0; min-height: 0;
    display: flex; flex-direction: column;
  }
  .doc-scroll {
    flex: 1 1 auto; min-height: 160px; overflow-y: auto;
    background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
    padding: 16px 18px; white-space: pre-wrap; word-wrap: break-word;
    font: 14.5px/1.7 ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  }
  .said { color: var(--accent); }
  .reply { color: var(--text); }
  .reply.live::after {
    content: "▍"; color: var(--accent-2); animation: blink 1s steps(2) infinite;
  }
  @keyframes blink { 50% { opacity: 0; } }
  .hint { color: var(--dim); }
  /* The edge of the model's window, drawn into the transcript where it falls.
     The label is a pseudo-element on purpose: text here would join the
     document's text, and the document's text is the transcript that the next
     prompt is built from and that Copy puts on the clipboard. */
  .cut {
    display: block; position: relative; height: 0;
    border-top: 1px dashed var(--accent); margin: 20px 0 12px;
  }
  .cut::after {
    content: attr(data-label); position: absolute; top: -9px; left: 2px;
    padding: 0 7px; background: var(--panel); color: var(--accent);
    font: 11.5px/1.4 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    white-space: nowrap;
  }
  /* What the model can no longer read stays legible, and recedes. */
  .unread { opacity: .45; }
  textarea {
    width: 100%; margin-top: 12px; background: var(--panel-2); color: var(--text);
    border: 1px solid var(--line); border-radius: 8px; padding: 11px 13px;
    font: 14px/1.6 ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    resize: vertical; min-height: 62px;
  }
  textarea:focus { outline: none; border-color: var(--accent); }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }
  button {
    background: var(--panel-2); color: var(--text); border: 1px solid var(--line);
    border-radius: 7px; padding: 8px 15px; font-size: 13.5px; cursor: pointer;
  }
  button:hover:not(:disabled) { border-color: var(--accent); }
  button:disabled { opacity: .4; cursor: default; }
  button.primary { background: var(--accent); color: #1b1508; border-color: var(--accent); font-weight: 600; }
  button.stop { border-color: #5a2f2f; color: var(--danger); }
  .status { color: var(--dim); font-size: 12.5px; margin-left: auto; font-variant-numeric: tabular-nums; }
  aside {
    border-left: 1px solid var(--line); padding: 16px 18px 26px;
    background: var(--panel); min-height: 0; overflow-y: auto;
  }
  aside h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .9px;
             color: var(--dim); margin: 0 0 12px; font-weight: 600; }
  .field { margin-bottom: 13px; }
  .field label { display: flex; justify-content: space-between; font-size: 12.5px;
                 color: #b9c0cc; margin-bottom: 4px; }
  .field output { color: var(--accent); font-variant-numeric: tabular-nums; }
  .field input[type=range] { width: 100%; accent-color: var(--accent); }
  .field input[type=number], .field input[type=text] {
    width: 100%; background: var(--panel-2); color: var(--text);
    border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px;
    font: 13px ui-monospace, Menlo, monospace;
  }
  .field input:focus { outline: none; border-color: var(--accent); }
  .pair { display: grid; grid-template-columns: 1fr 84px; gap: 7px; align-items: center; }
  .check { display: flex; gap: 8px; align-items: center; font-size: 12.5px;
           color: #b9c0cc; margin: 9px 0; }
  .check input { accent-color: var(--accent); }
  .presets { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
  .presets button { padding: 5px 10px; font-size: 12px; }
  .tokens {
    margin-top: 14px; border-top: 1px solid var(--line); padding-top: 12px;
    max-height: 26vh; overflow-y: auto;
    /* It carries the answer to "what does the model actually read", so it keeps
       its height: as a shrinkable flex item a tall transcript squeezed it to
       a sliver, chips and all. */
    flex: 0 0 auto;
  }
  .tokens .chip {
    display: inline-block; background: #202634; border: 1px solid var(--line);
    border-radius: 4px; padding: 1px 6px; margin: 0 3px 4px 0;
    font: 12px ui-monospace, Menlo, monospace; color: var(--accent-2);
  }
  .tokens .chip.space { color: #6f7787; }
  .tokens .chip.dropped { opacity: .4; border-style: dashed; }
  code { background: #202634; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
  .err { color: var(--danger); font-size: 12.5px; margin-top: 8px; }
  /* Breakpoints go last, and it matters: a media block that appears before the
     rule it means to override loses to it, since both have the same specificity
     and the later declaration wins. This block used to sit above `.doc-scroll`,
     so the stacked 38vh transcript pane it asks for never once applied. */
  @media (max-width: 900px) {
    /* Stacked: let the whole page scroll rather than squeezing two panes. The
       rows are sized to their content instead of being stretched to fit, or
       the token view is squeezed to nothing by the transcript above it. */
    body { overflow: auto; }
    main { grid-template-columns: 1fr; align-content: start; }
    .doc-scroll { flex: 0 0 38vh; }
    aside { border-left: 0; border-top: 1px solid var(--line); }
  }
</style>
</head>
<body>
<header>
  <h1>Local <span>prompt</span></h1>
  <div class="meta" id="meta">connecting…</div>
  <div class="meta window" id="window" hidden></div>
</header>
<div class="note">
  <strong>This is a base language model, not an assistant.</strong> It was trained
  for one job — continue text — so it does not answer questions or follow
  instructions. Whatever you type is the start of a passage it will keep writing,
  in the style of its corpus.
</div>

<main>
  <section class="doc">
    <div class="doc-scroll" id="doc"><span class="hint">A blank page. Type a line
below — a speaker's name, a scrap of dialogue — and the model will take it from
there. It only remembers the last part of the text (its context window), so a
long session will eventually drop the beginning — the line under the title
keeps count of how much has gone.</span></div>
    <textarea id="prompt" rows="2" placeholder="KING HENRY:"></textarea>
    <div class="row">
      <button class="primary" id="send">Continue</button>
      <button class="stop" id="stop" disabled>Stop</button>
      <button id="reset">Reset context</button>
      <button id="copy">Copy</button>
      <button id="showtokens">Tokens</button>
      <div class="status" id="status">idle</div>
    </div>
    <div class="err" id="err"></div>
    <div class="tokens" id="tokens" hidden></div>
  </section>

  <aside>
    <h2>Sampling</h2>
    <div class="presets">
      <button data-preset="default">Default</button>
      <button data-preset="steady">Steady</button>
      <button data-preset="wild">Wild</button>
      <button data-preset="greedy">Greedy</button>
    </div>

    <div class="field">
      <label>Temperature <output id="temperature-out">0.80</output></label>
      <div class="pair">
        <input type="range" id="temperature" min="0" max="2" step="0.05" value="0.8">
        <input type="number" id="temperature-n" min="0" max="5" step="0.05" value="0.8">
      </div>
    </div>

    <div class="field">
      <label>Top-k <output id="top_k-out">200</output></label>
      <div class="pair">
        <input type="range" id="top_k" min="0" max="512" step="1" value="200">
        <input type="number" id="top_k-n" min="0" max="100000" step="1" value="200">
      </div>
      <div class="hint" style="font-size:11.5px">0 disables the cutoff</div>
    </div>

    <div class="field">
      <label>Top-p (nucleus) <output id="top_p-out">off</output></label>
      <div class="pair">
        <input type="range" id="top_p" min="0.05" max="1" step="0.01" value="0.95" disabled>
        <input type="number" id="top_p-n" min="0.05" max="1" step="0.01" value="0.95" disabled>
      </div>
      <div class="check"><input type="checkbox" id="top_p-on"><label for="top_p-on"
        style="margin:0">keep the smallest set of tokens summing to p</label></div>
    </div>

    <div class="field">
      <label>Repetition penalty <output id="rep-out">1.00</output></label>
      <div class="pair">
        <input type="range" id="rep" min="1" max="1.5" step="0.01" value="1">
        <input type="number" id="rep-n" min="0.01" max="10" step="0.01" value="1">
      </div>
      <div class="hint" style="font-size:11.5px">1.0 disables it</div>
    </div>

    <div class="field">
      <label>Penalty window</label>
      <input type="number" id="penalty_window" min="1" max="4096" step="1" value="64">
    </div>

    <div class="field">
      <label>Max new tokens</label>
      <input type="number" id="max_new_tokens" min="1" max="2000" step="1" value="160">
    </div>

    <div class="field">
      <label>Stop when this appears</label>
      <input type="text" id="stop" placeholder="empty = never stop">
    </div>

    <div class="field">
      <label>Seed</label>
      <input type="text" id="seed" placeholder="empty = different each time">
    </div>

    <div class="check"><input type="checkbox" id="keep" checked>
      <label for="keep" style="margin:0">continue the text above</label></div>
    <div class="check"><input type="checkbox" id="seed_lock">
      <label for="seed_lock" style="margin:0">bump the seed each reply</label></div>

    <div class="hint" style="font-size:11.5px;margin-top:10px">
      Sampling rules and the repetition penalty are <code>chat.py</code>'s —
      this page calls the same <code>stream_reply</code>, so the terminal prompt
      and the browser behave identically.
    </div>
  </aside>
</main>

<script>
(() => {
  const $ = (id) => document.getElementById(id);
  const doc = $("doc"), statusEl = $("status"), errEl = $("err");
  const windowEl = $("window");
  let transcript = "", live = null, controller = null, streaming = false;
  let blockSize = 0;      // the model's window, from /api/info
  let lastStats = { tokens: 0, seconds: 0, in_context: 0, block_size: 0,
                    prompt_tokens: 0, dropped_tokens: 0, dropped_prompt_tokens: 0 };

  // ---------------------------------------------------------------- numbers
  // Setting .value in JS fires no events, so this cannot loop back on itself.
  const bind = (range, number, out, fmt) => {
    const show = (v) => { if (out) out.textContent = fmt(v); };
    range.addEventListener("input", () => { number.value = range.value; show(range.value); });
    number.addEventListener("input", () => { range.value = number.value; show(number.value); });
    show(range.value);
  };
  const f2 = (v) => Number(v).toFixed(2);
  bind($("temperature"), $("temperature-n"), $("temperature-out"), f2);
  bind($("top_k"), $("top_k-n"), $("top_k-out"), (v) => String(parseInt(v, 10) || 0));
  bind($("rep"), $("rep-n"), $("rep-out"), f2);

  const pRange = $("top_p"), pNumber = $("top_p-n"), pOut = $("top_p-out"), pOn = $("top_p-on");
  const pEnabled = () => pOn.checked;
  const renderP = () => {
    pOut.textContent = pEnabled() ? f2(pRange.value) : "off";
    pRange.disabled = pNumber.disabled = !pEnabled();
  };
  pRange.addEventListener("input", () => { pNumber.value = pRange.value; renderP(); });
  pNumber.addEventListener("input", () => { pRange.value = pNumber.value; renderP(); });
  pOn.addEventListener("change", renderP);
  renderP();

  // ---------------------------------------------------------------- presets
  const PRESETS = {
    default: { temperature: 0.8, top_k: 200, top_p: false, rep: 1 },
    steady:  { temperature: 0.6, top_k: 60,  top_p: false, rep: 1.05 },
    wild:    { temperature: 1.15, top_k: 400, top_p: true,  rep: 1 },
    greedy:  { temperature: 1, top_k: 1,    top_p: false, rep: 1 },
  };
  for (const b of document.querySelectorAll("[data-preset]")) {
    b.addEventListener("click", () => {
      const p = PRESETS[b.dataset.preset];
      $("temperature").value = $("temperature-n").value = p.temperature;
      $("top_k").value = $("top_k-n").value = p.top_k;
      pOn.checked = p.top_p;
      $("rep").value = $("rep-n").value = p.rep;
      $("temperature-out").textContent = f2(p.temperature);
      $("top_k-out").textContent = String(p.top_k);
      $("rep-out").textContent = f2(p.rep);
      renderP();
    });
  }

  // ----------------------------------------------------------------- render
  const scrollDown = () => {
    const nearBottom = doc.scrollTop + doc.clientHeight >= doc.scrollHeight - 60;
    if (nearBottom) doc.scrollTop = doc.scrollHeight;
  };
  // The window holds the last block_size tokens and nothing older, so the two
  // counts that matter are how much is in view and how much has fallen out.
  // "667/64" would be nonsense, and "all 64 tokens in view" while 603 sat out
  // of view was worse.
  const renderWindow = (s) => {
    if (!s || !s.block_size) { windowEl.hidden = true; return; }
    windowEl.hidden = false;
    windowEl.className = "meta window" + (s.dropped_tokens ? " warn" : "");
    const parts = s.dropped_tokens
      ? [`context window full: the last ${s.block_size} tokens are what the model`
         + ` reads — ${s.dropped_tokens} of ${s.in_context} written are out of view`]
      : [`context window: ${s.in_context}/${s.block_size} tokens in view`];
    if (s.dropped_prompt_tokens) {
      parts.push(`your prompt was ${s.prompt_tokens} tokens, so the model saw`
                 + ` only its last ${s.block_size} — the opening of what you`
                 + ` typed was never read`);
    }
    windowEl.textContent = parts.join(" · ");
  };
  const setStatus = () => {
    if (streaming) return;
    const s = lastStats;
    renderWindow(s);
    if (!s.tokens) { statusEl.textContent = "idle"; return; }
    const window_ = s.dropped_tokens
      ? `${s.block_size} in view · ${s.dropped_tokens} out`
      : `${s.in_context}/${s.block_size} in view`;
    statusEl.textContent = `${s.tokens} tokens · `
      + `${(s.tokens / Math.max(s.seconds, 1e-9)).toFixed(0)} tok/s · ${window_}`;
  };
  const fail = (message) => { errEl.textContent = message; };
  const clearFail = () => { errEl.textContent = ""; };
  // The token view is a view of *this* transcript, so it goes when the
  // transcript does: chips left over from the previous one describe text that
  // is no longer on screen.
  const clearTokens = () => {
    const box = $("tokens");
    box.hidden = true;
    box.innerHTML = "";
  };
  // The counts under the title say how much of the transcript the model can no
  // longer read. This says *where* the reading stops: a rule drawn into the
  // text at the offset the server reports, with what is above it dimmed.
  const cutOut = () => {
    for (const el of [...doc.querySelectorAll(".cut")]) el.remove();
    for (const child of doc.childNodes) child.classList.remove("unread");
  };
  const drawCut = (mark) => {
    cutOut();
    if (!mark) return;
    const marker = document.createElement("span");
    marker.className = "cut";
    marker.dataset.label =
      `outside the model's window · ${mark.dropped} tokens above`;
    marker.title = `everything above this line is further back than`
      + ` ${mark.block} tokens, so the model no longer reads it`;
    // Land it between characters: split whichever span the offset falls inside,
    // so every sibling is wholly above the line or wholly below it.
    let seen = 0, placed = false;
    for (const child of [...doc.childNodes]) {
      const len = child.textContent.length;
      if (seen + len >= mark.offset) {
        const at = mark.offset - seen;
        if (at === 0) {
          doc.insertBefore(marker, child);
        } else {
          const tail = child.cloneNode(false);   // same kind of span
          tail.textContent = child.textContent.slice(at);
          child.textContent = child.textContent.slice(0, at);
          doc.insertBefore(marker, child.nextSibling);
          doc.insertBefore(tail, marker.nextSibling);
        }
        placed = true;
        break;
      }
      seen += len;
    }
    // The offset is not in the document on screen — a stale reply, say. No line
    // is better than one in the wrong place.
    if (!placed) { cutOut(); return; }
    let above = true;
    for (const child of doc.childNodes) {
      if (child === marker) { above = false; continue; }
      child.classList.toggle("unread", above);
    }
  };

  function append(kind, text) {
    const span = document.createElement("span");
    span.className = kind;
    span.textContent = text;
    doc.appendChild(span);
    return span;
  }

  // -------------------------------------------------------------- settings
  const settings = () => ({
    temperature: parseFloat($("temperature").value),
    top_k: parseInt($("top_k").value, 10) || 0,
    top_p: pEnabled() ? parseFloat(pRange.value) : null,
    repetition_penalty: parseFloat($("rep").value),
    penalty_window: parseInt($("penalty_window").value, 10) || 64,
    max_new_tokens: parseInt($("max_new_tokens").value, 10) || 160,
    stop: $("stop").value || null,
    seed: $("seed").value.trim() === "" ? null : parseInt($("seed").value, 10),
  });

  // ---------------------------------------------------------------- the ask
  async function send() {
    if (streaming) return;
    clearFail();
    const line = $("prompt").value;
    if (!line.trim() && !$("keep").checked) { fail("type something to continue"); return; }
    $("prompt").value = "";        // the line is consumed, as in chat.py

    // The prompt is the text to be continued: the running transcript when
    // "continue the text above" is on, otherwise just this line.
    const base = $("keep").checked ? transcript : "";
    const prompt = base + line + "\\n";

    if (base === "") { doc.innerHTML = ""; transcript = ""; clearTokens(); }
    append("said", line + "\\n");
    live = append("reply live", "");
    transcript = prompt;
    // The line described the transcript as it stood before this line was added,
    // so it goes now and comes back once the reply has moved the window again.
    cutOut();

    const body = settings();
    body.prompt = prompt;
    if (body.seed !== null && $("seed_lock").checked) {
      body.seed = body.seed + 1;
      $("seed").value = String(body.seed);
    }

    controller = new AbortController();
    streaming = true;
    $("send").disabled = true;
    $("stop").disabled = false;
    statusEl.textContent = "sampling…";
    let count = 0, closing = "", cut = null;

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try { detail = (await response.json()).error || detail; } catch (e) {}
        throw new Error(detail);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\\n");
        buffer = lines.pop();
        for (const raw of lines) {
          if (!raw.trim()) continue;
          const event = JSON.parse(raw);
          if (event.t === "chunk") {
            live.textContent += event.text;
            count += 1;
            scrollDown();
          } else if (event.t === "done") {
            transcript = prompt + event.text;
            lastStats = {
              tokens: event.steps,
              seconds: event.total_seconds || event.seconds || 0,
              in_context: event.in_context,
              block_size: event.block_size,
              prompt_tokens: event.prompt_tokens,
              dropped_tokens: event.dropped_tokens,
              dropped_prompt_tokens: event.dropped_prompt_tokens,
            };
            // Where the window begins, if the server could place it: null means
            // it could not, and then nothing is drawn.
            cut = event.dropped_tokens && event.boundary_chars !== null
                  && event.boundary_chars !== undefined
              ? { offset: event.boundary_chars, dropped: event.dropped_tokens,
                  block: event.block_size }
              : null;
          } else if (event.t === "error") {
            throw new Error(event.message);
          }
        }
      }
    } catch (exc) {
      if (exc.name === "AbortError") {
        transcript = prompt + live.textContent;
        // Keep what was sampled; report the stop rather than the last reply's
        // statistics, which is what setStatus() would otherwise show.
        closing = `stopped after ${count} tokens`;
      } else {
        fail(String(exc.message || exc));
        closing = "failed";
      }
    } finally {
      streaming = false;
      live.classList.remove("live");
      live = null;
      controller = null;
      $("send").disabled = false;
      $("stop").disabled = true;
      // Drawn once the reply is finished: mid-stream the window is a moving
      // target, and an abandoned reply leaves no line at all, because the text
      // it would have described is no longer the text on screen.
      drawCut(cut);
      if (closing) statusEl.textContent = closing; else setStatus();
      scrollDown();
    }
  }

  $("send").addEventListener("click", send);
  $("stop").addEventListener("click", () => { if (controller) controller.abort(); });
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  $("reset").addEventListener("click", () => {
    transcript = "";
    doc.innerHTML = '<span class="hint">cleared.</span>';
    statusEl.textContent = "idle";
    clearFail();
    clearTokens();
    lastStats = { tokens: 0, seconds: 0, in_context: 0, block_size: 0,
                  prompt_tokens: 0, dropped_tokens: 0, dropped_prompt_tokens: 0 };
    renderWindow(lastStats);
  });
  $("copy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(transcript); statusEl.textContent = "copied"; }
    catch (e) { fail("clipboard blocked by the browser"); }
  });

  const chip = (label, text) => {
    const el = document.createElement("span");
    el.className = "chip" + (/^\\s+$/.test(text) ? " space" : "");
    el.title = `id ${label}`;
    el.textContent = text.replace(/\\n/g, "\\\\n") || "(blank)";
    return el;
  };
  $("showtokens").addEventListener("click", async () => {
    const box = $("tokens");
    if (!box.hidden) { box.hidden = true; return; }
    const text = $("keep").checked ? transcript : $("prompt").value;
    if (!text) { fail("nothing to tokenise yet"); return; }
    clearFail();
    try {
      const response = await fetch("/api/tokens", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const info = await response.json();
      if (info.error) throw new Error(info.error);
      // The same encode the model does, so these chips line up with its window:
      // everything before the first `out` chips is no longer attended to.
      const out = blockSize ? Math.max(0, info.count - blockSize) : 0;
      box.innerHTML = `<div class="meta" style="margin-bottom:8px">${info.count} tokens
        for ${info.chars} characters — ${(info.chars / Math.max(info.count, 1)).toFixed(2)}
        characters each</div>`
        + (out ? `<div class="meta" style="margin:-4px 0 8px">the model no
          longer sees the first ${out} of these — only the last ${blockSize} tokens
          are in its window</div>` : "");
      info.tokens.forEach((t, i) => {
        const el = chip(t.id, t.text);
        if (i < out) el.classList.add("dropped");   // out of the window
        box.appendChild(el);
      });
      box.hidden = false;
    } catch (exc) { fail(String(exc.message || exc)); }
  });

  // ------------------------------------------------------------------ boot
  fetch("/api/info").then((r) => r.json()).then((info) => {
    if (info.error) throw new Error(info.error);
    blockSize = info.block_size;
    $("meta").textContent =
      `${info.checkpoint} · ${info.tokenizer} vocab ${info.vocab_size} · `
      + `${info.num_params.toLocaleString()} params · ${info.block_size}-token context`;
  }).catch((exc) => { $("meta").textContent = "no model: " + (exc.message || exc); });
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help="checkpoint to serve; it carries its own tokenizer")
    ap.add_argument("--host", default=DEFAULT_HOST,
                    help="loopback by default: anyone who can reach the port can "
                         "use the model")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help="0 picks a free port and prints it")
    ap.add_argument("--open", action="store_true",
                    help="open the page in a browser once it is listening")
    args = ap.parse_args()

    # Loading the checkpoint takes a few seconds on a CPU-only box, and the
    # socket is not bound until it is done — so say so, flushed, rather than
    # leaving the user staring at a URL that is not answering yet.
    print(f"loading    : {args.checkpoint}", flush=True)
    server = build_server(args.checkpoint, args.host, args.port)
    host, port = server.server_address[0], server.server_address[1]
    url = f"http://{host}:{port}/"

    print(f"tokenizer  : {server.session.tokenizer.name}, "
          f"vocab {server.session.tokenizer.vocab_size}", flush=True)
    print(f"model      : {server.session.model.num_params():,} params, "
          f"{server.session.model.block_size}-token context", flush=True)
    print(f"listening  : {url}", flush=True)
    print("Ctrl-C to stop.", flush=True)

    if args.open:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
