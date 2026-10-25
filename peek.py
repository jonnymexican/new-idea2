#!/usr/bin/env python3
"""Watch the checkpoint read: what each token attended to, where it guessed.

`chat.py` answers what the model says; `serve.py` streams it. This answers a
different question — what was it doing — by reading the same weights from the
inside: every layer's attention weights, captured as they are computed, plus the
probability the model gave to the token that actually came next.

    .venv/bin/python peek.py                                   # runs_tok_bpe/model.pt
    .venv/bin/python peek.py --checkpoint runs_v3/model.pt
    .venv/bin/python peek.py --port 0 --no-open

Nothing here trains or changes the model. The capture is a list handed to
`GPT.forward`; the weights are detached after the softmax they were already
computed for, so the logits are bit-identical to a run that captures nothing
(the tests pin that, because a viewer that quietly altered the model would be
worse than no viewer).

Three things come back for one prompt:

* per-token surprise — the bits the model was wrong by *before* seeing each
  token, which is where it was guessing rather than reading;
* attention, layer by layer and head by head, so "the" can be shown finding the
  noun it belongs to — or failing to;
* the next-token distribution, top-k, because that is the decision the whole
  window was feeding.

A checkpoint's `block_size` is a hard ceiling, and the window is the last
`block_size` tokens of the prompt, so the beginning of a long passage is
reported as dropped rather than silently ignored.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PAGE = HERE / "peek"
DEFAULT_CHECKPOINT = "runs_tok_bpe/model.pt"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
DEFAULT_TOP_K = 10

# What one request may ask the server to tokenise and read. The model's own
# block_size bounds the read; this bounds the tokenising, which happens first.
MAX_TEXT_CHARS = 4_000
MAX_BODY_BYTES = 64 * 1024

# The checkpoint's own sample, offered as a prompt to read back. Only the first
# few hundred characters: it is an example, not an archive.
SAMPLE_CHARS = 400

# Attention is published as integer thousandths rather than floats: a hundred
# thousand weights at three decimal places is a megabyte of JSON of which the
# page shows one row at a time, and the precision difference is invisible at
# one-pixel lines.
ATTENTION_SCALE = 1000

PAGE_FILES = ("index.html", "peek.js", "peek.css")


def _thousandths(weights: torch.Tensor) -> list:
    """(T, Tk) attention -> nested ints in 0..1000, rows still summing to ~1000."""
    scaled = (weights * ATTENTION_SCALE).round().to(torch.int32)
    return scaled.tolist()


def look(model, tokenizer, text, top_k=DEFAULT_TOP_K) -> dict:
    """Everything the page needs about `text`, as plain JSON-able data.

    Kept free of HTTP on purpose: the tests call this directly with a tiny
    random model, and the server is a thin wrapper that reads `text` out of a
    query string.
    """
    ids = list(tokenizer.encode(text))
    dropped = 0
    if len(ids) > model.block_size:
        # The positional embeddings and the mask only exist for block_size
        # tokens, so the window is the tail — and the loss of the head is
        # reported rather than hidden.
        dropped = len(ids) - model.block_size
        ids = ids[-model.block_size :]
    if not ids:
        return {"tokens": [], "attention": [], "next": [], "dropped": 0,
                "window": model.block_size, "layers": 0, "heads": 0}

    idx = torch.tensor([ids])
    captured: list[torch.Tensor] = []
    with torch.no_grad():
        logits, _ = model(idx, capture=captured)

    # (T, vocab) — the distribution the model held *before* each token, so row
    # t predicts token t+1. Row -1 is the one still waiting for its answer.
    probs = F.softmax(logits[0], dim=-1)

    tokens = []
    for index, token_id in enumerate(ids):
        entry = {
            "id": int(token_id),
            "text": tokenizer.decode([token_id]),
            "surprise": None,
        }
        if index:
            p = float(probs[index - 1, token_id])
            entry["surprise"] = round(-math.log2(max(p, 1e-12)), 2)
        tokens.append(entry)

    last = probs[-1]
    top = torch.topk(last, min(top_k, last.numel()))
    next_tokens = [
        {"id": int(i), "text": tokenizer.decode([int(i)]), "p": round(float(p), 4)}
        for p, i in zip(top.values, top.indices)
    ]
    entropy = float(-(last * torch.log2(last.clamp_min(1e-12))).sum())

    layers = []
    for layer in captured:
        # (B, n_head, T, Tk); Tk == T here because nothing is cached.
        layers.append([_thousandths(layer[0, head]) for head in range(layer.shape[1])])

    return {
        "decoded": tokenizer.decode(ids),
        "tokens": tokens,
        "attention": layers,
        "layers": len(layers),
        "heads": len(layers[0]) if layers else 0,
        "window": model.block_size,
        "dropped": dropped,
        "next": next_tokens,
        "entropy_bits": round(entropy, 2),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "new-idea2-peek"

    def __init__(self, *args, model=None, tokenizer=None, checkpoint="",
                 sample="", **kwargs):
        self._model = model
        self._tokenizer = tokenizer
        self._checkpoint = checkpoint
        self._sample = sample
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200):
        self._send(json.dumps(payload).encode(), "application/json; charset=utf-8",
                   status=status)

    def do_GET(self):  # noqa: N802 - stdlib names this
        url = urlparse(self.path)
        if url.path == "/api/info":
            return self._json({
                "checkpoint": self._checkpoint,
                "layers": len(self._model.blocks),
                "heads": self._model.blocks[0].attn.n_head,
                "block_size": self._model.block_size,
                "vocab": self._tokenizer.vocab_size,
                "params": self._model.num_params(),
                "sample": self._sample,
            })
        name = "index.html" if url.path in ("/", "/index.html") else url.path[1:]
        if name in PAGE_FILES:
            file = PAGE / name
            if file.is_file():
                content_type = ("text/html; charset=utf-8" if name.endswith(".html")
                                else "text/javascript; charset=utf-8" if name.endswith(".js")
                                else "text/css; charset=utf-8")
                return self._send(file.read_bytes(), content_type)
        self._send(b"not found\n", "text/plain; charset=utf-8", status=404)

    def do_POST(self):  # noqa: N802 - stdlib names this
        if urlparse(self.path).path != "/api/look":
            return self._send(b"not found\n", "text/plain; charset=utf-8", status=404)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            return self._json({"error": "body too large"}, 413)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = str(payload.get("text", ""))
        except (ValueError, AttributeError):
            return self._json({"error": "expected a JSON object with a text field"}, 400)
        if len(text) > MAX_TEXT_CHARS:
            return self._json(
                {"error": f"text longer than {MAX_TEXT_CHARS} characters"}, 413)
        return self._json(look(self._model, self._tokenizer, text))


def sample_beside(checkpoint) -> str:
    """The checkpoint's own `sample.txt`, as a prompt worth reading back.

    A run writes the text the model produced when it finished, which is a far
    better default than anything this file could invent: it is the model's own
    output, warts and invented characters included.
    """
    path = Path(checkpoint).parent / "sample.txt" if checkpoint else None
    if path and path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")[:SAMPLE_CHARS]
    return ""


def serve(model, tokenizer, checkpoint="", host=DEFAULT_HOST, port=DEFAULT_PORT,
          open_browser=True, sample=None):
    """Start the server and return it, without blocking."""
    if not (PAGE / "index.html").is_file():
        raise SystemExit(f"the page is missing from {PAGE}")
    handler = functools.partial(
        Handler, model=model, tokenizer=tokenizer, checkpoint=str(checkpoint),
        sample=sample_beside(checkpoint) if sample is None else sample)
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        raise SystemExit(
            f"cannot listen on {host}:{port}: {exc}\n"
            f"another port: --port {port + 1}") from None
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"peek     : {url}  ({checkpoint}, {model.block_size}-token window, "
          f"Ctrl-C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help=f"checkpoint to read (default {DEFAULT_CHECKPOINT})")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port to serve on, 0 picks a free one "
                             f"(default {DEFAULT_PORT})")
    parser.add_argument("--no-open", action="store_true",
                        help="serve without opening a browser")
    args = parser.parse_args(argv)

    # Imported here so the module can be tested without a checkpoint on disk.
    from chat import load_model

    model, tokenizer, _ = load_model(args.checkpoint)
    server = serve(model, tokenizer, args.checkpoint, args.host, args.port,
                   open_browser=not args.no_open)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
