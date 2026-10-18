#!/usr/bin/env python3
"""
Talk to a trained model from the terminal.

Read this before you type: nothing here was instruction-tuned. The model was
trained on exactly one job — continue text — so it does **not** answer questions,
follow instructions, or hold a conversation the way a chat assistant does. Ask it
"what is 2+2?" and it will continue the sentence in the style of its corpus. That
is not a bug in this script; it is the honest behaviour of a base language model,
and it is what makes the prompt box the most interesting way to look at one.

    .venv/bin/python chat.py                          # the BPE model by default
    .venv/bin/python chat.py --checkpoint runs_v3/model.pt
    .venv/bin/python chat.py --prompt "KING HENRY:" --max-new-tokens 200
    .venv/bin/python chat.py --context 24        # a deliberately smaller window
    .venv/bin/python chat.py --refresh 1         # re-read the window every token
    .venv/bin/python chat.py --spec 2            # the lower layers guess ahead
    echo "To be or not" | .venv/bin/python chat.py

In the prompt, an empty line generates from a fresh start, and anything else is
appended to the running context and continued. /help lists the controls.

`--context` (or `/window` at the prompt) sets how much of the end of the text
the model is given. It cannot exceed the checkpoint's `block_size`, which is what
the positional embeddings were trained for; below that, it is a handicap you can
choose, and a useful one — the model keeps writing, but with less to go on.

The replies come from `infer.py`, which keeps the attention cache: the window is
read once and the next tokens are written into it. `--refresh` is how many tokens
that lasts before the window moves on and is re-read; the default (a quarter of
the window) is about three times faster than re-reading on every token, and
`--refresh 1` is the old behaviour, token for token. `--spec` adds speculative
drafts — exact, because the acceptance rule corrects for the draft — which is
interesting to watch and, on this CPU, slower than the cache alone.
"""

import argparse
import os
import sys
import time

import torch

import infer
from model import GPT, sample_next
from train import tokenizer_from_state


def load_model(path):
    """Rebuild the model and its tokenizer from a checkpoint.

    The checkpoint carries both, which is the point: the weights are only
    meaningful next to the vocabulary they were trained against.
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"no checkpoint at {path}\n"
            "train one first, e.g.  .venv/bin/python train.py --max-iters 1000\n"
            "or point at an existing run:  --checkpoint runs_v3/model.pt"
        )

    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("tokenizer")
    if state is None and "itos" in ckpt:
        # Checkpoints written before the tokenizer became pluggable.
        state = {"name": "char",
                 "itos": [ckpt["itos"][i] for i in sorted(ckpt["itos"])]}
    if state is None:
        raise SystemExit(f"{path} has no tokenizer in it; cannot rebuild the model")

    tokenizer = tokenizer_from_state(state)
    cfg = ckpt["config"]
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        block_size=cfg["block_size"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, ckpt


def context_window(model, requested=None):
    """How many tokens of the running text the model is actually given.

    `block_size` is a ceiling, not a requirement: the positional embeddings and
    the attention mask were trained for at most that many tokens, so asking for
    more is not a thing the model can do, and asking for fewer is a trade the
    caller is entitled to make. Everything downstream measures against this
    number rather than against the model's maximum.
    """
    if requested is None:
        return model.block_size
    if requested < 1:
        raise ValueError("the context window must be at least one token")
    return min(int(requested), model.block_size)


def stream_reply(model, tokenizer, ids, args, out, context=None, refresh=None,
                 spec=0, draft_layers=2, guesses=4):
    """Generate a continuation of `ids`, printing it as it arrives.

    `context` is how much of the *end* of `ids` the model gets to see. It
    defaults to the model's own `block_size`; passing less is how the window is
    shortened deliberately, and it is the reason the crop below is not written
    as `-block` any more.

    The token loop itself lives in `infer.py`, where the cache and the optional
    speculation are; this function is the part that knows about text, stopping
    and the caller's stream.
    """
    window = context_window(model, context)
    prompt = [int(t) for t in ids[-window:]]
    base = tokenizer.decode(prompt)
    started = time.time()

    sampler = infer.Sampler(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        penalty_window=args.penalty_window,
    )
    if spec:
        decoder = infer.SpeculativeDecoder(model, prompt, window, sampler,
                                           draft_layers=draft_layers,
                                           guesses=guesses, refresh=refresh)
    else:
        decoder = infer.Decoder(model, prompt, window, sampler, refresh=refresh)

    steps, shown, text = 0, 0, ""
    for _ in decoder.tokens(args.max_new_tokens):
        steps += 1
        text = tokenizer.decode(decoder.ids)[len(base):]
        if args.stop and args.stop in text:
            # Cut before the stop string, and print only what was kept, so
            # the streamed output matches what is returned.
            text = text[:text.index(args.stop)]
            if len(text) > shown:
                out.write(text[shown:])
                out.flush()
            break
        if len(text) > shown:
            out.write(text[shown:])
            out.flush()
            shown = len(text)

    return text, steps, time.time() - started


def show_tokens(tokenizer, text, out):
    """Show how the tokenizer splits a line — subwords are visible this way."""
    ids = tokenizer.encode(text)
    pieces = " | ".join(f"{tokenizer.decode([i])!r}" for i in ids)
    print(f"  {len(ids)} tokens ({len(text) / max(1, len(ids)):.2f} chars each):",
          file=out)
    print(f"  {pieces}", file=out)


HELP = """\
commands
  /help              this list
  /reset             forget the running context
  /context           show what the model will see next, and how full it is
  /tokens <text>     how the tokenizer splits a line
  /temp <float>      temperature (lower is safer, higher is wilder)
  /k <int>           top-k cutoff (/k 0 disables it)
  /p <float>         nucleus sampling cutoff
  /len <int>         tokens to generate per reply
  /window <int>      how many tokens of the text above the model is given
                     (its block_size is the ceiling) — lower it and watch the
                     model lose the thread
  /penalty <float>   repetition penalty (1.0 disables it)
  /seed <int>        reseed the sampler
  /save <file>       write the transcript so far
  /quit              leave
blank line           generate from a fresh start"""


def repl(model, tokenizer, args):
    context = ""
    transcript = []
    print(f"checkpoint : {args.checkpoint}")
    print(f"tokenizer  : {tokenizer.name}, vocab {tokenizer.vocab_size}")
    print(f"model      : {model.num_params():,} params, "
          f"{model.block_size}-token context")
    if args.context is not None:
        window = context_window(model, args.context)
        print(f"window     : the model is given the last {window} tokens"
              + (" (its own block_size)" if window == model.block_size else ""))
    print("note       : base model — it continues text, it does not answer questions")
    print("Type /help for commands; a blank line generates from a fresh start.\n")

    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            continue

        stripped = line.strip()
        if stripped.startswith("/"):
            parts = stripped.split(maxsplit=1)
            command, rest = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

            if command in ("/quit", "/exit", "/q"):
                return 0
            elif command == "/help":
                print(HELP)
            elif command == "/reset":
                context = ""
                print("(context cleared)")
            elif command in ("/window", "/w"):
                try:
                    args.context = context_window(model, int(rest))
                except ValueError as exc:
                    print(f"({exc})")
                else:
                    print(f"(the model will be given the last {args.context} "
                          f"tokens; its block_size is {model.block_size})")
            elif command == "/context":
                ids = tokenizer.encode(context)
                shown = context[-200:].replace("\n", "\\n")
                window = context_window(model, args.context)
                hidden = max(0, len(ids) - window)
                print(f"  {len(ids)}/{window} tokens in the window"
                      + (f" ({hidden} above it the model cannot read)"
                         if hidden else "") + f": {shown!r}")
            elif command == "/tokens":
                show_tokens(tokenizer, rest or context, sys.stdout)
            elif command in ("/temp", "/temperature"):
                args.temperature = float(rest or args.temperature)
                print(f"(temperature {args.temperature})")
            elif command == "/k":
                args.top_k = int(rest) or None
                print(f"(top-k {args.top_k})")
            elif command == "/p":
                args.top_p = float(rest) or None
                print(f"(top-p {args.top_p})")
            elif command == "/len":
                args.max_new_tokens = int(rest or args.max_new_tokens)
                print(f"(generating {args.max_new_tokens} tokens per reply)")
            elif command == "/penalty":
                args.repetition_penalty = float(rest or args.repetition_penalty)
                print(f"(repetition penalty {args.repetition_penalty})")
            elif command == "/seed":
                torch.manual_seed(int(rest))
                print(f"(seed {rest})")
            elif command == "/save":
                target = rest or "chat_transcript.txt"
                with open(target, "w", encoding="utf-8") as f:
                    f.write("".join(transcript))
                print(f"(wrote {target})")
            else:
                print(f"(unknown command {command}; /help lists them)")
            continue

        # A blank line means "start fresh from a newline", like training sampling.
        if not stripped:
            context = ""
            prompt_text = "\n"
        else:
            prompt_text = context + line + "\n"

        try:
            reply, steps, seconds = stream_reply(
                model, tokenizer, tokenizer.encode(prompt_text), args, sys.stdout,
                context=args.context, refresh=args.refresh, spec=args.spec,
                draft_layers=args.draft_layers,
            )
        except KeyboardInterrupt:
            print("\n(stopped)")
            context = prompt_text
            continue

        context = prompt_text + reply
        transcript += [line + "\n", reply]

        window = context_window(model, args.context)
        status = (f"[{len(tokenizer.encode(context))}/{window} in context"
                  f" · temp {args.temperature}"
                  f" · {args.max_new_tokens} max new")
        if args.stats:
            status += (f" · {steps} in {seconds:.1f}s"
                       f" ({steps / max(seconds, 1e-9):.0f} tok/s)")
        print(f"\n{status}]\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", default="runs_tok_bpe/model.pt",
                    help="checkpoint to load; it carries its own tokenizer")
    ap.add_argument("--prompt", default=None,
                    help="one-shot: complete this text and exit")
    ap.add_argument("--repl", action="store_true",
                    help="force the interactive prompt even when stdin is piped")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--top-p", type=float, default=None,
                    help="nucleus sampling cutoff; off by default")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help=">1 discourages repeating recent tokens (1.0 disables)")
    ap.add_argument("--penalty-window", type=int, default=64)
    ap.add_argument("--context", type=int, default=None,
                    help="tokens of the text above that the model is given; "
                         "defaults to the checkpoint's block_size, and a "
                         "smaller number is a deliberate handicap")
    ap.add_argument("--stop", default=None,
                    help="stop generating when this string appears")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stats", action="store_true",
                    help="report seconds and tokens/s per reply")
    ap.add_argument("--refresh", type=int, default=None,
                    help="tokens generated before the cached window is re-read "
                         "(default: a quarter of the window; 1 re-reads on every "
                         "token, which is exactly the old uncached behaviour)")
    ap.add_argument("--spec", type=int, default=0,
                    help="tokens the model's own lower layers guess ahead each "
                         "round; 0 (the default) disables speculation")
    ap.add_argument("--draft-layers", type=int, default=2,
                    help="how many of the model's layers the draft keeps")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tokenizer, _ = load_model(args.checkpoint)

    if args.prompt is not None:
        _, steps, seconds = stream_reply(
            model, tokenizer, tokenizer.encode(args.prompt), args, sys.stdout,
            context=args.context, refresh=args.refresh, spec=args.spec,
            draft_layers=args.draft_layers,
        )
        print()
        if args.stats:
            print(f"[{steps} tokens in {seconds:.1f}s "
                  f"({steps / max(seconds, 1e-9):.0f} tokens/s)]")
        return 0

    if not args.repl and not sys.stdin.isatty():
        # Piped input with no --prompt: complete it and exit.
        text = sys.stdin.read()
        _, steps, seconds = stream_reply(
            model, tokenizer, tokenizer.encode(text), args, sys.stdout,
            context=args.context, refresh=args.refresh, spec=args.spec,
            draft_layers=args.draft_layers,
        )
        print()
        if args.stats:
            print(f"[{steps} tokens in {seconds:.1f}s]")
        return 0

    return repl(model, tokenizer, args)


if __name__ == "__main__":
    sys.exit(main())
