#!/usr/bin/env python3
"""What the cache and the speculation actually buy, measured.

Every number here is wall-clock on the machine that ran it, because the point of
the file is to stop claims about speed being made from the shape of the code. Run
it with a checkpoint whose tokenizer you have:

    .venv/bin/python bench_decode.py                       # the BPE checkpoint
    .venv/bin/python bench_decode.py --checkpoint runs/model.pt --tokens 120

Three things are timed.

`uncached` is the reference loop: every token re-reads the whole window. It is
what generation did before `infer.py`, and it is the denominator.

`cached` reads the window once and then writes into it. `--refresh` is how many
tokens are generated before the window is re-read, so the cost per token is
roughly `read(window - refresh) / refresh + 1`. `refresh` equal to the window is
the strict sliding window: no room to write, so it re-reads every token and
should land on top of `uncached`.

`spec` lets the model's own lower layers guess ahead and checks the guesses in
one pass. It is measured rather than assumed, because on a CPU this is a
throughput question about batch sizes and dispatch overhead, not a matter of the
algorithm being clever: a pass over four tokens is not four times a pass over
one, and the draft has to be cheap enough to pay for itself.
"""

import argparse
import time

import torch

from chat import context_window, load_model
from infer import Decoder, SpeculativeDecoder, Sampler


def timed(make_decoder, tokens):
    """Run a decoder for `tokens` tokens and return (tokens/s, decoder)."""
    decoder = make_decoder()
    started = time.perf_counter()
    for _ in decoder.tokens(tokens):
        pass
    elapsed = time.perf_counter() - started
    return tokens / elapsed, decoder


def prompt_ids(tokenizer, window):
    """A real prompt of the checkpoint's own text, `window` tokens long."""
    text = ("KING HENRY: The gentle mind is not the weaker mind, and reason "
            "rules where anger cannot reach. What says the court to this? ")
    ids = tokenizer.encode(text)
    while len(ids) < window:
        ids = ids + ids
    return ids[-window:]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs_tok_bpe/model.pt")
    ap.add_argument("--tokens", type=int, default=80,
                    help="tokens to generate per measurement")
    ap.add_argument("--window", type=int, default=None,
                    help="context window (default: the checkpoint's block_size)")
    args = ap.parse_args()

    model, tokenizer, _ = load_model(args.checkpoint)
    window = context_window(model, args.window)
    ids = prompt_ids(tokenizer, window)
    sampler = Sampler(temperature=0.8, top_k=200)

    print(f"checkpoint : {args.checkpoint}")
    print(f"model      : {model.num_params():,} params, {model.n_layer} layers, "
          f"{model.block_size}-token block")
    print(f"window     : {window}, prompt {len(ids)} tokens, "
          f"{args.tokens} tokens generated")
    print(f"threads    : {torch.get_num_threads()}\n")

    # The reference: a hand-written loop that re-reads the window every token.
    def uncached():
        idx = torch.tensor([ids], dtype=torch.long)
        produced = []
        with torch.no_grad():
            for _ in range(args.tokens):
                logits, _ = model(idx[:, -window:])
                token = sampler.sample(logits[:, -1, :], idx[0].tolist())
                produced.append(token)
                idx = torch.cat((idx, torch.tensor([[token]])), dim=1)
        return produced

    started = time.perf_counter()
    reference = uncached()
    base = args.tokens / (time.perf_counter() - started)
    print(f"{'mode':22s} {'tokens/s':>9s} {'speedup':>8s}   detail")
    print(f"{'uncached':22s} {base:9.0f} {1.0:8.2f}x   re-reads the window")

    best = None
    for refresh in (1, max(1, window // 8), max(1, window // 4), max(1, window // 2)):
        rate, decoder = timed(lambda r=refresh: Decoder(model, ids, window, sampler,
                                                        refresh=r), args.tokens)
        label = f"cached refresh={refresh}"
        print(f"{label:22s} {rate:9.0f} {rate / base:8.2f}x   "
              f"{decoder.reads} window reads for {args.tokens} tokens")
        if refresh > 1 and (best is None or rate > best[1]):
            best = (refresh, rate)

    # Speculation, on top of the best cache setting, with the draft's depth and
    # lookahead swept because both are guesses about the machine.
    if best is not None:
        refresh = best[0]
        for layers in (1, 2):
            if layers >= model.n_layer:
                continue
            for guesses in (2, 4, 6):
                rate, decoder = timed(
                    lambda l=layers, g=guesses: SpeculativeDecoder(
                        model, ids, window, sampler, draft_layers=l,
                        guesses=g, refresh=refresh), args.tokens)
                accept = decoder.accepted / max(decoder.proposed, 1)
                print(f"{'spec ' + str(layers) + 'L/' + str(guesses) + ' ahead':22s} "
                      f"{rate:9.0f} {rate / base:8.2f}x   "
                      f"{accept:.0%} of guesses accepted, "
                      f"{decoder.accepted / max(decoder.rounds, 1):.2f} tokens/round")

    print("\nThe cached run must produce exactly the tokens the uncached one did "
          "when refresh=1;\nthe tests check that. Above refresh=1 the window "
          "re-anchors in chunks, so the text\nis allowed to differ — the numbers "
          "above are what that difference buys.")


if __name__ == "__main__":
    main()
