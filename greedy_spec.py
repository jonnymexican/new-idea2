#!/usr/bin/env python3
"""Greedy speculative decoding: the target's exact text, in fewer passes.

`infer.SpeculativeDecoder` answers a *sampling* question: it accepts a draft
token with probability min(1, p/q) and samples from the residual on rejection,
so the sequence it emits is distributed exactly as the target's own samples
would be — but it is not any particular sequence, and comparing its text with a
baseline's would prove nothing.

This file answers the other question, the one a deterministic decoder asks. When
generation is greedy — always the argmax — the target's output is a single fixed
string, so there is something to be *identical* to, and the acceptance rule
collapses into a comparison:

    guess d_1, d_2, ... from the draft
    run the target once over all of them
    keep the guesses whose token equals the target's argmax at that position
    replace the first one that does not, and stop the round there

Every kept token is the target's own argmax at its position, and the replacement
is the argmax the plain loop would have emitted next, so the text is the same by
construction rather than by luck. The bench still checks it, because "by
construction" is the sort of claim that survives right up until an off-by-one in
the cache — and because a check that has never been seen to fail is not a check,
`verdict` is small enough to test on a deliberately corrupted decode.

The draft is not a second model: it is the target's own first few layers, which
share the vocabulary and the bottom of the stack exactly, so the target's cache
primes it for free. It is therefore *biased* — three layers asked to predict what
six would — which costs acceptance rate and nothing else, since a wrong guess is
discarded rather than believed.

The draft's cost is the whole question, so the bench also runs an *oracle*: the
target's own continuation, handed over as guesses, costing no pass at all. Every
guess is right, so a round returns one more token per target pass than the plain
loop does. That rate is the ceiling — what speculation would reach here if
drafting were free — and the gap between it and a real draft's row is exactly
what the draft is spending.

The one thing the identity depends on is room. Positional embeddings were trained
to `block_size`, so the target cannot verify guesses past the end of its window;
if the decode runs long enough to fill it, `_reanchor` slides the window forward
exactly as `infer.Decoder` does, and the text from that point matches a baseline
that slides at the same points — not one that read a full window every token.
Inside the window, which is where speculation is meant to run and where the bench
stays, the guarantee is unconditional.

    .venv/bin/python greedy_spec.py                       # the BPE checkpoint
    .venv/bin/python greedy_spec.py --reps 5 --tokens 24

Each row is the best of `--reps` runs. Best-of is stated rather than hidden,
because it biases upward: this is a laptop, and one scheduling hiccup in a run
this short moves the number.
"""

import argparse
import os

import torch

from chat import context_window, load_model


def argmax_token(row):
    """The greedy choice for one position's logits, shape (1, vocab)."""
    return int(row.argmax(dim=-1))


class GreedySpeculativeDecoder:
    """Greedy decoding where a cheaper draft proposes and the target disposes.

    `guesses` is how many tokens the draft proposes per round, `draft_layers`
    how many of the target's own layers it keeps. Both are guesses about the
    machine — a deeper draft accepts more and costs more — so the bench sweeps
    them rather than picking one and calling it best.

    Counters, all of which the bench reports and the tests check:
      `rounds`          target passes that ran a speculation round
      `proposed`        tokens the draft guessed
      `accepted`        guesses the target agreed with
      `target_passes`   passes over the target model, the cost that matters
    """

    def __init__(self, model, ids, window, draft_layers=2, guesses=4,
                 oracle=None):
        if not ids:
            raise ValueError("cannot decode from an empty prompt")
        self.model = model.eval()
        self.window = max(2, int(window))
        self.guesses = max(1, int(guesses))
        self.draft_layers = max(1, min(int(draft_layers), model.n_layer))
        self.ids = [int(t) for t in ids]
        self.prompt_len = len(self.ids)
        # A draft that is handed the answer, for measuring the ceiling: `None`
        # means the ordinary draft, the target's own first `draft_layers`.
        self.oracle = None if oracle is None else [int(t) for t in oracle]
        self.cache = None
        self.logits = None
        self.pending = None      # emitted, but not yet a position in the cache
        self.rounds = 0
        self.proposed = 0
        self.accepted = 0
        self.target_passes = 0
        self._prefill()

    # ------------------------------------------------------------- reading
    @torch.inference_mode()
    def _read(self, tokens):
        """Read `tokens` into a fresh cache, keeping the last logits."""
        self.cache = self.model.new_cache()
        idx = torch.tensor([tokens], dtype=torch.long)
        self.logits, _ = self.model(idx, cache=self.cache)
        self.target_passes += 1

    @torch.inference_mode()
    def _prefill(self):
        self._read(self.ids[-self.window:])

    @torch.inference_mode()
    def _reanchor(self):
        """Slide the window, leaving room for one pending token and the guesses.

        This is the only place the identity guarantee weakens: the target is
        then reading a shorter context than a full-window loop would, so its
        argmax can differ. It happens once the decode outgrows `block_size`.
        """
        keep = max(1, self.window - self.guesses - 1)
        self._read(self.ids[-keep:])
        self.pending = None

    # ------------------------------------------------------------- guessing
    @torch.inference_mode()
    def _draft(self, count):
        """`count` greedy guesses, in order.

        With an oracle the guesses are the answer itself and no pass is spent
        producing them, which is what makes that row a ceiling rather than a
        draft. Positions are counted from the prompt, so the first guess after
        `k` emitted tokens is the oracle's token `k`.
        """
        if self.oracle is not None:
            start = len(self.ids) - self.prompt_len
            return self.oracle[start:start + count]
        base = len(self.cache)
        draft_cache = self.cache.copy_prefix(self.draft_layers)
        if self.pending is not None:
            feed, offset = self.pending, base
        else:
            # The cached logits came from the full-depth model, so the draft
            # cannot reuse them; it re-reads the last token through its own
            # shallower stack and starts from there.
            draft_cache.truncate(base - 1)
            feed, offset = self.ids[-1], base - 1

        idx = torch.tensor([[feed]], dtype=torch.long)
        logits, _ = self.model(idx, cache=draft_cache, pos_offset=offset,
                               n_layers=self.draft_layers)

        guesses = []
        for _ in range(count):
            token = argmax_token(logits[:, -1, :])
            guesses.append(token)
            idx = torch.tensor([[token]], dtype=torch.long)
            logits, _ = self.model(idx, cache=draft_cache,
                                   pos_offset=len(draft_cache),
                                   n_layers=self.draft_layers)
        return guesses

    # --------------------------------------------------------- one round
    @torch.inference_mode()
    def round(self):
        """One verification pass. Returns the tokens it produced, in order."""
        base = len(self.cache)
        shift = 0 if self.pending is None else 1
        count = min(self.guesses, self.window - base - shift)
        if count <= 0:
            self._reanchor()
            base = len(self.cache)
            shift = 0
            count = min(self.guesses, self.window - base - 1)
        if count <= 0:
            # A window too small to hold even one guess: one plain greedy token.
            token = argmax_token(self.logits[:, -1, :])
            self.ids.append(token)
            self.pending = token
            self.rounds += 1
            return [token]

        guesses = self._draft(count)
        verify = ([] if self.pending is None else [self.pending]) + guesses
        idx = torch.tensor([verify], dtype=torch.long)
        out, _ = self.model(idx, cache=self.cache, pos_offset=base)
        self.target_passes += 1

        def predict(j):
            """The logits that predict guess j (0-based)."""
            if self.pending is None and j == 0:
                return self.logits[:, -1, :]
            return out[:, j - 1 + shift]

        kept = []
        for j, token in enumerate(guesses):
            predicted = argmax_token(predict(j))
            if predicted == token:
                kept.append(token)
                continue
            final = predicted
            break
        else:
            # Every guess survived, so the target gives one token for free.
            final = argmax_token(predict(len(guesses)))

        # Guesses that lost were computed and are now about text that will
        # never be written; drop those positions from the cache.
        self.cache.truncate(base + shift + len(kept))
        self.pending = final
        self.rounds += 1
        self.proposed += len(guesses)
        self.accepted += len(kept)
        emitted = kept + [final]
        self.ids.extend(emitted)
        return emitted

    def tokens(self, count):
        produced = 0
        while produced < count:
            for token in self.round():
                if produced >= count:
                    break
                produced += 1
                yield token


# --------------------------------------------------------------- baselines
@torch.inference_mode()
def greedy_cached(model, ids, window, count):
    """The target alone, cache growing: the fast reference to beat."""
    cache = model.new_cache()
    idx = torch.tensor([ids[-window:]], dtype=torch.long)
    logits, _ = model(idx, cache=cache)
    out = []
    for _ in range(count):
        token = argmax_token(logits[:, -1, :])
        out.append(token)
        idx = torch.tensor([[token]], dtype=torch.long)
        logits, _ = model(idx, cache=cache, pos_offset=len(cache))
    return out


@torch.inference_mode()
def greedy_uncached(model, ids, window, count):
    """The naive loop: every token re-reads the whole window.

    This is what generation did before the cache, and it is the denominator the
    README's speedups are stated against.
    """
    context = list(ids)
    out = []
    for _ in range(count):
        idx = torch.tensor([context[-window:]], dtype=torch.long)
        logits, _ = model(idx)
        token = argmax_token(logits[:, -1, :])
        out.append(token)
        context.append(token)
    return out


# ------------------------------------------------------------------- bench
def measure(model, ids, window, args):
    """Time the baselines and every draft setting, and check the text.

    Returns the number of mismatches, so the caller can exit non-zero: a bench
    that prints a speedup over wrong text is worse than no bench.
    """
    # Keep the whole run inside the window, which is where the identity
    # guarantee holds and where speculative decoding is defined at all.
    guesses_list = [int(g) for g in args.guesses.split(",") if g.strip()]
    largest = max(guesses_list)
    tokens = min(args.tokens, window - len(ids) - largest)
    if tokens < 1:
        raise SystemExit(
            f"window {window} leaves no room: prompt {len(ids)} + "
            f"{largest} guesses is already {len(ids) + largest}"
        )

    print(f"{tokens} tokens per run, best of {args.reps}, window {window}\n")
    print(f"{'mode':26s} {'tokens/s':>9s} {'vs cached':>10s} {'vs uncached':>12s}"
          f"   detail")
    print("-" * 92)

    from bench_decode import best_of

    want_cached = greedy_cached(model, ids, window, tokens)
    want_uncached = greedy_uncached(model, ids, window, tokens)
    mismatches = 0
    if want_cached != want_uncached:
        print("!! the two baselines disagree, so there is nothing to match")
        mismatches += 1

    rate_uncached, _ = best_of(
        lambda: greedy_uncached(model, ids, window, tokens), tokens, args.reps)
    rate_cached, _ = best_of(
        lambda: greedy_cached(model, ids, window, tokens), tokens, args.reps)
    print(f"{'uncached':26s} {rate_uncached:9.0f} {'-':>10s} {1.0:11.2f}x"
          f"   re-reads the window")
    print(f"{'cached':26s} {rate_cached:9.0f} {1.0:9.2f}x "
          f"{rate_cached / rate_uncached:11.2f}x   one pass per token")

    for layers in [int(x) for x in args.draft_layers.split(",") if x.strip()]:
        for guesses in guesses_list:
            def run(layers=layers, guesses=guesses):
                decoder = GreedySpeculativeDecoder(
                    model, ids, window, draft_layers=layers, guesses=guesses)
                produced = list(decoder.tokens(tokens))
                return (produced, decoder.accepted, decoder.proposed,
                        decoder.rounds, decoder.target_passes)

            rate, (produced, accepted, proposed, rounds, passes) = best_of(
                run, tokens, args.reps)
            result = verdict(produced, want_cached)
            if result != "identical":
                mismatches += 1
            label = f"spec {layers}L/{guesses} ahead"
            print(f"{label:26s} {rate:9.0f} {rate / rate_cached:9.2f}x "
                  f"{rate / rate_uncached:11.2f}x   "
                  f"{accepted / max(proposed, 1):4.0%} accepted "
                  f"({accepted}/{proposed}), "
                  f"{(accepted + rounds) / max(rounds, 1):.2f} tok/round, "
                  f"{passes} passes, {result}")

    # The ceiling: the target's own continuation, handed over as guesses for
    # nothing. Every guess is right, so this is the most speculation could ever
    # gain on this machine, and the gap to a real draft's row is its cost.
    for guesses in guesses_list:
        def run_oracle(guesses=guesses):
            decoder = GreedySpeculativeDecoder(
                model, ids, window, guesses=guesses, oracle=want_cached)
            produced = list(decoder.tokens(tokens))
            return (produced, decoder.accepted, decoder.proposed,
                    decoder.rounds, decoder.target_passes)

        rate, (produced, accepted, proposed, rounds, passes) = best_of(
            run_oracle, tokens, args.reps)
        result = verdict(produced, want_cached)
        if result != "identical":
            mismatches += 1
        print(f"{f'oracle {guesses} ahead':26s} {rate:9.0f} "
              f"{rate / rate_cached:9.2f}x {rate / rate_uncached:11.2f}x   "
              f"draft is free and always right: {passes} passes, "
              f"{(accepted + rounds) / max(rounds, 1):.2f} tok/round, "
              f"{result}")

    print()
    if mismatches:
        print(f"{mismatches} run(s) did NOT reproduce the greedy baseline")
    else:
        print("Every speculative row reproduced the cached baseline token for "
              "token.")
        print("The draft proposes, the target decides; acceptance is what the "
              "guess cost.")
        print("The oracle rows are the ceiling: what free, perfect drafting "
              "would reach here. A real draft's row pays the difference.")
    return mismatches


def first_difference(got, want):
    """Where two token sequences stop agreeing, in words."""
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return f"token {i} ({a} vs {b})"
    return f"length ({len(got)} vs {len(want)})"


def verdict(got, want):
    """The bench's only claim, as a string: is this the target's own text?

    Pulled out of `measure` so the check itself can be tested: a benchmark whose
    failure path has never run is a benchmark that cannot report a failure.
    """
    if got == want:
        return "identical"
    return f"MISMATCH after {first_difference(got, want)}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs_tok_bpe/model.pt")
    ap.add_argument("--tokens", type=int, default=32,
                    help="tokens to generate per run (clamped to the window)")
    ap.add_argument("--prompt", type=int, default=24,
                    help="prompt length in tokens, taken from the corpus text")
    ap.add_argument("--window", type=int, default=None,
                    help="context window (default: the checkpoint's block_size)")
    ap.add_argument("--reps", type=int, default=3,
                    help="runs per row; the best is reported")
    ap.add_argument("--draft-layers", default="1,2,3",
                    help="comma-separated draft depths to sweep")
    ap.add_argument("--guesses", default="2,4,6",
                    help="comma-separated lookaheads to sweep")
    args = ap.parse_args()

    model, tokenizer, _ = load_model(args.checkpoint)
    window = context_window(model, args.window)

    text = ("KING HENRY: The gentle mind is not the weaker mind, and reason "
            "rules where anger cannot reach. What says the court to this?")
    ids = tokenizer.encode(text)[-args.prompt:]
    if not ids:
        raise SystemExit("the tokenizer produced no prompt tokens")

    print(f"checkpoint : {args.checkpoint} "
          f"({os.path.basename(os.path.dirname(args.checkpoint))})")
    print(f"model      : {model.num_params():,} params, {model.n_layer} layers, "
          f"{model.block_size}-token block")
    print(f"window     : {window}, prompt {len(ids)} tokens")
    print(f"threads    : {torch.get_num_threads()}\n")

    return 1 if measure(model, ids, window, args) else 0


if __name__ == "__main__":
    raise SystemExit(main())
