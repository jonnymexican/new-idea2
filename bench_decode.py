#!/usr/bin/env python3
"""What the cache, the speculation and the batching actually buy, measured.

Every number here is wall-clock on the machine that ran it, because the point of
the file is to stop claims about speed being made from the shape of the code. Run
it with a checkpoint whose tokenizer you have:

    .venv/bin/python bench_decode.py                       # the BPE checkpoint
    .venv/bin/python bench_decode.py --checkpoint runs/model.pt --tokens 120

Each row is the best of `--reps` runs (three by default): this is a laptop, and a
single run of eighty tokens is short enough that another process waking up moves
the number. Best-of is stated rather than hidden, because it biases upward.

`uncached` is the reference loop: every token re-reads the whole window. It is
what generation did before `infer.py`, and it is the denominator.

`cached` reads the window once and then writes into it. `refresh` is how many
tokens are generated before the window is re-read, so the cost per token is
roughly `read(window - refresh) / refresh + 1`. `refresh` equal to the window is
the strict sliding window: no room to write, so it re-reads every token and
should land on top of `uncached`.

`spec` lets the model's own lower layers guess ahead and checks the guesses in
one pass. It is measured rather than assumed, because on a CPU this is a
throughput question about batch sizes and dispatch overhead, not a matter of the
algorithm being clever: a pass over four tokens is not four times a pass over
one, and the draft has to be cheap enough to pay for itself.

`batch` is the serving question, and it is not the same question. A cached
decoder serves one reply at about 300 tokens/s because every token reads every
weight once. A batch of eight reads them once for all eight rows, so it serves
eight replies at once. Both served throughput and per-reply speed are printed,
because batching raises the first while lowering the second, and only the second
is what one waiting user feels. Reading the table together with the one above it
answers the question directly: batching multiplies the cache (4 rows are worth
more than 3x) and does not replace it (8 rows without the cache are still slower
than 1 row with it).

Everything above is best-of-N over a run of under a second, which is the right
way to compare two decoders and the wrong way to describe a machine. `--sustained
SECONDS` runs the same decode for minutes and buckets the tokens per second, so
the number that gets quoted can be compared with the number a user actually
experiences: the peak is what a warm start shows, the plateau is what a long
reply gets. The gap is reported only when it is larger than the run-to-run
jitter, because on a shared laptop a 4% dip after a minute is usually another
process rather than a thermal limit.
"""

import argparse
import math
import statistics
import time

import torch

from chat import context_window, load_model
from infer import BatchDecoder, Decoder, Sampler, SpeculativeDecoder

SHADES = "▁▂▃▄▅▆▇█"


def best_of(run, tokens, reps, units=1):
    """Time `run()`, which must produce `tokens` per row, and keep the best.

    Returns (units * tokens / best elapsed, whatever the fastest run returned).
    `units` is the number of rows a batch produced tokens for.
    """
    best_rate, best_info = 0.0, None
    for _ in range(reps):
        started = time.perf_counter()
        info = run()
        rate = units * tokens / (time.perf_counter() - started)
        if rate > best_rate:
            best_rate, best_info = rate, info
    return best_rate, best_info


def sustained(steps, seconds, clock=time.perf_counter):
    """Run `steps` until `seconds` have passed, keeping when each token landed.

    `steps` yields a list of tokens per step — one element per row of a batch,
    so a pass that serves eight rows counts eight tokens at one instant. Each
    sample is (elapsed, tokens produced so far), and the clock is a parameter
    so the loop can be tested without waiting for it.
    """
    start = clock()
    produced = 0
    samples = [(0.0, 0)]
    for tokens in steps:
        produced += len(tokens)
        elapsed = clock() - start
        samples.append((elapsed, produced))
        if elapsed >= seconds:
            break
    return samples


def bucket_rates(samples, seconds=1.0):
    """Tokens per second in each whole `seconds` of the run.

    A token is attributed to the bucket its completion time falls in, so a
    partial final bucket is dropped rather than reported as a slow one — the
    last fraction of a second is not a second. A run shorter than one bucket
    therefore has no measurement in it, and says so by returning nothing.

    Buckets are half-open on the left and closed on the right, `(start, end]`,
    because a token either completes in a second or it does not: with `[start,
    end)` a token landing exactly on the last boundary would belong to a bucket
    that was never finished, and a fast run would quietly lose its last few
    tokens from every bucket boundary it hit. The rounding is there because
    `0.05 * 1200` is not exactly 60.
    """
    total = samples[-1][0] if samples else 0.0
    counts = [0] * max(0, int(total // seconds))
    previous = 0
    for elapsed, produced in samples:
        index = math.ceil(round(elapsed / seconds, 9)) - 1
        done = produced - previous
        if 0 <= index < len(counts):
            counts[index] += done
        previous = produced
    return [count / seconds for count in counts]


def summarise(samples, seconds=1.0):
    """Peak, plateau and drift, with the noise needed to believe them.

    Peak is the best bucket, which is the number a benchmark table would quote.
    Plateau is the median of the second half, which is the number a long reply
    gets. Drift compares medians rather than extremes so one scheduling hiccup
    cannot manufacture a thermal story.

    The bar it has to clear is the standard error of those two medians, not the
    spread of the buckets: a noisy minute can have 20% bucket-to-bucket jitter
    *and* a real 30% fall from start to finish, and comparing the drift against
    the jitter would call that steady — which is exactly the mistake this mode
    exists to avoid. The error is the usual `1.253 * MAD / sqrt(n)` for a
    median, so it shrinks as the run gets longer and a slow decline becomes
    visible instead of being averaged away.
    """
    rates = bucket_rates(samples, seconds)
    if not rates:
        # No bucket means no rate, but the run still produced tokens: reporting
        # zero would look like a decode that did nothing rather than a window
        # too short to measure one.
        return {"rates": [], "peak": 0.0, "plateau": 0.0, "drift": 0.0,
                "jitter": 0.0, "steady": False,
                "tokens": samples[-1][1] if samples else 0,
                "elapsed": samples[-1][0] if samples else 0.0}
    half = max(1, len(rates) // 2)
    first, tail = rates[:half], rates[-half:]
    head = statistics.median(first)
    plateau = statistics.median(tail)
    drift = 0.0 if head <= 0 else max(0.0, 1.0 - plateau / head)

    def median_error(values):
        """Standard error of the median, in tokens/s."""
        mid = statistics.median(values)
        mad = statistics.median([abs(v - mid) for v in values])
        return 1.253 * mad / math.sqrt(len(values))

    error = math.hypot(median_error(first), median_error(tail))
    noise = error / head if head else 0.0
    return {"rates": rates, "peak": max(rates), "plateau": plateau,
            "drift": drift, "jitter": noise, "error": error,
            "steady": drift <= max(0.05, 2 * noise),
            "tokens": samples[-1][1], "elapsed": samples[-1][0]}


def sparkline(rates):
    """The shape of a run in one line, so a cooling slope is visible at a glance."""
    if not rates:
        return ""
    low, high = min(rates), max(rates)
    if high <= low:
        return SHADES[3] * len(rates)
    return "".join(SHADES[int((r - low) / (high - low) * (len(SHADES) - 1))]
                   for r in rates)


def prompt_ids(tokenizer, window):
    """A real prompt of the checkpoint's own text, `window` tokens long."""
    text = ("KING HENRY: The gentle mind is not the weaker mind, and reason "
            "rules where anger cannot reach. What says the court to this? ")
    ids = tokenizer.encode(text)
    while len(ids) < window:
        ids = ids + ids
    return ids[-window:]


def measure(model, ids, window, sampler, args):
    """The sweeps: uncached, cache, speculation, batching, mixed lengths.

    Returns the refresh setting the cache sweep preferred, which is what the
    sustained run below should use — the server does not run a setting that
    only wins over eighty tokens.
    """
    print(f"{args.tokens} tokens per row, best of {args.reps}\n")

    # The reference: a hand-written loop that re-reads the window every token.
    def uncached():
        idx = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            for _ in range(args.tokens):
                logits, _ = model(idx[:, -window:])
                token = sampler.sample(logits[:, -1, :], idx[0].tolist())
                idx = torch.cat((idx, torch.tensor([[token]])), dim=1)

    base, _ = best_of(uncached, args.tokens, args.reps)
    print(f"{'mode':22s} {'tokens/s':>9s} {'speedup':>8s}   detail")
    print(f"{'uncached':22s} {base:9.0f} {1.0:8.2f}x   re-reads the window")

    best_refresh, best_rate = None, 0.0
    for refresh in (1, max(1, window // 8), max(1, window // 4),
                    max(1, window // 2)):
        def run(refresh=refresh):
            decoder = Decoder(model, ids, window, sampler, refresh=refresh)
            for _ in decoder.tokens(args.tokens):
                pass
            return decoder.reads

        rate, reads = best_of(run, args.tokens, args.reps)
        print(f"cached refresh={refresh:<2d}   {rate:9.0f} {rate / base:8.2f}x   "
              f"{reads} window reads for {args.tokens} tokens")
        if refresh > 1 and rate > best_rate:
            best_refresh, best_rate = refresh, rate

    # Speculation, on top of the best cache setting, with the draft's depth and
    # lookahead swept because both are guesses about the machine.
    if best_refresh is not None:
        for layers in (1, 2):
            if layers >= model.n_layer:
                continue
            for guesses in (2, 4, 6):
                def run(layers=layers, guesses=guesses):
                    decoder = SpeculativeDecoder(
                        model, ids, window, sampler, draft_layers=layers,
                        guesses=guesses, refresh=best_refresh)
                    for _ in decoder.tokens(args.tokens):
                        pass
                    return (decoder.accepted / max(decoder.proposed, 1),
                            decoder.accepted / max(decoder.rounds, 1))

                rate, (accept, per_round) = best_of(run, args.tokens, args.reps)
                print(f"{'spec ' + str(layers) + 'L/' + str(guesses) + ' ahead':22s} "
                      f"{rate:9.0f} {rate / base:8.2f}x   "
                      f"{accept:.0%} of guesses accepted, "
                      f"{per_round:.2f} tokens/round")

    # Batching: several rows in one model pass, at both ends of the window
    # policy, because the interesting number is how it composes with the cache.
    print(f"\nbatched serving ({args.tokens} tokens per row, best of {args.reps})")
    print(f"{'rows x refresh':22s} {'served/s':>9s} {'per-row/s':>10s}   detail")
    for refresh in (1, best_refresh if best_refresh else 16):
        for batch in (1, 2, 4, 8, 16):
            def run(batch=batch, refresh=refresh):
                decoder = BatchDecoder(model, [ids] * batch, window, sampler,
                                       refresh=refresh)
                for _ in decoder.steps(args.tokens):
                    pass
                return decoder.reads

            offered, reads = best_of(run, args.tokens, args.reps, units=batch)
            label = f"{batch} {'row' if batch == 1 else 'rows'} x refresh={refresh}"
            print(f"{label:22s} {offered:9.0f} {offered / batch:10.0f}   "
                  f"{reads} window reads, {offered / base:.1f}x uncached")

    # One row shorter than the others means padding, and padding means the
    # shared cache re-anchors when the longest row needs it to. Worth showing
    # rather than hiding: it is the price of serving mixed traffic together.
    mixed = [ids] + [ids[-n:] for n in (8, 40, 56)]

    def run_mixed():
        decoder = BatchDecoder(model, mixed, window, sampler,
                               refresh=best_refresh if best_refresh else 16)
        for _ in decoder.steps(args.tokens):
            pass
        return decoder.reads

    offered, reads = best_of(run_mixed, args.tokens, args.reps, units=4)
    print(f"{'4 mixed lengths':22s} {offered:9.0f} {offered / 4:10.0f}   "
          f"{reads} window reads")

    print("\nThe cached run must produce exactly the tokens the uncached one did "
          "when refresh=1;\nthe tests check that. Above refresh=1 the window "
          "re-anchors in chunks, so the text\nis allowed to differ — the numbers "
          "above are what that difference buys. Batching is\nexact in a stronger "
          "sense: a row's tokens are the same as if it had been decoded\nalone, "
          "with or without padding, and the tests check that too.")
    return best_refresh


def sustained_run(model, ids, window, sampler, args, refresh):
    """Repeat the cached decode for minutes and report the plateau.

    The default row counts are the two headline cases: one reply, and the
    eight-row batch the table above singles out. `refresh` comes from the sweep
    above, because a setting that wins over eighty tokens is not necessarily the
    one worth running for a minute.
    """
    rows_list = [max(1, min(int(part), 64))
                 for part in args.sustained_rows.split(",") if part.strip()]
    if not rows_list:
        raise SystemExit("--sustained-rows needs at least one row count")

    print(f"\nsustained ({args.sustained:g} s per row count, refresh={refresh}, "
          f"buckets of one second)")
    print(f"{'rows':>4s} {'tokens':>8s} {'peak/s':>7s} {'plateau/s':>10s} "
          f"{'drift':>6s} {'±':>6s}  verdict")
    for rows in rows_list:
        decoder = BatchDecoder(model, [ids] * rows, window, sampler,
                               refresh=refresh)
        samples = sustained(decoder.steps(1 << 62), args.sustained)
        stats = summarise(samples)
        if not stats["rates"]:
            print(f"{rows:4d} {stats['tokens']:8d}       -          -     -"
                  f"       -  shorter than one bucket")
            continue
        verdict = "steady" if stats["steady"] else "falls off"
        print(f"{rows:4d} {stats['tokens']:8d} {stats['peak']:7.0f} "
              f"{stats['plateau']:10.0f} {stats['drift']:5.0%} "
              f"{stats['jitter']:5.1%}  {verdict}")
        print(f"     {sparkline(stats['rates'])}")

    print("\nPeak is the best single second, which is what a table like the one "
          "above quotes;\nplateau is the median of the second half, which is "
          "what a long reply gets. `±` is\nthe error on those two medians, so a "
          "drift below it is not a measurement — it is\nthis machine doing "
          "something else, and is worth running again before believing.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs_tok_bpe/model.pt")
    ap.add_argument("--tokens", type=int, default=80,
                    help="tokens to generate per measurement (per row for a batch)")
    ap.add_argument("--window", type=int, default=None,
                    help="context window (default: the checkpoint's block_size)")
    ap.add_argument("--reps", type=int, default=3,
                    help="runs per row; the best is reported")
    ap.add_argument("--sustained", type=float, default=0.0, metavar="SECONDS",
                    help="also decode for this long per row count and report the "
                         "plateau rather than the peak (0 turns it off)")
    ap.add_argument("--sustained-rows", default="1,8", metavar="LIST",
                    help="comma-separated row counts for the sustained run")
    ap.add_argument("--no-table", dest="table", action="store_false",
                    help="skip the sweeps and run only the sustained part")
    args = ap.parse_args()

    if not args.table and args.sustained <= 0:
        ap.error("--no-table only makes sense with --sustained")

    model, tokenizer, _ = load_model(args.checkpoint)
    window = context_window(model, args.window)
    ids = prompt_ids(tokenizer, window)
    sampler = Sampler(temperature=0.8, top_k=200)

    print(f"checkpoint : {args.checkpoint}")
    print(f"model      : {model.num_params():,} params, {model.n_layer} layers, "
          f"{model.block_size}-token block")
    print(f"window     : {window}, prompt {len(ids)} tokens")
    print(f"threads    : {torch.get_num_threads()}\n")

    best_refresh = measure(model, ids, window, sampler, args) if args.table else None
    if args.sustained > 0:
        sustained_run(model, ids, window, sampler, args,
                      best_refresh or max(1, window // 4))


if __name__ == "__main__":
    main()
