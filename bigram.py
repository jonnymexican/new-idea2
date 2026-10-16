#!/usr/bin/env python3
"""
A bigram character model: the simplest possible language model.

It predicts the next character using only the *previous* character, by counting
how often each pair occurs in the training text. There is no learning, no
attention, and no context beyond a single character.

That makes it a useful floor: anything fancier should beat it. Running this
next to `train.py` shows how much the transformer's attention is actually
buying, and gives a principled reference line for the loss curve.

    .venv/bin/python bigram.py
"""

import argparse
import json
import math
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/input.txt")
    ap.add_argument("--out", default="runs/bigram_baseline.json")
    ap.add_argument("--smoothing", type=float, default=0.1,
                    help="add-k smoothing, so unseen pairs are not assigned zero probability")
    ap.add_argument("--tokenizer", choices=("char", "bpe"), default="char",
                    help="count pairs of characters, or of subword tokens")
    ap.add_argument("--bpe-tokenizer", default=None,
                   help="existing tokenizer.json to reuse "
                        "(default: <out-dir>/tokenizer.json when present)")
    ap.add_argument("--bpe-vocab-size", type=int, default=512)
    args = ap.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    if args.tokenizer == "bpe":
        # Imported here so the character path stays free of any dependency.
        from bpe import BPETokenizer, train as train_bpe

        path = args.bpe_tokenizer or os.path.join(
            os.path.dirname(args.out) or ".", "tokenizer.json"
        )
        if os.path.exists(path):
            tokenizer = BPETokenizer.load(path)
        else:
            tokenizer = BPETokenizer(train_bpe(text, args.bpe_vocab_size))
            tokenizer.save(path)
        vocab = tokenizer.vocab_size
        ids = tokenizer.encode(text)
    else:
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        vocab = len(chars)
        ids = [stoi[ch] for ch in text]

    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]

    # How much text one token stands for, so the floor can also be quoted in
    # bits per character and compared against a differently tokenised model.
    # Measured on the validation split, exactly as train.py does, so the two
    # numbers are on the same footing.
    chars_per_token = 1.0
    if args.tokenizer == "bpe":
        chars_per_token = len(tokenizer.decode(val_ids)) / len(val_ids)
    val_bits_per_char = None

    # Count every (previous, next) character pair in the training split.
    counts = [[0] * vocab for _ in range(vocab)]
    for a, b in zip(train_ids, train_ids[1:]):
        counts[a][b] += 1
    row_totals = [sum(row) for row in counts]

    k = args.smoothing

    def cross_entropy(seq):
        """Mean negative log-likelihood of `seq` under the bigram model."""
        total = 0.0
        denom_cache = {}
        for a, b in zip(seq, seq[1:]):
            denom = denom_cache.get(a)
            if denom is None:
                denom = row_totals[a] + k * vocab
                denom_cache[a] = denom
            total += -math.log((counts[a][b] + k) / denom)
        return total / (len(seq) - 1)

    train_loss = cross_entropy(train_ids)
    val_loss = cross_entropy(val_ids)
    uniform = math.log(vocab)

    unit = "tokens" if args.tokenizer == "bpe" else "characters"
    print(f"tokenizer      : {args.tokenizer}, vocab {vocab} {unit}")
    print(f"train / val    : {len(train_ids):,} / {len(val_ids):,} {unit}")
    print(f"uniform baseline (ln V) : {uniform:.4f}")
    print(f"bigram  baseline  train : {train_loss:.4f}")
    print(f"bigram  baseline  val   : {val_loss:.4f}   "
          f"(perplexity {math.exp(val_loss):.2f})")
    if args.tokenizer == "bpe":
        val_bits_per_char = (val_loss / math.log(2))
        val_bits_per_char /= chars_per_token
        print(f"chars per token (val)   : {chars_per_token:.4f}")
        print(f"bigram val in bits/char : {val_bits_per_char:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "uniform_loss": round(uniform, 5),
                "bigram_val_perplexity": round(math.exp(val_loss), 3),
                "tokenizer": args.tokenizer,
                "vocab_size": vocab,
                "chars_per_token": round(chars_per_token, 4),
                "val_bits_per_char": (round(val_bits_per_char, 4)
                                      if val_bits_per_char is not None else None),
                "smoothing": k,
            },
            f,
            indent=2,
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
