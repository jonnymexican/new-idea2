#!/usr/bin/env python3
"""
A byte-level BPE tokenizer, trained from scratch on the corpus.

Byte-pair encoding starts from raw bytes (256 possible symbols) and repeatedly
merges the most frequent adjacent pair into a new symbol. Frequent strings like
" the" or "ing" end up as single tokens, so the same text costs far fewer tokens
than one-token-per-character.

Nothing here is imported from a tokenizer library: the trainer, the encoder and
the decoder are all implemented below. Text is pretokenised with the pattern
GPT-2 uses so that merges never cross word boundaries in silly ways.

    .venv/bin/python bpe.py --vocab-size 512 --out runs_bpe/tokenizer.json
"""

import argparse
import json
import os
import re
from collections import Counter

# GPT-2's pretokenisation pattern. Contractions and words keep their leading
# space, numbers are separate, runs of other characters are separate.
PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)

BYTE_VOCAB = 256


def pretokenize(text):
    """Split text into the units BPE is allowed to merge inside."""
    return PATTERN.findall(text)


def _apply_merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(text, vocab_size, verbose=False):
    """Learn BPE merges until the vocabulary reaches `vocab_size` symbols.

    Each merge is chosen by counting adjacent pairs weighted by how often the
    word containing them occurs, which is what "most frequent pair" means.
    """
    if vocab_size <= BYTE_VOCAB:
        raise ValueError(f"vocab_size must exceed {BYTE_VOCAB} (bytes)")

    counts = Counter(pretokenize(text))
    # Each distinct pretoken is stored once as a list of byte ids.
    words = {w: list(w.encode("utf-8")) for w in counts}

    merges = []
    while BYTE_VOCAB + len(merges) < vocab_size:
        pairs = Counter()
        for word, ids in words.items():
            weight = counts[word]
            for i in range(len(ids) - 1):
                pairs[(ids[i], ids[i + 1])] += weight
        if not pairs:
            break
        (a, b), freq = pairs.most_common(1)[0]
        merged = _apply_merge  # local alias, the loop below is hot
        new_id = BYTE_VOCAB + len(merges)
        merges.append([a, b])
        for word, ids in words.items():
            if len(ids) > 1 and a in ids:
                words[word] = merged(ids, (a, b), new_id)
        if verbose:
            print(f"  merge {len(merges):4d}: {a}+{b} -> {new_id}  "
                  f"(freq {freq})  "
                  f"{_expand_pair(merges, a)}+{_expand_pair(merges, b)}")

    return merges


def _expand_pair(merges, token_id):
    """The bytes a token id stands for, for readable training output."""
    if token_id < BYTE_VOCAB:
        return repr(bytes([token_id]))[2:-1]
    a, b = merges[token_id - BYTE_VOCAB]
    return _expand_pair(merges, a) + _expand_pair(merges, b)


class BPETokenizer:
    """Encodes text by replaying the learned merges in the order they were made."""

    name = "bpe"

    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.vocab_size = BYTE_VOCAB + len(self.merges)
        # Earlier merges win, so ranking by position gives the greedy choice.
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._word_cache = {}
        self._expand_cache = {}

    # ------------------------------------------------------------- encoding
    def _encode_word(self, ids):
        while len(ids) >= 2:
            best_rank, best_at = None, None
            for i in range(len(ids) - 1):
                rank = self.ranks.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_at = rank, i
            if best_at is None:
                break
            ids = ids[:best_at] + [BYTE_VOCAB + best_rank] + ids[best_at + 2:]
        return ids

    def _encode_pretoken(self, word):
        ids = self._word_cache.get(word)
        if ids is None:
            ids = self._encode_word(list(word.encode("utf-8")))
            self._word_cache[word] = ids
        return ids

    def encode(self, text):
        return [i for w in pretokenize(text) for i in self._encode_pretoken(w)]

    # ------------------------------------------------------------- decoding
    def _expand(self, token_id):
        if token_id < BYTE_VOCAB:
            return bytes([token_id])
        cached = self._expand_cache.get(token_id)
        if cached is None:
            a, b = self.merges[token_id - BYTE_VOCAB]
            cached = self._expand(a) + self._expand(b)
            self._expand_cache[token_id] = cached
        return cached

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            out += self._expand(int(i))
        return out.decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- disk
    def save(self, path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "name": "bpe", "merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["merges"])

    def state(self):
        """Everything needed to rebuild this tokenizer (for a checkpoint)."""
        return {"name": "bpe", "merges": [list(m) for m in self.merges]}

    def show(self, text):
        """Token-by-token breakdown, for looking at what the merges learned."""
        pieces = []
        for word in pretokenize(text):
            for i in self._encode_pretoken(word):
                pieces.append((i, self._expand(i).decode("utf-8", "replace")))
        return pieces


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/input.txt")
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--out", default="runs_bpe/tokenizer.json")
    ap.add_argument("--demo", default="GLOUCESTER: the king hath sent for me.\n")
    ap.add_argument("--verbose", action="store_true",
                    help="print every merge as it is learned")
    args = ap.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"corpus      : {len(text):,} characters")
    print(f"pretokens   : {len(set(pretokenize(text))):,} distinct")
    print(f"training    : byte-level BPE to vocab {args.vocab_size} ...")
    merges = train(text, args.vocab_size, verbose=args.verbose)
    tok = BPETokenizer(merges)

    ids = tok.encode(text)
    ratio = len(text) / len(ids)
    print(f"vocab       : {tok.vocab_size} ({len(merges)} merges over 256 bytes)")
    print(f"encoded     : {len(ids):,} tokens for {len(text):,} characters")
    print(f"compression : {ratio:.2f} characters per token")

    # Round-trip must be exact, including the characters outside the vocab.
    back = tok.decode(ids)
    print(f"round-trip  : {'exact' if back == text else 'MISMATCH'}"
          f" ({len(back):,} chars returned)")
    assert back == text, "BPE round-trip is not lossless"

    print("\nlongest tokens learned (most characters per token):")
    ranked = sorted(range(BYTE_VOCAB, tok.vocab_size),
                    key=lambda i: -len(tok._expand(i)))
    for i in ranked[:12]:
        piece = tok._expand(i).decode("utf-8", "replace")
        print(f"  id {i:4d}  {piece!r}")

    print(f"\nhow it tokenizes: {args.demo!r}")
    for i, piece in tok.show(args.demo):
        print(f"  id {i:4d}  {piece!r}")

    tok.save(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
