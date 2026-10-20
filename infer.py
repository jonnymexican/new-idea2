"""Decoding: turning a trained model into text, one token at a time.

Two costs used to be paid on every token, and neither has to be.

The first is the past. Attention is causal, so the key and value a position
contributes never change once it has been computed; an uncached loop recomputes
all of them for every new token, which is the same arithmetic repeated T times
for no new information. `model.KVCache` keeps them instead, so a token costs one
position of work rather than a window of it. The window is still honoured: when
it fills, the oldest position has to fall out, and because positions are absolute
the cache is rebuilt once per window rather than shifted.

The second is the ordering of that work. A token has to be sampled before the
next one can be computed, so the loop is serial and the CPU is idle between
steps. Speculative decoding breaks the seriality: a cheap draft guesses several
tokens ahead, and the target model checks all the guesses in a *single* pass,
because checking g guesses is just running the model on g tokens — which it can
do at once. Accepted guesses are free tokens; the first rejection is replaced by
the target's own sample and the rest are thrown away.

The draft here is not a second trained model. It is this model's own first few
layers, which is a smaller model that happens to share the target's vocabulary
and its bottom half exactly, so the target's cache can prime it for nothing. That
also means the draft is *biased* — it is the target with three quarters of its
depth missing — which is precisely the case speculative sampling is designed to
survive: the acceptance rule below is exact for any draft distribution.

The rule, for a draft distribution q and target distribution p:

    sample x ~ q;  accept with probability min(1, p(x) / q(x))

and on rejection sample from the residual `normalise(max(0, p - q))`. The
resulting marginal is exactly p, whatever q is — that is the theorem, and the
test suite checks it by sampling thousands of tokens from a deliberately wrong
draft and comparing the histogram against p.

`Sampler` holds the sampling rules so the plain path and the speculative path
cannot drift apart: the same temperature, cutoffs and repetition penalty define
both p and q, which is what makes the comparison meaningful.

`BatchDecoder` is the serving answer to a different cost: not the past, but the
idle time. A CPU generating one token at a time reads every weight in the model
to produce a single row of output, so the arithmetic units are mostly waiting on
memory. Give it eight rows and the same weights serve all eight — the pass costs
more, but far less than eight passes. That is why batching and caching are not
rivals: the cache removes work, batching makes the remaining work pay for more
rows. `bench_decode.py` measures both, and their product.
"""

import torch
from torch.nn import functional as F

from model import sample_next, top_p_filter


class Sampler:
    """The documented sampling rules, as both a draw and a distribution.

    `sample` is what the plain path calls and it goes through `sample_next`, so
    generation stays exactly what it was. `distribution` applies the same rules
    without drawing, which is what speculative decoding needs: acceptance is a
    ratio of probabilities, so both models have to be able to state theirs.
    """

    def __init__(self, temperature=1.0, top_k=None, top_p=None,
                 repetition_penalty=1.0, penalty_window=64):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.penalty_window = penalty_window

    def penalise(self, row, history):
        """The repetition penalty, applied to a copy of one position's logits.

        Written here rather than in chat.py because the speculative path has to
        apply it to the *draft* as well: a q that ignored the penalty while p
        used it would be compared against the wrong distribution.
        """
        row = row.clone()
        if self.repetition_penalty == 1.0:
            return row
        recent = set(int(t) for t in history[-self.penalty_window:])
        for token in recent:
            if row[0, token] > 0:
                row[0, token] /= self.repetition_penalty
            else:
                row[0, token] *= self.repetition_penalty
        return row

    def distribution(self, row, history):
        """The probability of each next token, shape (1, vocab)."""
        logits = self.penalise(row, history) / max(self.temperature, 1e-8)
        if self.top_k is not None and self.top_k > 0:
            values, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[..., [-1]], float("-inf"))
        if self.top_p is not None:
            logits = top_p_filter(logits, self.top_p)
        return F.softmax(logits, dim=-1)

    def sample(self, row, history, generator=None):
        """One token id, drawn exactly as `sample_next` draws it.

        `generator` lets a batched row draw from its own stream, so its reply
        does not depend on what it happened to be batched with.
        """
        return int(sample_next(self.penalise(row, history), self.temperature,
                               self.top_k, self.top_p, generator=generator))

    def draw(self, probs):
        return int(torch.multinomial(probs, num_samples=1)[0, 0])


def accepted(p, q, token):
    """Does `token`, proposed by q, survive comparison with the target p?

    Drawn with probability min(1, p(token) / q(token)). The ratio is clamped so
    a token q barely proposed and p does not want at all is rejected instead of
    dividing by zero.
    """
    ratio = float(p[0, token] / q[0, token].clamp_min(1e-30))
    return ratio >= 1.0 or float(torch.rand(())) < ratio


def residual(p, q):
    """What is left of the target's distribution after the draft's is removed.

    Sampling from this on rejection is what makes the round exact: the accepted
    mass from q plus the residual covers p exactly, whatever q was.
    """
    rest = (p - q).clamp_min(0.0)
    total = float(rest.sum())
    return p if total <= 0.0 else rest / total


class Decoder:
    """One context, one cache, and one token at a time out of it.

    `refresh` is the part of the window that is re-read at once, and it is not a
    detail — it is what makes the cache worth having. The positional embeddings
    are absolute and trained up to `block_size`, so a cache holding a full window
    cannot accept another position at all: a naive implementation re-reads the
    whole window on every single token, which is exactly the cost it was meant to
    remove. Re-anchoring `refresh` tokens at a time reads the window once and then
    writes `refresh - 1` tokens into it, so the model always sees between
    `window - refresh` and `window` tokens, and the re-read is amortised.

    `refresh=1` is the strict sliding-by-one window the uncached path uses: same
    text, no saving. Larger values trade a coarser left edge for the saving, and
    the tests pin both ends of that trade.
    """

    def __init__(self, model, ids, window, sampler, refresh=None):
        self.model = model
        self.sampler = sampler
        self.window = max(1, int(window))
        self.refresh = max(1, min(int(refresh) if refresh else max(1, self.window // 4),
                                  self.window))
        self.ids = [int(t) for t in ids]
        if not self.ids:
            raise ValueError("cannot decode from an empty prompt")
        self.cache = None
        self.logits = None
        self.reads = 0        # full-window reads, the thing the cache exists to cut
        self.prefill()

    @torch.no_grad()
    def read(self, tokens):
        """Read `tokens` into a fresh cache and keep its last logits."""
        self.reads += 1
        self.cache = self.model.new_cache()
        idx = torch.tensor([tokens], dtype=torch.long)
        self.logits, _ = self.model(idx, cache=self.cache)

    @torch.no_grad()
    def prefill(self):
        """Read the whole window in once."""
        self.read(self.ids[-self.window:])

    @torch.no_grad()
    def reanchor(self):
        """Re-read the window, leaving `refresh - 1` free positions to write into.

        `refresh=1` keeps the whole window and so has no room at all: it re-reads
        on every token, which is the uncached behaviour exactly — the reference
        the tests hold the faster settings to.
        """
        keep = max(1, self.window - self.refresh + 1)
        self.read(self.ids[-keep:])

    @torch.no_grad()
    def advance(self, token):
        """Extend the cache with a token whose next-token logits are already known."""
        if len(self.cache) + 1 > self.window:
            self.reanchor()
        else:
            idx = torch.tensor([[token]], dtype=torch.long)
            self.logits, _ = self.model(idx, cache=self.cache,
                                        pos_offset=len(self.cache))

    @torch.no_grad()
    def step(self):
        token = self.sampler.sample(self.logits[:, -1, :], self.ids)
        self.ids.append(token)
        self.advance(token)
        return token

    def tokens(self, count):
        for _ in range(count):
            yield self.step()


class SpeculativeDecoder(Decoder):
    """The same decode, with the model's own lower layers guessing ahead.

    Each round costs one target pass over a pending token plus `guesses` drafted
    ones, and returns however many survived verification. The counters
    (`rounds`, `proposed`, `accepted`, `target_passes`) are kept so the speedup
    can be explained rather than merely observed.
    """

    def __init__(self, model, ids, window, sampler, draft_layers=2, guesses=4,
                 refresh=None):
        super().__init__(model, ids, window, sampler, refresh=refresh)
        self.draft_layers = max(1, min(int(draft_layers), model.n_layer - 1))
        self.guesses = max(1, int(guesses))
        self.pending = None
        self.rounds = 0
        self.proposed = 0
        self.accepted = 0
        self.target_passes = 0

    @torch.no_grad()
    def _draft(self):
        """`guesses` tokens from the truncated model, and the q behind each."""
        room = self.window - len(self.cache) - 1
        count = min(self.guesses, room)
        if count <= 0:
            return [], []

        # The first draft pass primes the copied cache. With a pending token,
        # feeding it is all that is needed; without one, the draft has to be
        # told about the last cached token, because the target's logits came
        # from the full-depth model and are not the draft's.
        draft_cache = self.cache.copy_prefix(self.draft_layers)
        if self.pending is not None:
            feed, offset = self.pending, len(draft_cache)
        else:
            feed, offset = self.ids[-1], len(draft_cache) - 1
            draft_cache.truncate(offset)
        idx = torch.tensor([[feed]], dtype=torch.long)
        logits, _ = self.model(idx, cache=draft_cache, pos_offset=offset,
                               n_layers=self.draft_layers)

        history = list(self.ids)
        guessed, draft_probs = [], []
        for _ in range(count):
            q = self.sampler.distribution(logits[:, -1, :], history)
            token = self.sampler.draw(q)
            guessed.append(token)
            draft_probs.append(q)
            history.append(token)
            idx = torch.tensor([[token]], dtype=torch.long)
            logits, _ = self.model(idx, cache=draft_cache,
                                   pos_offset=len(draft_cache),
                                   n_layers=self.draft_layers)
        return guessed, draft_probs

    @torch.no_grad()
    def round(self):
        """One verification pass. Returns the tokens it produced, in order."""
        if len(self.cache) + 1 > self.window:
            # The window is full, so the pending token pushes the oldest one
            # out. It is already the last element of `ids`, so re-reading
            # reads it back in and there is nothing left pending.
            self.reanchor()
            self.pending = None

        pending = self.pending
        guessed, draft_probs = self._draft()
        prefix = [] if pending is None else [pending]
        verify = prefix + guessed
        if not verify:
            # No room left to speculate: the ordinary cached step is the answer.
            self.pending = None
            return [self.step()]

        base = len(self.cache)
        idx = torch.tensor([verify], dtype=torch.long)
        out, _ = self.model(idx, cache=self.cache, pos_offset=base)
        self.target_passes += 1
        self.rounds += 1
        self.proposed += len(guessed)

        # `self.ids` already ends with `pending`: it was emitted last round.
        history = list(self.ids)

        # `out[:, j]` follows verify[j]. Without a pending token the
        # distribution for the first guess is the one already in hand, and
        # everything shifts by one when there is one.
        shift = 0 if pending is None else 1

        def after(j):
            """The target's distribution once verify[:j + 1] has been read."""
            if pending is None and j == 0:
                return self.sampler.distribution(self.logits[:, -1, :], history)
            return self.sampler.distribution(out[:, j - 1 + shift], history)

        kept = []
        final = None
        for j, token in enumerate(guessed):
            p = after(j)
            q = draft_probs[j]
            if accepted(p, q, token):
                kept.append(token)
                history.append(token)
                self.accepted += 1
                continue
            final = self.sampler.draw(residual(p, q))
            break
        else:
            # Every guess survived, so the target gets one token for free.
            final = self.sampler.draw(after(len(guessed)))

        # The guesses that were rejected were computed and are now about text
        # that will never be written; drop those positions from the cache.
        self.cache.truncate(base + len(prefix) + len(kept))
        self.pending = final
        return kept + [final]

    def tokens(self, count):
        produced = 0
        while produced < count:
            for token in self.round():
                if produced >= count:
                    break
                self.ids.append(token)
                produced += 1
                yield token


def decode(model, ids, window, sampler, max_tokens, spec=0, draft_layers=2,
           guesses=4, refresh=None):
    """A decoder for `ids`, cached, optionally with speculative drafts."""
    if spec and spec > 0:
        decoder = SpeculativeDecoder(model, ids, window, sampler,
                                     draft_layers=draft_layers, guesses=guesses,
                                     refresh=refresh)
    else:
        decoder = Decoder(model, ids, window, sampler, refresh=refresh)
    yield from decoder.tokens(max_tokens)


class BatchDecoder:
    """Several prompts decoded together, in one model pass per step.

    The rows do not have to be the same length. They are right-padded to a
    common width, and the pads are hidden by an attention mask rather than by a
    sentinel the model might have been trained to notice; the padded positions
    keep the absolute positions they would have had alone, so a row's logits
    are the same ones it would get on its own. There is a test that holds a
    batched row to a solo decode token for token, because that is the whole
    claim batching is allowed to make.

    Caching works exactly as it does for one row — a re-anchor is one pass over
    the batch instead of one per row — so the two savings compose. What does
    not compose is the *stopping*: rows finish at different times and the batch
    keeps computing for the ones still going, which is why `steps` yields the
    same number of tokens for every row whether they were used or not. The
    caller decides what to keep.
    """

    def __init__(self, model, prompts, window, samplers=None, refresh=None,
                 generators=None):
        if not prompts:
            raise ValueError("a batch needs at least one prompt")
        self.model = model
        self.window = max(1, int(window))
        self.refresh = max(1, min(int(refresh) if refresh else max(1, self.window // 4),
                                  self.window))
        self.ids = []
        for prompt in prompts:
            ids = [int(t) for t in prompt]
            if not ids:
                raise ValueError("cannot decode from an empty prompt")
            self.ids.append(ids[-self.window:])
        self.rows = len(self.ids)

        if samplers is None or isinstance(samplers, Sampler):
            samplers = [samplers or Sampler()] * self.rows
        self.samplers = list(samplers)
        if len(self.samplers) != self.rows:
            raise ValueError("give one sampler per row, or one for the batch")
        self.generators = [None] * self.rows if generators is None else list(generators)
        if len(self.generators) != self.rows:
            raise ValueError("give one generator per row, or none")

        # The pads are masked out, so their id carries no information at all.
        self.pad = 0
        self.cache = None
        self.logits = None
        self.real = [0] * self.rows      # real tokens currently in the cache
        self.at = [0] * self.rows        # which logit row predicts each next token
        self.dropped = [0] * self.rows   # tokens each row has left behind
        self.padded = False              # whether the cache holds any pad at all
        self.live = None                 # which cache positions hold real tokens
        self.reads = 0
        self.prefill()

    def _matrix(self, rows):
        """Right-pad `rows` to one width. Returns the ids and the real lengths."""
        width = max(len(row) for row in rows)
        idx = torch.full((len(rows), width), self.pad, dtype=torch.long)
        lengths = []
        for b, row in enumerate(rows):
            kept = row[-width:]
            idx[b, :len(kept)] = torch.tensor(kept, dtype=torch.long)
            lengths.append(len(kept))
        return idx, lengths

    @staticmethod
    def _live(lengths, width):
        """Which of `width` cache positions hold a real token in each row."""
        live = torch.zeros((len(lengths), width), dtype=torch.bool)
        for b, n in enumerate(lengths):
            live[b, :n] = True
        return live

    @staticmethod
    def _causal(live, T, Tk):
        """Who may attend to whom: earlier positions, and no pad ever.

        `live` rather than a length, because a pad is not always at the end:
        once a short row keeps decoding beside a long one, new tokens land
        *after* its pads. Counting real tokens would then include the pads.
        """
        keys = torch.arange(Tk).view(1, 1, Tk)
        queries = torch.arange(T).view(1, T, 1) + (Tk - T)
        return (keys <= queries) & live[:, None, :]

    @torch.no_grad()
    def read(self, rows):
        """Read `rows` into a fresh cache and keep the last logits of each."""
        self.reads += 1
        idx, lengths = self._matrix(rows)
        width = idx.size(1)
        self.padded = any(n != width for n in lengths)
        self.live = self._live(lengths, width)
        self.cache = self.model.new_cache()
        if self.padded:
            positions = torch.arange(width).view(1, width).expand(len(rows), width)
            self.logits, _ = self.model(
                idx, cache=self.cache, positions=positions,
                attend=self._causal(self.live, width, width))
        else:
            # Equal lengths: the ordinary path, unchanged down to the mask, so a
            # batch of one costs what a single decode costs.
            self.logits, _ = self.model(idx, cache=self.cache)
        self.real = list(lengths)
        self.at = [n - 1 for n in lengths]

    @torch.no_grad()
    def prefill(self):
        self.read(self.ids)

    @torch.no_grad()
    def reanchor(self):
        """One re-read for the whole batch, but each row keeps its own schedule.

        A row still inside its window has nothing to drop, so it is re-read in
        full — which costs nothing in accuracy, because a prefix read at the
        same positions is the same computation. Only rows that have actually
        filled the window give up their oldest tokens, exactly as `Decoder`
        does. Without that distinction a long sibling would silently shorten a
        short prompt's text, which is the one thing batching must not do.
        """
        keep = max(1, self.window - self.refresh + 1)
        rows = []
        for b, ids in enumerate(self.ids):
            available = len(ids) - self.dropped[b]
            if available > self.window:
                take = keep
                self.dropped[b] = len(ids) - take
            else:
                take = available
            rows.append(ids[-take:])
        self.read(rows)

    @torch.no_grad()
    def advance(self, tokens):
        """Extend every row's cache with one token."""
        if len(tokens) != self.rows:
            raise ValueError(f"expected {self.rows} tokens, got {len(tokens)}")
        # The trigger is any row's own fill, not the cache's length: pads
        # occupy cache slots without using up anyone's window.
        if any(n + 1 > self.window for n in self.real):
            self.reanchor()
            return
        Tk = len(self.cache) + 1
        idx = torch.tensor([[int(t)] for t in tokens], dtype=torch.long)
        self.live = torch.cat([self.live,
                               torch.ones((self.rows, 1), dtype=torch.bool)],
                              dim=1)
        if self.padded:
            positions = torch.tensor([[n] for n in self.real], dtype=torch.long)
            self.logits, _ = self.model(
                idx, cache=self.cache, positions=positions, pos_offset=Tk - 1,
                attend=self._causal(self.live, 1, Tk))
        else:
            self.logits, _ = self.model(idx, cache=self.cache,
                                        pos_offset=Tk - 1)
        self.real = [n + 1 for n in self.real]
        # A one-token pass returns one position, so the predicting logit is 0 —
        # not the cache index it sits at.
        self.at = [0] * self.rows

    @torch.no_grad()
    def step(self):
        """One token per row, drawn from that row's own logits and history."""
        tokens = []
        for b in range(self.rows):
            row = self.logits[b:b + 1, self.at[b], :]
            token = self.samplers[b].sample(row, self.ids[b],
                                            generator=self.generators[b])
            self.ids[b].append(token)
            tokens.append(token)
        self.advance(tokens)
        return tokens

    def steps(self, count):
        """`count` steps, each a list of one token per row."""
        for _ in range(count):
            yield self.step()
