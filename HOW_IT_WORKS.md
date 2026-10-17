# How this transformer works

A walkthrough of the mechanics, pointing at the actual code. Read this next to
[`model.py`](model.py) and [`train.py`](train.py).

The whole model has one job: **given some text, predict what comes next**.
Everything else is machinery for doing that well.

Numbers below come from the default configuration (4 layers, 128-wide) so they
match the code you are reading. The runs quoted in the README use a larger one
(6 layers, 192-wide) and a subword tokenizer; nothing about the explanation
changes, only the sizes and the units.

---

## 1. Turning text into numbers

A model only does arithmetic, so text becomes integers. We take every distinct
character in the corpus, sort them, and assign each an index:

```python
chars = sorted(set(text))          # 65 characters: newline, space, !, ', ...
stoi = {ch: i for i, ch in enumerate(chars)}
```

That is the entire tokenizer — a lookup table. This is why it is called
*character-level*: the vocabulary is 65 characters, not tens of thousands of
sub-word tokens like a production LLM.

Each character then becomes a **learned vector** of 128 numbers (a token
embedding). Similar characters end up with similar vectors, which is how the
model handles the fact that vowels behave alike.

---

## 2. The forward pass

`GPT.forward` in [`model.py`](model.py) is the whole pipeline, top to bottom:

1. **Embed** each character index into a 128-dim vector.
2. **Add position** information, because attention on its own is order-blind —
   without it, "abc" and "cba" would look identical. Position also gets its own
   learned embedding, added to the token embedding.
3. **Run four transformer blocks** — the part that does the thinking.
4. **Project to vocabulary size** (a linear layer to 65 numbers) giving a
   **logit** per character: a raw score for "how plausible is this character
   next".

Softmax over those 65 logits turns scores into probabilities.

---

## 3. Attention — the one genuinely new idea

Everything before transformers either looked at a fixed window or processed
tokens one at a time. Attention lets **every position look at every earlier
position at once**, and *choose how much to care about each one*.

In `CausalSelfAttention.forward`, each token produces three vectors:

| Vector | Meaning |
| --- | --- |
| **Query** (`q`) | "what am I looking for?" |
| **Key** (`k`) | "what do I contain?" |
| **Value** (`v`) | "what do I actually pass on if attended to?" |

The mechanic is one line:

```python
att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
```

The dot product of a query with every key measures *relevance*. Dividing by
`sqrt(head_dim)` keeps the numbers from blowing up as dimensionality grows.

Then the causal mask:

```python
att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
```

Masked positions become `-inf`, and `softmax` turns `-inf` into exactly `0`.
So when predicting character 10, the model is *structurally incapable* of
looking at characters 11+. This is what makes the model honest — without it,
training would be cheating by reading the answer.

Finally `att @ v` is a weighted average of the values, weighted by relevance.
That is the output: each position's new representation is a mixture of the
things it decided were relevant.

**Multiple heads** (4 here) run in parallel on different slices of the 128
dimensions, so the model can attend to several kinds of relationships at once —
one head might track "am I inside a word", another "am I inside a quotation".

---

## 4. The rest of the block

`Block` wraps attention with two pieces that make deep stacks trainable:

```python
x = x + self.attn(self.ln1(x))   # attend, then add the input back
x = x + self.mlp(self.ln2(x))    # transform, then add the input back
```

- **Residual connections** (`x + ...`): the block only has to learn a
  *correction* to `x`, not a replacement. Gradients flow straight through the
  `+`, which is what stops them vanishing in a deep network.
- **LayerNorm** (`ln1`, `ln2`): rescales activations to a stable range before
  each sub-layer, so the numbers stay well-behaved.
- **The MLP** (`FeedForward`): expands 128 → 512 → 128 with a nonlinearity.
  Attention *mixes information between positions*; the MLP *processes each
  position on its own*.

---

## 5. Measuring wrongness

The loss is cross-entropy:

```python
loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
```

For the correct next character, it takes `-log(probability the model assigned)`.

- Confident and right → loss near `0`.
- Confident and wrong → loss very large.
- Uninformed (uniform over 65 characters) → `ln(65) = 4.17`, our starting point.

Loss is measured in **nats**. Perplexity is `e^loss` and is easier to read: it
is roughly *how many characters the model is choosing between*. Perplexity 7.76
means the model behaves about as uncertainly as if it were picking from ~8
characters, rather than all 65.

---

## 6. Learning

`loss.backward()` (PyTorch autograd) computes, for every parameter, the
direction that would reduce the loss. `AdamW` then nudges each
parameter a little that way, with two refinements: it tracks a running average
of recent gradients (momentum) and adapts the step size per parameter.

We also clip gradients to a maximum norm of 1.0 — one unusually bad batch can
otherwise produce an enormous gradient and wreck the weights:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Repeat ~1,500 times and the model gets good at the corpus.

---

## 7. The learning rate: how big a step to take

AdamW decides the *direction* to move in. The learning rate decides *how far*.
Too large and training bounces around or diverges; too small and it crawls.

Two ideas do most of the work here:

- **Warmup** — start at a tiny rate and ramp up over the first ~100 steps.
  Adam's moment estimates are averages of almost nothing at the very start, so
  a large first step can throw the weights somewhere unhelpful.
- **Cosine decay** — after the peak, glide the rate down towards a floor along a
  cosine curve. Late in training the model is near a minimum, where large steps
  only kick it back out; smaller steps let it settle in. A good part of the
  final loss improvement comes from this, which is why a run that ends at its
  peak rate still looks like it was cut off mid-descent.

```python
progress = (global_step - warmup) / (total_steps - warmup)
lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
```

A subtlety worth internalising: the schedule is defined over the model's *whole
life*, not over one session. Resuming from a checkpoint continues the decay
where it left off, because restarting at the peak rate would undo the settling
the decay had already bought. That is why `train.py` plans against
`prior_steps + max_iters` rather than just `max_iters`.

---

## 8. Generating text

`GPT.generate` runs the model, takes the final position's probability
distribution, samples one token from it, appends it, and repeats — feeding its own
output back in. The knobs all live in `sample_next`:

- **temperature** divides the logits before softmax. Below 1.0 sharpens the
  distribution (safer, more repetitive); above 1.0 flattens it (more surprising,
  more mistakes). Training samples use 0.8.
- **top-k** discards everything outside the k most likely tokens before sampling,
  which removes the long tail of nonsense. Training samples use 200.
- **top-p** (nucleus) keeps the smallest set of tokens whose probabilities add up
  to p. Unlike a fixed k it adapts to the model's confidence: one or two options
  when it is sure of itself, hundreds when it is not. Off by default.

A small model also tends to loop, so there is a **repetition penalty** that
divides down the logits of tokens it has just used. It is a blunt instrument —
too much of it and the model starts avoiding words it needs — which is why it
defaults to off.

Note the context is cropped to the last `block_size` tokens: 64 characters for a
character-level model, around 124 characters for the BPE one. That is the model's
entire memory, and it is why a long session in `chat.py` eventually pushes the
start of the conversation out of view.

The page does not leave that implicit. It counts how much has fallen out and
says so in words, and the token view dims the chips the model can no longer
attend to — so the window is a visible property of the transcript rather than a
number you have to interpret.

The window is a ceiling rather than a requirement, and that distinction is easy
to miss: `block_size` is what the positional embeddings were trained for, so the
model cannot be *given* more than that — but it can be given less. The page's
Context control does exactly that. Nothing about the attention changes; what
changes is how many tokens the model is allowed to attend to, which is the
difference between a model that has read the passage and one that has read the
last line of it.

The page also says *where* the reading stops, not only how much has fallen out.
`done` reports the window's opening as a character offset, and the transcript
gets a dashed rule at that offset with the text above it dimmed — so the model's
memory has an edge you can point at, and the characters below the line tokenise
to exactly as many tokens as the window holds. The offset is measured from the visible tail
rather than from the dropped prefix on purpose: a byte-level token can cut a
multi-byte character in half, and the two ways of slicing then disagree by
exactly that character. A reply that was stopped part-way leaves no line at all,
because its boundary was never computed, and a line in the wrong place is worse
than none.

That token view also has to survive being on screen. It once did not: as a flex
item that was allowed to shrink, it gave its height up to the transcript pane
above it and collapsed to about twelve pixels, with hundreds of chips still in
the DOM and none of them visible. `flex: 0 0 auto` is the fix — the panel keeps
its height and the pane above gives up its own instead — and because no HTTP test
can see a box, `test_all.py` resolves the served stylesheet and asserts the
contract: who shrinks, who caps, and that the panel and the pane above it really
are items of the same flex column. One of those tests removes the declaration
and shows the check would have caught the old layout. The breakpoint block lives
at the end of the stylesheet for a related reason: a `@media` rule that appears
before the rule it means to override loses to it at equal specificity, which is
what once kept the stacked layout's transcript pane from ever getting the height
it asked for.

[`chat.py`](chat.py) is this same loop with a prompt in front of it: the knobs
above are commands, the running text is kept as context, and each sampled token
is written as it arrives so you can watch a sentence form.
[`serve.py`](serve.py) is the same loop again, with a web page on the front: it
calls `stream_reply` unchanged and gives it a file-like object that writes each
flushed chunk to the browser as a line of JSON. The streaming is the whole point,
so it is worth being precise about where it comes from — the sampling loop
already decides how much text is ready to show and flushes exactly that, which is
why the browser needs no buffering logic of its own. Nothing about the model
changes between the terminal and the page; only the sink does. It is worth being
clear that **this is a completion model, not an assistant** — nothing here was
instruction-tuned, so the prompt is text to be continued, not a question to be
answered.

---

## 9. Reading the loss curve

The curve is generated by `render_curve.py` into `runs/loss_curve.svg` and a
fuller `runs/loss_curve.html`. Those are build artifacts and are not committed,
so open the HTML locally to see the live chart.

- **Blue** is training loss, **orange** is validation loss — the same measurement
  on 10% of text the model never trains on.
- **The gap between them** is the signal. They track each other closely here,
  which means the model is *generalising*, not memorising. If orange started
  rising while blue kept falling, that would be overfitting.
- **The flat grey line at 4.17** is the uniform baseline: guessing uniformly.
- **The flat amber line at 2.48** is the bigram model from [`bigram.py`](bigram.py),
  which only ever sees the previous character.
- **The green line** is the learning rate, on the right-hand axis. When a run was
  resumed, the flat part is the earlier session's constant rate and the decaying
  part is the cosine schedule; the dashed violet vertical line marks the step
  where the checkpoint was picked up. Watch what the loss does right after it.

The curve's shape is the classic one: a very steep drop in the first ~100 steps
while the model learns basic character statistics, then a long slow grind as it
learns words, names, names-in-capitals, line breaks and punctuation.

**Why the transformer wins:** the bigram model has no memory beyond one
character. The transformer can look at 64 characters and decide which ones
matter — so it can finish `'tis` after seeing `quo`, and can tell a speaker name
from dialogue.

---

## 10. Subword tokenisation (BPE)

Section 1 treated the vocabulary of 65 characters as given. It is not: it is a
choice, and it is the one choice that changes what the model's job *is*.

A character-level model spends a lot of capacity relearning how letters behave —
that `q` is followed by `u`, that `t`, `h`, `e` appear in that order. Byte-pair
encoding (BPE) removes that work by promoting frequent strings to single tokens.
[`bpe.py`](bpe.py) implements it from scratch in about a hundred lines.

It works purely by counting:

1. Start with 256 symbols, one per byte of UTF-8.
2. Count every adjacent pair of symbols across the corpus, weighted by how often
   the word containing them appears.
3. Take the most frequent pair, replace every occurrence of it with a new symbol,
   and record the merge.
4. Repeat until the vocabulary is the size you asked for.

After 256 merges on Tiny Shakespeare, the longest tokens are whole words:

| token id | characters |
| --- | --- |
| 414 | ` shall` |
| 502 | ` there` |
| 504 | ` would` |
| 342 | ` thou` |

This is why production tokenizers contain oddities like ` ind` and `entation`:
those are the strings a real corpus actually repeats. Note the leading spaces —
because the pretokenizer keeps a word's leading space attached, ` the` and `the`
are different tokens, which is how the model can see word boundaries at all.

**The payoff is compression.** The corpus is 1,115,394 characters; with a
512-token BPE vocabulary it is 575,345 tokens, or **1.94 characters per token**.
So the same 64-token context window now covers about 124 characters of text
instead of 64 — the model sees roughly twice as far for the same cost in
attention.

**The cost is that the loss stops being directly comparable.** Cross-entropy is
per *token*, and a token is now about two characters, so a lower number is not by
itself a better model. Worse, a uniform guess costs `ln(vocab)` nats: 4.17 for 65
characters but 6.24 for 512 tokens. The two losses are not the same quantity.

The fix is to convert both to information per character:

```
bits per character = (loss in nats / ln 2) / characters per token
```

That is the only axis on which runs with different tokenizers can honestly be
drawn together, which is what `render_curve.py --metric bits-char` does. It is
also the unit the wider literature reports, for exactly this reason.

Vocabulary size is a genuine knob with a trade-off: a larger vocabulary means
fewer tokens per sentence (cheaper attention, more text in context) but a bigger
embedding and output head, and rarer tokens that get less training signal.

**What it bought here.** A matched pair — same model, same schedule, same 1,000
steps, only the tokenizer changed:

| | char-level | BPE-512 |
| --- | --- | --- |
| val loss | 2.1774 nats/char | 3.5878 nats/token |
| **bits / char** | **3.1413** | **2.7269** |
| bigram floor, bits/char | 3.5834 | 2.7768 |
| attention's gain over its own bigram | **0.4421** | **0.0499** |

Two things stand out. The bigram over subwords beats the *transformer* over
characters outright, and the subword transformer sits only 0.05 bits/char above
the subword bigram — once the units encode the spelling, there is far less left
for attention to discover. That is not an argument against attention; it is an
argument that character-level modelling is the harder problem, and therefore the
more revealing place to look for what the architecture is doing.

---

## 11. What "from scratch" means here

Written by hand in this repo:

- the causal self-attention and its mask
- the multi-head split and re-assembly
- the residual/LayerNorm block layout
- the language-model head and the sampling loop
- the whole training loop, batching and evaluation

Taken as given:

- the **backward pass**. We call `loss.backward()` rather than deriving the
  gradients by hand. Autograd is a general-purpose tool for exactly this, and
  hand-deriving 816k partial derivatives would add pages of algebra without
  teaching more about *transformers*.
- the **optimiser** (AdamW), matrix multiplication, and the embedding table.

To go fully manual — deriving and coding the gradient of attention yourself —
the natural next step is reimplementing this in NumPy alone, with no autograd.

---

## Try it yourself

```bash
# the same prompt, in a browser (open http://127.0.0.1:8765/)
.venv/bin/python serve.py

# train (a 10-minute session: --max-seconds stops it there, wherever it has got to)
.venv/bin/python train.py --max-iters 2000

# the comparison floor
.venv/bin/python bigram.py

# rebuild the chart
.venv/bin/python render_curve.py

# talk to the trained model
.venv/bin/python chat.py

# check the invariants above still hold (about 90 seconds)
.venv/bin/python test_all.py
```

To see what tokenisation changes, teach a tokenizer and train against it:

```bash
# learn a 512-token byte-level BPE vocabulary (about 30 seconds)
.venv/bin/python bpe.py --vocab-size 512 --out runs_bpe/tokenizer.json

# the same model, same settings, subword tokens instead of characters
.venv/bin/python train.py --tokenizer bpe \
    --bpe-tokenizer runs_bpe/tokenizer.json --out-dir runs_tok_bpe

# and the comparison floor on those same tokens
.venv/bin/python bigram.py --tokenizer bpe --out runs_bpe/bigram_baseline.json

# plot both tokenizers on one axis
.venv/bin/python render_curve.py --metric bits-char \
    --metrics runs_tok_bpe/metrics.json \
    --compare runs_tok_char/metrics.json --compare-label "char" \
    --baselines runs_bpe/bigram_baseline.json --out-dir runs_tok_bpe
```

Things worth changing to build intuition:

- `--block-size 8` shrinks the context to a few characters. The loss will end up
  much closer to the bigram line — direct evidence that context is doing the work.
- `--n-layer 1` removes the depth. It will still learn something, just less.
- `--max-iters 200` shows how much of the curve is early and cheap.
- Delete the causal mask in `model.py` and watch the loss collapse to near zero.
  That is the model cheating by reading the future, not learning.
