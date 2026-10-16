#!/usr/bin/env python3
"""
Train a character-level GPT on Tiny Shakespeare and record its loss curve.

Writes incremental progress to runs/metrics.json so training can be watched
from outside, plus a checkpoint and a generated sample when it finishes.

Example:
    .venv/bin/python train.py --max-iters 1500
"""

import argparse
import json
import math
import os
import time

import torch

from bpe import BPETokenizer, train as train_bpe
from model import GPT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--seed", type=int, default=1337)

    # Tokenisation.
    p.add_argument("--tokenizer", choices=("char", "bpe"), default="char",
                   help="char: one token per character; bpe: byte-level subwords")
    p.add_argument("--bpe-vocab-size", type=int, default=512,
                   help="vocabulary to learn when training a new BPE tokenizer")
    p.add_argument("--bpe-tokenizer", default=None,
                   help="reuse an existing tokenizer.json rather than learning "
                        "one (default: <out-dir>/tokenizer.json when present)")

    # Model shape (deliberately small: this runs on 2 CPU cores).
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    # Optimisation.
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-schedule", choices=("cosine", "const"), default="cosine",
                   help="cosine: linear warmup, then decay to --min-lr")
    p.add_argument("--warmup-iters", type=int, default=100,
                   help="linear warmup steps; a resumed run is already warm")
    p.add_argument("--min-lr", type=float, default=3e-5,
                   help="floor the cosine decay settles on")
    p.add_argument("--max-iters", type=int, default=1500)
    p.add_argument("--max-seconds", type=float, default=600.0,
                   help="hard wall-clock budget; training stops early if exceeded")
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-iters", type=int, default=30)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--sample-tokens", type=int, default=600)

    # Continuing a run across sessions.
    p.add_argument("--resume", default=None,
                   help="checkpoint to continue from; restores the weights, the "
                        "optimiser state and the loss history")
    p.add_argument("--checkpoint-interval", type=int, default=200,
                   help="write the checkpoint every N steps so a run that is "
                        "interrupted can be resumed from where it stopped")
    return p.parse_args()


class CharTokenizer:
    """One token per character: the tokenizer the first experiments used."""

    name = "char"

    def __init__(self, text=None, itos=None):
        if itos is not None:
            self.itos = {i: c for i, c in enumerate(itos)}
        else:
            self.itos = {i: ch for i, ch in enumerate(sorted(set(text)))}
        self.stoi = {c: i for i, c in self.itos.items()}

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def state(self):
        return {"name": "char", "itos": [self.itos[i] for i in sorted(self.itos)]}


def tokenizer_from_state(state):
    if state["name"] == "bpe":
        return BPETokenizer(state["merges"])
    return CharTokenizer(itos=state["itos"])


def build_tokenizer(args, text, prior):
    """Choose the tokenizer, and say which one was chosen and why.

    The checkpoint's tokenizer wins on resume: the weights were trained against
    those exact token ids, so a different vocabulary would make them meaningless.
    Returns (tokenizer, description).
    """
    if prior is not None:
        if prior.get("tokenizer"):
            return tokenizer_from_state(prior["tokenizer"]), "from checkpoint"
        if "stoi" in prior:
            # Checkpoints written before the tokenizer was pluggable.
            itos = [prior["itos"][i] for i in sorted(prior["itos"])]
            return CharTokenizer(itos=itos), "from checkpoint (legacy format)"

    if args.tokenizer == "bpe":
        path = args.bpe_tokenizer or os.path.join(args.out_dir, "tokenizer.json")
        if os.path.exists(path):
            return BPETokenizer.load(path), f"reused {path}"
        merges = train_bpe(text, args.bpe_vocab_size)
        tok = BPETokenizer(merges)
        tok.save(path)
        return tok, f"trained {args.bpe_vocab_size}-token BPE, wrote {path}"

    return CharTokenizer(text=text), "one token per character"


def load_sibling_metrics(checkpoint):
    """Read the metrics.json next to a checkpoint, if one is there.

    Checkpoints written by earlier versions of this script did not carry the
    loss history, so the sibling metrics file is the only surviving record of
    the steps already trained.
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(checkpoint)), "metrics.json"
    )
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------- resumed state
    # Loaded before the data so the checkpoint can dictate the tokenizer.
    prior = None
    if args.resume:
        prior = torch.load(args.resume, map_location="cpu")

    # ---------------------------------------------------------------- data
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer, tokenizer_source = build_tokenizer(args, text, prior)
    vocab_size = tokenizer.vocab_size
    encode, decode = tokenizer.encode, tokenizer.decode

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # Loss per token is not comparable across tokenizers: a subword model sees
    # fewer, larger tokens. Bits per character is.
    val_chars = len(decode(val_data.tolist()))
    chars_per_token = val_chars / max(1, len(val_data))

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([d[i : i + args.block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + args.block_size + 1] for i in ix])
        return x, y

    # --------------------------------------------------------------- model
    # The checkpoint's shape wins: weights only load into the model they were
    # trained in, so shape flags passed alongside --resume would do nothing.
    shape_keys = ("block_size", "n_embd", "n_head", "n_layer", "dropout")
    if prior is not None:
        for key in shape_keys:
            if key in prior.get("config", {}):
                setattr(args, key, prior["config"][key])

    model = GPT(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    prior_steps, total_elapsed = 0, 0.0
    history, step_log = [], []
    if prior is not None:
        model.load_state_dict(prior["model_state"])
        if "optimizer_state" in prior:
            optimizer.load_state_dict(prior["optimizer_state"])
        sibling = load_sibling_metrics(args.resume)
        prior_steps = prior.get("step")
        if prior_steps is None:
            prior_steps = (sibling.get("final") or {}).get("steps_completed")
            if prior_steps is None and sibling.get("history"):
                prior_steps = sibling["history"][-1]["step"]
            prior_steps = prior_steps or 0
        total_elapsed = prior.get("elapsed_total")
        if total_elapsed is None:
            # Same story as prior_steps: older checkpoints did not record it,
            # and the sibling metrics file did.
            total_elapsed = (sibling.get("final") or {}).get("elapsed", 0.0)
        # Carry the history over: this is the same model, trained further, so
        # its loss curve should continue rather than start again at zero.
        history = prior.get("history") or sibling.get("history") or []
        step_log = prior.get("steps") or sibling.get("steps") or []

    print(f"device          : cpu (threads={torch.get_num_threads()})", flush=True)
    print(f"tokenizer       : {tokenizer.name}, vocab {vocab_size} "
          f"({tokenizer_source})", flush=True)
    print(f"corpus          : {len(text):,} chars -> {len(data):,} tokens "
          f"({len(text) / len(data):.2f} chars/token)", flush=True)
    print(f"train / val     : {len(train_data):,} / {len(val_data):,} tokens", flush=True)
    print(f"parameters      : {model.num_params():,}", flush=True)
    print(f"context window  : {args.block_size} tokens", flush=True)
    if prior is not None:
        best_so_far = min(
            (h["val_loss"] for h in history), default=float("inf")
        )
        print(
            f"resumed         : {args.resume} at step {prior_steps}, "
            f"best val so far {best_so_far:.4f}",
            flush=True,
        )
    print(f"lr schedule     : {args.lr_schedule}, "
          f"lr {args.lr:g} -> {args.min_lr:g}, warmup {args.warmup_iters}", flush=True)
    print("-" * 64, flush=True)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                x, y = get_batch(split)
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    metrics = {
        "config": vars(args),
        "n_threads": torch.get_num_threads(),
        "vocab_size": vocab_size,
        "tokenizer": {
            "name": tokenizer.name,
            "vocab_size": vocab_size,
            "source": tokenizer_source,
        },
        "num_params": model.num_params(),
        "corpus_chars": len(text),
        "corpus_tokens": len(data),
        "chars_per_token": round(chars_per_token, 4),
        "resumed_from": args.resume,
        "prior_steps": prior_steps,
        "history": history,   # evaluated points: step, train_loss, val_loss, elapsed
        "steps": step_log,    # per-log-interval mini-batch loss: step, loss
        "sample": "",
        "final": {},
    }

    def save_metrics(stage):
        tmp = os.path.join(args.out_dir, "metrics.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def save_checkpoint(step_value, best_val):
        """Write the checkpoint atomically so an interrupted run cannot
        truncate the last good one."""
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {k: getattr(args, k) for k in shape_keys},
            "tokenizer": tokenizer.state(),
            "step": step_value,
            "best_val": best_val,
            "elapsed_total": round(total_elapsed + (time.time() - start), 2),
            "history": metrics["history"],
            "steps": metrics["steps"],
        }
        path = os.path.join(args.out_dir, "model.pt")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            torch.save(payload, f)
        os.replace(tmp, path)

    save_metrics("init")

    # ---------------------------------------------------- learning-rate plan
    # The schedule spans the whole life of the model, not just this session,
    # so a resumed run picks the decay up where it left off instead of jumping
    # back to the peak rate. On a fresh run prior_steps is 0 and this is the
    # textbook warmup-then-cosine-decay.
    total_steps = prior_steps + args.max_iters

    def lr_at(global_step):
        if args.lr_schedule == "const":
            return args.lr
        if global_step < args.warmup_iters:
            return args.lr * (global_step + 1) / max(1, args.warmup_iters)
        span = max(1, total_steps - args.warmup_iters)
        progress = min(1.0, (global_step - args.warmup_iters) / span)
        return args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    def set_lr(value):
        for group in optimizer.param_groups:
            group["lr"] = value

    # ------------------------------------------------------------ training
    start = time.time()
    if prior is not None:
        best_val = prior.get("best_val")
        if best_val is None:
            # Older checkpoints did not record it; the carried-over history
            # still knows the best validation loss the model has seen.
            best_val = min(
                (h["val_loss"] for h in metrics["history"]), default=float("inf")
            )
    else:
        best_val = float("inf")
    stopped_early = False
    lr, lr_start = args.lr, None

    for step in range(args.max_iters + 1):
        global_step = prior_steps + step
        lr = lr_at(global_step)
        if lr_start is None:
            lr_start = lr
        set_lr(lr)

        # Evaluate periodically so we get a clean train/val curve.
        if step % args.eval_interval == 0:
            losses = estimate_loss()
            elapsed = time.time() - start
            point = {
                "step": global_step,
                "train_loss": round(losses["train"], 5),
                "val_loss": round(losses["val"], 5),
                "elapsed": round(total_elapsed + elapsed, 2),
                "lr": round(lr, 8),
            }
            # A resumed run evaluates the step it resumed from again. Replace
            # that point rather than recording the same step twice, which would
            # put two values at one x position on the loss curve.
            if metrics["history"] and metrics["history"][-1]["step"] == global_step:
                metrics["history"][-1] = point
            else:
                metrics["history"].append(point)
            best_val = min(best_val, losses["val"])
            save_metrics("train")
            if step and step % args.checkpoint_interval == 0:
                save_checkpoint(global_step, best_val)
            print(
                f"step {global_step:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | lr {lr:.2e} | {elapsed:6.1f}s",
                flush=True,
            )

        if step == args.max_iters:
            break
        if time.time() - start > args.max_seconds:
            print(
                f"time budget of {args.max_seconds:.0f}s reached at step {global_step}; "
                "stopping early",
                flush=True,
            )
            stopped_early = True
            break

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_interval == 0:
            metrics["steps"].append(
                {"step": global_step, "loss": round(loss.item(), 5)}
            )

    total = time.time() - start
    global_step = prior_steps + step

    # -------------------------------------------------------------- output
    print("-" * 64, flush=True)
    print(
        f"finished {global_step} steps total ({step} this session) in {total:.1f}s",
        flush=True,
    )

    save_checkpoint(global_step, best_val)

    start_ids = encode("\n")
    x = torch.tensor(start_ids, dtype=torch.long)[None, ...]
    out = model.generate(x, max_new_tokens=args.sample_tokens, temperature=0.8, top_k=200)
    sample = decode(out[0].tolist())

    final = estimate_loss()
    metrics["sample"] = sample
    metrics["final"] = {
        "steps_completed": global_step,
        "session_steps": step,
        "elapsed": round(total, 2),
        "elapsed_total": round(total_elapsed + total, 2),
        "train_loss": round(final["train"], 5),
        "val_loss": round(final["val"], 5),
        "best_val_loss": round(best_val, 5),
        "final_val_perplexity": round(float(torch.exp(torch.tensor(final["val"]))), 3),
        "stopped_early": stopped_early,
        "tokens_per_second": round(
            step * args.batch_size * args.block_size / max(total, 1e-9), 1
        ),
        "tokens_per_second_total": round(
            global_step * args.batch_size * args.block_size
            / max(total_elapsed + total, 1e-9),
            1,
        ),
        "chars_per_token": round(chars_per_token, 4),
        "val_bits_per_char": round(
            (final["val"] / math.log(2)) / chars_per_token, 4
        ),
        "lr_schedule": args.lr_schedule,
        "lr_start": round(lr_start, 8),
        "lr_final": round(lr, 8),
    }
    save_metrics("final")

    # Loss curve data for the plotter.
    with open(os.path.join(args.out_dir, "loss_curve.csv"), "w", encoding="utf-8") as f:
        f.write("step,train_loss,val_loss\n")
        for h in metrics["history"]:
            f.write(f"{h['step']},{h['train_loss']},{h['val_loss']}\n")

    with open(os.path.join(args.out_dir, "sample.txt"), "w", encoding="utf-8") as f:
        f.write(sample)

    print(f"final train {final['train']:.4f} | final val {final['val']:.4f}", flush=True)
    print(f"wrote {args.out_dir}/metrics.json, model.pt, loss_curve.csv, sample.txt", flush=True)
    print("\n--- sample ---", flush=True)
    print(sample[:400], flush=True)


if __name__ == "__main__":
    main()
