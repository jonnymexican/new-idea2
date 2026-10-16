#!/usr/bin/env python3
"""
Render the training loss curve from runs/metrics.json.

Uses only the standard library: the chart is hand-built as inline SVG, so there
is no plotting dependency to install. Writes runs/loss_curve.svg and a
self-contained runs/loss_curve.html suitable for opening in a browser.

If runs/bigram_baseline.json exists (see bigram.py), its losses are drawn as
reference floors so the transformer's improvement is visible on the chart.

    .venv/bin/python render_curve.py
"""

import argparse
import html
import json
import math
import os

BASELINE_COLORS = {
    "bigram": "#d29922",
    "uniform": "#6e7681",
}

# Configuration fields that actually change how a model trains, used to say
# precisely what differs between two runs.
TRAINING_KEYS = (
    "n_layer", "n_embd", "n_head", "block_size", "dropout",
    "batch_size", "lr", "lr_schedule", "warmup_iters", "min_lr",
)
LR_GROUP = ("lr_schedule", "warmup_iters", "min_lr")
PRETTY = {
    "n_layer": "depth",
    "n_embd": "embedding width",
    "n_head": "attention heads",
    "block_size": "context window",
    "dropout": "dropout",
    "batch_size": "batch size",
    "lr": "peak learning rate",
    "lr_schedule": "learning-rate schedule",
}


def human_join(names):
    """'a', 'a and b', or 'a, b and c'."""
    if not names:
        return "no configuration change"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def config_diff(a, b):
    """Name the training-relevant fields where two runs differ.

    Warmup, floor and decay-shape are all part of one idea, so they are
    reported as a single change rather than three.
    """
    diff = [k for k in TRAINING_KEYS if k not in LR_GROUP and a.get(k) != b.get(k)]
    if any(a.get(k) != b.get(k) for k in LR_GROUP):
        diff.append("lr_schedule")
    return diff


def nice_ticks(lo, hi, count=5):
    """Return a small set of rounded tick values spanning [lo, hi]."""
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    raw = span / max(count - 1, 1)
    mag = 10 ** (len(str(int(raw))) - 1) if raw >= 1 else 0.1
    while mag > raw:
        mag /= 10
    for mult in (1, 2, 2.5, 5, 10):
        step = mag * mult
        if step >= raw:
            break
    start = step * int(lo / step)
    ticks, v = [], start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            ticks.append(round(v, 6))
        v += step
    return ticks


def load_baselines(path):
    """Load the bigram/uniform reference floors, if they have been computed."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        b = json.load(f)
    # The floors were measured on one tokenizer's tokens; remember which, so
    # they can be put on a bits-per-character footing when asked.
    cpt = b.get("chars_per_token") or 1.0
    suffix = "" if b.get("tokenizer", "char") == "char" else f" ({b['tokenizer']})"
    out = []
    if "uniform_loss" in b:
        out.append({"value": b["uniform_loss"], "label": f"uniform baseline{suffix}",
                    "color": BASELINE_COLORS["uniform"], "cpt": cpt})
    if "val_loss" in b:
        out.append({"value": b["val_loss"], "label": f"bigram baseline{suffix}",
                    "color": BASELINE_COLORS["bigram"], "cpt": cpt})
    return out


def build_svg(metrics, baselines=None, compare=None, compare_label="baseline",
              width=960, height=560, metric="nats"):
    baselines = baselines or []
    hist = metrics["history"]
    cmp_hist = (compare or {}).get("history", [])
    steps = metrics.get("steps", [])
    if not hist:
        raise SystemExit("metrics.json has no history yet -- run train.py first")

    if metric == "bits-char":
        # Per-token loss is measured in different units when the vocabulary
        # changes. Dividing nats by ln(2) and by characters-per-token puts every
        # run on one axis: how many bits it costs to predict a character.
        per_token = 1.0 / math.log(2)
        this_cpt = metrics.get("chars_per_token") or 1.0
        cmp_cpt = (compare or {}).get("chars_per_token") or 1.0
        to_bpc = lambda loss, cpt: loss * per_token / cpt  # noqa: E731
        hist = [{**h, "train_loss": to_bpc(h["train_loss"], this_cpt),
                 "val_loss": to_bpc(h["val_loss"], this_cpt)} for h in hist]
        cmp_hist = [{**h, "val_loss": to_bpc(h["val_loss"], cmp_cpt)}
                    for h in cmp_hist]
        steps = [{**s, "loss": to_bpc(s["loss"], this_cpt)} for s in steps]
        baselines = [{**b, "value": to_bpc(b["value"], b.get("cpt", 1.0))}
                     for b in baselines]
        y_label = "cross-entropy (bits per character)"
    else:
        y_label = "cross-entropy loss (nats)"

    # Learning-rate overlay. Histories written before lr was logged per
    # evaluation simply have no lr points, and the overlay is skipped.
    lr_points = [(h["step"], h["lr"]) for h in hist if "lr" in h]
    peak_lr = metrics["config"].get("lr") or 0.0
    prior_steps = metrics.get("prior_steps") or 0
    lr_max = max([v for _, v in lr_points] + [peak_lr, 0.0])

    left, right = 78, (78 if lr_points else 34)
    top, bottom = 58, 66
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_max = max(h["step"] for h in hist) or 1
    y_vals = [h["train_loss"] for h in hist] + [h["val_loss"] for h in hist]
    y_vals += [s["loss"] for s in steps if s["loss"] < 50]
    y_vals += [b["value"] for b in baselines]
    y_vals += [h["val_loss"] for h in cmp_hist]
    y_lo, y_hi = min(y_vals), max(y_vals)
    pad = (y_hi - y_lo) * 0.08 or 0.1
    y_lo, y_hi = y_lo - pad, y_hi + pad

    def sx(s):
        return left + (s / x_max) * plot_w

    # Name the tokenizer in the chart title: the same script draws both.
    kind = (metrics.get("tokenizer") or {}).get("name", "char")
    svg_title = ("Training a character-level transformer from scratch"
                 if kind == "char" else
                 f"Training a {kind.upper()}-tokenized transformer from scratch")

    def sy(v):
        return top + plot_h - ((v - y_lo) / (y_hi - y_lo)) * plot_h

    def sy_lr(v):
        """Right-hand axis: the learning rate, scaled to the plot height."""
        return top + plot_h - (v / lr_max) * plot_h if lr_max > 0 else top + plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" aria-label="Training and validation loss curve">',
        "<style>"
        ".bg{fill:#0d1117}.grid{stroke:#21262d;stroke-width:1}"
        ".axis{stroke:#30363d;stroke-width:1.5}"
        ".tick{fill:#7d8590;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}"
        ".lab{fill:#adbac7;font:12px ui-sans-serif,system-ui,sans-serif}"
        ".title{fill:#e6edf3;font:600 17px ui-sans-serif,system-ui,sans-serif}"
        ".sub{fill:#7d8590;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}"
        ".leg{fill:#adbac7;font:12px ui-sans-serif,system-ui,sans-serif}"
        ".bl{font:11.5px ui-monospace,SFMono-Regular,Menlo,monospace}"
        "</style>",
        f'<rect class="bg" width="{width}" height="{height}" rx="10"/>',
        f'<text class="title" x="{left}" y="30">'
        f'{html.escape(svg_title)}</text>',
    ]

    cfg = metrics["config"]
    fin = metrics.get("final", {})
    budget = " (hit time budget)" if fin.get("stopped_early") else ""
    resumed = f' | resumed at step {prior_steps}' if prior_steps else ""
    schedule = cfg.get("lr_schedule", "const")
    tok = metrics.get("tokenizer") or {}
    cpt = metrics.get("chars_per_token", 1.0)
    tok_note = (f' | {tok["name"]} vocab {tok["vocab_size"]} ({cpt:.2f} chars/token)'
                if tok.get("name", "char") != "char" else "")
    if metric == "bits-char":
        tok_note += " | plotted in bits/char"
    sub = (
        f'{metrics["num_params"]:,} params | block {cfg["block_size"]} | '
        f'{cfg["n_layer"]} layers | {cfg["n_head"]} heads | n_embd {cfg["n_embd"]} | '
        f'CPU ({metrics.get("n_threads", "?")} threads){tok_note} | '
        f'{fin.get("steps_completed", "?")} steps in '
        f'{fin.get("elapsed_total", fin.get("elapsed", "?"))}s{budget}'
        f' | {schedule} lr{resumed}'
    )
    parts.append(f'<text class="sub" x="{left}" y="48">{html.escape(sub)}</text>')

    for t in nice_ticks(y_lo, y_hi, 6):
        if not (y_lo <= t <= y_hi):
            continue
        y = sy(t)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{t:.2f}</text>')

    for i in range(6):
        s = x_max * i / 5
        x = sx(s)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle">{int(s)}</text>')

    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    parts.append(f'<text class="lab" x="{left + plot_w / 2:.0f}" y="{height - 16}" text-anchor="middle">training step</text>')
    parts.append(
        f'<text class="lab" x="20" y="{top + plot_h / 2:.0f}" text-anchor="middle" '
        f'transform="rotate(-90 20 {top + plot_h / 2:.0f})">{y_label}</text>'
    )

    # Right-hand axis for the learning rate, when the run recorded one.
    if lr_points:
        for t in nice_ticks(0, lr_max, 4):
            y = sy_lr(t)
            if not (top <= y <= top + plot_h):
                continue
            parts.append(
                f'<text class="tick" x="{left + plot_w + 8}" y="{y + 4:.1f}" '
                f'fill="#3fb950">{t:.1e}</text>'
            )
        parts.append(
            f'<text class="lab" x="{width - 14}" y="{top + plot_h / 2:.0f}" '
            f'text-anchor="middle" fill="#3fb950" '
            f'transform="rotate(90 {width - 14} {top + plot_h / 2:.0f})">'
            f'learning rate</text>'
        )

    # Where a resumed run picked up: the schedule changed here.
    if prior_steps and prior_steps <= x_max:
        x = sx(prior_steps)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" '
            f'stroke="#8957e5" stroke-width="1.4" stroke-dasharray="5 4" '
            f'opacity="0.8"/>'
        )
        parts.append(
            f'<text class="bl" x="{x - 6:.1f}" y="{top + 14}" text-anchor="end" '
            f'fill="#8957e5">resumed here</text>'
        )

    # Reference floors: what models without real context can reach.
    for b in baselines:
        y = sy(b["value"])
        if not (top <= y <= top + plot_h):
            continue
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="{b["color"]}" stroke-width="1.4" stroke-dasharray="2 3" opacity="0.9"/>'
        )
        parts.append(
            f'<text class="bl" x="{left + plot_w - 4}" y="{y - 6:.1f}" text-anchor="end" '
            f'fill="{b["color"]}">{html.escape(b["label"])} {b["value"]:.2f}</text>'
        )

    def polyline(points, color, width_, dash=None, opacity=1.0):
        if len(points) < 2:
            return ""
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<polyline fill="none" stroke="{color}" stroke-width="{width_}" '
            f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"'
            f'{dash_attr} points="{d}"/>'
        )

    # Learning rate: flat before the seam, decaying after it.
    if lr_points:
        if prior_steps:
            parts.append(polyline(
                [(sx(0), sy_lr(peak_lr)), (sx(min(prior_steps, x_max)), sy_lr(peak_lr))],
                "#3fb950", 1.8, opacity=0.55,
            ))
        parts.append(polyline(
            [(sx(s), sy_lr(v)) for s, v in lr_points], "#3fb950", 1.8, dash="3 3",
            opacity=0.9,
        ))

    if cmp_hist:
        parts.append(polyline([(sx(h["step"]), sy(h["val_loss"])) for h in cmp_hist],
                              "#8957e5", 1.8, dash="4 4", opacity=0.9))
    parts.append(polyline([(sx(s["step"]), sy(s["loss"])) for s in steps if s["loss"] < 50],
                          "#3b6ea5", 1.1, opacity=0.5))
    parts.append(polyline([(sx(h["step"]), sy(h["train_loss"])) for h in hist], "#58a6ff", 2.2))
    parts.append(polyline([(sx(h["step"]), sy(h["val_loss"])) for h in hist], "#f0883e", 2.2, dash="6 4"))
    parts.append("".join(
        f'<circle cx="{sx(h["step"]):.1f}" cy="{sy(h["train_loss"]):.1f}" r="2.6" fill="#58a6ff"/>'
        for h in hist
    ))
    parts.append("".join(
        f'<circle cx="{sx(h["step"]):.1f}" cy="{sy(h["val_loss"]):.1f}" r="2.6" fill="#f0883e"/>'
        for h in hist
    ))

    lx, ly = left + 22, top + 20
    legend = [
        ("#58a6ff", "train loss (evaluated)", None),
        ("#f0883e", "val loss (evaluated)", "6 4"),
        ("#3b6ea5", "train loss (per mini-batch)", None),
    ]
    if cmp_hist:
        legend.append(("#8957e5", f"val loss, {compare_label}", "4 4"))
    if lr_points:
        legend.append(("#3fb950", "learning rate (right axis)", "3 3"))
    for i, (color, label, dash) in enumerate(legend):
        y = ly + i * 20
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 26}" y2="{y}" stroke="{color}" stroke-width="2.5"{dash_attr}/>')
        parts.append(f'<text class="leg" x="{lx + 34}" y="{y + 4}">{html.escape(label)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def summarize_run(metrics):
    """The headline numbers for a run, falling back to the latest evaluation.

    metrics.json is rewritten while a run is still training, so the renderer can
    be pointed at a run in progress, which has no final block yet.
    """
    fin = metrics.get("final") or {}
    hist = metrics.get("history") or []
    last = hist[-1] if hist else {}
    val = fin.get("val_loss", last.get("val_loss"))
    ppl = fin.get("final_val_perplexity")
    if ppl is None and val is not None:
        ppl = math.exp(val)
    # Bits per character is the loss that survives a change of tokenizer.
    cpt = cpt_of(metrics)
    bpc = fin.get("val_bits_per_char")
    if bpc is None and val is not None:
        bpc = (val / math.log(2)) / cpt
    return {
        "params": metrics.get("num_params", 0),
        "vocab": metrics.get("vocab_size", "?"),
        "cpt": cpt,
        "bpc": bpc,
        "steps": fin.get("steps_completed", last.get("step", "?")),
        "wall": fin.get("elapsed_total",
                       fin.get("elapsed", last.get("elapsed", "?"))),
        "train": fin.get("train_loss", last.get("train_loss")),
        "val": val,
        "ppl": ppl,
        "done": "val_loss" in fin,
    }


def fmt(value, spec):
    """Format a number, or a dash when a run has not produced it yet."""
    return format(value, spec) if isinstance(value, (int, float)) else "—"


def cpt_of(metrics):
    """Characters per token for a run; 1.0 for a character-level one."""
    cpt = metrics.get("chars_per_token")
    if cpt is None:
        chars, tokens = metrics.get("corpus_chars"), metrics.get("corpus_tokens")
        cpt = (chars / tokens) if chars and tokens else 1.0
    return cpt


def build_html(metrics, svg, baselines, compare=None, compare_label="baseline",
               runs=None, metric="nats"):
    fin = metrics.get("final", {})
    cfg = metrics["config"]
    summary = summarize_run(metrics)
    # Older runs predate the corpus_tokens key; derive it from the ratio.
    corpus_tokens = metrics.get("corpus_tokens") or round(
        metrics["corpus_chars"] / cpt_of(metrics)
    )
    tok_name = metrics.get("tokenizer", {}).get("name", "char")
    title = ("Character-level transformer, trained from scratch" if tok_name == "char"
             else f"{tok_name.upper()}-tokenized transformer, trained from scratch")
    vocab = metrics["vocab_size"]
    uniform = math.log(vocab)

    def stat(label, value, color=None):
        style = f' style="color:{color}"' if color else ""
        return (
            f'<div class="stat"><div class="k">{html.escape(label)}</div>'
            f'<div class="v"{style}>{html.escape(str(value))}</div></div>'
        )

    cells = [
        stat("parameters", f'{metrics["num_params"]:,}'),
        stat("steps trained", summary["steps"]),
        stat("wall clock", f'{summary["wall"]} s'),
        stat("train loss", fmt(summary["train"], ".4f")),
        stat("val loss", fmt(summary["val"], ".4f"), "#f0883e"),
        stat("best val loss", fin.get("best_val_loss", "?")),
        stat("val perplexity", fmt(summary["ppl"], ".2f")),
        stat("tokens / second", fin.get("tokens_per_second_total",
                                         fin.get("tokens_per_second", "?"))),
    ]
    if fin.get("lr_final") is not None:
        cells.append(stat("final learning rate", f'{fin["lr_final"]:.1e}', "#3fb950"))
    if isinstance(summary["bpc"], float):
        cells.append(stat("bits / char", f'{summary["bpc"]:.4f}'))
    if metrics.get("prior_steps"):
        cells.append(stat("resumed at step", metrics["prior_steps"]))
    stats = "".join(cells)

    # Comparison table against the reference floors. In bits-per-character mode
    # every number is converted from its own tokenizer, so unlike the raw loss
    # the rows stay comparable.
    big = next((b for b in baselines if "bigram" in b["label"]), None)
    comparison = ""
    if big:
        cpt = cpt_of(metrics)
        deci = lambda v, c=1.0: (v * (1.0 / math.log(2)) / c)  # noqa: E731
        if metric == "bits-char":
            u_val, b_val = deci(uniform, cpt), deci(big["value"], big.get("cpt", 1.0))
            t_val = (summary["bpc"] if isinstance(summary["bpc"], float)
                     else deci(summary["val"], cpt))
        else:
            u_val, b_val = uniform, big["value"]
            t_val = (summary["val"] if isinstance(summary["val"], float)
                     else b_val)
        delta = b_val - t_val
        rows = [
            (f"uniform (guess any of {vocab} tokens)", u_val, "#6e7681"),
            ("bigram (previous token only)", b_val, "#d29922"),
            (f'transformer ({cfg["block_size"]}-token context)', t_val, "#f0883e"),
        ]
        if metric == "bits-char":
            head = f'<tr><th>model</th><th>bits / char</th></tr>'
            body = "".join(
                f'<tr><td><span class="dot" style="background:{c}"></span>'
                f'{html.escape(n)}</td><td class="num">{v:.4f}</td></tr>'
                for n, v, c in rows
            )
            why = ("Perplexity is left out on purpose: it counts possibilities per"
                   " token, and these rows do not share a tokenizer.")
            unit = "bits per character"
        else:
            head = ('<tr><th>model</th><th>val loss (nats)</th><th>perplexity</th></tr>')
            body = "".join(
                f'<tr><td><span class="dot" style="background:{c}"></span>'
                f'{html.escape(n)}</td><td class="num">{v:.4f}</td>'
                f'<td class="num">{math.exp(v):.2f}</td></tr>'
                for n, v, c in rows
            )
            why = ("Perplexity is <code>e^loss</code> &mdash; roughly \"how many"
                   " options the model is choosing between\".")
            unit = "nats"
        comparison = f"""
  <div class="card">
    <h2>What the context buys</h2>
    <table>
      <thead>{head}</thead>
      <tbody>{body}</tbody>
    </table>
    <p class="note">
      Validation cross-entropy, lower is better. {why}
      The transformer beats the bigram floor by
      <strong>{delta:.3f} {unit}</strong> because it can condition on a whole
      context window instead of just the previous token.
    </p>
  </div>"""

    # One row per run, so the effect of each change is readable side by side.
    scaling = ""
    if runs:
        shades = ["#8957e5", "#3b6ea5", "#f0883e", "#58a6ff"]
        entries = []
        for i, (label, m) in enumerate(runs):
            s = summarize_run(m)
            entries.append({
                **s,
                "label": label,
                "color": shades[i % len(shades)],
                "cfg": m.get("config", {}),
                "prior_steps": m.get("prior_steps") or 0,
            })
        body = "".join(
            f'<tr><td><span class="dot" style="background:{e["color"]}"></span>'
            f'{html.escape(e["label"])}</td>'
            f'<td class="num">{e["params"]:,}</td>'
            f'<td class="num">{e["vocab"]}</td>'
            f'<td class="num">{e["steps"]}</td>'
            f'<td class="num">{e["wall"]} s</td>'
            f'<td class="num">{fmt(e["val"], ".4f")}</td>'
            f'<td class="num">{fmt(e["ppl"], ".2f")}</td>'
            f'<td class="num">{fmt(e["bpc"], ".4f")}</td></tr>'
            for e in entries
        )
        note = "Same corpus and the same 64-token context window throughout."
        scored = [e for e in entries if isinstance(e["val"], float)]
        mixed_vocab = len({e["vocab"] for e in entries}) > 1
        if mixed_vocab:
            note += (
                " These rows do not share a tokenizer, so a token is a different"
                " amount of text in each and per-token loss is not comparable"
                " between them."
            )
        if len(scored) >= 2:
            a, b = scored[-2], scored[-1]
            diff = config_diff(a["cfg"], b["cfg"])
            names = (["the tokenizer"] if a["vocab"] != b["vocab"] else [])
            names += [PRETTY.get(k, k) for k in diff]
            what = human_join(names)
            # Only compare numbers that mean the same thing in both rows.
            if metric == "bits-char":
                unit = "bits per character"
                delta = a["bpc"] - b["bpc"]
                shown = f"{delta:+.4f}"
            elif a["vocab"] == b["vocab"]:
                unit = "nats"
                delta = a["val"] - b["val"]
                shown = f"{delta:.3f}"
            else:
                unit = shown = None
                note += (
                    " Their per-token losses are in different units, so no"
                    " difference is quoted here; re-render with"
                    " <code>--metric bits-char</code> to compare them."
                )
            note += f" The last two scored runs differ only in {what}"
            if shown is not None:
                note += f", worth <strong>{shown} {unit}</strong>"
            note += "."
            if b["prior_steps"]:
                note += (
                    f" The final run continues the previous one from step"
                    f" {b['prior_steps']} rather than starting over."
                )
        if mixed_vocab:
            note += (" Bits per character is the measure that can be compared"
                     " across rows.")
        note += " Runs that hit their wall-clock budget are still descending."
        scaling = f"""
  <div class="card">
    <h2>Runs</h2>
    <table>
      <thead><tr><th>run</th><th>params</th><th>vocab</th><th>steps</th>
      <th>wall clock</th><th>val loss (nats / token)</th><th>perplexity</th>
      <th>bits / char</th></tr></thead>
      <tbody>{body}</tbody>
    </table>
    <p class="note">{note}</p>
  </div>"""

    sample = html.escape(metrics.get("sample", "").strip()
                         or "(no sample generated yet)")
    early = fin.get("stopped_early")
    early_note = (
        " This run stopped at its wall-clock budget rather than a fixed step "
        "count, so the curve is still descending when it ends."
        if early else ""
    )
    if metric == "bits-char":
        unit_note = (
            f"Plotted in bits per character. A uniform guess over the"
            f" {vocab}-token vocabulary costs"
            f" {uniform / math.log(2) / cpt_of(metrics):.4f} bits/char."
        )
    else:
        unit_note = (
            f"Cross-entropy is measured in nats per token; a uniform guess over"
            f" the {vocab}-token vocabulary scores {uniform:.2f}."
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} - loss curve</title>
<style>
  :root {{
    --bg: #010409; --panel: #0d1117; --line: #21262d;
    --fg: #e6edf3; --dim: #7d8590; --accent: #58a6ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 28px; background: var(--bg); color: var(--fg);
    font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 1040px; margin: 0 auto; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; letter-spacing: -0.01em; }}
  .lede {{ color: var(--dim); margin: 0 0 22px; font-size: 13.5px; }}
  .card {{
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 18px; margin-bottom: 18px;
  }}
  .stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1px; background: var(--line); border: 1px solid var(--line);
    border-radius: 12px; overflow: hidden; margin-bottom: 18px;
  }}
  .stat {{ background: var(--panel); padding: 12px 14px; }}
  .stat .k {{ color: var(--dim); font-size: 11.5px; text-transform: uppercase;
             letter-spacing: .05em; }}
  .stat .v {{ font: 600 17px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
             color: var(--accent); }}
  h2 {{ font-size: 13px; text-transform: uppercase; letter-spacing: .06em;
       color: var(--dim); margin: 0 0 12px; font-weight: 600; }}
  pre {{
    margin: 0; white-space: pre-wrap; word-wrap: break-word;
    font: 13.5px/1.7 ui-monospace, SFMono-Regular, Menlo, monospace;
    color: #c9d1d9;
  }}
  code {{ font: 12.5px ui-monospace, SFMono-Regular, Menlo, monospace;
         color: #adbac7; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--line);
           font-size: 13.5px; }}
  th {{ color: var(--dim); font-size: 11.5px; text-transform: uppercase;
       letter-spacing: .05em; font-weight: 600; }}
  td.num {{ text-align: right;
           font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         margin-right: 9px; vertical-align: middle; }}
  .note {{ color: var(--dim); font-size: 12.5px; margin: 12px 0 0; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{html.escape(title)}</h1>
  <p class="lede">
    Tiny Shakespeare &middot; {metrics["corpus_chars"]:,} characters &middot;
    {html.escape(metrics.get("tokenizer", {}).get("name", "char"))} vocab {vocab}
    &middot; {cpt_of(metrics):.2f} characters per token &middot;
    {corpus_tokens:,} tokens &middot; context {cfg["block_size"]} tokens
    &middot; trained on CPU only, no pretrained weights
  </p>

  <div class="stats">{stats}</div>

  <div class="card">
    {svg}
  </div>
{comparison}
{scaling}
  <div class="card">
    <h2>Sample generation (temperature 0.8, top-k 200)</h2>
    <pre>{sample}</pre>
    <p class="note">{unit_note}{early_note}</p>
  </div>
</div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default="runs/metrics.json")
    ap.add_argument("--baselines", default="runs/bigram_baseline.json")
    ap.add_argument("--compare", default="runs_baseline/metrics.json")
    ap.add_argument("--compare-label", default="")
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=PATH",
                    help='a row in the runs table, e.g. "816K=runs_baseline/metrics.json"; '
                         "repeatable, listed in the order given")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--metric", choices=("nats", "bits-char"), default="nats",
                    help="nats: loss per token, the raw training objective; "
                         "bits-char: the same loss in bits per character, the "
                         "only axis on which runs with different tokenizers "
                         "can be compared")
    args = ap.parse_args()

    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    baselines = load_baselines(args.baselines)

    compare, compare_label = None, args.compare_label
    if os.path.exists(args.compare):
        with open(args.compare, "r", encoding="utf-8") as f:
            compare = json.load(f)
        compare_label = args.compare_label or (
            f'{compare["num_params"]:,} param'
        )

    # Rows for the runs table: whatever --run says, or the compared run.
    runs = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        if not path:
            label, path = path, label
        if not os.path.exists(path):
            raise SystemExit(f"--run {spec}: no such metrics file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            runs.append((label.strip() or path, json.load(f)))
    if not runs:
        runs = [("this run", metrics)]
        if compare:
            runs.insert(0, (compare_label or "compare", compare))

    svg = build_svg(metrics, baselines=baselines, compare=compare,
                    compare_label=compare_label, metric=args.metric)

    os.makedirs(args.out_dir, exist_ok=True)
    svg_path = os.path.join(args.out_dir, "loss_curve.svg")
    html_path = os.path.join(args.out_dir, "loss_curve.html")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html(metrics, svg, baselines, compare, compare_label,
                           runs=runs, metric=args.metric))

    print(f"wrote {svg_path}")
    print(f"wrote {html_path}")
    print(f"baselines: {[b['label'] for b in baselines] or 'none'}")
    print(f"metric: {args.metric}")
    print(f"points: {len(metrics['history'])} eval, {len(metrics.get('steps', []))} batch")


if __name__ == "__main__":
    main()
