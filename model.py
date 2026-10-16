"""
A minimal decoder-only Transformer (GPT) for character-level language modelling.

Everything structural is written from scratch here: causal self-attention, the
multi-head projection, pre-norm residual blocks, and the language-model head.
Only the backward pass (autograd) is taken as given.

The architecture follows "Attention Is All You Need" (Vaswani et al., 2017) with
the pre-norm residual layout popularised by GPT-2 (Radford et al., 2019).
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def top_p_filter(logits, top_p):
    """Keep the smallest set of tokens whose probabilities reach `top_p`.

    This is nucleus sampling: the long tail of implausible tokens is removed
    before sampling, while the cutoff adapts to how confident the model is.
    """
    sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Drop every token past the point where the nucleus is already covered.
    drop = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
    return logits.scatter(-1, order, sorted_logits)


def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    """Turn one position's logits into a single sampled token id.

    Shared by training-time sampling and the interactive prompt in chat.py so
    both use exactly the same sampling rules.
    """
    logits = logits / max(temperature, 1e-8)
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[..., [-1]], float("-inf"))
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention over a fixed-size context window.

    Each token emits a query, key and value. Attention weights are the scaled
    dot product of queries and keys, masked so position t can only see tokens
    at positions <= t (this is what makes the model usable for generation).
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # A single fused projection producing queries, keys and values.
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Lower-triangular mask: entry (i, j) is 1 when j <= i.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with the causal mask applied.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v                                     # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads
        return self.dropout(self.proj(y))


class FeedForward(nn.Module):
    """Position-wise MLP with the standard 4x expansion."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)); x + mlp(ln(x))."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Character-level GPT: token + position embeddings, N blocks, LM head."""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, (
            f"sequence length {T} exceeds block_size {self.block_size}"
        )
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1), targets.view(B * T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Sample tokens one at a time, cropping context to block_size."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            idx_next = sample_next(logits[:, -1, :], temperature, top_k, top_p)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
