"""
Shared fixtures for the plasticity test suite.

The suite runs with no model download. Every test here drives a toy network
whose module tree mirrors the parts of GPT-2 that `plasticity.py` reaches for:
dotted paths of the form `transformer.h.<i>.mlp.c_proj`, and Conv1D-style
weights of shape (n_in, n_out) with `y = x @ W + b`.

That shape convention is the thing under test as much as the learning rule is.
HuggingFace's Conv1D stores (n_in, n_out); `nn.Linear` stores (n_out, n_in).
`OjaPlasticity` handles the first natively and the second via `transposed=True`,
so both live here.

Nothing in this file imports `transformers`. A local Conv1D reimplementation is
cheap and keeps the default suite offline; tests that genuinely need real GPT-2
weights belong behind the `slow` marker.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Conv1D, as HuggingFace defines it
# --------------------------------------------------------------------------

class Conv1D(nn.Module):
    """
    Mirror of `transformers.pytorch_utils.Conv1D`.

    weight : (n_in, n_out)   -- note the order; this is not nn.Linear
    forward: y = x @ W + b
    """

    def __init__(self, n_out: int, n_in: int):
        super().__init__()
        self.n_out = n_out
        self.weight = nn.Parameter(torch.empty(n_in, n_out))
        self.bias = nn.Parameter(torch.zeros(n_out))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.shape[:-1] + (self.n_out,)
        y = torch.addmm(self.bias, x.reshape(-1, x.shape[-1]), self.weight)
        return y.view(size_out)


# --------------------------------------------------------------------------
# A toy transformer with GPT-2's module names
# --------------------------------------------------------------------------

class ToyMLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.c_fc = Conv1D(4 * d_model, d_model)
        self.c_proj = Conv1D(d_model, 4 * d_model)

    def forward(self, x):
        return self.c_proj(torch.nn.functional.gelu(self.c_fc(x)))


class ToyAttn(nn.Module):
    """
    Single-head attention over the sequence. Not GPT-2's attention -- it exists
    so `attn.c_attn` and `attn.c_proj` are present as candidate sites with the
    right weight shapes, and so activity actually flows through them.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.c_attn = Conv1D(3 * d_model, d_model)
        self.c_proj = Conv1D(d_model, d_model)

    def forward(self, x):
        q, k, v = self.c_attn(x).split(self.d_model, dim=-1)
        w = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_model)
        mask = torch.tril(torch.ones(w.shape[-2:], dtype=torch.bool))
        w = w.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        return self.c_proj(w @ v)


class ToyBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = ToyAttn(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = ToyMLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class ToyTransformer(nn.Module):
    def __init__(self, d_model: int, n_layer: int):
        super().__init__()
        self.h = nn.ModuleList([ToyBlock(d_model) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x):
        for block in self.h:
            x = block(x)
        return self.ln_f(x)


class ToyModel(nn.Module):
    """Exposes `transformer.h.<i>.{mlp,attn}.{c_fc,c_proj,c_attn}`."""

    def __init__(self, d_model: int = 16, n_layer: int = 2):
        super().__init__()
        self.d_model = d_model
        self.transformer = ToyTransformer(d_model, n_layer)

    def forward(self, x):
        return self.transformer(x)


class LinearModel(nn.Module):
    """
    An `nn.Linear` target, for the `transposed=True` path.

    `nn.Linear.weight` is (n_out, n_in) and forward is `y = x @ W.T`, the
    transpose of the Conv1D convention the learning rules are written in.

    Both layers are deliberately NON-SQUARE. A square weight makes the two
    conventions indistinguishable by shape, so a transpose bug survives every
    assertion that only checks shapes and silently applies the update in the
    wrong orientation. Non-square is the only honest test of `transposed=True`.
    """

    def __init__(self, d_model: int = 16, d_hidden: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.proj = nn.Linear(d_model, d_hidden, bias=False)   # weight (32, 16)
        self.out = nn.Linear(d_hidden, d_model, bias=False)    # weight (16, 32)

    def forward(self, x):
        return self.out(torch.tanh(self.proj(x)))


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

SITE = "transformer.h.1.mlp.c_proj"


@pytest.fixture(autouse=True)
def _determinism():
    """Every test starts from the same RNG state; nothing here uses the GPU."""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield


@pytest.fixture
def toy_model() -> ToyModel:
    torch.manual_seed(1234)
    m = ToyModel(d_model=16, n_layer=2)
    m.eval()
    m.requires_grad_(False)
    return m


@pytest.fixture
def linear_model() -> LinearModel:
    torch.manual_seed(1234)
    m = LinearModel(d_model=16)
    m.eval()
    m.requires_grad_(False)
    return m


@pytest.fixture
def site() -> str:
    """The default target: the MLP down-projection, mid-stack."""
    return SITE


@pytest.fixture
def r0() -> torch.Tensor:
    """A residual-stream state, shape (batch=1, tokens=4, d_model=16)."""
    torch.manual_seed(7)
    r = torch.randn(1, 4, 16)
    return r / r.norm()


@pytest.fixture
def atr_step():
    """
    A stand-in for the parent repo's tested engine, with its signature:
    `atr_step(model, r) -> r_next`.

    One forward pass, renormalised. Deterministic and side-effect free, so any
    trajectory difference a test observes comes from the plasticity layer and
    nowhere else.
    """

    def _step(model, r: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = model(r)
        return out / (out.norm() + 1e-12)

    return _step
