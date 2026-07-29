"""
Shared fixtures for the plasticity test suite.

Everything here runs against real GPT-2 small. There is no toy model any more.
The toy's `Conv1D` was our own reimplementation, so it could disagree with
HuggingFace without any test noticing, and the experiment this repo exists for
runs on the real thing. The suite is correspondingly no longer offline and no
longer a few seconds; that trade was made deliberately.

The model fixture is SESSION-scoped -- 124M parameters is not something to
reload per test -- which makes weight hygiene a correctness requirement rather
than a courtesy. A test that leaves `transformer.h.6.mlp.c_proj` modified
silently re-runs every later test against a different model. Two things guard
that: every mutating test uses `try/finally: p.revert()`, and
`_target_weight_unchanged` below is autouse and fails whichever test broke the
rule.

Convention under test as much as the learning rule is: HuggingFace's Conv1D
stores `(n_in, n_out)` and computes `y = x @ W + b`. `nn.Linear` stores
`(n_out, n_in)`. `OjaPlasticity` handles the first natively and the second via
`transposed=True`. GPT-2 is Conv1D throughout, so the `transposed=True` path has
no real-model site; the module that exercises it lives in `test_plasticity.py`
and is labelled there as a code-path fixture rather than a model.

Run with:  .venv/bin/pytest
"""

from __future__ import annotations

import os

import pytest
import torch


# The site README nominates as the first target: MLP down-projection, mid-stack
# (layer 6 of 12).
SITE = "transformer.h.6.mlp.c_proj"
# The same matrix, as TransformerLens names it. Both spellings address
# W_out of block 6's MLP; only the module tree around it differs.
TL_SITE = "blocks.6.mlp"
TL_LAYER = 6

# Every matrix this suite can write to, per backend, as (path, attribute).
# The MLP down-projection is the default site; the attention output projections
# are where the per-head sites live -- block 11 because that is the block the
# parent found carrying the period-2 oscillation, so it is the one head tests
# reach for. `_target_weight_unchanged` snapshots these and nothing else.
_HF_WRITABLE = (
    ("transformer.h.6.mlp.c_proj", "weight"),
    ("transformer.h.11.attn.c_proj", "weight"),
)
_TL_WRITABLE = (
    ("blocks.6.mlp", "W_out"),
    ("blocks.11.attn", "W_O"),
)

N_LAYER = 12
D_MODEL = 768
D_MLP = 4 * D_MODEL      # 3072

# The log schema DESIGN.md's measurement plan writes every iteration.
REPORT_TYPES = {
    "site": str,
    "mode": str,
    "eta": float,
    "n_applied": int,
    "delta_norm": float,
    "delta_frac": float,
    "last_update_norm": float,
    "clipped": bool,
    "nonfinite": bool,
}


def resolve(model, path: str):
    """Resolve a dotted site path independently of `OjaPlasticity._resolve`."""
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def weight_snapshot(model, path: str) -> torch.Tensor:
    return resolve(model, path).weight.detach().clone()


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _determinism():
    """Every test starts from the same RNG state; nothing here uses the GPU."""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield


def _unavailable(reason: str):
    """
    Skip, or fail if the environment says the model must be there.

    Skipping is right on a laptop with no network: a red suite on a machine that
    was never going to have the model tells you nothing about the code. It is
    exactly wrong in CI. Every test here needs GPT-2, so a runner without it
    skips the entire suite and exits 0 -- measured: 84 of 91 skipped, green.
    A check that cannot fail is worse than no check, which is the same argument
    this suite makes about the controls themselves.

    CI sets ATR_REQUIRE_MODEL=1, and then an absent model is a failure.
    """
    if os.environ.get("ATR_REQUIRE_MODEL"):
        pytest.fail(f"ATR_REQUIRE_MODEL is set but {reason}")
    pytest.skip(reason)


@pytest.fixture(scope="session")
def gpt2():
    """GPT-2 small, loaded once for the whole session, frozen and in eval mode."""
    try:
        # Not importorskip: a broken install (present, but its own imports fail)
        # must be treated the same as an absent one.
        import transformers
    except ImportError as exc:
        _unavailable(f"transformers is not importable: {exc}")
    try:
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    except (OSError, ImportError) as exc:      # incl. hub HTTP errors (OSError)
        _unavailable(f"GPT-2 small not loadable offline: {type(exc).__name__}: {exc}")
    model.eval()
    model.requires_grad_(False)
    return model


@pytest.fixture(scope="session")
def hf_conv1d():
    """`transformers.pytorch_utils.Conv1D` -- the class the rules are written for."""
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError as exc:
        _unavailable(f"transformers.pytorch_utils is not importable: {exc}")
    return Conv1D


@pytest.fixture(scope="session")
def tl_gpt2():
    """
    GPT-2 small as a TransformerLens `HookedTransformer` -- the object the ATR
    engine actually runs on, and therefore the one the plasticity layer has to
    attach to for any experiment to happen.

    Same weights as the `gpt2` fixture, a different module tree: the MLP output
    matrix is a bare `nn.Parameter` called `W_out` on `blocks.{L}.mlp`, with no
    enclosing module carrying a 2-D `.weight`.
    """
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        _unavailable(f"transformer_lens is not importable: {exc}")
    try:
        model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    except (OSError, ImportError) as exc:
        _unavailable(f"GPT-2 small not loadable for TransformerLens: {exc}")
    model.eval()
    model.requires_grad_(False)
    return model


@pytest.fixture(scope="session")
def tl_site() -> str:
    """The TransformerLens spelling of the default target."""
    return TL_SITE


@pytest.fixture(scope="session")
def site() -> str:
    """The default target: the MLP down-projection, mid-stack."""
    return SITE


@pytest.fixture(scope="session")
def r0() -> torch.Tensor:
    """
    A residual-stream state at GPT-2 small's real width: (1, 4, 768).

    Four token positions, not more: every forward in this suite costs ~20ms and
    the learning rules average over positions, so a longer sequence buys no
    coverage and multiplies the wall clock.
    """
    g = torch.Generator().manual_seed(7)
    r = torch.randn(1, 4, D_MODEL, generator=g)
    return r / r.norm()


@pytest.fixture(scope="session")
def atr_step():
    """
    TEST DOUBLE for the parent project's engine, not an ATR implementation.

    `plasticity.py` and README are both explicit that the real loop must be
    imported from the ATR repo and never reimplemented here; this is a
    deterministic, side-effect-free one-step map with the required signature
    `atr_step(model, r) -> r_next`, and nothing more. Any trajectory difference
    a test sees therefore comes from the plasticity layer alone.

    `model.transformer`, not `model`: the lm_head matmul is 38M multiply-adds
    that no test reads, and skipping it roughly halves the suite.
    """

    def _step(model, r: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = model.transformer(inputs_embeds=r).last_hidden_state
        return out / (out.norm() + 1e-12)

    return _step


@pytest.fixture(autouse=True)
def _target_weight_unchanged(request):
    """
    Session-scoped model plus mutating tests equals cross-test contamination.

    Every test that touches the model must hand it back exactly as it found it.
    A failure here is not a failure of the test that trips it in isolation --
    it means that test was corrupting every test that ran after it, and every
    result downstream of this file is suspect until it is fixed.

    Guards only tests that actually asked for a model, so the ones needing
    nothing but the `transformers` package still run on a cold cache.

    Watches more than the default site. Per-head sites made 144 more matrices
    writable across both backends, and a guard on one MLP down-projection would
    not see a test that left an attention output projection dirty -- so the
    attention projections the head tests target are watched too, on whichever
    model the test actually requested. Snapshotting every parameter would be
    correct and far too slow at ~124M parameters per model per test; this is
    the set the suite can currently write to.
    """
    watched = []
    for fixture, paths in (("gpt2", _HF_WRITABLE), ("tl_gpt2", _TL_WRITABLE)):
        if fixture not in request.fixturenames:
            continue
        model = request.getfixturevalue(fixture)
        for path, attr in paths:
            watched.append((f"{fixture}:{path}.{attr}",
                            (w := getattr(resolve(model, path), attr)),
                            w.detach().clone()))
        # A guard that silently watches nothing passes everything. If a path
        # above stops resolving -- a renamed attribute, a TransformerLens
        # layout change -- the AttributeError from `resolve` should surface
        # here as a loud failure rather than leave the suite unguarded.
        assert watched, f"{fixture} requested but no weights were snapshotted"

    yield

    for name, w, before in watched:
        assert torch.equal(w, before), f"{name} left modified for subsequent tests"
