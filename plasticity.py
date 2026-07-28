"""
Activation-driven weight plasticity for iterated-map experiments on frozen
transformers.

Design principle: this module knows NOTHING about the ATR loop. It installs
hooks on one weight matrix, watches the activations flowing through it, and
applies a local learning rule on request. You wrap it around your own,
already-tested iteration code:

    from plasticity import OjaPlasticity

    with OjaPlasticity(model, site="transformer.h.6.mlp.c_proj",
                       eta=1e-5, mode="oja") as plast:
        r = initial_tensor
        for i in range(n_iter):
            r = my_atr_step(model, r)      # YOUR existing engine
            if (i + 1) % 4 == 0:
                plast.apply()              # commit accumulated update
            log(i, plast.report())

Do not port the ATR loop into this repo. Import it, or copy the tested engine
verbatim. The whole point of this scaffold is that the plasticity layer is the
only new, untested thing.

Two model backends are supported, addressed by two spellings of the same site:

    HuggingFace   OjaPlasticity(model, "transformer.h.6.mlp.c_proj")
    TransformerLens
                  OjaPlasticity(model, "blocks.6.mlp")

The parent ATR project runs TransformerLens, where the MLP down-projection is a
bare `nn.Parameter` named `W_out` on `blocks.{L}.mlp` -- there is no module with
a 2-D `.weight`, and no module whose forward maps x -> y for that matrix alone.
The learning rules are unchanged by this; only three things differ (where the
live weight lives, how to write it, how to capture x and y) and they are behind
`_SiteAdapter` below. See "Site adapters".

STATUS: written but never executed against real weights. The eta=0 identity
check (see README, Control C0) is the first thing you should run, and it must
pass bit-exactly before any result here means anything.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Learning rules
# --------------------------------------------------------------------------
# Convention throughout: the target weight W has shape (n_in, n_out) and the
# module computes y = x @ W (+ bias). This matches HuggingFace's Conv1D, used
# for every linear map inside GPT-2. For nn.Linear, weight is (n_out, n_in),
# so set `transposed=True` and we handle it.
#
#   x : (N, n_in)   flattened over batch and token positions
#   y : (N, n_out)
#
# Raw Hebbian:  dW = <x y^T>                     -- diverges, included to show why
# Oja subspace: dW = <x y^T> - W <y y^T>         -- intrinsically normalised
# Anti-Hebbian: dW = -<x y^T> - W <y y^T>        -- erodes, and still braked
#
# The Oja decay term is what makes this the right rule here rather than a
# convenience: Oja's rule is Hebbian learning with normalisation built in, and
# it performs power iteration on the input correlation structure. ATR is
# nonlinear power iteration on activations. Same mathematics, one loop up.
#
# ANTI-HEBBIAN IS NOT A NEGATIVE ETA, and the difference is the decay term.
# `eta = -e` scales the whole rule, so it flips BOTH terms: the reinforcement
# term decorrelates (wanted) and the brake `-W <y y^T>` becomes `+W <y y^T>`
# (not wanted). `<y y^T>` is positive semi-definite, so `+W <y y^T>` points
# along W and the weight grows without bound -- the one property Oja exists to
# provide is exactly the one a sign-flipped eta destroys. The anti-Hebbian mode
# flips the reinforcement term only and keeps the brake with its stabilising
# sign, which makes it a linear map with a bounded fixed point at
# W* = -<x y^T> <y y^T>^-1 rather than a divergence.
# `tests/test_antihebbian.py` measures both arms on real GPT-2 weights.


def _hebb_term(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """<x y^T>, shape (n_in, n_out)."""
    n = x.shape[0]
    return (x.transpose(0, 1) @ y) / max(n, 1)


def _oja_decay(w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """W <y y^T>, shape (n_in, n_out)."""
    n = y.shape[0]
    yy = (y.transpose(0, 1) @ y) / max(n, 1)   # (n_out, n_out)
    return w @ yy


# --------------------------------------------------------------------------
# Site adapters
# --------------------------------------------------------------------------
# A "site" is one weight matrix plus the pre- and post-synaptic activity that
# flows through it. Everything else in this module -- the rules, the delta, the
# ceiling, revert, report -- is written against those three facts and nothing
# else. An adapter supplies exactly them:
#
#   .weight        the live 2-D tensor (read)
#   .write(w)      overwrite it in place
#   .install(sink) start feeding sink(x, y) on every forward; return a handle
#   .remove(h)     undo exactly that, and nothing else
#
# and, optionally, `.supports_transposed = False` if the adapter's addressing
# assumes the (n_in, n_out) layout -- the per-head ones do, because a head owns
# rows there and columns in the other layout.
#
# Two backends:
#
#   _WeightModuleSite       a dotted path to an nn.Module with a 2-D .weight,
#                           whose forward maps x -> y. HuggingFace GPT-2's
#                           Conv1D, and nn.Linear via transposed=True. This is
#                           the original behaviour, unchanged.
#   _TransformerLensMLPSite "blocks.{L}.mlp", meaning that MLP's W_out. The
#                           matrix is a bare Parameter and no single module's
#                           forward is the matmul, so x and y are read from the
#                           two TransformerLens hook points that bracket it.
#   _HeadSliceSite          "...attn.c_proj.head.{h}": one attention head's
#                           stripe of the block's output projection, on
#                           HuggingFace.
#   _TransformerLensHeadSite
#                           "blocks.{L}.attn.head.{h}": the same head, on the
#                           model where W_O is already stored per head.
#
# Adding a third backend means writing four methods, not touching a rule.


# Suffix that turns a matrix site into one head of it. Spelled the same way on
# both backends so a site string reads the same in a log whichever model
# produced it.
_HEAD_MARK = ".head."


def _split_head(site: str) -> tuple[str, int]:
    """Split `<path>.head.{h}` into (path, head index)."""
    path, _, index = site.rpartition(_HEAD_MARK)
    if not path or not index.isdigit():
        raise TypeError(
            f"{site!r} is not a per-head site; spell it '<path>.head.{{h}}', "
            f"e.g. 'transformer.h.11.attn.c_proj.head.7'"
        )
    return path, int(index)


def _n_heads(module: nn.Module, site: str) -> int:
    """
    How many heads share this projection, asked of the module that owns them.

    Never inferred from the matrix: (768, 768) is twelve heads of 64 on GPT-2
    small and would be equally consistent with any other factorisation, so a
    guess here would silently slice the wrong rows on the next model.
    """
    for attr in ("num_heads", "n_heads", "n_head"):
        n = getattr(module, attr, None)
        if isinstance(n, int) and n > 0:
            return n
    raise TypeError(
        f"{site}: cannot determine the head count from {type(module).__name__}; "
        "it exposes none of num_heads / n_heads / n_head, so which rows belong "
        "to which head is unknown"
    )


def _resolve_path(model: nn.Module, path: str):
    """Resolve a dotted path, indexing ModuleLists on numeric components."""
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _is_hooked_transformer(model: nn.Module) -> bool:
    """
    Duck-typed, so importing this module never requires transformer_lens.

    `hook_dict` (name -> HookPoint) is what makes a model a TransformerLens
    `HookedRootModule`; `blocks` is what makes it a `HookedTransformer`.
    """
    return (
        hasattr(model, "hook_dict")
        and hasattr(model, "add_hook")
        and hasattr(model, "blocks")
    )


class _WeightModuleSite:
    """Site = an nn.Module with a 2-D `.weight` whose forward maps x -> y."""

    backend = "module"

    def __init__(self, model: nn.Module, site: str):
        self.module = _resolve_path(model, site)
        w = getattr(self.module, "weight", None)
        if not torch.is_tensor(w) or w.dim() != 2:
            raise TypeError(f"{site} has no 2-D .weight; not a supported target")

    @property
    def weight(self) -> torch.Tensor:
        return self.module.weight

    def write(self, w: torch.Tensor) -> None:
        with torch.no_grad():
            self.module.weight.copy_(w)

    def install(self, sink):
        # The sink IS a torch forward hook here, registered directly. Not
        # wrapped: `tests/test_controls.py` injects its C0 defect by
        # subclassing OjaPlasticity and overriding `_hook`, and that override
        # only bites if the method torch calls is the one the subclass
        # replaced. See the note on the sink shape in `install()` below.
        return self.module.register_forward_hook(sink)

    @staticmethod
    def remove(handle) -> None:
        handle.remove()


class _TransformerLensMLPSite:
    """
    Site = `blocks.{L}.mlp` on a `HookedTransformer`, meaning that MLP's W_out.

    W_out is a bare `nn.Parameter` of shape (d_mlp, d_model) -- already the
    (n_in, n_out) convention the rules are written in, same as Conv1D -- so no
    transpose is involved anywhere on this path.

    x and y come from the two hook points that bracket the matmul:

        x  blocks.{L}.mlp.hook_post   (d_mlp)   post-activation, input to W_out
        y  blocks.{L}.hook_mlp_out    (d_model) MLP output, x @ W_out + b_out

    hook_post fires first and its tensor is held until hook_mlp_out fires, at
    which point the pair is handed to the sink. If a forward ever reached
    hook_mlp_out without hook_post -- it cannot, in a single-threaded pass --
    the pair is dropped rather than mismatched.

    HOOK HYGIENE. These are plain `torch` forward hooks on the HookPoint
    modules, registered with `register_forward_hook` and removed by their own
    handles. Deliberately NOT `model.add_hook`, whose only documented teardown
    is `model.reset_hooks()`: that clears every hook on the model, including the
    injection hook the caller's ATR engine reinstalls every iteration. This
    layer must be removable without touching anything it did not install, and
    symmetrically it stays out of TransformerLens's own `fwd_hooks` bookkeeping,
    so the engine's `reset_hooks()` does not silently detach the instrument
    mid-run either. `remove()` here is exactly the inverse of `install()`.
    """

    backend = "transformer_lens"

    def __init__(self, model: nn.Module, site: str):
        layer = self._parse(site)
        try:
            self.module = model.blocks[layer].mlp
        except (IndexError, AttributeError) as exc:
            raise TypeError(f"{site} does not name an MLP on this model: {exc}") from exc

        w = getattr(self.module, "W_out", None)
        if not torch.is_tensor(w) or w.dim() != 2:
            raise TypeError(f"{site} has no 2-D W_out; not a supported target")

        hooks = getattr(model, "hook_dict", {})
        self._x_point = hooks.get(f"blocks.{layer}.mlp.hook_post")
        self._y_point = hooks.get(f"blocks.{layer}.hook_mlp_out")
        if self._x_point is None or self._y_point is None:
            raise TypeError(
                f"{site}: model lacks blocks.{layer}.mlp.hook_post / "
                f"blocks.{layer}.hook_mlp_out, so pre- and post-synaptic "
                "activity cannot be observed"
            )
        self._pending = None

    @staticmethod
    def _parse(site: str) -> int:
        """
        Canonical spelling is `blocks.{L}.mlp`; `blocks.{L}.mlp.W_out` is
        accepted as an alias because it is the spelling someone reading the
        module tree reaches for first, and rejecting it costs a debugging hour
        for no benefit.
        """
        parts = site.split(".")
        if parts and parts[-1] == "W_out":
            parts = parts[:-1]
        if len(parts) == 3 and parts[0] == "blocks" and parts[1].isdigit() \
                and parts[2] == "mlp":
            return int(parts[1])
        raise TypeError(
            f"{site!r} is not a supported TransformerLens site. Spell it "
            "'blocks.{L}.mlp' -- the site names the MLP module and means its "
            "W_out. A whole attention layer is not a site: TransformerLens "
            "stores W_Q/W_K/W_V/W_O per head, 3-D, and the rules here are 2-D. "
            "One head of W_O is a site -- spell it 'blocks.{L}.attn.head.{h}'."
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.module.W_out

    def write(self, w: torch.Tensor) -> None:
        with torch.no_grad():
            self.module.W_out.copy_(w)

    def install(self, sink):
        self._pending = None

        def _on_x(_module, _inputs, output):
            if torch.is_tensor(output):
                self._pending = output

        def _on_y(_module, _inputs, output):
            x, self._pending = self._pending, None
            if torch.is_tensor(x) and torch.is_tensor(output):
                # Called in the torch forward-hook shape, (module, inputs,
                # output), even though no single module computed this matmul.
                # That is deliberate: it keeps ONE capture entry point across
                # both backends, so a subclass overriding `_hook` -- which is
                # how the C0 fail-direction test injects its defect -- bites
                # here too rather than silently doing nothing.
                sink(self.module, (x,), output)

        return (
            self._x_point.register_forward_hook(_on_x),
            self._y_point.register_forward_hook(_on_y),
        )

    def remove(self, handles) -> None:
        for h in handles:
            h.remove()
        self._pending = None


class _HeadSliceSite:
    """
    Site = one attention head's stripe of a block's output projection.

    HuggingFace packs GPT-2's twelve heads into ONE Conv1D of shape
    (n_heads*d_head, d_model) = (768, 768) at `transformer.h.{L}.attn.c_proj`.
    The head merge upstream is a reshape, so head h owns rows
    [h*d_head : (h+1)*d_head] of the input axis and nothing else. A per-head
    site is therefore a row slice of a shared matrix rather than a parameter of
    its own, which is why it needs an adapter rather than a longer site string.

    Spelling: `transformer.h.{L}.attn.c_proj.head.{h}`.

        x   this head's slice of the c_proj input,  (N, d_head)
        y   the FULL c_proj output,                 (N, d_model)

    y is not sliced, and must not be. The post-synaptic activity of an output
    unit is what that unit actually does, and all twelve heads write to all 768
    of them; this head's isolated contribution is an activity the model never
    computes. That choice has an exact consequence worth relying on: because
    both rules are row-wise in the (n_in, n_out) convention -- `<x y^T>` row i
    uses x_i, and `W <y y^T>` row i uses W row i -- the update this site
    produces is bit-for-bit the corresponding row slice of the update the
    whole-matrix site would have produced. Confining plasticity to one head is a
    restriction of the same rule, not a different rule.

    W0, delta and the `max_delta_frac` ceiling are all the stripe's, so
    `delta_frac` is drift relative to ||W_head||_F, not to the whole matrix.
    """

    backend = "module"
    supports_transposed = False

    def __init__(self, model: nn.Module, site: str):
        path, head = _split_head(site)
        self.module = _resolve_path(model, path)
        w = getattr(self.module, "weight", None)
        if not torch.is_tensor(w) or w.dim() != 2:
            raise TypeError(f"{path} has no 2-D .weight; not a supported target")

        owner_path = path.rsplit(".", 1)[0]
        if owner_path == path:
            raise TypeError(f"{site}: cannot find the module that owns {path}")
        n_heads = _n_heads(_resolve_path(model, owner_path), site)
        if not 0 <= head < n_heads:
            raise TypeError(f"{site}: head {head} is out of range for {n_heads} heads")
        if w.shape[0] % n_heads:
            raise TypeError(
                f"{site}: input axis {w.shape[0]} is not divisible by {n_heads} "
                "heads, so the head stripes are not contiguous row blocks"
            )

        self.head = head
        self.d_head = w.shape[0] // n_heads
        self._lo = head * self.d_head
        self._hi = self._lo + self.d_head

    @property
    def weight(self) -> torch.Tensor:
        return self.module.weight[self._lo:self._hi]

    def write(self, w: torch.Tensor) -> None:
        # A slice assignment, so the other heads' rows are not written at all --
        # not rewritten with the same values, which would still round-trip
        # through float32 and is not the same guarantee.
        with torch.no_grad():
            self.module.weight[self._lo:self._hi].copy_(w)

    def install(self, sink):
        lo, hi = self._lo, self._hi

        def _on_forward(module, inputs, output):
            x = inputs[0]
            if torch.is_tensor(x):
                # Same forward-hook shape the sink sees everywhere else, with x
                # narrowed to this head's rows. The C0 fail-direction test
                # overrides `_hook`, and the sink is that method, so it still
                # bites here.
                sink(module, (x[..., lo:hi],), output)

        return self.module.register_forward_hook(_on_forward)

    @staticmethod
    def remove(handle) -> None:
        handle.remove()


class _TransformerLensHeadSite:
    """
    Site = `blocks.{L}.attn.head.{h}` on a `HookedTransformer`: head h's block
    of that attention layer's W_O.

    TransformerLens stores W_O as (n_heads, d_head, d_model) -- already
    per-head, so there is no slicing arithmetic to get wrong here, and W_O[h] is
    (d_head, d_model), which is the rules' (n_in, n_out) convention with no
    transpose. The HuggingFace adapter above has to reconstruct by hand what
    this backend hands over; `tests/test_head_sites.py` checks the two agree.

        x  blocks.{L}.attn.hook_z    (n_heads, d_head), this head's slice taken
        y  blocks.{L}.hook_attn_out  (d_model), the summed attention output

    y is the whole layer's output for the reason given on `_HeadSliceSite`: it
    is what the output units actually do, and `hook_result` -- the per-head
    contribution -- is off by default in TransformerLens precisely because
    materialising it costs n_heads times the memory for a quantity the model
    never forms.

    Hook hygiene as for `_TransformerLensMLPSite`: plain torch forward hooks on
    the HookPoint modules, removed by their own handles, never a model-wide
    `reset_hooks()`.
    """

    backend = "transformer_lens"
    supports_transposed = False

    def __init__(self, model: nn.Module, site: str):
        path, head = _split_head(site)
        layer = self._parse(path, site)
        try:
            self.module = model.blocks[layer].attn
        except (IndexError, AttributeError) as exc:
            raise TypeError(f"{site} does not name an attention layer: {exc}") from exc

        w = getattr(self.module, "W_O", None)
        if not torch.is_tensor(w) or w.dim() != 3:
            raise TypeError(f"{site} has no 3-D W_O; not a supported target")
        if not 0 <= head < w.shape[0]:
            raise TypeError(f"{site}: head {head} is out of range for {w.shape[0]} heads")

        hooks = getattr(model, "hook_dict", {})
        self._x_point = hooks.get(f"blocks.{layer}.attn.hook_z")
        self._y_point = hooks.get(f"blocks.{layer}.hook_attn_out")
        if self._x_point is None or self._y_point is None:
            raise TypeError(
                f"{site}: model lacks blocks.{layer}.attn.hook_z / "
                f"blocks.{layer}.hook_attn_out, so pre- and post-synaptic "
                "activity cannot be observed"
            )
        self.head = head
        self.d_head = w.shape[1]
        self._pending = None

    @staticmethod
    def _parse(path: str, site: str) -> int:
        """`blocks.{L}.attn` is the only accepted prefix; `W_O` is an alias."""
        parts = path.split(".")
        if parts and parts[-1] == "W_O":
            parts = parts[:-1]
        if len(parts) == 3 and parts[0] == "blocks" and parts[1].isdigit() \
                and parts[2] == "attn":
            return int(parts[1])
        raise TypeError(
            f"{site!r} is not a supported TransformerLens per-head site. Spell "
            "it 'blocks.{L}.attn.head.{h}' -- the site names one head's block "
            "of that layer's W_O. Q, K and V are not offered: they are read by "
            "the attention pattern rather than written to the residual stream, "
            "so there is no post-synaptic activity of theirs to Hebb against."
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.module.W_O[self.head]

    def write(self, w: torch.Tensor) -> None:
        with torch.no_grad():
            self.module.W_O[self.head].copy_(w)

    def install(self, sink):
        self._pending = None
        head = self.head

        def _on_x(_module, _inputs, output):
            if torch.is_tensor(output) and output.dim() >= 2:
                # (batch, pos, n_heads, d_head) -> this head's (batch, pos, d_head)
                self._pending = output[..., head, :]

        def _on_y(_module, _inputs, output):
            x, self._pending = self._pending, None
            if torch.is_tensor(x) and torch.is_tensor(output):
                sink(self.module, (x,), output)

        return (
            self._x_point.register_forward_hook(_on_x),
            self._y_point.register_forward_hook(_on_y),
        )

    def remove(self, handles) -> None:
        for h in handles:
            h.remove()
        self._pending = None


def _make_site(model: nn.Module, site: str):
    """Pick the adapter from the model, not from the site string."""
    if _is_hooked_transformer(model):
        if _HEAD_MARK in site:
            return _TransformerLensHeadSite(model, site)
        return _TransformerLensMLPSite(model, site)
    if _HEAD_MARK in site:
        return _HeadSliceSite(model, site)
    return _WeightModuleSite(model, site)


# --------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------

class OjaPlasticity:
    """
    Local activation-driven plasticity on a single weight matrix.

    Parameters
    ----------
    model : nn.Module
        The transformer. Left otherwise untouched.
    site : str
        The target weight matrix, spelled for the model's backend:
          HuggingFace     "transformer.h.6.mlp.c_proj" -- a dotted path to a
                          module with a 2-D `.weight`.
          TransformerLens "blocks.6.mlp" -- names the MLP module and means its
                          `W_out`. `candidate_sites(model)` lists the options
                          in the right spelling for whichever model you pass.
        A `.head.{h}` suffix narrows the target to one attention head's stripe
        of a block's output projection -- "transformer.h.11.attn.c_proj.head.7"
        or "blocks.11.attn.head.7". Everything downstream (W0, delta, the
        ceiling, report) is then the stripe's, not the whole matrix's;
        `candidate_sites(model, heads=True)` lists them.
    eta : float
        Learning rate. Start absurdly small (1e-6) and work up. There is no
        gradient here and no optimiser to save you.
    mode : {"oja", "hebb", "anti_hebb", "random", "off"}
        "oja"       - Oja subspace rule (the real experiment)
        "hebb"      - raw Hebbian, no decay (will diverge; pedagogical)
        "anti_hebb" - Oja with the reinforcement term negated and the decay
                      term left alone: dW = -<x y^T> - W <y y^T>. Erodes what
                      the loop has settled into while staying bounded. NOT the
                      same as eta < 0, which flips the brake too; see the note
                      above the learning rules.
        "random"    - random update matched in Frobenius norm to what Oja would
                      have applied (Control C2: is the *direction* doing work?)
        "off"       - accumulate statistics, apply nothing (Control C0/C1)
    cadence : int
        Informational only; you decide when to call apply().
    max_delta_frac : float
        Ceiling on ||delta||_F / ||W_0||_F. Updates are scaled down to respect
        it and `clipped` is flagged in report(). This is the guard against
        silently destroying the model. Held to float32 precision rather than
        exactly -- expect overshoot of order 1e-8 relative on a large matrix.
    transposed : bool
        Set True for nn.Linear (weight is (n_out, n_in)). False for Conv1D and
        for TransformerLens W_out, both of which are already (n_in, n_out).
    device / dtype
        Inferred from the target weight.
    """

    VALID_MODES = ("oja", "hebb", "anti_hebb", "random", "off")

    def __init__(
        self,
        model: nn.Module,
        site: str,
        eta: float = 1e-6,
        mode: str = "oja",
        cadence: int = 1,
        max_delta_frac: float = 0.05,
        transposed: bool = False,
        seed: Optional[int] = 0,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")

        self.model = model
        self.site = site
        self.eta = float(eta)
        self.mode = mode
        self.cadence = int(cadence)
        self.max_delta_frac = float(max_delta_frac)
        self.transposed = bool(transposed)

        # The adapter is the only thing that knows which backend this is; it
        # raises TypeError here if the site is not a 2-D matrix with observable
        # pre- and post-synaptic activity.
        self._site = _make_site(model, site)
        if self.transposed and not getattr(self._site, "supports_transposed", True):
            # A head owns ROWS of an (n_in, n_out) matrix and COLUMNS of an
            # (n_out, n_in) one. The slicing here is row-wise and would take the
            # wrong 64 numbers rather than fail, so this combination is refused
            # instead of supported approximately.
            raise ValueError(
                f"transposed=True is not supported for the per-head site {site!r}: "
                "the head's stripe is rows in the (n_in, n_out) layout and "
                "columns in the transposed one"
            )
        self.backend = self._site.backend
        # The enclosing module of the target matrix: the Conv1D/Linear itself on
        # the HuggingFace path, the MLP on the TransformerLens one.
        self.module = self._site.module

        # Frozen reference copy. Everything is measured against this.
        self.W0 = self._site.weight.detach().clone()
        # float64 throughout for the norms: a float32 Frobenius norm of a
        # 3072x768 matrix carries ~3e-4 relative error from summation alone,
        # and delta_frac -- the number every run logs -- is a ratio of two of
        # them. The ceiling is enforced on the same quantity report() prints.
        self.W0_norm = self.W0.double().norm().item()

        # The accumulated change, kept explicitly so it can be inspected,
        # measured, and reverted. (A dense stored delta gives the same
        # inspect/revert properties as a low-rank adapter, with less machinery
        # and no rank assumption -- Oja updates are not low-rank.)
        self.delta = torch.zeros_like(self.W0)

        self._acc: Optional[torch.Tensor] = None   # pending update, pre-eta
        self._n_batches = 0
        self._handle = None
        self._rng = torch.Generator(device=self.W0.device)
        if seed is not None:
            self._rng.manual_seed(seed)

        # Diagnostics
        self.n_applied = 0
        self.clipped = False
        self.nonfinite = False
        self._last_update_norm = 0.0

    # ---------------------------------------------------------------- setup

    @staticmethod
    def _resolve(model: nn.Module, path: str) -> nn.Module:
        return _resolve_path(model, path)

    def install(self) -> "OjaPlasticity":
        if self._handle is not None:
            return self
        # `_hook` keeps torch's forward-hook signature on BOTH backends, even
        # though the TransformerLens one has no single module behind the
        # matmul. One capture entry point, one name to override: the C0
        # fail-direction test subclasses this class and replaces `_hook` to
        # prove the gate can fail, and a second name would quietly break that.
        self._handle = self._site.install(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            # Removes precisely what install() added -- see the hook-hygiene
            # note on _TransformerLensMLPSite. Never a model-wide reset.
            self._site.remove(self._handle)
            self._handle = None

    def __enter__(self) -> "OjaPlasticity":
        return self.install()

    def __exit__(self, *exc) -> None:
        self.remove()

    # ----------------------------------------------------------- collection

    def _hook(self, module, inputs, output):
        """Capture pre- and post-synaptic activity and accumulate an update."""
        x = inputs[0]
        y = output
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            return

        x = x.detach().reshape(-1, x.shape[-1]).to(self.W0.dtype)
        y = y.detach().reshape(-1, y.shape[-1]).to(self.W0.dtype)

        if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
            self.nonfinite = True
            return

        # Everything here is in the rule's (n_in, n_out) convention, including
        # for nn.Linear -- `_effective_W` hands the decay term a flipped view.
        # The flip back into the weight's own layout happens in apply(), which
        # is the only place that touches module.weight.
        upd = _hebb_term(x, y)
        if self.mode == "anti_hebb":
            # The reinforcement term is negated; the decay term is NOT. Note
            # this is `-hebb - decay`, not `-(hebb - decay)`: the second is what
            # a negative eta computes, and it turns the brake into an
            # accelerator. Everything downstream -- eta, the ceiling, delta,
            # revert -- is identical to the other modes; only this sign differs.
            w_eff = self._effective_W()
            upd = -upd - _oja_decay(w_eff, y)
        elif self.mode in ("oja", "random"):
            # "random" subtracts the decay too: apply() norm-matches the noise
            # to whatever is accumulated here, and the docstring promises that
            # target is "what Oja would have applied". Matching the raw Hebb
            # term instead biases C2 -- and the bias grows with weight scale,
            # since the Hebb term does not depend on W and the decay does.
            w_eff = self._effective_W()
            upd = upd - _oja_decay(w_eff, y)

        self._acc = upd if self._acc is None else self._acc + upd
        self._n_batches += 1

    def _effective_W(self) -> torch.Tensor:
        w = self.W0 + self.delta
        return w.transpose(0, 1) if self.transposed else w

    # ---------------------------------------------------------------- apply

    def apply(self) -> dict:
        """
        Commit the accumulated update to the live weight. Returns report().

        No-ops safely if nothing has been accumulated or mode == "off".
        """
        if self._acc is None or self._n_batches == 0:
            return self.report()

        upd = self._acc / self._n_batches
        self._acc = None
        self._n_batches = 0

        if self.mode == "off":
            return self.report()

        if self.mode == "random":
            # Match in float64. C2's entire verdict rests on these two arms
            # carrying the same magnitude, and a float32 match leaves ~3e-4
            # relative error on a matrix this size -- small, but avoidable
            # error in the control the README calls decisive.
            target_norm = upd.double().norm().item()
            noise = torch.randn(
                upd.shape, generator=self._rng, device=upd.device, dtype=upd.dtype
            )
            upd = noise * (target_norm / (noise.double().norm().item() + 1e-12))

        if self.transposed:
            # Rule convention is (n_in, n_out); nn.Linear stores (n_out, n_in).
            # Flip once, here, so delta and the ceiling below are both in the
            # weight's own layout. Frobenius norm is transpose-invariant, so
            # the random norm-match above is unaffected by where this sits.
            upd = upd.transpose(0, 1)

        step = self.eta * upd
        if not torch.isfinite(step).all():
            self.nonfinite = True
            return self.report()

        new_delta = self.delta + step

        # Enforce the ceiling on total drift from W0.
        #
        # The norm is taken in float64 because a float32 one overflows to inf
        # on a delta that has blown up -- and the rescale below would then be
        # ceiling/inf, i.e. exactly zero, which wipes the delta and reports
        # delta_frac == 0.0 with nonfinite == False. A blow-up that reads as
        # "nothing happened" is the worst possible failure for a diagnostic,
        # and c3_divergence_demo runs with max_delta_frac=1e9 by design, so
        # this is reachable from inside the repo rather than only in theory.
        ceiling = self.max_delta_frac * self.W0_norm
        nd = new_delta.double().norm().item()
        if not math.isfinite(nd):
            # Past float range entirely: reject the update, keep the last good
            # delta, and say so rather than silently zeroing it.
            self.nonfinite = True
            return self.report()
        if ceiling > 0 and nd > ceiling:
            # Holds to float32 precision, not exactly: rounding the rescaled
            # matrix back into float32 can leave the norm a hair over the
            # ceiling (measured 2.7e-8 relative on GPT-2's 3072x768 matrix),
            # and correcting again is a no-op because the correction factor
            # rounds to 1.0 in float32. Immaterial at any usable eta.
            new_delta = new_delta * (ceiling / (nd + 1e-12))
            self.clipped = True

        self.delta = new_delta
        self._last_update_norm = step.double().norm().item()
        self._write_weight()
        self.n_applied += 1
        return self.report()

    def _write_weight(self) -> None:
        # Through the adapter, not `self.module.weight`: on TransformerLens the
        # target is a bare Parameter named W_out and there is no `.weight` to
        # copy into. Every write to the live matrix goes through here.
        self._site.write(self.W0 + self.delta)

    def revert(self) -> None:
        """
        Restore the original weight exactly and reset to a clean slate.

        The diagnostics reset too, and they have to: `report()` is the
        per-iteration log schema, and an instance reused across a revert would
        otherwise report `clipped: true` for a run that never clipped, or a
        `nonfinite` flag raised by a previous eta. `n_applied` counts applies
        since the last reset, matching every other field in `report()`, which
        is measured against the current run rather than the object's lifetime.

        The controls dodge this by constructing a fresh instance per arm. A
        caller logging one instance across a sweep does not.
        """
        self.delta = torch.zeros_like(self.W0)
        self._acc = None
        self._n_batches = 0
        self.n_applied = 0
        self.clipped = False
        self.nonfinite = False
        self._last_update_norm = 0.0
        self._site.write(self.W0)

    # --------------------------------------------------------------- report

    def report(self) -> dict:
        dn = self.delta.double().norm().item()
        return {
            "site": self.site,
            "mode": self.mode,
            "eta": self.eta,
            "n_applied": self.n_applied,
            "delta_norm": dn,
            "delta_frac": dn / self.W0_norm if self.W0_norm else float("nan"),
            "last_update_norm": self._last_update_norm,
            "clipped": self.clipped,
            "nonfinite": self.nonfinite,
        }

    def __repr__(self) -> str:
        r = self.report()
        return (
            f"OjaPlasticity(site={r['site']!r}, mode={r['mode']}, "
            f"eta={r['eta']:.2e}, applied={r['n_applied']}, "
            f"delta_frac={r['delta_frac']:.3e})"
        )


# --------------------------------------------------------------------------
# Convenience: enumerate plausible target sites in a GPT-2-like model
# --------------------------------------------------------------------------

def candidate_sites(
    model: nn.Module, prefix: Optional[str] = None, heads: bool = False
) -> list[str]:
    """
    List the plasticity targets this model offers, spelled for its backend.

    On a HuggingFace GPT-2, sites are dotted paths to modules with a 2-D
    `.weight`, and `prefix` filters them (default `transformer.h`). Preferred
    first targets, in order:
      1. `mlp.c_proj`  -- pre and post activity both cleanly defined, and the
         MLP down-projection is the least entangled place to perturb.
      2. `attn.c_proj` -- the OV output circuit.
      3. `mlp.c_fc`    -- pre-nonlinearity; post-synaptic activity is less
         cleanly interpretable.
    Avoid `attn.c_attn` initially: it packs Q, K and V into one matrix, so a
    Hebbian update there is three different experiments at once.

    On a TransformerLens `HookedTransformer` -- the model the ATR engine
    actually runs -- the only offered matrix sites are `blocks.{L}.mlp`, meaning
    that MLP's `W_out`. That is deliberately narrower than the HuggingFace list:
    `W_in` has no post-synaptic activity that is cleanly the output of a single
    matmul (the nonlinearity sits in between), and the attention matrices are
    stored per-head and 3-D, which the 2-D rules cannot address whole.
    Returning a site this module cannot actually attach to would be worse than
    returning fewer.

    `heads=True` appends the per-head output-projection sites -- one per
    (layer, head), 144 of them on GPT-2 small -- after the matrix sites. They
    are off by default because they are a different granularity of experiment
    rather than a longer list of the same one: a whole-matrix run and a
    per-head run at the same eta are not comparable, since `max_delta_frac` is a
    fraction of whatever W0 the site names. Reach for them when the object of
    study is a head, which is the case issue #25 raises -- the parent project's
    period-2 oscillation is carried by a single head in the last block.
    """
    wanted = prefix if prefix is not None else "transformer.h"

    if _is_hooked_transformer(model):
        sites = []
        for layer, block in enumerate(model.blocks):
            w = getattr(getattr(block, "mlp", None), "W_out", None)
            if torch.is_tensor(w) and w.dim() == 2:
                sites.append(f"blocks.{layer}.mlp")
        if heads:
            for layer, block in enumerate(model.blocks):
                w = getattr(getattr(block, "attn", None), "W_O", None)
                if torch.is_tensor(w) and w.dim() == 3:
                    sites += [f"blocks.{layer}.attn.head.{h}" for h in range(w.shape[0])]
        return sites

    out = []
    for name, mod in model.named_modules():
        if not name.startswith(wanted):
            continue
        w = getattr(mod, "weight", None)
        if torch.is_tensor(w) and w.dim() == 2:
            out.append(name)

    if heads:
        # Only the output projection: c_attn packs Q, K and V, whose head
        # stripes run along the OUTPUT axis three times over, so the same suffix
        # would mean something different there and mean it silently.
        for name in list(out):
            if not name.endswith("attn.c_proj"):
                continue
            owner = _resolve_path(model, name.rsplit(".", 1)[0])
            try:
                n = _n_heads(owner, name)
            except TypeError:
                continue
            out += [f"{name}{_HEAD_MARK}{h}" for h in range(n)]
    return out
