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

STATUS: executed against real GPT-2 small, and continuously. Every committed
experiment in this repository runs on this module, and the eta=0 identity check
(see README, Control C0) passes bit-exactly on the real model -- max abs
deviation 0.0, not "small". It is also not a one-time acceptance test that was
passed once and is now remembered: it is re-run as a gate, by the suite
(`tests/test_controls.py::test_c0_identity_passes_on_healthy_setup`, on real
weights) and by the runners themselves before they are allowed to report
anything -- `experiments/exp002_distributed.py`'s `gate:eta0` refuses to continue
if the trajectory or any weight matrix moved at step size zero, and the step-size
map spends an `eta=0, mode="off"` cell on the same question.

C0 remains THE gate, and re-running it is the point: a result taken without it is
not a result, whatever else the run measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Union

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
# Raw Hebbian:  dW = <x y^T>                     -- unbraked; see the note below
# Oja subspace: dW = <x y^T> - W <y y^T>         -- intrinsically normalised
# Anti-Hebbian: dW = -<x y^T> - W <y y^T>        -- erodes, and still braked
#
# The Oja decay term is what makes this the right rule here rather than a
# convenience: Oja's rule is Hebbian learning with normalisation built in, and
# it performs power iteration on the input correlation structure. ATR is
# nonlinear power iteration on activations. Same mathematics, one loop up.
#
# TWO CORRECTIONS TO THE PARAGRAPH ABOVE, both from this repo's own artifacts
# and both carried in CLAIMS.md. The theoretical argument stands; the empirical
# claims that used to sit beside it do not.
#
#   1. "Raw Hebb diverges" is retired (C-15). Across all ten `hebb` cells of
#      the step-size map the non-finite count is zero, and at the working point
#      (eta 7.07e-05, 120 updates) the ceiling never fires and ||W||_F moves
#      +0.03%. Hebb's drift keeps growing only at large eta with the ceiling
#      lifted, which is the regime c3_divergence_demo runs in -- and note the
#      limit of that evidence: C3 runs a fixed, small number of applications,
#      so it records continued growth over the run measured, not an unbounded
#      limit. Nothing here shows Hebb never levels off. At every stable
#      eta Oja's update is in fact ~100x LARGER than Hebb's, because at
#      ||W||_F = 164.9 the decay term dominates the reinforcement term rather
#      than correcting it.
#
#   2. "The right rule here" is not what the measurements say (C-13). Oja was
#      run at eight step sizes over five orders of magnitude and moved the
#      loop's settled basin at none of them, including the ceiling-silent cells
#      up to 2.9% drift. Every result in which the loop's behaviour CHANGED
#      comes from `hebb`; the other rules are recorded throughout the
#      step-size map and the C3 traces, and what they record is that nothing
#      moved. Why oja is inert is NOT established: the leading explanation,
#      that the brake outweighs the reinforcement term ~110:1 at this site, is
#      C-14 and is untested as an explanation.
#
# ANTI-HEBBIAN IS NOT A NEGATIVE ETA, and the difference is the decay term.
# `eta = -e` scales the whole rule, so it flips BOTH terms: the reinforcement
# term decorrelates (wanted) and the brake `-W <y y^T>` becomes `+W <y y^T>`
# (not wanted). `<y y^T>` is positive semi-definite, so `+W <y y^T>` points along
# W and the weight GROWS -- the one property Oja exists to provide is exactly the
# one a sign-flipped eta destroys. The anti-Hebbian mode flips the reinforcement
# term only and keeps the brake with its stabilising sign, which makes it a linear
# map with a bounded fixed point at W* = -<x y^T> <y y^T>^-1, and on real weights
# it erodes rather than grows.
#
# HOW FAR THAT GROWTH CLAIM GOES, because the unqualified version is what C-15
# retires for `hebb` and the same discipline applies here. What is measured is
# `tests/test_antihebbian.py::TestBoundedness` on real GPT-2 weights, at eta 3e-5
# with the ceiling lifted (`max_delta_frac=1e9`): over SIXTEEN applications the
# negative-eta arm's ||W||_F rises monotonically at every step, 164.9 -> 6.7e+04,
# while `anti_hebb` falls monotonically at every step and is still falling at 50
# (163.6). Sixteen applications is a short run and it establishes no unbounded
# limit -- it says the two spellings separate by orders of magnitude rather than
# percent, in that regime, which is what the mode exists to be distinguished from.
# Do not upgrade it to "grows without bound" anywhere.
#
# COMPOSING TERMS. The modes above are fixed recipes over two primitives:
# H = <x y^T> (reinforcement) and D = W <y y^T> (the brake). Issue #25's third
# row -- "sign, per subspace: reinforce inside the target subspace, erode
# outside it" -- is none of them, because `mode` is one string and selects one
# branch, so a site can carry reinforcement OR erosion and never both. `terms=`
# opens the recipe up. Each term names a primitive, a sign, an optional
# projector of its own and an optional scale, and one hook firing contributes
#
#     dW = sum_i scale_i * sign_i * (T_i P_i)      T_i in {H, D}
#
# which makes issue #24 step 2's Hebbian/anti-Hebbian balance a single site's
# rule: `[+H P, -H (I - P), -D]` reinforces inside P, erodes outside it, and
# keeps one brake over both. Composed per firing, not per apply(), for the same
# reason the single-mode path is -- D reads the live effective weight.
#
# The modes are NOT reimplemented on top of this. `terms=None` runs exactly the
# arithmetic it always did, in the order it always did, so every number already
# recorded in this repo stays reproducible bit-for-bit; the composed path is a
# second branch alongside it rather than a refactor of it.


def _hebb_term(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """<x y^T>, shape (n_in, n_out)."""
    n = x.shape[0]
    return (x.transpose(0, 1) @ y) / max(n, 1)


def _oja_decay(w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """W <y y^T>, shape (n_in, n_out)."""
    n = y.shape[0]
    yy = (y.transpose(0, 1) @ y) / max(n, 1)   # (n_out, n_out)
    return w @ yy


def subspace_projector(basis: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal projector onto the span of `basis`'s rows, (n, n).

    `basis` is (k, n): k directions in the output space of the target matrix --
    an unembedding row, a mean of embeddings, a J-space basis vector. The rows
    need not be orthonormal, but they must be linearly independent; a rank-
    deficient basis makes the projector ill-defined and is rejected rather than
    silently reduced.

    Built in float64 and returned in the basis's dtype: P is `Q Q^T` from a QR,
    and in float32 the residual non-idempotency of that product is of the same
    order as the tolerance `OjaPlasticity` checks it against.
    """
    if basis.dim() != 2 or basis.shape[0] == 0:
        raise ValueError(f"basis must be (k, n) with k >= 1, got {tuple(basis.shape)}")
    b = basis.detach().double()
    if torch.linalg.matrix_rank(b).item() < b.shape[0]:
        raise ValueError("basis rows are linearly dependent; the span is smaller "
                         "than the basis and the projector would be ambiguous")
    q, _ = torch.linalg.qr(b.transpose(0, 1))          # (n, k), orthonormal columns
    return (q @ q.transpose(0, 1)).to(basis.dtype)


def _as_projector(
    project: torch.Tensor, n_out: int, like: torch.Tensor, what: str
) -> torch.Tensor:
    """
    Check `project` is an (n_out, n_out) projection and return it on `like`'s
    device and dtype. `what` names the offending argument in the errors.

    Checked here rather than trusted: a matrix that is not idempotent is not a
    projection, and passing one -- a raw basis, say, instead of
    `subspace_projector(basis)` -- would rescale and rotate every update while
    every number in report() stayed plausible. Held to float32 precision, the
    dtype the product is formed in.
    """
    if not torch.is_tensor(project):
        # Before `.dim()`, or a list or ndarray -- which `TermSpec.coerce`
        # accepts from a dict without inspecting -- raises AttributeError
        # instead of the named ValueError every other path here gives.
        raise ValueError(
            f"{what} must be a torch.Tensor, got {type(project).__name__}; "
            "build it with subspace_projector()"
        )
    if project.dim() != 2 or project.shape != (n_out, n_out):
        raise ValueError(
            f"{what} must be ({n_out}, {n_out}) -- the update's output "
            f"axis in the rules' (n_in, n_out) convention -- got "
            f"{tuple(project.shape)}"
        )
    p = project.detach().to(device=like.device, dtype=like.dtype)
    residual = (p @ p - p).double().norm().item()
    if residual > 1e-4 * max(p.double().norm().item(), 1e-12):
        raise ValueError(
            f"{what} is not idempotent to float32 precision "
            f"(||PP - P||_F / ||P||_F = "
            f"{residual / max(p.double().norm().item(), 1e-12):.2e}), "
            "so it is not a projection; build it with subspace_projector()"
        )
    return p


# --------------------------------------------------------------------------
# Composed rules
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TermSpec:
    """
    One signed, optionally-projected primitive of a composed update rule.

    A list of these is what `OjaPlasticity(terms=...)` takes, and the list reads
    as the rule: `[TermSpec("hebb", +1, P), TermSpec("hebb", -1, P_perp),
    TermSpec("decay", -1)]` is "reinforce inside P, erode outside it, brake over
    both" -- issue #25's per-subspace sign, at one site.

    primitive : {"hebb", "decay"}
        Which of the two quantities the rules are built from:
          "hebb"  -- `<x y^T>`, the reinforcement term. Independent of W.
          "decay" -- `W <y y^T>`, the Oja brake, read against the LIVE effective
                     weight `W0 + delta` at the firing, exactly as the built-in
                     modes read it. Positive semi-definite `<y y^T>` is what
                     makes the sign of this term the difference between a brake
                     and an accelerator, so it is spelled unsigned here and the
                     sign is yours: `sign=-1` brakes, `sign=+1` diverges.
    sign : {+1, -1}
        Exactly +1 or -1, nothing else. Magnitude belongs in `scale`, and a
        `sign` that quietly doubled as a coefficient would put the same number
        in two places and make neither readable in a log.
    project : torch.Tensor, optional
        An (n_out, n_out) orthogonal projector applied to THIS TERM ALONE, as
        `T @ P`, before the sign and scale. `subspace_projector(basis)` builds
        one; `torch.eye(n_out) - P` is its complement, which is how "outside the
        target subspace" is spelled. Composes with `OjaPlasticity(project=...)`:
        the per-term projectors shape each term, the whole-update one is applied
        afterwards to the sum (see `OjaPlasticity`).
    scale : float
        A positive coefficient on the term, default 1.0 -- the knob for an
        unbalanced pair, e.g. eroding outside a direction at a third of the rate
        it is reinforced inside it. `eta` scales the whole rule; this scales one
        term relative to the others. Must be finite and > 0; a negative or zero
        scale is a `sign` or a deleted term written the wrong way round.
    """

    HEBB = "hebb"
    DECAY = "decay"
    VALID_PRIMITIVES = ("hebb", "decay")

    primitive: str
    sign: float = 1.0
    project: Optional[torch.Tensor] = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        # Everything except the projector, which needs the target matrix's width
        # and is therefore checked by `OjaPlasticity`. These need nothing, so
        # they fail at the point the spec is written rather than an import later.
        if self.primitive not in self.VALID_PRIMITIVES:
            raise ValueError(
                f"primitive must be one of {self.VALID_PRIMITIVES}, got "
                f"{self.primitive!r}"
            )
        try:
            sign = float(self.sign)
        except (TypeError, ValueError):
            sign = None
        if sign not in (1.0, -1.0):
            raise ValueError(
                f"sign must be exactly +1 or -1, got {self.sign!r}; a coefficient "
                "belongs in scale="
            )
        # Store the coerced value back, not just validate it. `"1"` floats to a
        # valid sign, so validating a local and discarding it would accept the
        # spec here and then fail in `_composed_update`'s `t.sign * t.scale`
        # with a TypeError, hours into a sweep -- the failure this class
        # validates eagerly to prevent.
        object.__setattr__(self, "sign", sign)
        try:
            scale = float(self.scale)
        except (TypeError, ValueError):
            scale = float("nan")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"scale must be a finite positive number, got {self.scale!r}; "
                "the direction of a term is sign=, not a negative scale"
            )
        object.__setattr__(self, "scale", scale)

    @classmethod
    def coerce(cls, term: Union["TermSpec", dict]) -> "TermSpec":
        """Accept a `TermSpec` as-is, or a plain dict of its fields -- the same
        contract `multi_site.SiteSpec.coerce` offers, so a composed rule can be
        written in whatever a sweep script already builds its cells in. An
        unknown key is a `TypeError` from the constructor, not a silently
        ignored field."""
        if isinstance(term, TermSpec):
            return term
        if isinstance(term, dict):
            return cls(**term)
        raise TypeError(
            f"each term must be a TermSpec or a dict of its fields, got "
            f"{type(term).__name__}"
        )

    def __repr__(self) -> str:
        s = "+" if float(self.sign) > 0 else "-"
        bits = f"{s}{self.primitive}"
        if float(self.scale) != 1.0:
            bits = f"{s}{self.scale:g}*{self.primitive}"
        if self.project is not None:
            bits += "@P"
        return f"TermSpec({bits})"


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
        """The live weight, in the module's own layout.

        HuggingFace `Conv1D` stores (n_in, n_out), which is already the rules'
        convention; `nn.Linear` stores (n_out, n_in) and is handled by the
        `transposed` flip in `apply()`, not here.
        """
        return self.module.weight

    def write(self, w: torch.Tensor) -> None:
        """Overwrite the live weight in place, under `no_grad`.

        In place because the module holds the `Parameter` object -- rebinding
        would leave any hook or optimiser pointing at the old tensor.
        """
        with torch.no_grad():
            self.module.weight.copy_(w)

    def install(self, sink):
        """Register `sink` as a forward hook and return its handle.

        The handle is the only teardown path -- see the hook-hygiene note on
        `_TransformerLensMLPSite` for why `model.reset_hooks()` is never used.
        """
        # The sink IS a torch forward hook here, registered directly. Not
        # wrapped: `tests/test_controls.py` injects its C0 defect by
        # subclassing OjaPlasticity and overriding `_hook`, and that override
        # only bites if the method torch calls is the one the subclass
        # replaced. See the note on the sink shape in `install()` below.
        return self.module.register_forward_hook(sink)

    @staticmethod
    def remove(handle) -> None:
        """Remove exactly the hook `install` added, and nothing else."""
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
        """The MLP's output projection, `W_out`, shape (d_mlp, d_model).

        A bare `nn.Parameter` rather than a module's `.weight`, and already in
        the rules' (n_in, n_out) convention -- so no transpose is needed here,
        unlike the HuggingFace path.
        """
        return self.module.W_out

    def write(self, w: torch.Tensor) -> None:
        """Overwrite `W_out` in place, under `no_grad`."""
        with torch.no_grad():
            self.module.W_out.copy_(w)

    def install(self, sink):
        """Capture x and y from two hook points and return both handles.

        There is no single module behind this matmul, so the pre-activation and
        the projection output are captured separately and paired: `_on_x`
        stashes the input, `_on_y` consumes it and calls `sink` once, in torch's
        forward-hook shape.
        """
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
        """Remove both hooks and drop any half-captured pair.

        Clearing `_pending` matters: a removal between `_on_x` and `_on_y`
        would otherwise leave a stale input to be paired with the next run's
        output.
        """
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

    `shared_post_activity = True` marks that y is the shared full projection
    output rather than this site's own matmul. `offline_control.replay_offline`
    reads it to reconstruct the drifted full output additively when recomputing
    y (`record.y + x @ delta`), instead of rebuilding a single head's
    contribution -- see that module for why the two are not the same tensor.
    """

    backend = "module"
    supports_transposed = False
    shared_post_activity = True

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
        """This head's contiguous row block of the packed (n_in, n_out) matrix.

        A view, not a copy: heads own rows of the *input* axis on HuggingFace,
        which is why this adapter sets `supports_transposed = False` -- in the
        (n_out, n_in) layout a head would own columns and the slice arithmetic
        would silently address the wrong entries.
        """
        return self.module.weight[self._lo:self._hi]

    def write(self, w: torch.Tensor) -> None:
        # A slice assignment, so the other heads' rows are not written at all --
        # not rewritten with the same values, which would still round-trip
        # through float32 and is not the same guarantee.
        with torch.no_grad():
            self.module.weight[self._lo:self._hi].copy_(w)

    def install(self, sink):
        """Hook the owning projection and hand the sink this head's slice of x.

        y is the *full* projection output, not this head's contribution to it:
        the heads share one post-synaptic activity, and that choice is what
        makes a head update bit-for-bit the row slice of the whole-matrix
        update. `tests/test_head_sites.py` asserts that with `torch.equal`.
        """
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

    `shared_post_activity = True` marks that y is the whole layer's attention
    output, not this head's block of it. `offline_control.replay_offline` reads
    the flag to reconstruct the drifted full output additively when recomputing
    y (`record.y + x @ delta`) rather than rebuilding one head's contribution.
    """

    backend = "transformer_lens"
    supports_transposed = False
    shared_post_activity = True

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
        self.n_heads = w.shape[0]
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
        """This head's (d_head, d_model) block of `W_O`.

        No slice arithmetic is needed on this backend: TransformerLens already
        stores `W_O` as (n_heads, d_head, d_model), so a head is an index.
        """
        return self.module.W_O[self.head]

    def write(self, w: torch.Tensor) -> None:
        """Write back into this head's block only, leaving the others untouched.

        Indexed assignment rather than a whole-tensor write, so the other heads'
        entries are never rewritten -- restoring them to identical values would
        still round-trip through float32, which is a weaker guarantee than not
        touching them.
        """
        with torch.no_grad():
            self.module.W_O[self.head].copy_(w)

    def install(self, sink):
        """Capture this head's slice of `hook_z` and pair it with the block output.

        Same convention as the HuggingFace head adapter: x is the head's slice,
        y is the shared full output.
        """
        self._pending = None
        head = self.head

        n_heads = self.n_heads

        def _on_x(_module, _inputs, output):
            # The head axis is second-to-last, and is checked rather than
            # assumed: hook_z is (batch, pos, n_heads, d_head), and indexing the
            # wrong axis of it stays in range on GPT-2 small -- 7 is a valid
            # index into 64 as well as into 12 -- so a silent wrong answer is
            # available here and a loud one is not.
            if (torch.is_tensor(output) and output.dim() >= 3
                    and output.shape[-2] == n_heads):
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
        "oja"       - Oja subspace rule. The rule the project was designed
                      around, but measured inert at the one site tested: it
                      moved the loop's settled basin at none of the eight
                      step sizes swept, including the ceiling-silent cells up
                      to 2.9% drift (register row C-13). Why is not
                      established (C-14).
        "hebb"      - raw Hebbian, no decay. NOT "will diverge": at the
                      step sizes this repo runs at it is bounded and finite
                      (working point 7.07e-05: 0 non-finite, 0.0% clip,
                      ||W||_F +0.03%), and it is the mode that produced every
                      behaviour-changing result. Its drift keeps growing only
                      at large eta with the ceiling lifted, over the fixed
                      small number of applications C3 measures -- which is
                      continued growth in that regime, not an unbounded limit.
                      See the note above the learning rules; register row C-15
                      retires the old unqualified claim.
        "anti_hebb" - Oja with the reinforcement term negated and the decay
                      term left alone: dW = -<x y^T> - W <y y^T>. Erodes what
                      the loop has settled into while staying bounded. NOT the
                      same as eta < 0, which flips the brake too; see the note
                      above the learning rules.
        "random"    - random update matched in Frobenius norm to what Oja would
                      have applied (Control C2). DIAGNOSTIC, not decisive, and
                      the reason is the quantity it is matched on: an isotropic
                      matrix spreads its mass over all 768 singular values, so
                      across the whole step-size sweep this arm holds
                      sigma1/||dW||_F = 0.054 against `hebb`'s 0.979 and its
                      operator norm never reaches `hebb`'s anywhere -- 4x to 11x
                      short. Two arms that agree in Frobenius norm and differ by
                      an order of magnitude in the norm that actually moves a
                      state cannot settle whether the *direction* is doing the
                      work; register row C-23 is retired for exactly that.
                      The control that can is a rank-1 direction matched on the
                      loop-state displacement instead:
                      `experiments/rank1_random_control.py`, which found that
                      arbitrary directions usually DO move the basin (4 of the 6
                      matchable seeds) but never to `hebb`'s destination
                      `comrade`, and only at 66x-171x `hebb`'s weight cost
                      (C-55).
        "off"       - accumulate statistics, apply nothing (Control C0/C1)
        Mutually exclusive with `terms`; see there.
    cadence : int
        Informational only; you decide when to call apply().
    max_delta_frac : float
        Ceiling on ||delta||_F / ||W_0||_F, the guard against silently destroying
        the model. Two details in how apply() enforces it, both of which change
        what a clipped run is:

        What gets rescaled is the ACCUMULATED delta, not the incoming update. The
        candidate `delta + eta*upd` is measured against the ceiling and, if it is
        over, the whole sum is multiplied down by `ceiling/||delta + eta*upd||`.
        So the rescale also shrinks drift accumulated by earlier applies, pulling
        the matrix back toward W0 along directions it had already committed to. A
        clipped run is therefore not "the same trajectory with a smaller last
        step", and a clipped cell is not a smaller-eta cell -- which is why the
        standing prohibition is to never quote one, rather than to quote it with a
        caveat.

        `clipped` in report() is a LATCHING boolean and not a rate: it is set the
        first time any apply() rescales and is cleared only by revert(), so it
        answers "did the ceiling ever bite in this run", not "is it biting now"
        and not "how often". Register row C-46 retires the claim that the library
        records a clipping rate; the rates the step-size map quotes are that
        script's own bookkeeping over semi-private state.

        Held to float32 precision rather than exactly -- expect overshoot of order
        1e-8 relative on a large matrix.
    transposed : bool
        Set True for nn.Linear (weight is (n_out, n_in)). False for Conv1D and
        for TransformerLens W_out, both of which are already (n_in, n_out).
    project : torch.Tensor, optional
        An (n_out, n_out) orthogonal projector, in the rules' convention, that
        the update is multiplied by before the ceiling: `upd = upd @ P`. Drift
        is then confined to the subspace P projects onto -- a direction in
        residual-stream space, say -- and every column of the update outside it
        is exactly zero. `subspace_projector(basis)` builds one from directions.
        Applied to the "random" arm too, so C2's two arms sit inside the SAME
        subspace rather than differing by which subspace they occupy. That removes
        one confound from a control whose scope is otherwise limited by what it is
        matched on -- see `mode="random"` above, and C-23: keeping the arms in one
        subspace does not make the Frobenius match decisive about direction.
        Composes with per-term projectors: those shape each term as it is
        accumulated, this one is applied to the averaged sum in `apply()`, so
        the net effect is `(sum_i scale_i sign_i (T_i P_i)) P`. Both are linear,
        so the order is immaterial to the result and only to where it is legible.
    terms : list of TermSpec (or dicts), optional
        A composed rule, replacing the fixed recipe `mode` selects. One firing
        contributes `sum_i scale_i * sign_i * (T_i P_i)`, with the terms taken
        in the order given; everything downstream -- the whole-update projector,
        eta, the ceiling, delta, revert, report -- is unchanged and identical to
        the built-in modes. This is what makes issue #25's per-subspace sign
        expressible at ONE site:

            P = subspace_projector(gpt2.lm_head.weight[:1])     # a direction
            I = torch.eye(P.shape[0])
            OjaPlasticity(model, site, eta=1e-6, terms=[
                TermSpec("hebb",  +1, P),          # reinforce inside it
                TermSpec("hebb",  -1, I - P),      # erode outside it
                TermSpec("decay", -1),             # one brake over both
            ])

        PRECEDENCE, and why it is an error rather than a rule: `mode` and
        `terms` are two spellings of the same thing, so passing both is
        ambiguous and raises. `terms` is only accepted while `mode` is left at
        its default ("oja") -- an explicit `mode="anti_hebb"` alongside `terms`
        is a `ValueError`, not a silently-ignored argument. An explicit
        `mode="oja"` is indistinguishable from the default and is therefore
        accepted, with the terms winning; that is the one case where the two can
        be written together and the composed rule is what runs.
        `report()["mode"]` is then "terms", because none of the five named rules
        is the one that ran and a log row claiming "oja" would be false.
        An empty list is rejected: it would accumulate zero on every firing and
        report a run that did nothing.
    device / dtype
        Inferred from the target weight.
    """

    VALID_MODES = ("oja", "hebb", "anti_hebb", "random", "off")
    # The value of `mode` that counts as "not set", so `terms=` may be used, and
    # the value `mode` takes when it is.
    DEFAULT_MODE = "oja"
    TERMS_MODE = "terms"

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
        project: Optional[torch.Tensor] = None,
        terms: Optional[Iterable[Union["TermSpec", dict]]] = None,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        if terms is not None and mode != self.DEFAULT_MODE:
            # Two spellings of the same thing, and no defensible way to merge
            # them: silently letting `terms` win would run a rule the caller's
            # `mode=` and every log row disagree with. Refused at construction,
            # naming both, rather than found out in the analysis.
            raise ValueError(
                f"mode={mode!r} and terms= both specify the update rule; pass one "
                f"or the other. `terms` is accepted only while `mode` is left at "
                f"its default {self.DEFAULT_MODE!r}, and then the composed rule "
                f"runs and report()['mode'] is {self.TERMS_MODE!r}"
            )

        self.model = model
        self.site = site
        self.eta = float(eta)
        self.mode = self.TERMS_MODE if terms is not None else mode
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

        # The update's output width in the rules' (n_in, n_out) convention --
        # what every projector, whole-update or per-term, has to match.
        n_out = self.W0.shape[0] if self.transposed else self.W0.shape[1]

        # Checked rather than trusted -- see `_as_projector`, which is where the
        # shape and idempotency tests live so that the whole-update projector
        # and the per-term ones are held to one standard and one message.
        self.project = None
        if project is not None:
            self.project = _as_projector(project, n_out, self.W0, "project")

        # The composed rule, if there is one. Validated here, where W0 fixes the
        # output width and the device/dtype each per-term projector must live
        # on; the rest of each term was checked when the TermSpec was built.
        self.terms: Optional[tuple[TermSpec, ...]] = None
        if terms is not None:
            specs = [TermSpec.coerce(t) for t in terms]
            if not specs:
                raise ValueError(
                    "terms= is empty. A composed rule with no terms accumulates "
                    "zero on every firing and reports a run that did nothing, "
                    "which is indistinguishable in a log from mode='off' and "
                    "from a site that never fired. Pass at least one term, or "
                    "leave terms unset and use mode="
                )
            self.terms = tuple(
                t if t.project is None else replace(
                    t,
                    project=_as_projector(
                        t.project, n_out, self.W0, f"terms[{i}].project"
                    ),
                )
                for i, t in enumerate(specs)
            )

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
        """Capture pre- and post-synaptic activity and accumulate an update.

        Kept as the torch forward-hook signature and as the single override
        point -- the C0 fail-direction test subclasses this class and replaces
        this method. It unpacks the hook arguments and delegates to `observe`,
        which is where the rule actually lives.
        """
        self.observe(inputs[0], output)

    def observe(self, x, y) -> None:
        """
        Accumulate one update from a pre/post activation pair.

        The public entry point for feeding activations to the rule, and the
        supported way to drive it without a live forward pass -- the offline
        replay arm in `offline_control.py` calls this to push a recording
        through the identical rule with no feedback. `_hook` is the same thing
        wearing torch's hook signature.

        Two kinds of input are dropped rather than raising, and they are NOT
        equivalent -- an earlier version of this docstring conflated them:

        - **Non-finite values** set `nonfinite`, so `report()` shows the batch
          was seen and rejected.
        - **Non-tensor** `x` or `y` returns silently, leaving no trace. That
          branch exists because a forward hook on some sites is handed a tuple
          rather than a tensor, where skipping is correct and unremarkable. But
          on the replay path it is a place a bug can hide: feed the wrong thing
          and the arm accumulates nothing while `report()` looks healthy.
          `n_applied` is the check -- an arm that observed nothing still
          reports zero updates.
        """
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
        if self.terms is not None:
            # A composed rule. A SEPARATE branch, not a generalisation of the
            # one below: rewriting the five modes as term lists would reorder
            # their float arithmetic, and every number recorded in this repo so
            # far was produced by the arithmetic below, in this order.
            upd = self._composed_update(x, y)
        else:
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

    def _composed_update(self, x, y) -> torch.Tensor:
        """
        One firing's contribution from a `terms` rule:
        `sum_i scale_i * sign_i * (T_i P_i)`, summed in the order given.

        Composed per firing rather than per apply() because the decay term reads
        the live effective weight, which moves between applies -- exactly the
        reason the single-mode path composes here too.

        `_effective_W()` is read at most once, and only if some term asks for
        the decay primitive: a purely Hebbian composed rule never touches the
        weight, which is what keeps it identical to `mode="hebb"`.
        """
        total = None
        w_eff = None
        for t in self.terms:
            if t.primitive == TermSpec.HEBB:
                term = _hebb_term(x, y)
            else:
                if w_eff is None:
                    w_eff = self._effective_W()
                term = _oja_decay(w_eff, y)
            if t.project is not None:
                # This term's own subspace, on the output axis, the same side
                # and the same convention as the whole-update projector in
                # apply(). Applied BEFORE the sign and scale, so `sign` is the
                # sign of the term that actually lands.
                term = term @ t.project
            term = (t.sign * t.scale) * term
            total = term if total is None else total + term
        return total

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

        if self.project is not None:
            # Confine the drift to the chosen subspace. Before the random arm,
            # not after: norm-matching an unprojected Oja update and then
            # projecting the noise would leave C2's two arms differing in
            # magnitude by the fraction of the noise that survives projection
            # (~sqrt(k/n_out) -- a factor of 28 for a single direction at
            # d_model=768), which is precisely the confound C2 exists to remove.
            upd = upd @ self.project

        if self.mode == "random":
            # Match in float64. The Frobenius match is the one property this arm
            # does have, so it should at least be exact in the quantity it claims:
            # a float32 match leaves ~3e-4 relative error on a matrix this size --
            # small, but avoidable, and an arm already limited in what it can
            # decide should not also be sloppy about what it can.
            #
            # What the match does NOT buy is direction-specificity, and no
            # precision here would. Frobenius norm is the wrong quantity: this
            # noise spreads it across the spectrum (sigma1/||dW||_F 0.054) where
            # `hebb` concentrates it (0.979), so the two arms never carry the same
            # operator norm and C2 cannot separate direction from magnitude.
            # C-23 is retired on that ground; the decisive comparison is the
            # rank-1 arm matched on loop displacement in
            # `experiments/rank1_random_control.py` (C-55), not this one.
            target_norm = upd.double().norm().item()
            noise = torch.randn(
                upd.shape, generator=self._rng, device=upd.device, dtype=upd.dtype
            )
            if self.project is not None:
                noise = noise @ self.project
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
