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
#
# The Oja decay term is what makes this the right rule here rather than a
# convenience: Oja's rule is Hebbian learning with normalisation built in, and
# it performs power iteration on the input correlation structure. ATR is
# nonlinear power iteration on activations. Same mathematics, one loop up.


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
#
# Adding a third backend means writing four methods, not touching a rule.


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
            "W_out. Attention sites are not offered: TransformerLens stores "
            "W_Q/W_K/W_V/W_O per head, 3-D, and the rules here are 2-D."
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


def _make_site(model: nn.Module, site: str):
    """Pick the adapter from the model, not from the site string."""
    if _is_hooked_transformer(model):
        return _TransformerLensMLPSite(model, site)
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
    eta : float
        Learning rate. Start absurdly small (1e-6) and work up. There is no
        gradient here and no optimiser to save you.
    mode : {"oja", "hebb", "random", "off"}
        "oja"    - Oja subspace rule (the real experiment)
        "hebb"   - raw Hebbian, no decay (will diverge; pedagogical)
        "random" - random update matched in Frobenius norm to what Oja would
                   have applied (Control C2: is the *direction* doing work?)
        "off"    - accumulate statistics, apply nothing (Control C0/C1)
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

    VALID_MODES = ("oja", "hebb", "random", "off")

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
        if self.mode in ("oja", "random"):
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

def candidate_sites(model: nn.Module, prefix: Optional[str] = None) -> list[str]:
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
    actually runs -- the only offered sites are `blocks.{L}.mlp`, meaning that
    MLP's `W_out`. That is deliberately narrower than the HuggingFace list:
    `W_in` has no post-synaptic activity that is cleanly the output of a single
    matmul (the nonlinearity sits in between), and the attention matrices are
    stored per-head and 3-D, which the 2-D rules here cannot address at all.
    Returning a site this module cannot actually attach to would be worse than
    returning fewer.
    """
    if _is_hooked_transformer(model):
        sites = []
        for layer, block in enumerate(model.blocks):
            w = getattr(getattr(block, "mlp", None), "W_out", None)
            if torch.is_tensor(w) and w.dim() == 2:
                sites.append(f"blocks.{layer}.mlp")
        return sites

    out = []
    for name, mod in model.named_modules():
        if not name.startswith(prefix if prefix is not None else "transformer.h"):
            continue
        w = getattr(mod, "weight", None)
        if torch.is_tensor(w) and w.dim() == 2:
            out.append(name)
    return out
