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
        Dotted path to the target module, e.g. "transformer.h.6.mlp.c_proj".
        Must be a module with a 2-D `.weight`.
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
        Set True for nn.Linear (weight is (n_out, n_in)). False for Conv1D.
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

        self.module = self._resolve(model, site)
        if not hasattr(self.module, "weight") or self.module.weight.dim() != 2:
            raise TypeError(f"{site} has no 2-D .weight; not a supported target")

        # Frozen reference copy. Everything is measured against this.
        self.W0 = self.module.weight.detach().clone()
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
        obj = model
        for part in path.split("."):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    def install(self) -> "OjaPlasticity":
        if self._handle is not None:
            return self
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
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
        self._last_update_norm = step.norm().item()
        self._write_weight()
        self.n_applied += 1
        return self.report()

    def _write_weight(self) -> None:
        with torch.no_grad():
            self.module.weight.copy_(self.W0 + self.delta)

    def revert(self) -> None:
        """Restore the original weight exactly and zero the accumulated delta."""
        self.delta = torch.zeros_like(self.W0)
        self._acc = None
        self._n_batches = 0
        with torch.no_grad():
            self.module.weight.copy_(self.W0)

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

def candidate_sites(model: nn.Module, prefix: str = "transformer.h") -> list[str]:
    """
    List 2-D weight matrices that make sensible plasticity targets.

    Preferred first targets, in order:
      1. `mlp.c_proj`  -- pre and post activity both cleanly defined, and the
         MLP down-projection is the least entangled place to perturb.
      2. `attn.c_proj` -- the OV output circuit.
      3. `mlp.c_fc`    -- pre-nonlinearity; post-synaptic activity is less
         cleanly interpretable.
    Avoid `attn.c_attn` initially: it packs Q, K and V into one matrix, so a
    Hebbian update there is three different experiments at once.
    """
    out = []
    for name, mod in model.named_modules():
        if not name.startswith(prefix):
            continue
        w = getattr(mod, "weight", None)
        if torch.is_tensor(w) and w.dim() == 2:
            out.append(name)
    return out
