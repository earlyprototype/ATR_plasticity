"""
Per-step bridge to the parent ATR engine.

`controls.py` needs `atr_step(model, r) -> r_next`. The parent project
(`atr_engine.run_atr_loop`) drives the whole loop internally, so there is no
point between two iterations at which a weight update can be applied. This
module opens that seam and nothing else.

**This is an extraction, not a second ATR implementation.** Every line of
`atr_step` below is `run_atr_loop`'s loop body, copied: the same hook points,
the same full-tensor state, the same rescale-to-initial-norm before injection,
the same `add_hook` / `reset_hooks()`-in-`finally`. No new normalisation
options, no extra features, no improvements. If the parent's loop changes,
this file is wrong until it is re-copied, and the equivalence test in
`tests/test_atr_bridge.py` is what will tell you so.

The map, for reference (parent's `docs/TECHNICAL.md`):

    x₀ = f(embed(prompt))                    read at blocks.{layer_end}.hook_resid_post
    xₙ₊₁ = f(normalise(xₙ))                  injected at blocks.{layer_start}.hook_resid_pre
    normalise(x) = x · (‖x₀‖₂ / ‖x‖₂)

`initial_norm` is **loop** state, not step state: it is `‖x₀‖` captured once
from the first forward pass and held fixed thereafter. Recomputing it per step
would turn the rescale into a no-op and silently change what the loop means, so
it is threaded explicitly — through the closure, through `ATRState`, and
through a saved `.pt` file when resuming.

Two things a caller must know, both inherited from the parent:

- The step calls `model.reset_hooks()` on the way out. That removes *every*
  non-permanent TransformerLens hook on the model, not just the injection hook.
  A plasticity layer attached with `model.add_hook(...)` will be gone after the
  first step; attach it with `is_permanent=True`, or as a plain
  `torch.nn.Module.register_forward_hook`, which `reset_hooks()` does not touch.
- After iteration 0 the prompt string is inert except for its token count. The
  injection overwrites `blocks.{layer_start}.hook_resid_pre` wholesale, so
  embeddings and positional embeddings are discarded; the prompt survives only
  as sequence length. Resuming a saved state therefore requires a prompt that
  tokenises to the same number of positions (the parent's states carry theirs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch

AtrStep = Callable[[object, torch.Tensor], torch.Tensor]

__all__ = [
    "ATRState",
    "AtrStep",
    "hook_points",
    "renormalise",
    "initial_state",
    "load_state",
    "make_atr_step",
    "make_atr_step_from_state",
]


@dataclass(frozen=True)
class ATRState:
    """One point on an ATR trajectory, plus the loop state needed to continue it.

    `tensor` is the full residual stream over all positions, (seq, d_model) --
    the parent iterates the whole tensor, not the last position. `initial_norm`
    belongs to the trajectory, not to the tensor: carrying the two separately
    is the bug this dataclass exists to prevent.
    """

    tensor: torch.Tensor
    initial_norm: float
    prompt: str
    iteration: int = 0
    label: Optional[str] = None


def hook_points(layer_start: int, layer_end: int) -> tuple[str, str]:
    """(read, write) hook names, exactly as `run_atr_loop` spells them."""
    return (
        f"blocks.{layer_end}.hook_resid_post",
        f"blocks.{layer_start}.hook_resid_pre",
    )


def _layer_end(model, layer_end: Optional[int]) -> int:
    return model.cfg.n_layers - 1 if layer_end is None else layer_end


def renormalise(tensor: torch.Tensor, initial_norm: float) -> torch.Tensor:
    """`x · (‖x₀‖ / ‖x‖)` -- rescale to the trajectory's initial energy.

    Not unit norm. A caller who normalises to 1.0 instead is running a
    different dynamical system: the parent's attractors are measured at
    ‖x₀‖ (1468.5 for `Divine`), and the map is not scale-invariant.

    Zero-norm input is passed through unchanged, as in the parent.
    """
    current_norm = tensor.norm().item()
    if current_norm > 0:
        return tensor * (initial_norm / current_norm)
    return tensor


def initial_state(model, prompt: str, layer_end: Optional[int] = None) -> ATRState:
    """Iteration 0: one clean forward pass, read at `blocks.{layer_end}.hook_resid_post`.

    Returns the tensor and the `initial_norm` every later step rescales to. If
    this norm is wrong, every subsequent iterate is on the wrong energy shell
    and no comparison against the parent's recorded attractors is meaningful.
    """
    layer_end = _layer_end(model, layer_end)
    hook_point_read, _ = hook_points(0, layer_end)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            prompt,
            names_filter=lambda n: n == hook_point_read,
        )

    tensor = cache[hook_point_read][0].clone()
    return ATRState(
        tensor=tensor,
        initial_norm=tensor.norm().item(),
        prompt=prompt,
        iteration=0,
    )


def load_state(path: str, map_location: str = "cpu") -> ATRState:
    """Resume from one of the parent's saved `state_*.pt` checkpoints.

    Those files hold `label`, `prompt`, `iteration`, `current_tensor`,
    `initial_norm`, `prev_last`, `snapshots`; only the first five are loop
    state and `prev_last`/`snapshots` are dropped here. The point of resuming
    is that `state_divine.pt` already sits *on* the period-2 cycle at iteration
    1000 -- reproducing it from the prompt costs 1000 forward passes and, worse,
    would be a different tensor if anything in the stack has shifted since.
    """
    raw = torch.load(path, map_location=map_location, weights_only=False)
    missing = {"current_tensor", "initial_norm", "prompt"} - set(raw)
    if missing:
        raise KeyError(f"{path} is not an ATR state checkpoint; missing {sorted(missing)}")
    return ATRState(
        tensor=raw["current_tensor"].clone(),
        initial_norm=float(raw["initial_norm"]),
        prompt=raw["prompt"],
        iteration=int(raw.get("iteration", 0)),
        label=raw.get("label"),
    )


def make_atr_step(
    model,
    prompt: str,
    layer_start: int = 0,
    layer_end: Optional[int] = None,
    initial_norm: Optional[float] = None,
) -> AtrStep:
    """Build the per-step closure `controls.py` expects.

    One call to the returned `step` is one iteration of `run_atr_loop`, in the
    same order: rescale to `initial_norm`, inject the whole tensor at
    `blocks.{layer_start}.hook_resid_pre`, forward, read
    `blocks.{layer_end}.hook_resid_post`. Iterating it must reproduce
    `run_atr_loop`'s trajectory bit for bit; if it does not, the plasticity
    results are being compared against a loop the parent never ran.

    `initial_norm` defaults to `‖x₀‖` from a fresh forward pass on `prompt`
    (what `run_atr_loop` does). Pass it explicitly when resuming a saved
    trajectory -- see `make_atr_step_from_state`.

    The `model` argument to `step` is the one used, not the one captured here,
    so a control may hand it a modified model; `prompt`, the hook points and
    `initial_norm` are fixed at construction.
    """
    layer_end = _layer_end(model, layer_end)
    hook_point_read, hook_point_write = hook_points(layer_start, layer_end)

    if initial_norm is None:
        initial_norm = initial_state(model, prompt, layer_end).initial_norm
    initial_norm = float(initial_norm)

    def step(model, r: torch.Tensor) -> torch.Tensor:
        # --- run_atr_loop body, verbatim -------------------------------------
        current_norm = r.norm().item()
        if current_norm > 0:
            r = r * (initial_norm / current_norm)

        inject_tensor = r.clone()

        def injection_hook(resid, hook, tensor=inject_tensor):
            resid[0, :, :] = tensor
            return resid

        model.add_hook(hook_point_write, injection_hook)
        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    prompt,
                    names_filter=lambda n: n == hook_point_read,
                )
        finally:
            # Unconditional: a step that raises with the injection hook still
            # installed poisons every later forward pass on this model.
            model.reset_hooks()

        return cache[hook_point_read][0].clone()
        # --- end of copied body ----------------------------------------------

    step.prompt = prompt
    step.initial_norm = initial_norm
    step.hook_point_read = hook_point_read
    step.hook_point_write = hook_point_write
    return step


def make_atr_step_from_state(
    model,
    state: ATRState,
    layer_start: int = 0,
    layer_end: Optional[int] = None,
) -> AtrStep:
    """`make_atr_step` continuing an existing trajectory.

    Takes the prompt and -- critically -- `initial_norm` from the checkpoint
    rather than recomputing them. Recomputing `initial_norm` from a resumed
    tensor would rescale the attractor to its own norm and destroy the thing
    being measured.
    """
    return make_atr_step(
        model,
        state.prompt,
        layer_start=layer_start,
        layer_end=layer_end,
        initial_norm=state.initial_norm,
    )


if __name__ == "__main__":
    raise SystemExit(
        "Nothing to run here. Import make_atr_step and pass it to controls.py; "
        f"the loop itself lives in the parent repo "
        f"({os.environ.get('ATR_PARENT_PATH', 'ATR_PARENT_PATH unset')})."
    )
