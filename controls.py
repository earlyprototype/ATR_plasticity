"""
Controls for the plasticity experiments.

Run these before believing anything. C0 in particular: if installing the hooks
changes the trajectory at eta=0, every downstream result is contaminated and
nothing else in this repo means anything yet.

Each function takes an `atr_step` callable with the signature

    atr_step(model, r) -> r_next

which should be YOUR tested engine, imported from the main ATR repo. Nothing
here reimplements the loop.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from plasticity import OjaPlasticity

AtrStep = Callable[[object, torch.Tensor], torch.Tensor]


def _trajectory(
    model,
    r0: torch.Tensor,
    atr_step: AtrStep,
    n_iter: int,
    plast: Optional[OjaPlasticity] = None,
    apply_every: int = 1,
) -> list[torch.Tensor]:
    """Run n_iter steps, returning every intermediate state."""
    r = r0.clone()
    states = []
    for i in range(n_iter):
        r = atr_step(model, r)
        if plast is not None and (i + 1) % apply_every == 0:
            plast.apply()
        states.append(r.detach().clone())
    return states


def c0_identity(
    model, r0: torch.Tensor, atr_step: AtrStep, site: str, n_iter: int = 50
) -> dict:
    """
    C0 -- THE GATE. With eta=0 and mode="off", installing the hooks must not
    perturb the trajectory by a single bit.

    Hooks that read activations should be side-effect free. If this fails, the
    likely causes are: a dtype cast leaking back into the graph, an in-place
    op on a captured tensor, or the hook firing on a module whose output is
    later mutated. Fix before proceeding.
    """
    baseline = _trajectory(model, r0, atr_step, n_iter)

    with OjaPlasticity(model, site=site, eta=0.0, mode="off") as plast:
        hooked = _trajectory(model, r0, atr_step, n_iter, plast=plast)

    max_abs = max(
        (a - b).abs().max().item() for a, b in zip(baseline, hooked)
    )
    bit_exact = all(torch.equal(a, b) for a, b in zip(baseline, hooked))

    return {
        "control": "C0_identity",
        "bit_exact": bit_exact,
        "max_abs_deviation": max_abs,
        "verdict": "PASS" if bit_exact else "FAIL -- do not proceed",
    }


def c1_revert(
    model,
    r0: torch.Tensor,
    atr_step: AtrStep,
    site: str,
    eta: float,
    n_iter: int = 50,
) -> dict:
    """
    C1 -- reversibility. Run with plasticity, revert, run again. The
    post-revert trajectory must match the original baseline.

    Catches: weights not actually restored, hidden state accumulating
    elsewhere, RNG drift.
    """
    baseline = _trajectory(model, r0, atr_step, n_iter)

    # `with` removes the hook, `finally` restores the weights -- both have to
    # hold even when atr_step raises, or the next control in the sweep runs on
    # a model this one quietly left modified.
    with OjaPlasticity(model, site=site, eta=eta, mode="oja") as plast:
        try:
            _trajectory(model, r0, atr_step, n_iter, plast=plast)
            drifted = plast.report()["delta_frac"]
        finally:
            plast.revert()

    after = _trajectory(model, r0, atr_step, n_iter)

    max_abs = max((a - b).abs().max().item() for a, b in zip(baseline, after))
    return {
        "control": "C1_revert",
        "delta_frac_before_revert": drifted,
        "max_abs_deviation_after_revert": max_abs,
        "bit_exact": all(torch.equal(a, b) for a, b in zip(baseline, after)),
    }


def c2_random_direction(
    model,
    r0: torch.Tensor,
    atr_step: AtrStep,
    site: str,
    eta: float,
    n_iter: int = 200,
) -> dict:
    """
    C2 -- is the update DIRECTION doing the work, or just its magnitude?

    Compare mode="oja" against mode="random", which applies a random matrix
    matched in Frobenius norm to what Oja would have applied. If the two give
    the same landscape change, you have measured "perturbing weights by this
    much changes things", which is not a finding about Hebbian learning.

    This is the control that decides whether the branch is interesting.
    """
    out = {}
    for mode in ("oja", "random"):
        with OjaPlasticity(model, site=site, eta=eta, mode=mode) as plast:
            try:
                states = _trajectory(model, r0, atr_step, n_iter, plast=plast)
                out[mode] = {
                    "delta_frac": plast.report()["delta_frac"],
                    "final_state": states[-1],
                }
            finally:
                # The second arm must start from the original weights, not from
                # wherever the first one left them.
                plast.revert()

    a, b = out["oja"]["final_state"], out["random"]["final_state"]
    cos = torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()
    return {
        "control": "C2_random_direction",
        "cos_oja_vs_random_final": cos,
        "delta_frac_oja": out["oja"]["delta_frac"],
        "delta_frac_random": out["random"]["delta_frac"],
        "note": "cos near 1.0 means the direction is NOT doing the work",
    }


def c3_divergence_demo(
    model,
    r0: torch.Tensor,
    atr_step: AtrStep,
    site: str,
    eta: float,
    n_iter: int = 100,
) -> dict:
    """
    C3 -- pedagogical: show that raw Hebbian diverges and Oja does not.

    Not a control on the result; a demonstration that the decay term is doing
    the job it is there to do. Worth one figure.
    """
    traces = {}
    for mode in ("hebb", "oja"):
        with OjaPlasticity(
            model, site=site, eta=eta, mode=mode, max_delta_frac=1e9
        ) as plast:
            fracs = []
            try:
                r = r0.clone()
                for _ in range(n_iter):
                    r = atr_step(model, r)
                    plast.apply()
                    fracs.append(plast.report()["delta_frac"])
                    if not torch.isfinite(r).all():
                        break
            finally:
                # This control deliberately runs with the ceiling off, so an
                # un-reverted hebb arm would hand the oja arm a wrecked matrix.
                traces[mode] = fracs
                plast.revert()
    return {"control": "C3_divergence_demo", "delta_frac_traces": traces}


if __name__ == "__main__":
    raise SystemExit(
        "Import these and pass in your own atr_step; there is no default loop "
        "here on purpose. See README, 'Running the controls'."
    )
