"""Stimulation on the 12x12 grid: current delivered at chosen sites.

The companion to `mea_grid.py`. That module reads the 144 sites; this one writes
to them. In an electrode array the same electrodes do both, which is why Wagenaar
and Potter had to write a separate paper on suppressing the artefact of recording
through an electrode you have just stimulated (J Neurosci Methods, real-time
artifact suppression by local curve fitting). The same hazard exists here and is
handled the same way: a site that was stimulated on an iteration must not have its
activity read naively on that iteration, because the reading would include the
injection. `stimulated_sites` is carried on every step so the measurement side can
exclude them.

WHY THE LOOP IS NOT REIMPLEMENTED HERE. The project's standing rules forbid a
second ATR implementation. This module builds a step whose body is
`atr_bridge.make_atr_step`'s, with additional hooks installed alongside the
injection the bridge already performs. With no stimulation configured it must
produce a bit-identical trajectory, and `tests/test_mea_stim.py` asserts exactly
that against the bridge, max deviation 0.0. That test is the gate: if it fails,
this module is not studying the same loop.

THE TWO PLACES CURRENT CAN BE DELIVERED

  A head site, written `(layer, head)`. The vector is added to that head's output
  before the output projection, at `blocks.{L}.attn.hook_z`, index `head`. This is
  the faithful analogue of an electrode: a signal introduced at one location, which
  then propagates through whatever the network does next.

  A stream site, written `(layer, None)`. The vector is added to the whole residual
  stream arriving at that block, at `blocks.{L}.hook_resid_pre`. This is what an
  earlier draft of the pre-registration assumed was the only option.

STRENGTH IS ALWAYS RELATIVE TO LOCAL ACTIVITY. `beta` multiplies a unit vector that
is then scaled by the length of whatever is already at that site on that forward
pass. A `beta` of 0.01 means the injected signal is one percent as large as the
activity already there, wherever there is. This matters because the blocks differ
substantially in activation scale: EXP-002 found more than a 200x spread across
layers and had to anchor its step sizes per layer to cope. A single absolute scale
would repeat that problem, and would silently make "the same strength" mean
different things at different depths.

MATCHING SPREAD-OUT AGAINST CONCENTRATED. When one total strength is divided among
several sites, each site receives `beta / sqrt(n_sites)`, so that the sum of
squares is held equal rather than the plain sum. Sum of squares is the right
invariant because vectors drawn independently in a high-dimensional space are close
to orthogonal, so their combined length grows as the square root of their number
rather than linearly. `plan_sites` implements this and the alternative is available
through `match="sum"` so the choice can be reported as a sensitivity check rather
than buried.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence

import torch

from atr_bridge import hook_points, _layer_end, initial_state


__all__ = ["StimSite", "StimPlan", "plan_sites", "make_stim_step"]


@dataclass(frozen=True)
class StimSite:
    """One place current is delivered.

    `head` None means the whole residual stream arriving at `layer`; an integer
    means that head's slice of the attention output at `layer`.
    """

    layer: int
    head: Optional[int] = None

    @property
    def is_head(self) -> bool:
        return self.head is not None

    def __str__(self) -> str:
        return f"L{self.layer}.H{self.head}" if self.is_head else f"L{self.layer}.stream"


@dataclass
class StimPlan:
    """Where to stimulate, how hard, and how often.

    `beta_per_site` is already divided down for the number of sites; `beta_total`
    is kept alongside it so a record of the run says what was asked for as well as
    what each site got.

    `every` is the rate: current is delivered on iterations where
    `iteration % every == 0`, so `every=1` is every iteration and `every=16` is one
    in sixteen. This is the axis carrying the biological crossing, because the
    measurement being copied is stated in stimuli per second.
    """

    sites: tuple[StimSite, ...]
    beta_per_site: float
    beta_total: float
    every: int = 1
    seed: int = 20260802
    match: Literal["rms", "sum"] = "rms"
    _vectors: dict = field(default_factory=dict, repr=False)

    def vector_for(self, site: StimSite, length: int, width: int) -> torch.Tensor:
        """A fixed unit direction per site, drawn once and reused every iteration.

        Fixed rather than fresh each step because the protocol being copied
        delivered the same pulse repeatedly. A fresh-noise variant is named in the
        pre-registration as the obvious follow-up and is not this.
        """
        key = (site.layer, site.head, length, width)
        if key not in self._vectors:
            gen = torch.Generator().manual_seed(
                self.seed + 1009 * site.layer + 17 * (site.head if site.is_head else 99)
            )
            v = torch.randn(length, width, generator=gen)
            self._vectors[key] = v / v.norm()
        return self._vectors[key]


def plan_sites(
    sites: Sequence[StimSite],
    beta_total: float,
    every: int = 1,
    seed: int = 20260802,
    match: Literal["rms", "sum"] = "rms",
) -> StimPlan:
    """Divide one total strength among sites so the arms are comparable.

    `rms` (the registered default) holds the sum of squares equal, giving each site
    `beta_total / sqrt(n)`. `sum` holds the plain sum equal, giving each
    `beta_total / n`. The two differ by a factor of sqrt(n), which at three sites is
    1.73 and at twenty-four is 4.9, so the choice is not cosmetic and is reported.
    """
    sites = tuple(sites)
    if not sites:
        raise ValueError("a stimulation plan needs at least one site")
    if len({(s.layer, s.head) for s in sites}) != len(sites):
        raise ValueError("duplicate stimulation site; each site may appear once")
    if every < 1:
        raise ValueError("`every` is an iteration count and must be at least 1")

    n = len(sites)
    per = beta_total / (n ** 0.5) if match == "rms" else beta_total / n
    return StimPlan(sites=sites, beta_per_site=per, beta_total=beta_total,
                    every=every, seed=seed, match=match)


def make_stim_step(
    model,
    prompt: str,
    plan: Optional[StimPlan] = None,
    layer_start: int = 0,
    layer_end: Optional[int] = None,
    initial_norm: Optional[float] = None,
) -> Callable:
    """An ATR step that also delivers current at the planned sites.

    Returns `step(model, r, iteration=0) -> r_next`. With `plan` None, or with
    `beta_total` exactly zero, the trajectory is bit-identical to
    `atr_bridge.make_atr_step`'s, which is the gate the test suite enforces.

    `step.stimulated_sites(iteration)` reports which sites fired on a given
    iteration, so the measurement side can exclude them and avoid reading its own
    injection back.
    """
    layer_end = _layer_end(model, layer_end)
    hook_point_read, hook_point_write = hook_points(layer_start, layer_end)

    if initial_norm is None:
        initial_norm = initial_state(model, prompt, layer_end).initial_norm
    initial_norm = float(initial_norm)

    active = plan is not None and plan.beta_total != 0.0 and len(plan.sites) > 0

    def fires(iteration: int) -> bool:
        return active and (iteration % plan.every == 0)

    def _stim_hook_for(site: StimSite):
        """Build the hook that adds current at one site.

        The added vector is scaled by the length of what is already there, so
        `beta` always means a fraction of local activity.
        """
        def head_hook(z, hook):
            # z is (batch, pos, head, d_head)
            slice_ = z[0, :, site.head, :]
            unit = plan.vector_for(site, slice_.shape[0], slice_.shape[1]).to(z.dtype)
            z[0, :, site.head, :] = slice_ + plan.beta_per_site * slice_.norm() * unit
            return z

        def stream_hook(resid, hook):
            # resid is (batch, pos, d_model)
            cur = resid[0, :, :]
            unit = plan.vector_for(site, cur.shape[0], cur.shape[1]).to(resid.dtype)
            resid[0, :, :] = cur + plan.beta_per_site * cur.norm() * unit
            return resid

        return head_hook if site.is_head else stream_hook

    def _hook_name(site: StimSite) -> str:
        return (f"blocks.{site.layer}.attn.hook_z" if site.is_head
                else f"blocks.{site.layer}.hook_resid_pre")

    def step(model, r: torch.Tensor, iteration: int = 0) -> torch.Tensor:
        # --- atr_bridge.make_atr_step body, unchanged -----------------------
        current_norm = r.norm().item()
        if current_norm > 0:
            r = r * (initial_norm / current_norm)

        inject_tensor = r.clone()

        def injection_hook(resid, hook, tensor=inject_tensor):
            resid[0, :, :] = tensor
            return resid

        model.add_hook(hook_point_write, injection_hook)
        # --- the only addition: current at the planned sites ----------------
        if fires(iteration):
            for site in plan.sites:
                name = _hook_name(site)
                if name == hook_point_write:
                    # The loop's own injection overwrites this tensor wholesale,
                    # so a stimulation hook registered at the same point would be
                    # silently erased. Refusing is better than a silent no-op.
                    raise ValueError(
                        f"stimulation site {site} collides with the loop's "
                        f"injection point {hook_point_write}; the loop overwrites "
                        f"that tensor, so the stimulation would be discarded. "
                        f"Use hook_resid_post of the same block, or a head site."
                    )
                model.add_hook(name, _stim_hook_for(site))
        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    prompt,
                    names_filter=lambda n: n == hook_point_read,
                )
        finally:
            model.reset_hooks()

        return cache[hook_point_read][0].clone()

    step.prompt = prompt
    step.initial_norm = initial_norm
    step.hook_point_read = hook_point_read
    step.hook_point_write = hook_point_write
    step.plan = plan
    step.stimulated_sites = lambda iteration: (
        tuple(plan.sites) if fires(iteration) else ()
    )
    return step
