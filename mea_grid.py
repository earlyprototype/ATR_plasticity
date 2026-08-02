"""The 12x12 grid: per-head activity, and the centre of activity taken from it.

This module ports one measurement from Chao, Bakkum and Potter (2007), J Neural
Eng 4(3):294-308, where it was used to detect network plasticity in cultured
cortical networks that firing-rate statistics could not detect. Their equation 1,
quoted from the reprint:

    [CA_X(n), CA_Y(n)] = ( sum_k FRH_Ek(n) * [Col(E_k) - R_col, Row(E_k) - R_row] )
                         / ( sum_k FRH_Ek )

"Intuitively, CA is analogous to the center of mass, where the 'mass' at an
electrode location is determined by the recorded FR. CAT is the sequence of CAs
over successive time intervals."

WHY THIS IS PORTABLE AT ALL. Their array was 60 electrodes on an 8 by 8 grid with
real physical geometry, which is what makes an average position meaningful. A
transformer's residual stream has no comparable geometry across its width, and a
centroid over d_model would be meaningless because that basis is arbitrary. But
GPT-2 small has 12 blocks of 12 attention heads, which is a 12 by 12 grid of 144
addressable sites, and the same sites both carry activity and can be stimulated,
exactly as an electrode both records and stimulates.

THE ONE ASYMMETRY, STATED UP FRONT BECAUSE IT LIMITS WHAT MAY BE CLAIMED. Only
one axis of the grid has a metric. Layer index is ordered and means something:
layer 3 really does sit between layer 2 and layer 4, and activity really does
flow along that axis. Head index does not. Heads within a layer have no canonical
order and permuting their labels changes nothing about the model, so a centroid
along the head axis has no fixed value.

This module therefore computes both and treats them differently:

  - `ca_depth` is the load-bearing quantity. Layer index weighted by how much each
    layer writes. Meaningful.
  - `ca_head` is computed only as an internal consistency check. It has no
    interpretation, and the check is that shuffling head labels must leave every
    downstream statistic unchanged. If it does not, the implementation is wrong.

That second use is the honest version of Chao's own shuffle control (their
CAT-ELS, supplement S2), which randomised electrode positions to test whether the
spatial embedding was doing the work or whether the statistic merely benefited
from compressing many channels into few well-conditioned numbers. Their reported
effect: detectable change roughly doubled and sensitivity fell from 88.7% to
35.4%. Here the layer shuffle is the real version of that test, and the head
shuffle has a known correct answer (no change), which makes it a control that
cannot silently pass.

WHAT THE MASS IS. For head h of layer L, the mass is the length of what that head
writes into the residual stream on that forward pass:

    w[L, h] = || z[L][:, h, :] @ W_O[L][h] ||

where z is `blocks.L.attn.hook_z`, the per-head output before the output
projection, and W_O[L][h] is that head's slice of the projection. This is the
head's actual contribution to the stream, not a proxy for it. The MLP write per
layer is captured separately by `mlp_mass` because the MLPs are twelve further
sites that are not part of the head grid, and every published result in this
repository so far moved an MLP rather than a head.

BIT-EXACTNESS. `grid_step` must produce the same trajectory as
`atr_bridge.make_atr_step`. It is the same loop body with a wider `names_filter`
on the cache, so the arithmetic that produces the next state is untouched.
`tests/test_mea_grid.py` asserts equality tensor for tensor against the bridge,
max deviation exactly 0.0. If that test fails, this module is measuring a
different trajectory from the one the rest of the project studies and nothing it
reports is comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from atr_bridge import hook_points, _layer_end


__all__ = [
    "GridActivity",
    "grid_step",
    "ca_depth",
    "ca_head",
    "ca_2d",
    "separation_ratio",
    "shuffle_layers",
    "shuffle_heads",
]


@dataclass
class GridActivity:
    """One iteration's activity over the 12x12 grid, plus the MLP writes.

    `heads` is (n_layers, n_heads): the length of each head's write into the
    residual stream. `mlp` is (n_layers,): the same for each block's MLP output.
    Both are float64 so that ratios of small numbers are not quantised.
    """

    heads: torch.Tensor
    mlp: torch.Tensor

    @property
    def n_layers(self) -> int:
        return int(self.heads.shape[0])

    @property
    def n_heads(self) -> int:
        return int(self.heads.shape[1])


def _head_masses(cache, model, layer_end: int) -> torch.Tensor:
    """Per-head write lengths, (n_layers, n_heads), float64.

    z is (batch, pos, head, d_head) and W_O is (head, d_head, d_model), so head
    h's write is z[..., h, :] @ W_O[h]. Summed over positions by taking the norm
    of the whole (pos, d_model) block, which is the length of that head's total
    contribution across the sequence rather than at any one position. Position
    uniformity means the choice matters less here than it would in general, but
    it is stated rather than assumed.
    """
    rows = []
    for layer in range(layer_end + 1):
        z = cache[f"blocks.{layer}.attn.hook_z"][0].double()      # (pos, head, d_head)
        w_o = model.blocks[layer].attn.W_O.double()                # (head, d_head, d_model)
        # (head, pos, d_head) @ (head, d_head, d_model) -> (head, pos, d_model)
        per_head = torch.bmm(z.permute(1, 0, 2), w_o)
        rows.append(per_head.reshape(per_head.shape[0], -1).norm(dim=1))
    return torch.stack(rows)


def _mlp_masses(cache, layer_end: int) -> torch.Tensor:
    """Per-layer MLP write lengths, (n_layers,), float64."""
    return torch.stack([
        cache[f"blocks.{layer}.hook_mlp_out"][0].double().norm()
        for layer in range(layer_end + 1)
    ])


def grid_step(
    model,
    prompt: str,
    layer_start: int = 0,
    layer_end: Optional[int] = None,
    initial_norm: Optional[float] = None,
) -> Callable:
    """An ATR step that also returns the grid activity of the pass it ran.

    The returned callable has the signature `step(model, r) -> (r_next, activity)`.
    Its trajectory is identical to `atr_bridge.make_atr_step`'s, which the test
    suite asserts bit for bit; the only difference is that the cache is asked for
    more names on the way past.

    Note the ordering, which matters and is the bridge's: the state is rescaled to
    `initial_norm` BEFORE injection, so the activity measured on this pass is the
    activity produced by the rescaled state, not the raw one.
    """
    from atr_bridge import initial_state

    layer_end = _layer_end(model, layer_end)
    hook_point_read, hook_point_write = hook_points(layer_start, layer_end)

    if initial_norm is None:
        initial_norm = initial_state(model, prompt, layer_end).initial_norm
    initial_norm = float(initial_norm)

    wanted = {hook_point_read}
    for layer in range(layer_end + 1):
        wanted.add(f"blocks.{layer}.attn.hook_z")
        wanted.add(f"blocks.{layer}.hook_mlp_out")

    def step(model, r: torch.Tensor):
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
                    names_filter=lambda n: n in wanted,
                )
        finally:
            model.reset_hooks()

        activity = GridActivity(
            heads=_head_masses(cache, model, layer_end),
            mlp=_mlp_masses(cache, layer_end),
        )
        return cache[hook_point_read][0].clone(), activity

    step.prompt = prompt
    step.initial_norm = initial_norm
    step.hook_point_read = hook_point_read
    step.hook_point_write = hook_point_write
    return step


# --------------------------------------------------------------- the statistic

def ca_depth(activity: GridActivity, include_mlp: bool = True) -> float:
    """Centre of activity along depth: the mean layer index, weighted by write size.

    Returns a number between 0 and n_layers-1 saying where in the stack the work
    is being done. This is the axis with a real metric and it is the only one
    whose value is interpreted.

    `include_mlp` adds each block's MLP write to that block's total. It defaults
    to true because the MLPs are where every committed result in this repository
    was measured, and excluding them would measure a different system from the one
    the register describes. The alternative is reported alongside rather than
    chosen silently.
    """
    per_layer = activity.heads.sum(dim=1)
    if include_mlp:
        per_layer = per_layer + activity.mlp
    idx = torch.arange(per_layer.shape[0], dtype=torch.float64)
    total = per_layer.sum()
    if total <= 0:
        return float("nan")
    return float((idx * per_layer).sum() / total)


def ca_head(activity: GridActivity) -> float:
    """Centre of activity along the head axis. HAS NO INTERPRETATION.

    Head indices are arbitrary labels, so this number would change if someone
    permuted them. It exists only so that the head-shuffle control has something
    to act on, and the control's correct outcome is that every downstream
    statistic is unchanged by such a permutation.
    """
    per_head = activity.heads.sum(dim=0)
    idx = torch.arange(per_head.shape[0], dtype=torch.float64)
    total = per_head.sum()
    if total <= 0:
        return float("nan")
    return float((idx * per_head).sum() / total)


def ca_2d(activity: GridActivity) -> tuple[float, float]:
    """The full two-dimensional centroid, in the form Chao's equation 1 gives.

    Returned as (depth, head). The second component carries the caveat on
    `ca_head` and is reported only for completeness with the source.
    """
    return ca_depth(activity, include_mlp=False), ca_head(activity)


# ------------------------------------------------------------------- controls

def shuffle_layers(activity: GridActivity, generator: torch.Generator) -> GridActivity:
    """Chao's CAT-ELS control, on the axis that has a metric.

    Permuting layer labels should destroy any information the depth centroid
    carries. If the statistic survives this, its apparent power came from
    summarising activity size rather than from where in the stack that activity
    was, and it must be discarded.
    """
    perm = torch.randperm(activity.n_layers, generator=generator)
    return GridActivity(heads=activity.heads[perm], mlp=activity.mlp[perm])


def shuffle_heads(activity: GridActivity, generator: torch.Generator) -> GridActivity:
    """The control whose correct answer is known in advance.

    Head labels are arbitrary, so permuting them must leave `ca_depth` exactly
    unchanged. This is not a test of the model, it is a test of this module: a
    change here means the depth statistic is picking up a labelling that carries
    no information.
    """
    perm = torch.randperm(activity.n_heads, generator=generator)
    return GridActivity(heads=activity.heads[:, perm], mlp=activity.mlp)


# ------------------------------------------------------------- the validity gate

def separation_ratio(values: dict[str, list[float]]) -> float:
    """Between-group spread over within-group spread, one number per grouping.

    This is the change-to-drift ratio of Chao, Bakkum and Potter (2007), which
    they introduced because cultures reorganise continuously on their own and an
    absolute change therefore means nothing. Their reading: a value near 1 means
    the groups are indistinguishable from the ordinary scatter inside them.

    The same quantity already appears in this repository as register row C-07,
    arrived at independently: the gap between the two nearest end-state labels is
    2.874e-03 and the spread within one label is 3.319e-03, so the token labels
    themselves score 0.87 on this scale. Any statistic proposed as a replacement
    has to beat that, which is why the registered gate is 1.5.

    Groups of size below 2 contribute to the between-group term but cannot
    contribute a within-group spread, and are skipped in the denominator rather
    than counted as zero scatter, which would inflate the ratio.
    """
    import statistics

    centres, within = {}, []
    for name, vals in values.items():
        clean = [v for v in vals if v == v]      # drop NaN
        if not clean:
            continue
        centres[name] = statistics.fmean(clean)
        if len(clean) >= 2:
            within.extend(abs(v - centres[name]) for v in clean)

    if len(centres) < 2 or not within:
        return float("nan")

    grand = statistics.fmean(centres.values())
    between = statistics.fmean(abs(c - grand) for c in centres.values())
    scatter = statistics.fmean(within)
    if scatter <= 0:
        return float("inf")
    return between / scatter
