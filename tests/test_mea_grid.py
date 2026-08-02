"""Gates on the grid measurement, against real GPT-2 small.

The load-bearing test is the first one: `grid_step` must reproduce
`atr_bridge.make_atr_step`'s trajectory bit for bit. If it does not, every number
this module produces describes a different trajectory from the one the rest of
the project studies, and none of them are comparable to anything in the register.

The rest test the two controls in both directions, which the project's standing
rules require: a control that cannot fail is worse than no control.
"""

from __future__ import annotations

import pytest
import torch

import atr_bridge
import mea_grid


PROMPT = "The implications of quantum entanglement suggest that"


@pytest.fixture(scope="module")
def steps(tl_gpt2):
    """The bridge's step and the grid's step, built identically."""
    model = tl_gpt2
    plain = atr_bridge.make_atr_step(model, PROMPT)
    grid = mea_grid.grid_step(model, PROMPT, initial_norm=plain.initial_norm)
    return plain, grid


def test_grid_step_reproduces_the_bridge_trajectory_bit_for_bit(tl_gpt2, steps):
    """The acceptance bar. Not a tolerance: exactly zero deviation.

    A wider `names_filter` on the cache must not change the arithmetic that
    produces the next state. If this drifts, the measurement is being taken on a
    loop the project never ran.
    """
    model = tl_gpt2
    plain, grid = steps
    s0 = atr_bridge.initial_state(model, PROMPT)

    r_plain = s0.tensor.clone()
    r_grid = s0.tensor.clone()

    for _ in range(4):
        r_plain = plain(model, r_plain)
        r_grid, _activity = grid(model, r_grid)
        assert torch.equal(r_plain, r_grid), (
            "grid_step diverged from atr_bridge.make_atr_step; max abs diff "
            f"{(r_plain - r_grid).abs().max().item()}"
        )


def test_the_grid_is_twelve_by_twelve_and_finite(tl_gpt2, steps):
    model = tl_gpt2
    _plain, grid = steps
    s0 = atr_bridge.initial_state(model, PROMPT)
    _r, activity = grid(model, s0.tensor.clone())

    assert activity.heads.shape == (12, 12)
    assert activity.mlp.shape == (12,)
    assert torch.isfinite(activity.heads).all()
    assert torch.isfinite(activity.mlp).all()
    assert (activity.heads > 0).all(), "a head with exactly zero write is suspicious"


def test_head_masses_sum_to_the_attention_output(tl_gpt2, steps):
    """The mass really is the head's write, not a proxy.

    Summing the twelve per-head contributions of a layer must reconstruct that
    layer's attention output, which is what makes the centroid a centroid of the
    thing it claims to weigh. Checked on layer 6, where every committed result in
    this repository was measured. Tolerance rather than equality because the sum
    is taken in a different order from the fused einsum, and b_O is added once by
    the model rather than once per head.
    """
    model = tl_gpt2
    s0 = atr_bridge.initial_state(model, PROMPT)
    wanted = {"blocks.6.attn.hook_z", "blocks.6.hook_attn_out"}
    with torch.no_grad():
        _, cache = model.run_with_cache(PROMPT, names_filter=lambda n: n in wanted)

    z = cache["blocks.6.attn.hook_z"][0].double()
    w_o = model.blocks[6].attn.W_O.double()
    rebuilt = torch.bmm(z.permute(1, 0, 2), w_o).sum(dim=0) + model.blocks[6].attn.b_O.double()
    actual = cache["blocks.6.hook_attn_out"][0].double()

    rel = (rebuilt - actual).norm() / actual.norm()
    assert rel < 1e-5, f"per-head decomposition does not rebuild attn_out, rel err {rel}"


def test_ca_depth_lands_inside_the_stack(tl_gpt2, steps):
    model = tl_gpt2
    _plain, grid = steps
    s0 = atr_bridge.initial_state(model, PROMPT)
    _r, activity = grid(model, s0.tensor.clone())
    depth = mea_grid.ca_depth(activity)
    assert 0.0 <= depth <= 11.0


def test_shuffling_head_labels_leaves_the_depth_statistic_exactly_unchanged(tl_gpt2, steps):
    """The control with a known correct answer.

    Head indices are arbitrary. If permuting them moves the depth centroid, the
    statistic is reading a labelling that carries no information and the
    implementation is wrong. Exact equality, because summing a permuted row is the
    same sum in a different order and the depth centroid never touches the head
    index at all.
    """
    model = tl_gpt2
    _plain, grid = steps
    s0 = atr_bridge.initial_state(model, PROMPT)
    _r, activity = grid(model, s0.tensor.clone())

    before = mea_grid.ca_depth(activity)
    gen = torch.Generator().manual_seed(20260802)
    for _ in range(5):
        shuffled = mea_grid.shuffle_heads(activity, gen)
        after = mea_grid.ca_depth(shuffled)
        assert after == pytest.approx(before, abs=1e-12), (
            "permuting arbitrary head labels changed the depth centroid"
        )


def test_shuffling_layer_labels_does_change_the_depth_statistic(tl_gpt2, steps):
    """The other direction, so the pair of controls can both fail.

    Layer indices are not arbitrary. If permuting them leaves the centroid alone,
    the statistic is not reading depth and the layer-shuffle control could never
    detect anything.
    """
    model = tl_gpt2
    _plain, grid = steps
    s0 = atr_bridge.initial_state(model, PROMPT)
    _r, activity = grid(model, s0.tensor.clone())

    before = mea_grid.ca_depth(activity)
    gen = torch.Generator().manual_seed(20260802)
    moved = [
        abs(mea_grid.ca_depth(mea_grid.shuffle_layers(activity, gen)) - before)
        for _ in range(20)
    ]
    assert max(moved) > 0.1, (
        "permuting layer labels never moved the depth centroid; it is not reading depth"
    )


# ---------------------------------------------------------- separation ratio

def test_separation_ratio_is_near_zero_for_groups_that_are_not_separated():
    """Two groups drawn from the same spread must not look separated."""
    a = [0.0, 1.0, -1.0, 0.5, -0.5]
    b = [0.1, 0.9, -1.1, 0.4, -0.6]
    assert mea_grid.separation_ratio({"a": a, "b": b}) < 0.3


def test_separation_ratio_is_large_for_groups_that_are_separated():
    """Two tight groups far apart must look separated."""
    a = [0.00, 0.01, -0.01]
    b = [10.0, 10.01, 9.99]
    assert mea_grid.separation_ratio({"a": a, "b": b}) > 100


def test_separation_ratio_does_not_inflate_on_singleton_groups():
    """A group of one has no internal scatter and must not be read as having none.

    The frozen census has a basin with a single member (`solidarity`), so this is
    the real case rather than a hypothetical. Counting its scatter as zero would
    drive the denominator down and the ratio up.
    """
    with_singleton = mea_grid.separation_ratio({"a": [0.0, 1.0, -1.0], "b": [5.0]})
    assert with_singleton == with_singleton      # not NaN
    assert with_singleton < 100
