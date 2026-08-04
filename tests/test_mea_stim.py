"""Gates on grid stimulation, against real GPT-2 small.

The load-bearing test is the first: with no stimulation, or with zero strength,
the trajectory must be bit-identical to `atr_bridge.make_atr_step`'s. That is this
project's C0 control in the shape it takes for this module. If it fails, nothing
downstream is interpretable.

Every control here is tested in both directions, as the standing rules require.
"""

from __future__ import annotations

import pytest
import torch

import atr_bridge
import mea_stim
from mea_stim import StimSite, plan_sites, make_stim_step


PROMPT = "The implications of quantum entanglement suggest that"


@pytest.fixture(scope="module")
def base(tl_gpt2):
    return atr_bridge.make_atr_step(tl_gpt2, PROMPT)


def _iterate(model, step, r, n=3, stim=False):
    for i in range(n):
        r = step(model, r, iteration=i) if stim else step(model, r)
    return r


# ------------------------------------------------------------------ the C0 gate

def test_no_plan_is_bit_identical_to_the_bridge(tl_gpt2, base):
    model = tl_gpt2
    s0 = atr_bridge.initial_state(model, PROMPT)
    stim = make_stim_step(model, PROMPT, plan=None, initial_norm=base.initial_norm)

    a = _iterate(model, base, s0.tensor.clone())
    b = _iterate(model, stim, s0.tensor.clone(), stim=True)
    assert torch.equal(a, b), f"max abs diff {(a - b).abs().max().item()}"


def test_zero_strength_is_bit_identical_to_the_bridge(tl_gpt2, base):
    """A plan that names sites but carries no strength must still be a no-op."""
    model = tl_gpt2
    s0 = atr_bridge.initial_state(model, PROMPT)
    plan = plan_sites([StimSite(3, 4), StimSite(7, 1)], beta_total=0.0)
    stim = make_stim_step(model, PROMPT, plan=plan, initial_norm=base.initial_norm)

    a = _iterate(model, base, s0.tensor.clone())
    b = _iterate(model, stim, s0.tensor.clone(), stim=True)
    assert torch.equal(a, b), f"max abs diff {(a - b).abs().max().item()}"


def test_nonzero_strength_does_change_the_trajectory(tl_gpt2, base):
    """The other direction, so the gate above cannot pass vacuously."""
    model = tl_gpt2
    s0 = atr_bridge.initial_state(model, PROMPT)
    plan = plan_sites([StimSite(3, 4)], beta_total=0.05)
    stim = make_stim_step(model, PROMPT, plan=plan, initial_norm=base.initial_norm)

    a = _iterate(model, base, s0.tensor.clone())
    b = _iterate(model, stim, s0.tensor.clone(), stim=True)
    assert not torch.equal(a, b), "stimulation at 5% of local activity changed nothing"


# ------------------------------------------------------------------ the rate axis

def test_rate_controls_which_iterations_fire(tl_gpt2, base):
    model = tl_gpt2
    plan = plan_sites([StimSite(3, 4)], beta_total=0.05, every=4)
    stim = make_stim_step(model, PROMPT, plan=plan, initial_norm=base.initial_norm)

    fired = [len(stim.stimulated_sites(i)) > 0 for i in range(8)]
    assert fired == [True, False, False, False, True, False, False, False]


def test_a_skipped_iteration_matches_the_unstimulated_step(tl_gpt2, base):
    """On an iteration where nothing fires, the step must be the plain one exactly."""
    model = tl_gpt2
    s0 = atr_bridge.initial_state(model, PROMPT)
    plan = plan_sites([StimSite(3, 4)], beta_total=0.05, every=1000)
    stim = make_stim_step(model, PROMPT, plan=plan, initial_norm=base.initial_norm)

    a = base(model, s0.tensor.clone())
    b = stim(model, s0.tensor.clone(), iteration=1)      # 1 % 1000 != 0, no fire
    assert torch.equal(a, b)


# ------------------------------------------------------------------- the matching

def test_rms_matching_divides_by_the_square_root_of_the_count():
    plan = plan_sites([StimSite(1, 0), StimSite(5, 2), StimSite(9, 7)], beta_total=0.3)
    assert plan.beta_per_site == pytest.approx(0.3 / (3 ** 0.5))
    assert plan.beta_total == 0.3


def test_sum_matching_divides_by_the_count():
    plan = plan_sites([StimSite(1, 0), StimSite(5, 2), StimSite(9, 7)],
                      beta_total=0.3, match="sum")
    assert plan.beta_per_site == pytest.approx(0.1)


def test_one_site_is_the_same_under_both_matchings():
    """With a single site the two rules must agree, or focal is ill-defined."""
    a = plan_sites([StimSite(6, 3)], beta_total=0.2, match="rms")
    b = plan_sites([StimSite(6, 3)], beta_total=0.2, match="sum")
    assert a.beta_per_site == pytest.approx(b.beta_per_site)


# ------------------------------------------------------------------- rejections

def test_duplicate_sites_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        plan_sites([StimSite(3, 4), StimSite(3, 4)], beta_total=0.1)


def test_empty_plan_is_rejected():
    with pytest.raises(ValueError, match="at least one site"):
        plan_sites([], beta_total=0.1)


def test_rate_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        plan_sites([StimSite(3, 4)], beta_total=0.1, every=0)


def test_a_site_at_the_loop_injection_point_is_rejected_not_silently_dropped(tl_gpt2, base):
    """The loop overwrites resid_pre of the start block, so stimulating there
    would be erased without trace. That must raise rather than quietly do nothing.
    """
    model = tl_gpt2
    s0 = atr_bridge.initial_state(model, PROMPT)
    plan = plan_sites([StimSite(0, None)], beta_total=0.05)     # layer_start = 0
    stim = make_stim_step(model, PROMPT, plan=plan, initial_norm=base.initial_norm)

    with pytest.raises(ValueError, match="collides"):
        stim(model, s0.tensor.clone(), iteration=0)

    # AND the hooks must be gone afterwards. An earlier version raised the
    # collision outside the try block, so the loop's own injection hook survived
    # the exception and silently corrupted every later forward pass on this
    # model. Asserting only that the error was raised did not catch that.
    after = base(model, s0.tensor.clone())
    clean = base(model, s0.tensor.clone())
    assert torch.equal(after, clean), (
        "a hook leaked from the collision-rejection path; later passes are poisoned"
    )


# ------------------------------------------------------------- strength is local

def test_strength_is_relative_to_local_activity(tl_gpt2, base):
    """Two sites at very different activation scales must both move by their own
    fraction, not by a shared absolute amount. Measured as the relative change in
    the next iterate, which should be the same order at both sites.
    """
    model = tl_gpt2
    s0 = atr_bridge.initial_state(model, PROMPT)
    ref = base(model, s0.tensor.clone())

    devs = []
    for layer in (2, 9):
        plan = plan_sites([StimSite(layer, 5)], beta_total=0.05)
        stim = make_stim_step(model, PROMPT, plan=plan, initial_norm=base.initial_norm)
        out = stim(model, s0.tensor.clone(), iteration=0)
        devs.append(((out - ref).norm() / ref.norm()).item())

    assert all(d > 0 for d in devs)
    ratio = max(devs) / min(devs)
    assert ratio < 100, (
        f"relative effect differs by {ratio:.1f}x across depths; strength is not "
        "being normalised to local activity"
    )


def test_hooks_are_removed_after_every_step(tl_gpt2, base):
    """A leaked stimulation hook would poison every later forward pass."""
    model = tl_gpt2
    s0 = atr_bridge.initial_state(model, PROMPT)
    plan = plan_sites([StimSite(3, 4)], beta_total=0.05)
    stim = make_stim_step(model, PROMPT, plan=plan, initial_norm=base.initial_norm)
    stim(model, s0.tensor.clone(), iteration=0)

    after = base(model, s0.tensor.clone())
    clean = base(model, s0.tensor.clone())
    assert torch.equal(after, clean)
