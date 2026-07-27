"""
Tests for `controls.py` -- the four gates and the trajectory helper, against
real GPT-2 small.

A control that cannot fail is worse than no control: it launders a contaminated
run into a green tick. So every control here is tested in both directions -- it
passes on a healthy setup, and it FAILS when the specific defect it exists to
catch is injected. The injections live in this file as deliberately-broken
`atr_step` wrappers and a leaky `OjaPlasticity` subclass; nothing under test is
modified.

`OjaPlasticity`'s internals belong to test_plasticity.py. What is asserted here
is only the contract `controls.py` publishes: returned keys, verdicts, weight
restoration, hook hygiene, and the snapshot schedule DESIGN.md insists on.

n_iter is kept at 2-5 throughout. The library defaults (50/200) are a sweep
budget, not a test budget, and a real GPT-2 forward is ~20ms, so every call
passes an explicit small value.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import controls
from conftest import resolve, weight_snapshot
from controls import (
    _trajectory,
    c0_identity,
    c1_revert,
    c2_random_direction,
    c3_divergence_demo,
)
from plasticity import OjaPlasticity

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Local helpers: the defect injections
# --------------------------------------------------------------------------

def target_module(model, site: str):
    return resolve(model, site)


class LeakyOja(OjaPlasticity):
    """
    An `OjaPlasticity` whose hook is NOT side-effect free.

    This is failure mode #1 from `c0_identity`'s own docstring and DESIGN.md's
    table: "an in-place op on a captured tensor". The rule still computes
    nothing (mode="off", eta=0) -- only the act of observing perturbs the
    forward pass, which is exactly the contamination C0 exists to detect.
    """

    def _hook(self, module, inputs, output):
        super()._hook(module, inputs, output)
        with torch.no_grad():
            output.add_(1e-6)


def hook_sensitive_step(base_step, model, site: str):
    """
    An `atr_step` that takes a different trajectory whenever a forward hook is
    registered on the target module. Stands in for any coupling between hook
    installation and the dynamics, without touching plasticity.py.
    """
    module = target_module(model, site)

    def _step(m, r):
        out = base_step(m, r)
        if module._forward_hooks:
            out = out + 1e-5
        return out

    return _step


def leaky_state_step(base_step, trip_after: int, magnitude: float = 1e-4):
    """
    An `atr_step` carrying hidden state that survives `revert()`.

    C1's docstring names this: "hidden state accumulating elsewhere". The
    counter is external to the model, so restoring the weights bit-exactly is
    not enough to restore the trajectory -- which is the whole reason C1
    compares trajectories rather than just diffing the weight matrix.
    """
    state = {"calls": 0}

    def _step(m, r):
        state["calls"] += 1
        out = base_step(m, r)
        if state["calls"] > trip_after:
            out = out + magnitude
        return out

    return _step


def diverging_step(base_step, r0: torch.Tensor, k: int):
    """
    An `atr_step` that goes non-finite on its k-th call, resetting whenever it
    is handed r0 again. C3 restarts from `r0.clone()` for each mode, so the
    reset makes both the hebb and the oja pass trip at the same iteration.
    Used to exercise C3's early-break path, which GPT-2's renormalised dynamics
    will not reach on their own at a usable eta.
    """
    state = {"i": 0}

    def _step(m, r):
        if torch.equal(r, r0):
            state["i"] = 0
        state["i"] += 1
        out = base_step(m, r)
        if state["i"] >= k:
            return torch.full_like(out, float("nan"))
        return out

    return _step


def raising_step(base_step, trip_after: int):
    """An `atr_step` that raises partway through, as a real engine can."""
    state = {"calls": 0}

    def _step(m, r):
        state["calls"] += 1
        if state["calls"] > trip_after:
            raise RuntimeError("engine blew up")
        return base_step(m, r)

    return _step


# --------------------------------------------------------------------------
# C0 -- the gate
# --------------------------------------------------------------------------

def test_c0_identity_passes_on_healthy_setup(gpt2, r0, atr_step, site):
    """
    THE GATE, on 124M real parameters. README: "A green suite is not C0
    passing. Control C0 must still pass bit-exactly against real weights before
    anything here produces a result worth recording." This is that run.

    Failure means that at eta=0 with mode="off", merely installing the hooks
    moved the trajectory -- every plasticity result is then confounded with an
    instrumentation artefact and nothing downstream is interpretable.

    KNOWN INTERMITTENCY, recorded so a future flake is recognised rather than
    re-investigated from scratch: this assertion has been observed to fail twice
    on the development machine, with max deviations of 8.6e-05 and 6.3e-05. It
    could not be reproduced in 80 controlled repeats (0/20 hooked, 0/20
    unhooked, on a quiet machine and under CPU load, plus 16 cold fresh
    processes), and an unhooked-vs-unhooked control never differed -- so the
    hooks provably do not mutate anything. Best current explanation is
    nondeterministic parallel float reduction order in the CPU BLAS, not
    contamination by this repo. The documented contract is bit-exactness, so the
    assertion stays strict and there is no retry: if it fires, reproduce it
    against an unhooked-vs-unhooked control before suspecting `plasticity.py`.
    """
    out = c0_identity(gpt2, r0, atr_step, site, n_iter=3)

    assert set(out) == {"control", "bit_exact", "max_abs_deviation", "verdict"}
    assert out["control"] == "C0_identity"        # the dict names itself
    assert out["bit_exact"] is True
    assert out["max_abs_deviation"] == 0.0
    assert out["verdict"] == "PASS"


def test_c0_detects_a_side_effecting_hook(gpt2, r0, atr_step, site, monkeypatch):
    """
    Failure means C0 is a tautology: a hook that provably mutates the forward
    pass slipped through the gate. This is the test that makes a PASS from
    test_c0_identity_passes_on_healthy_setup worth anything.
    """
    monkeypatch.setattr(controls, "OjaPlasticity", LeakyOja)
    out = c0_identity(gpt2, r0, atr_step, site, n_iter=3)

    assert out["bit_exact"] is False
    assert out["max_abs_deviation"] > 0.0
    assert out["verdict"].startswith("FAIL")


def test_c0_detects_a_trajectory_that_depends_on_hook_presence(
    gpt2, r0, atr_step, site
):
    """
    Second injection, from the other side: the hook is clean but the engine
    behaves differently while it is installed. Failure means C0 only notices
    contamination that originates inside plasticity.py, and the class of
    coupling that lives in the caller's engine goes ungated.
    """
    step = hook_sensitive_step(atr_step, gpt2, site)
    out = c0_identity(gpt2, r0, step, site, n_iter=3)

    assert out["bit_exact"] is False
    assert out["max_abs_deviation"] > 0.0
    assert out["verdict"].startswith("FAIL")


def test_c0_leaves_no_residue(gpt2, r0, atr_step, site):
    """
    Failure means C0 itself contaminates the model it just cleared -- a leaked
    hook or a nudged weight would silently ride along into C1, C2 and every
    sweep after them.
    """
    module = target_module(gpt2, site)
    before = weight_snapshot(gpt2, site)
    hooks_before = len(module._forward_hooks)

    c0_identity(gpt2, r0, atr_step, site, n_iter=3)

    assert torch.equal(before, module.weight)
    assert len(module._forward_hooks) == hooks_before == 0


# --------------------------------------------------------------------------
# C1 -- reversibility
# --------------------------------------------------------------------------

def test_c1_revert_passes_and_is_not_vacuous(gpt2, r0, atr_step, site):
    """
    Two failures in one. bit_exact False means revert() did not restore the run
    on a real 3072x768 matrix. delta_frac_before_revert == 0 means the weights
    never moved, so the PASS proves nothing at all -- a reversibility control
    that reverts nothing is a control that cannot fail.
    """
    out = c1_revert(gpt2, r0, atr_step, site, eta=1e-4, n_iter=3)

    assert set(out) == {
        "control",
        "delta_frac_before_revert",
        "max_abs_deviation_after_revert",
        "bit_exact",
    }
    assert out["control"] == "C1_revert"
    assert out["bit_exact"] is True
    assert out["max_abs_deviation_after_revert"] == 0.0
    # The non-vacuity clause: the plastic run must actually have drifted.
    assert out["delta_frac_before_revert"] > 1e-6


def test_c1_restores_the_weight_matrix_itself(gpt2, r0, atr_step, site):
    """
    Failure means C1 reported a matching trajectory while leaving the weights
    perturbed -- the model handed to the next control is not the one C0 passed.
    """
    before = weight_snapshot(gpt2, site)
    c1_revert(gpt2, r0, atr_step, site, eta=1e-4, n_iter=3)

    assert torch.equal(before, target_module(gpt2, site).weight)
    assert len(target_module(gpt2, site)._forward_hooks) == 0


def test_c1_detects_hidden_state_surviving_the_revert(gpt2, r0, atr_step, site):
    """
    Failure means C1 only checks the weights and would sign off on a run whose
    state lives somewhere revert() cannot reach -- an RNG stream, a cache, a
    counter. That is precisely the class of bug the README assigns to C1.
    """
    # Trip after the baseline and plastic passes, i.e. only the post-revert run
    # sees the shifted dynamics.
    n_iter = 3
    step = leaky_state_step(atr_step, trip_after=2 * n_iter)
    out = c1_revert(gpt2, r0, step, site, eta=1e-4, n_iter=n_iter)

    assert out["bit_exact"] is False
    assert out["max_abs_deviation_after_revert"] > 0.0


# --------------------------------------------------------------------------
# C2 -- direction versus magnitude
# --------------------------------------------------------------------------

def test_c2_random_direction_contract(gpt2, r0, atr_step, site):
    """
    Failure means C2 cannot answer its question. If either arm's drift is zero
    the comparison is between a perturbation and nothing; if the two drifts are
    orders apart the "norm-matched" premise is broken and any difference in the
    final states is a magnitude effect wearing a direction costume.
    """
    out = c2_random_direction(gpt2, r0, atr_step, site, eta=1e-6, n_iter=3, seeds=(0, 1))

    assert set(out) == {
        "control",
        "seeds",
        "cos_oja_vs_random_final",
        "cos_per_seed",
        "cos_min",
        "cos_max",
        "delta_frac_oja",
        "delta_frac_random",
        "delta_frac_random_per_seed",
        "note",
    }
    assert out["control"] == "C2_random_direction"

    d_oja = out["delta_frac_oja"]
    d_rnd = out["delta_frac_random"]
    assert d_oja > 0.0
    assert d_rnd > 0.0
    # Loose: per-apply norms are matched exactly, but Oja's successive updates
    # are correlated and random ones are not, so the accumulated deltas differ
    # by a walk-versus-drift factor. Same order of magnitude is the claim; the
    # exact factor is pinned by the next test.
    assert 1 / 5 < d_oja / d_rnd < 5

    cos = out["cos_oja_vs_random_final"]
    assert isinstance(cos, float)
    assert math.isfinite(cos)
    assert -1.0 - 1e-5 <= cos <= 1.0 + 1e-5


def test_c2_random_arm_actually_varies_with_the_seed(gpt2, r0, atr_step, site):
    """
    THE POINT OF THE SEEDS ARGUMENT. Before it existed, both arms were
    constructed without a seed, so the random arm always drew the same matrix:
    C2 was a single sample dressed as a control, and "the direction matters"
    could not be told apart from "that one direction happened to differ".

    Failure means the seeds are not reaching the random draw and every entry in
    cos_per_seed is the same number -- a distribution with no spread, which is
    exactly the failure this argument was added to remove.
    """
    seeds = (0, 1, 2)
    out = c2_random_direction(gpt2, r0, atr_step, site, eta=1e-4, n_iter=2, seeds=seeds)

    assert out["seeds"] == list(seeds)
    assert len(out["cos_per_seed"]) == len(seeds)
    assert len(out["delta_frac_random_per_seed"]) == len(seeds)

    # Distinct draws: no two seeds may give the same trajectory.
    assert len(set(out["cos_per_seed"])) == len(seeds), out["cos_per_seed"]

    # The reported headline is the mean of the per-seed values, and the min/max
    # bracket them -- so a reader who quotes only the headline is quoting
    # something the spread can contradict, and can see that it can.
    assert out["cos_oja_vs_random_final"] == pytest.approx(
        sum(out["cos_per_seed"]) / len(seeds)
    )
    assert out["cos_min"] == min(out["cos_per_seed"])
    assert out["cos_max"] == max(out["cos_per_seed"])
    assert out["cos_min"] <= out["cos_oja_vs_random_final"] <= out["cos_max"]


def test_c2_is_reproducible_for_the_same_seeds(gpt2, r0, atr_step, site):
    """
    Failure means C2's verdict moves between runs of the same configuration,
    so a rerun cannot confirm or refute the previous one.
    """
    kw = dict(eta=1e-4, n_iter=2, seeds=(0, 1))
    first = c2_random_direction(gpt2, r0, atr_step, site, **kw)
    second = c2_random_direction(gpt2, r0, atr_step, site, **kw)
    assert first["cos_per_seed"] == second["cos_per_seed"]
    assert first["delta_frac_random_per_seed"] == second["delta_frac_random_per_seed"]


def test_c2_rejects_an_empty_seed_list(gpt2, r0, atr_step, site):
    """Failure means C2 can be run with no random arm at all and still return."""
    with pytest.raises(ValueError, match="at least one seed"):
        c2_random_direction(gpt2, r0, atr_step, site, eta=1e-6, n_iter=2, seeds=())


def test_c2_cumulative_delta_frac_of_the_two_arms_is_not_norm_matched(
    gpt2, r0, atr_step, site
):
    """
    The norm match is PER APPLY. `c2_random_direction` applies every iteration,
    so the two arms' *cumulative* delta_frac must not match: Oja steps point the
    same way and add linearly, independent random steps add in quadrature,
    giving random/oja ~ 1/sqrt(n_applied).

    Recorded as a test because the gap looks exactly like the defect README
    lists as #2 and is not one. Deleting the fix would show up in
    test_plasticity's per-apply norm match, not here; "fixing" this one would
    break the control.
    """
    before = weight_snapshot(gpt2, site)
    res = c2_random_direction(gpt2, r0, atr_step, site, eta=1e-6, n_iter=2)

    assert res["control"] == "C2_random_direction"
    assert res["delta_frac_oja"] > 0 and res["delta_frac_random"] > 0
    assert math.isfinite(res["cos_oja_vs_random_final"])
    ratio = res["delta_frac_random"] / res["delta_frac_oja"]
    assert 0.6 < ratio < 0.85          # 1/sqrt(2) = 0.707; measured 0.717

    # Both arms must hand the weights back -- the second arm has to start from
    # the original matrix or the comparison is between two conditions that
    # differ by more than the update direction.
    assert torch.equal(before, target_module(gpt2, site).weight)


def test_c2_restores_the_original_weights(gpt2, r0, atr_step, site):
    """
    Failure means C2 leaves the model carrying whichever of its two arms ran
    last -- the deciding control would hand a corrupted model to the sweep.
    """
    before = weight_snapshot(gpt2, site)
    c2_random_direction(gpt2, r0, atr_step, site, eta=1e-6, n_iter=3)

    assert torch.equal(before, target_module(gpt2, site).weight)
    assert len(target_module(gpt2, site)._forward_hooks) == 0


# --------------------------------------------------------------------------
# C3 -- the divergence demonstration
# --------------------------------------------------------------------------
# Measured on GPT-2 small at this site, eta 1e-6 to 1e-3, n_iter 3-8: the Hebb
# trace grows monotonically and super-linearly, while the Oja trace saturates --
# at eta=1e-3 it *falls* after the first apply and then sits at ~0.17, which is
# the decay term pulling the weight to Oja's fixed point. That is the C3 claim.
#
# Two things a small-init toy network does not predict, both consequences of
# real weight scale:
#   * Oja's delta_frac is ~100x LARGER than Hebb's in absolute terms at every
#     stable eta. ||W||_F = 165 here, so W<y y^T> dominates <x y^T> instead of
#     correcting it. "Hebb diverges, Oja does not" is a statement about growth
#     over the run, not about which update is bigger at any one step -- so the
#     pointwise `hebb >= oja` form of the claim is FALSE on real weights and is
#     not asserted below.
#   * The claim has an upper eta bound. At eta >= 3e-3 the Oja arm itself
#     explodes because eta*lambda_max of <y y^T> leaves the stable range.
#     eta=1e-3 is inside the window; 3e-3 is not.

C3_ETA = 1e-3
C3_N_ITER = 4


def test_c3_hebb_diverges_faster_than_oja(gpt2, r0, atr_step, site):
    """
    README: "`mode='hebb'` is included so you can produce the divergence figure
    (control C3) rather than asserting it." This is that figure, in assertion
    form, on real weights. Failure means the decay term is not doing its job --
    the rule this repo argues for is not distinguishable from the one it is
    supposed to improve on, and the Oja-vs-Hebb framing has no support.
    """
    out = c3_divergence_demo(gpt2, r0, atr_step, site, eta=C3_ETA, n_iter=C3_N_ITER)

    assert set(out) == {"control", "delta_frac_traces"}
    assert out["control"] == "C3_divergence_demo"

    traces = out["delta_frac_traces"]
    assert set(traces) == {"hebb", "oja"}
    for name, trace in traces.items():
        assert isinstance(trace, list), name
        assert trace, name
        assert all(isinstance(v, float) for v in trace), name

    hebb, oja = traces["hebb"], traces["oja"]
    assert len(hebb) == len(oja) == C3_N_ITER
    assert all(v > 0 for v in hebb) and all(v > 0 for v in oja)

    # Hebb: monotone and compounding -- no fixed point.
    assert all(hebb[i] < hebb[i + 1] for i in range(len(hebb) - 1))
    hebb_growth = hebb[-1] / hebb[0]
    assert hebb_growth > C3_N_ITER - 1        # super-linear accumulation

    # Oja: bounded. The decay term pulls it back after the first apply, and it
    # never exceeds where it started.
    oja_growth = oja[-1] / oja[0]
    assert oja_growth < 1.0
    assert max(oja) / oja[0] <= 1.0 + 1e-6
    assert oja[-1] < 10.0

    # The gap widens rather than being a constant offset.
    assert hebb_growth > 4 * oja_growth


def test_c3_breaks_early_on_nonfinite_without_raising(gpt2, r0, atr_step, site):
    """
    Failure means C3 either crashes on the divergence it exists to display, or
    silently keeps iterating on NaN and reports a trace of garbage as if the run
    had completed.
    """
    k = 3
    n_iter = 6
    step = diverging_step(atr_step, r0, k=k)
    out = c3_divergence_demo(gpt2, r0, step, site, eta=1e-6, n_iter=n_iter)

    traces = out["delta_frac_traces"]
    for name, trace in traces.items():
        assert len(trace) == k, name          # broke on the non-finite state
        assert len(trace) < n_iter, name      # ... which is short of the budget
        assert all(math.isfinite(v) for v in trace), name


def test_c3_restores_the_original_weights(gpt2, r0, atr_step, site):
    """
    Failure means the demonstration is destructive: C3 runs at
    max_delta_frac=1e9 with no ceiling, so a missed revert() hands the next
    control -- and, with a session-scoped model, every later test -- a weight
    matrix wrecked by design.
    """
    before = weight_snapshot(gpt2, site)
    c3_divergence_demo(gpt2, r0, atr_step, site, eta=C3_ETA, n_iter=C3_N_ITER)

    assert torch.equal(before, target_module(gpt2, site).weight)
    assert len(target_module(gpt2, site)._forward_hooks) == 0


# --------------------------------------------------------------------------
# _trajectory -- the helper every control is built on
# --------------------------------------------------------------------------

def test_trajectory_returns_n_iter_states(gpt2, r0, atr_step):
    """
    Failure means every control's zip() silently truncates to the shorter
    trajectory and compares fewer iterations than it claims to.
    """
    for n_iter in (2, 3, 5):
        states = _trajectory(gpt2, r0, atr_step, n_iter)
        assert len(states) == n_iter
        assert all(torch.is_tensor(s) for s in states)


def test_trajectory_records_every_iteration_not_a_subsample(gpt2, r0, atr_step):
    """
    DESIGN.md, "Snapshot schedule": an even-only or otherwise aliased schedule
    samples a period-2 orbit at one phase and makes the oscillation invisible.
    That is how F9 stayed hidden for months. Failure here means the helper is
    aliasing, so a period-2 attractor reads as a fixed point.
    """
    n_iter = 4
    states = _trajectory(gpt2, r0, atr_step, n_iter)

    # Reference: the same loop, unrolled, recording consecutive iterations.
    r = r0.clone()
    reference = []
    for _ in range(n_iter):
        r = atr_step(gpt2, r)
        reference.append(r.detach().clone())

    assert len(states) == n_iter == len(reference)
    for i, (got, want) in enumerate(zip(states, reference)):
        assert torch.equal(got, want), f"state {i} is not iteration {i}"

    # Distinct objects, not one buffer aliased n_iter times.
    assert len({id(s) for s in states}) == n_iter
    for i in range(n_iter - 1):
        assert states[i] is not states[i + 1]
        assert states[i].data_ptr() != states[i + 1].data_ptr()


def test_trajectory_states_are_detached_clones(gpt2, r0, atr_step, site):
    """
    Failure means a caller that touches a returned state mutates the recorded
    history or the model itself -- and C0's bit-exactness check would then be
    comparing tensors it had already corrupted.
    """
    states = _trajectory(gpt2, r0, atr_step, 3)
    assert all(not s.requires_grad for s in states)

    first_before = states[0].clone()
    later_before = [s.clone() for s in states[1:]]
    weights_before = weight_snapshot(gpt2, site)

    with torch.no_grad():
        states[0].add_(1.0)

    for got, want in zip(states[1:], later_before):
        assert torch.equal(got, want)
    assert torch.equal(weights_before, target_module(gpt2, site).weight)

    # And the model still produces the same first state from r0: the mutation
    # touched a clone, not anything the forward pass reads.
    assert torch.equal(atr_step(gpt2, r0), first_before)


def test_trajectory_plast_none_runs_clean(gpt2, r0, atr_step, site):
    """
    Failure means the no-plasticity path touches the model, which would make
    every control's baseline arm the thing that perturbs the comparison.
    """
    before = weight_snapshot(gpt2, site)
    states = _trajectory(gpt2, r0, atr_step, 3, plast=None, apply_every=2)

    assert len(states) == 3
    assert torch.equal(before, target_module(gpt2, site).weight)
    assert len(target_module(gpt2, site)._forward_hooks) == 0


@pytest.mark.parametrize(
    "n_iter, apply_every, expected",
    [
        (6, 1, 6),    # every iteration
        (6, 2, 3),    # divides evenly
        (6, 3, 2),
        (7, 3, 2),    # does not divide: the trailing partial window is dropped
        (5, 2, 2),
        (5, 5, 1),
        (4, 9, 0),    # cadence longer than the run: never applied
    ],
)
def test_trajectory_apply_every_controls_cadence(
    gpt2, r0, atr_step, site, n_iter, apply_every, expected
):
    """
    Failure means cadence is not the timescale-separation knob DESIGN.md's
    "Cadence" note treats it as, and any k-sweep would be sweeping nothing.
    """
    plast = OjaPlasticity(gpt2, site=site, eta=1e-6, mode="oja").install()
    try:
        states = _trajectory(
            gpt2, r0, atr_step, n_iter, plast=plast, apply_every=apply_every
        )
        assert len(states) == n_iter
        assert plast.report()["n_applied"] == expected
    finally:
        plast.revert()
        plast.remove()

    assert len(target_module(gpt2, site)._forward_hooks) == 0


# --------------------------------------------------------------------------
# Cleanup when the engine raises
# --------------------------------------------------------------------------

def test_c0_cleans_up_when_the_engine_raises(gpt2, r0, atr_step, site):
    """
    Failure means a crashed C0 leaves its hook on the model. C0 uses
    `with OjaPlasticity(...)`, so this holds -- it is the reference behaviour
    the other three controls are measured against below.
    """
    module = target_module(gpt2, site)
    before = weight_snapshot(gpt2, site)

    with pytest.raises(RuntimeError):
        c0_identity(gpt2, r0, raising_step(atr_step, 4), site, n_iter=3)

    assert len(module._forward_hooks) == 0
    assert torch.equal(before, module.weight)


@pytest.mark.parametrize(
    "control, kwargs",
    [
        (c1_revert, {"eta": 1e-4}),
        (c2_random_direction, {"eta": 1e-4}),
        (c3_divergence_demo, {"eta": 1e-4}),
    ],
    ids=["c1", "c2", "c3"],
)
def test_controls_clean_up_when_the_engine_raises(
    gpt2, r0, atr_step, site, control, kwargs
):
    """
    Failure means a control that crashes mid-run hands the next control a model
    with a live hook and drifted weights. C0 would then be gating a model that
    no longer exists, and DESIGN.md's ordered sweep ("C0, C1 ... Gate") is
    running each stage on the wreckage of the last.
    """
    module = target_module(gpt2, site)
    before = weight_snapshot(gpt2, site)

    with pytest.raises(RuntimeError):
        control(gpt2, r0, raising_step(atr_step, 4), site, n_iter=3, **kwargs)

    assert len(module._forward_hooks) == 0
    assert torch.equal(before, module.weight)


# --------------------------------------------------------------------------
# Module entry point
# --------------------------------------------------------------------------

def test_running_controls_directly_refuses():
    """
    Failure means `python controls.py` runs something. The module deliberately
    has no default loop -- inventing one here would put an untested ATR
    reimplementation between the engine and the results, which is the single
    thing README.md's Architecture section forbids.
    """
    proc = subprocess.run(
        [sys.executable, "controls.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode != 0
    assert proc.stdout == ""
    assert "pass in your own atr_step" in proc.stderr
    assert "no default loop" in proc.stderr
    assert "Running the controls" in proc.stderr
