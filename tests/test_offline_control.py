"""
Tests for `offline_control` -- the arm that makes a closed-loop result mean
something.

Three of these are the point of the file:

`test_eta_zero_arms_are_bit_identical` is the harness's own C0. If the two arms
do not produce the same matrix when the rule is turned off, the harness itself
is introducing a difference and every divergence it later reports is that
difference plus noise. Exact by construction on any hardware -- `0 x anything`
is `0` -- so this one is `torch.equal` and stays that way.

`test_a_site_the_loop_does_not_route_through` is the detection limit. With the
feedback path severed the arms must not diverge, and how much they still do is
the floor beneath which no claim about feedback can be made. It asserts a bound
rather than a value, because the quantity is float accumulation order and its
magnitude is hardware-dependent.

`test_verify_flags_each_matched_axis` is the guard on the guard. The verifier is
the only thing standing between "the arms differed" and "the arms differed
because of feedback", and a verifier that cannot fail is worse than none: it
launders a mismatched comparison into a checked one. Every axis is broken in
turn and the verifier has to catch exactly that axis.

The measured eta > 0 divergence lives in
`test_eta_positive_divergence_is_measured_and_reported`. It asserts almost
nothing about the magnitude on purpose. A near-zero difference would mean
feedback contributes nothing detectable at this eta -- a real finding about this
substrate, and one this suite must not be able to turn red.

Everything runs against real GPT-2 small on the TransformerLens stack, because
that is the stack the ATR loop runs on and the offline arm has to attach to the
same one.

Run with:  ATR_REQUIRE_MODEL=1 .venv/bin/pytest tests/test_offline_control.py -q
"""

from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

import offline_control
from atr_bridge import initial_state, make_atr_step
from offline_control import (
    MATCHED_AXES,
    ActivationRecord,
    ArmConfig,
    ArmsMismatchError,
    compare_states,
    compare_weights,
    installed_weight,
    record_frozen_activations,
    replay_offline,
    run_closed_loop_arm,
    run_matched_arms,
    verify_arms_matched,
)
from plasticity import OjaPlasticity

# The `Divine` prompt: the one EXP-001 is written around, 9 tokens plus BOS.
PROMPT = "The cat sat on the mat and then the"
SITE = "blocks.6.mlp"
LAYER_END = 11

# A GPT-2 forward on this stack costs ~0.2-2s depending on machine load, and
# every arm is n_steps of them. Six steps is enough for the weight to move by
# something measurable at the etas below and cheap enough to run five times in
# one suite. The headline measurement uses more; see `matched_run`.
N_STEPS = 6

# The eta the headline number is measured at. Mid-ladder from EXP_001_SPEC's
# 1e-6 .. 1e-4, chosen before any arm was run and not revisited afterwards --
# tuning eta until the arms diverge would be manufacturing the finding.
ETA = 1e-5


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tl_loop(tl_gpt2):
    """`(r0, atr_step)` for the full-stack loop, layers 0 -> 11.

    Built once: `initial_state` is a forward pass and `make_atr_step` is another
    when `initial_norm` is not supplied.
    """
    state = initial_state(tl_gpt2, PROMPT, layer_end=LAYER_END)
    step = make_atr_step(
        tl_gpt2, PROMPT, layer_start=0, layer_end=LAYER_END,
        initial_norm=state.initial_norm,
    )
    return state.tensor, step


@pytest.fixture(scope="session")
def shallow_loop(tl_gpt2):
    """`(r0, atr_step)` for a loop that reads at layer 3, below the site.

    Blocks 4-11 still execute -- every forward runs the whole model -- but
    nothing they compute reaches the next iterate, because the state is read at
    `blocks.3.hook_resid_post`. So `blocks.6.mlp` is a site the loop does not
    route through: the state-feedback path from W to the next x is severed
    while everything else stays exactly as it was.
    """
    state = initial_state(tl_gpt2, PROMPT, layer_end=3)
    step = make_atr_step(
        tl_gpt2, PROMPT, layer_start=0, layer_end=3,
        initial_norm=state.initial_norm,
    )
    return state.tensor, step


@pytest.fixture(scope="session")
def matched_run(tl_gpt2, tl_loop):
    """The headline eta > 0 run, done once and read by several tests."""
    r0, step = tl_loop
    return run_matched_arms(
        tl_gpt2, r0, step, SITE, N_STEPS, eta=ETA, rerun_frozen=True,
    )


@pytest.fixture(scope="session")
def frozen_record(tl_gpt2, tl_loop):
    """One frozen recording, reused by the recorder tests."""
    r0, step = tl_loop
    return record_frozen_activations(tl_gpt2, r0, step, SITE, N_STEPS)


@pytest.fixture(autouse=True)
def _tl_weight_unchanged(request):
    """
    The TransformerLens model is session-scoped and every arm here writes to it.

    `conftest._target_weight_unchanged` guards the HuggingFace fixture only, so
    this is its counterpart for `tl_gpt2`. A test that leaves `blocks.6.mlp`'s
    `W_out` modified does not fail on its own -- it silently re-runs every later
    test against a different model, which is the failure mode hardest to find
    afterwards.
    """
    if "tl_gpt2" not in request.fixturenames:
        yield
        return
    model = request.getfixturevalue("tl_gpt2")
    w = model.blocks[6].mlp.W_out
    before = w.detach().clone()
    yield
    assert torch.equal(w, before), "blocks.6.mlp.W_out left modified for later tests"


# --------------------------------------------------------------------------
# The verifier -- no model needed, and the most important thing here
# --------------------------------------------------------------------------

def _closed_cfg(**over) -> ArmConfig:
    base = dict(
        arm="closed_loop", feedback=True, site=SITE, mode="oja", eta=1e-5,
        max_delta_frac=0.05, transposed=False, dtype="torch.float32", seed=0,
        rng_state_sha256="rng", w0_sha256="w0", centring="absent",
        n_steps=4, apply_every=1, n_updates=4, n_samples=4,
        samples_per_update=(1, 1, 1, 1), sample_order=(0, 1, 2, 3),
        y_source="live", store_dtype="torch.float32",
    )
    base.update(over)
    return ArmConfig(**base)


def _offline_cfg(**over) -> ArmConfig:
    return _closed_cfg(**{
        "arm": "offline", "feedback": False, "y_source": "recorded", **over
    })


def test_matched_axes_covers_the_prior_art_table():
    """Every row of PRIOR_ART.md's "must match" table has to be an axis here.

    The table is the specification. If a row is not in `MATCHED_AXES` the
    verifier does not check it, and an unchecked axis is exactly how a
    mismatched comparison gets reported as a matched one.
    """
    axes = {name for name, _ in MATCHED_AXES}
    # eta and the ceiling / update count / sample order / batching /
    # initial weight and seed / centring, in this module's spelling.
    required = {
        "eta", "max_delta_frac",
        "n_updates",
        "sample_order",
        "samples_per_update",
        "w0_sha256", "seed",
        "centring",
    }
    assert required <= axes, sorted(required - axes)


def test_every_matched_axis_is_a_field_on_ArmConfig():
    """The axes are first-class fields, not a free-text note.

    A verifier that reads axes off a dict it built itself proves nothing; these
    have to be attributes the arm runners filled in from the rule object.
    """
    cfg = _closed_cfg()
    for name, _ in MATCHED_AXES:
        assert hasattr(cfg, name), name
    assert set(cfg.axis_values()) == {name for name, _ in MATCHED_AXES}

    # And nothing on the config escapes the check by accident. Only the three
    # fields that distinguish the arms may sit outside the table; a field added
    # later without a decision about it shows up here rather than in a result.
    unchecked = {f.name for f in fields(cfg)} - set(cfg.axis_values())
    assert unchecked == {"arm", "feedback", "y_source"}, sorted(unchecked)


def test_verify_passes_on_two_properly_matched_arms():
    v = verify_arms_matched(_closed_cfg(), _offline_cfg())
    assert v["ok"] is True
    assert v["mismatched"] == []
    assert all(a["match"] for a in v["axes"])
    assert "MATCHED" in v["verdict"]


# One deliberately-wrong value per axis. Each must be caught, and each must be
# caught ALONE -- a verifier that fails everything whenever anything differs is
# no more use than one that fails nothing.
_BREAKAGES = {
    "site": "blocks.7.mlp",
    "mode": "hebb",
    "eta": 2e-5,
    "max_delta_frac": 0.10,
    "transposed": True,
    "dtype": "torch.float64",
    "n_steps": 5,
    "apply_every": 2,
    "n_updates": 3,
    "n_samples": 3,
    "samples_per_update": (2, 2),
    "sample_order": (3, 2, 1, 0),
    "w0_sha256": "a-different-starting-matrix",
    "seed": 1,
    "rng_state_sha256": "a-different-draw",
    "centring": "centre=True",
    "store_dtype": "torch.float16",
}


@pytest.mark.parametrize("axis", [name for name, _ in MATCHED_AXES])
def test_verify_flags_each_matched_axis(axis):
    """Break one axis; the verifier must fail, and name that axis and no other."""
    assert axis in _BREAKAGES, f"no breakage defined for the {axis!r} axis"
    v = verify_arms_matched(_closed_cfg(), _offline_cfg(**{axis: _BREAKAGES[axis]}))
    assert v["ok"] is False
    assert [a["axis"] for a in v["mismatched"]] == [axis]
    assert "MISMATCHED" in v["verdict"]


def test_verify_rejects_two_arms_that_are_not_one_of_each():
    """Comparing a closed arm against a second closed arm measures nothing.

    Every axis in the table matches, so the row-by-row check passes -- and the
    comparison is still meaningless. The structural check is what catches it.
    """
    v = verify_arms_matched(_closed_cfg(), _closed_cfg())
    assert v["ok"] is False
    assert v["mismatched"] == []
    assert v["structural_problems"]


def test_verify_reports_why_each_axis_is_checked():
    """Each axis carries the reason from PRIOR_ART.md, so a failure explains itself."""
    v = verify_arms_matched(_closed_cfg(), _offline_cfg())
    for a in v["axes"]:
        assert a["why"], a["axis"]
        assert isinstance(a["closed"], str) and isinstance(a["offline"], str)


# --------------------------------------------------------------------------
# The recorder
# --------------------------------------------------------------------------

def test_recording_does_not_perturb_the_frozen_trajectory(tl_gpt2, tl_loop):
    """C0 for the recorder: capturing must be bit-exactly side-effect free.

    The offline arm replays activations from this run. If installing the capture
    hooks moved the trajectory at all, the recording is of a loop the closed arm
    never ran, and the two arms differ by the instrument as well as by feedback.
    """
    r0, step = tl_loop
    r = r0.clone()
    baseline = []
    for _ in range(N_STEPS):
        r = step(tl_gpt2, r)
        baseline.append(r.clone())

    rec = record_frozen_activations(tl_gpt2, r0, step, SITE, N_STEPS)
    assert len(rec.states) == N_STEPS
    for i, (a, b) in enumerate(zip(baseline, rec.states)):
        assert torch.equal(a, b), f"recording perturbed the trajectory at step {i}"


def test_recorder_stores_every_sample_in_order_and_never_subsamples(frozen_record):
    """One sample per iteration, indexed by iteration, nothing dropped.

    `step_index` is the sample-order axis. If the recorder ever thinned a long
    run to save memory this would stop being `range(n_steps)` and the arms would
    stop being matched -- which is why the memory guard raises instead.
    """
    rec = frozen_record
    assert rec.n_samples == N_STEPS
    assert rec.step_index == tuple(range(N_STEPS))
    assert [rec.samples_for_step(i) for i in range(N_STEPS)] == [[i] for i in range(N_STEPS)]


def test_recorder_captures_the_two_tensors_the_rule_consumes(tl_gpt2, frozen_record):
    """x is post-activation into W_out, y is the MLP output, both already 2-D.

    Shapes and the relation `y = x W_out + b_out` are checked rather than
    assumed: they are the whole content of the claim that the recording is of
    what the rule would have seen.

    `allclose`, deliberately, and only here: the unfused `x @ W + b` is not the
    tensor the site computed, it is a different rounding of the same maths.
    `test_recompute_y_reproduces_the_sites_own_forward_bit_exactly` is the exact
    version, and it matters there because that arithmetic feeds the replay.
    """
    rec = frozen_record
    mlp = tl_gpt2.blocks[6].mlp
    d_mlp, d_model = mlp.W_out.shape
    for x, y in zip(rec.x, rec.y):
        assert x.shape[1] == d_mlp and y.shape[1] == d_model
        assert x.shape[0] == y.shape[0]
        assert x.device.type == "cpu" and y.device.type == "cpu"
        assert torch.allclose(y, x @ mlp.W_out + mlp.b_out, atol=1e-4)


def test_recorder_reports_its_own_memory_footprint(frozen_record):
    """N steps of (seq, 3072) float32 adds up; the record says how much."""
    rec = frozen_record
    expected = sum(t.numel() * t.element_size() for t in rec.x + rec.y)
    assert rec.bytes_stored == expected
    assert rec.precision == {}, "float32 storage should record no precision loss"


def test_memory_budget_refuses_rather_than_subsampling(tl_gpt2, tl_loop):
    """Over budget is an error with the arithmetic in it, never a thinned recording.

    Silently dropping steps would break the sample-order axis without breaking
    anything the verifier can see, because the offline arm would then be matched
    to its own thinned recording.
    """
    r0, step = tl_loop
    with pytest.raises(MemoryError) as exc:
        record_frozen_activations(
            tl_gpt2, r0, step, SITE, 1000, memory_budget_bytes=1024,
        )
    assert "subsample" in str(exc.value)


def test_float16_storage_records_the_precision_it_lost(tl_gpt2, tl_loop):
    """Half precision is offered, and it never comes for free or unrecorded.

    The offline arm's updates carry this rounding and the closed arm's do not,
    so the number has to travel with the record rather than being inferred from
    the format later.
    """
    r0, step = tl_loop
    rec = record_frozen_activations(
        tl_gpt2, r0, step, SITE, 2, store_dtype=torch.float16,
    )
    assert rec.store_dtype == torch.float16
    assert rec.x[0].dtype == torch.float16
    assert rec.precision["max_rel_round_trip_error"] > 0.0
    assert rec.precision["max_rel_round_trip_error"] < 1e-2
    assert "lossily" in rec.precision["note"]


def test_a_float16_recording_is_refused_as_an_offline_arm(tl_gpt2, tl_loop):
    """Half precision saves memory and costs the match; the runner says so.

    The offline arm's updates would carry rounding the closed arm's never saw,
    which is a difference that is not feedback. The memory escape hatch that
    keeps the arms matched is fewer steps, not lower precision, and the verifier
    is where that gets enforced rather than in a docstring nobody reads.
    """
    r0, step = tl_loop
    with pytest.raises(ArmsMismatchError) as exc:
        run_matched_arms(tl_gpt2, r0, step, SITE, 2, eta=ETA,
                         store_dtype=torch.float16, rerun_frozen=False)
    assert "store_dtype" in {a["axis"] for a in exc.value.verification["mismatched"]}


# --------------------------------------------------------------------------
# The two arms
# --------------------------------------------------------------------------

def test_a_single_step_replays_to_the_same_update_bit_exactly(tl_gpt2, tl_loop):
    """One step of each arm must land on the same matrix, bit for bit.

    Before the first `apply()` there is nothing for feedback to act through:
    both arms see the same weights and therefore the same x and the same y. So
    this is not a statement about feedback -- it is the proof that the replay
    path feeds `OjaPlasticity` exactly what the live hook feeds it. Any
    difference here is a defect in the recorder or the replayer, and would
    contaminate every later divergence number.
    """
    r0, step = tl_loop
    closed = run_closed_loop_arm(tl_gpt2, r0, step, SITE, 1, eta=ETA)
    rec = record_frozen_activations(tl_gpt2, r0, step, SITE, 1)
    offline = replay_offline(tl_gpt2, rec, eta=ETA)

    assert closed.config.n_updates == offline.config.n_updates == 1
    assert torch.equal(closed.weight, offline.weight)
    assert not torch.equal(closed.weight, closed.w0), "eta > 0 but nothing moved"


def test_arms_restore_the_weight_and_leave_no_hooks(tl_gpt2, tl_loop):
    """Each arm hands the model back exactly as it found it.

    `run_closed_loop_arm` writes to `W_out` on every apply and `replay_offline`
    does too, without running a single forward. Both revert in a `finally`; the
    final matrix comes back in the result rather than being left installed.
    """
    r0, step = tl_loop
    before = tl_gpt2.blocks[6].mlp.W_out.detach().clone()

    closed = run_closed_loop_arm(tl_gpt2, r0, step, SITE, 2, eta=1e-4)
    assert torch.equal(tl_gpt2.blocks[6].mlp.W_out, before)

    rec = record_frozen_activations(tl_gpt2, r0, step, SITE, 2)
    replay_offline(tl_gpt2, rec, eta=1e-4)
    assert torch.equal(tl_gpt2.blocks[6].mlp.W_out, before)
    assert not torch.equal(closed.weight, before)

    # TransformerLens's own bookkeeping, and torch's. The capture hooks are
    # plain `register_forward_hook`s, so they are invisible to the first and
    # would be left behind by a `reset_hooks()`-shaped teardown; the second is
    # where they actually live.
    left = {n: len(hp.fwd_hooks) for n, hp in tl_gpt2.hook_dict.items() if hp.fwd_hooks}
    assert left == {}
    for name in ("blocks.6.mlp.hook_post", "blocks.6.hook_mlp_out"):
        assert len(tl_gpt2.hook_dict[name]._forward_hooks) == 0, name


def test_installed_weight_restores_on_the_way_out_even_when_the_body_raises(tl_gpt2):
    """The frozen re-runs install an experimental matrix; it must never stick."""
    before = tl_gpt2.blocks[6].mlp.W_out.detach().clone()
    other = before + 1.0
    with pytest.raises(ZeroDivisionError):
        with installed_weight(tl_gpt2, SITE, other):
            assert torch.equal(tl_gpt2.blocks[6].mlp.W_out, other)
            raise ZeroDivisionError
    assert torch.equal(tl_gpt2.blocks[6].mlp.W_out, before)


def test_replay_rejects_an_unknown_y_source(tl_gpt2, frozen_record):
    with pytest.raises(ValueError, match="y_source"):
        replay_offline(tl_gpt2, frozen_record, eta=ETA, y_source="whatever")


def test_replay_is_deterministic(tl_gpt2, frozen_record):
    """Same recording, same rule, same matrix -- twice, bit for bit.

    Determinism is the premise of the whole comparison: a divergence between
    arms only means feedback if re-running an arm gives the same answer.
    """
    a = replay_offline(tl_gpt2, frozen_record, eta=ETA)
    b = replay_offline(tl_gpt2, frozen_record, eta=ETA)
    assert torch.equal(a.weight, b.weight)


def test_closed_arm_is_deterministic(tl_gpt2, tl_loop):
    r0, step = tl_loop
    a = run_closed_loop_arm(tl_gpt2, r0, step, SITE, 3, eta=ETA)
    b = run_closed_loop_arm(tl_gpt2, r0, step, SITE, 3, eta=ETA)
    assert torch.equal(a.weight, b.weight)


def test_batching_follows_the_cadence_in_both_arms(tl_gpt2, tl_loop):
    """apply_every=2 means two samples per update -- in both arms, identically."""
    r0, step = tl_loop
    closed = run_closed_loop_arm(tl_gpt2, r0, step, SITE, 4, eta=ETA, apply_every=2)
    rec = record_frozen_activations(tl_gpt2, r0, step, SITE, 4)
    offline = replay_offline(tl_gpt2, rec, eta=ETA, apply_every=2)

    assert closed.config.samples_per_update == (2, 2)
    assert offline.config.samples_per_update == (2, 2)
    assert closed.config.n_updates == offline.config.n_updates == 2
    assert verify_arms_matched(closed.config, offline.config)["ok"]


def test_a_reshuffled_replay_is_caught_by_the_verifier(tl_gpt2, tl_loop):
    """Oja is sequential, so a replay in a different order is a different arm.

    Constructed by mismatching the cadence, which is the realistic way the
    sample-order and batching axes drift apart: one arm applying every step and
    the other every two steps produces the same total drift budget and a
    completely different trajectory.
    """
    r0, step = tl_loop
    closed = run_closed_loop_arm(tl_gpt2, r0, step, SITE, 4, eta=ETA, apply_every=1)
    rec = record_frozen_activations(tl_gpt2, r0, step, SITE, 4)
    offline = replay_offline(tl_gpt2, rec, eta=ETA, apply_every=2)

    v = verify_arms_matched(closed.config, offline.config)
    assert v["ok"] is False
    flagged = {a["axis"] for a in v["mismatched"]}
    assert {"n_updates", "samples_per_update"} <= flagged


# --------------------------------------------------------------------------
# The degenerate cases -- these prove the harness works
# --------------------------------------------------------------------------

def test_eta_zero_arms_are_bit_identical(tl_gpt2, tl_loop):
    """THE GATE. With the rule turned off, the two arms must agree exactly.

    eta = 0 removes the only thing the arms can differ by. Anything left is the
    harness: a stray write, a different number of updates, a replay that fed the
    rule something the live hook did not. `torch.equal`, not `allclose` -- a
    tolerance here is how you lose the one check that cannot be argued with.
    """
    r0, step = tl_loop
    res = run_matched_arms(tl_gpt2, r0, step, SITE, N_STEPS, eta=0.0,
                           rerun_frozen=True, keep_states=True)

    assert res.verification["ok"] is True
    assert torch.equal(res.closed.weight, res.offline.weight)
    assert torch.equal(res.closed.weight, res.closed.w0), "eta=0 moved the weight"
    assert res.comparison["weight"]["bit_identical"] is True
    assert res.comparison["weight"]["rel_fro_diff"] == 0.0
    # ...and with the matrices identical, the loops under them are too.
    assert res.comparison["state"]["bit_identical"] is True
    # The rule still ran: this is "the update was zero", not "nothing happened".
    assert res.closed.config.n_updates == N_STEPS
    assert res.offline.config.n_updates == N_STEPS


def test_eta_zero_holds_for_the_recomputed_y_arm_too(tl_gpt2, tl_loop):
    """Both offline variants collapse onto the closed arm when the rule is off."""
    r0, step = tl_loop
    res = run_matched_arms(tl_gpt2, r0, step, SITE, 3, eta=0.0, rerun_frozen=False)
    assert res.offline_recomputed_y is not None
    assert torch.equal(res.closed.weight, res.offline_recomputed_y.weight)


def test_recompute_y_reproduces_the_sites_own_forward_bit_exactly(tl_gpt2, frozen_record):
    """`torch.addmm(b, x, W)`, not `x @ W + b`. They are not the same tensor.

    This is the root cause of what used to be this harness's noise floor. The
    two forms agree to float32 rounding and differ in the last bits, because the
    fused kernel accumulates the bias inside the reduction. On GPT-2's
    blocks.6.mlp the gap on y is max|diff| 1.9e-06 -- which then propagates into
    every offline update and compounds, and put ~5e-09 relative Frobenius under
    the whole arms comparison until it was matched.

    Both backends are addmm underneath (TransformerLens's `batch_addmm`
    flattens to 2-D and calls `torch.addmm`, written that way to match
    HuggingFace's `Conv1D`), so matching it is exact rather than merely closer.
    """
    mlp = tl_gpt2.blocks[6].mlp
    for x, y in zip(frozen_record.x, frozen_record.y):
        assert torch.equal(offline_control._recompute_y(x, mlp.W_out, mlp.b_out), y)
        # ...and the mathematically-equivalent form is not the same tensor.
        assert not torch.equal(x @ mlp.W_out + mlp.b_out, y)


def test_a_site_the_loop_does_not_route_through(tl_gpt2, shallow_loop):
    """With the state-feedback path severed, the arms must not diverge at all.

    THIS TEST IS THE HARNESS'S DETECTION LIMIT. The loop reads at
    `blocks.3.hook_resid_post`, so nothing `blocks.6.mlp` computes can reach the
    next iterate -- the x the rule sees is bit-identical in both arms at every
    step. Whatever the arms still differ by here is the floor, and no claim
    about feedback can be made underneath it.

    The only route left from a weight change back into an update is Oja's own
    `y = x W` recursion, which is internal to the rule and present in ordinary
    offline Oja on a fixed dataset. The two `y_source` modes separate exactly
    that, which is why this asserts different things about them:

      recomputed  y is rebuilt from the recorded x and the arm's own drifting
                  weight, so the recursion is live in both arms and nothing is
                  left to differ by. **Bit-identical.** This is the real
                  no-feedback control and the floor is zero.
      recorded    y is replayed frozen, so the offline arm's rule cannot see its
                  own weight move. The arms differ by that alone, with no
                  feedback anywhere in the system, and by a lot -- 6.8% of the
                  drift at eta=1e-5 over 6 steps, measured. That is the floor
                  for the default mode, and it is not small.

    The bit-identity is a defended property, not luck: it holds only because
    `_recompute_y` matches the site's fused addmm. The obvious `x @ W + b` left
    ~5e-09 relative Frobenius here instead -- hardware-dependent (2.9e-09 on a
    laptop, 4.8e-09 on a GitHub runner), harmless at any usable eta, and
    self-inflicted. The bound is asserted as well as the identity, so that if a
    future torch ever breaks the identity the failure message still says whether
    the detection limit survived.

    Reporting a `recorded`-mode divergence at a routed site without this number
    in hand would be attributing the rule's own recursion to feedback.
    """
    r0, step = shallow_loop
    res = run_matched_arms(tl_gpt2, r0, step, SITE, N_STEPS, eta=ETA,
                           rerun_frozen=False)
    assert res.verification["ok"] is True

    recomputed = res.comparison["weight_recomputed_y"]
    print(
        f"\n[detection limit] site={SITE} read_at=blocks.3 eta={ETA} "
        f"n_steps={N_STEPS}"
        f"\n  y_source=recomputed rel_fro_diff={recomputed['rel_fro_diff']:.3e} "
        f"bit_identical={recomputed['bit_identical']}"
        f"\n  y_source=recorded   rel_fro_diff={res.comparison['weight']['rel_fro_diff']:.6e} "
        f"diff_over_drift={res.comparison['weight']['diff_over_drift']:.6e}"
    )
    # The contract: a bound, never a value. This quantity is float
    # accumulation order and its magnitude is hardware-dependent, which is the
    # same trap that bit `initial_norm` (1468.48828125 on CI against
    # 1468.4886474609375 locally). Assert what the experiment needs to be true.
    assert recomputed["rel_fro_diff"] < 1e-8, (
        "the detection limit has regressed: no state feedback exists here, so "
        f"anything above float noise is a defect. rel_fro_diff="
        f"{recomputed['rel_fro_diff']:.3e}"
    )
    # ...and the stronger property, which currently holds by construction.
    assert recomputed["bit_identical"] is True, (
        "no state feedback and no frozen y, yet the arms differ: "
        f"rel_fro_diff={recomputed['rel_fro_diff']:.3e}. Below the 1e-8 bound "
        "above, so the detection limit is intact, but _recompute_y has stopped "
        "matching the site's fused addmm exactly -- see "
        "test_recompute_y_reproduces_the_sites_own_forward_bit_exactly."
    )

    # The default mode's floor, which is a real quantity and not noise.
    recorded = res.comparison["weight"]
    assert recorded["rel_fro_diff"] > recomputed["rel_fro_diff"]
    assert recorded["cos_delta"] == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------

def test_eta_positive_divergence_is_measured_and_reported(matched_run):
    """The real question, at eta = 1e-5 on the full-stack loop.

    Deliberately almost assertion-free on the magnitude. A difference near zero
    would mean feedback contributes nothing detectable at this eta -- a real
    result about this substrate, and one a test must not be able to fail. What
    IS asserted is that the measurement is well formed: the arms were matched,
    both actually moved, the ceiling did not silently truncate either of them,
    and the numbers are finite.

    The measured values are printed, not asserted, and belong in the run log
    rather than in a threshold.
    """
    res = matched_run
    w = res.comparison["weight"]

    assert res.verification["ok"] is True
    assert res.closed.report["clipped"] is False, "the ceiling truncated the closed arm"
    assert res.offline.report["clipped"] is False, "the ceiling truncated the offline arm"
    assert not res.closed.report["nonfinite"] and not res.offline.report["nonfinite"]
    assert w["drift_closed_rel"] > 0.0 and w["drift_offline_rel"] > 0.0
    for key in ("cos_weight", "cos_delta", "rel_fro_diff", "diff_over_drift"):
        assert torch.isfinite(torch.tensor(w[key])), key

    r = res.comparison["weight_recomputed_y"]
    s = res.comparison["state"]
    # Stated against the detection limit, which is what makes the number
    # readable: `test_a_site_the_loop_does_not_route_through` puts the
    # recomputed-y floor at zero (bit-identical), bounded below 1e-8.
    print(
        f"\n[offline arm] prompt={PROMPT!r} site={SITE} eta={ETA} "
        f"n_steps={N_STEPS} updates={res.closed.config.n_updates} cadence=1"
        f"\n  y_source=recorded    cos_delta={w['cos_delta']:.9f} "
        f"rel_fro_diff={w['rel_fro_diff']:.6e} diff_over_drift={w['diff_over_drift']:.6e}"
        f"\n  y_source=recomputed  cos_delta={r['cos_delta']:.9f} "
        f"rel_fro_diff={r['rel_fro_diff']:.6e} diff_over_drift={r['diff_over_drift']:.6e}"
        f"\n  drift closed={w['drift_closed_rel']:.6e} "
        f"offline={w['drift_offline_rel']:.6e}"
        f"\n  final state cos={s['cos']:.9f} rel_l2={s['rel_l2_diff']:.6e}"
        f"\n  clears the 1e-8 detection limit: recorded={w['rel_fro_diff'] > 1e-8} "
        f"recomputed={r['rel_fro_diff'] > 1e-8}"
    )


def test_summary_carries_every_number_a_run_log_needs(matched_run):
    """The result is structured, not a free-text note.

    Whoever reads a sweep afterwards gets the axes and the metrics as fields.
    """
    s = matched_run.summary()
    for key in ("site", "mode", "eta", "n_steps", "n_updates", "arms_matched",
                "cos_weight", "cos_delta", "rel_fro_diff", "diff_over_drift",
                "drift_closed_rel", "drift_offline_rel",
                "cos_delta_recomputed_y", "rel_fro_diff_recomputed_y",
                "diff_over_drift_recomputed_y",
                "clipped_closed", "clipped_offline"):
        assert key in s, key
    assert s["arms_matched"] is True
    assert s["eta"] == ETA
    # The pair with the zero floor is carried, not left in the nested dict:
    # it is the one a feedback claim should be read from.
    assert s["rel_fro_diff_recomputed_y"] is not None


def test_the_frozen_rerun_is_run_under_each_arms_matrix(matched_run):
    """"Install the resulting matrix, re-run the loop frozen, and compare."

    Both arms get the same treatment, so the two states differ only in which
    matrix produced them -- not in whether the weights were moving underneath
    the trajectory.
    """
    res = matched_run
    for key in ("state", "state_closed_vs_frozen_baseline",
                "state_offline_vs_frozen_baseline"):
        assert res.comparison[key] is not None, key
        assert torch.isfinite(torch.tensor(res.comparison[key]["cos"]))


# --------------------------------------------------------------------------
# The refusal
# --------------------------------------------------------------------------

def test_runner_refuses_to_report_a_mismatched_comparison(monkeypatch, tl_gpt2, tl_loop):
    """A mismatch is a dead result, not a caveated one.

    The runner must raise rather than return a number with a warning attached:
    a warning is the part that gets dropped on the way into a write-up. The
    offline arm is sabotaged after it runs, so the mismatch is discovered where
    it really would be -- in the verification step, on axes that only exist once
    both arms have executed.
    """
    r0, step = tl_loop
    real = offline_control.replay_offline

    def sabotaged(*args, **kwargs):
        arm = real(*args, **kwargs)
        return replace(arm, config=replace(arm.config,
                                           n_updates=arm.config.n_updates + 1))

    monkeypatch.setattr(offline_control, "replay_offline", sabotaged)

    with pytest.raises(ArmsMismatchError) as exc:
        run_matched_arms(tl_gpt2, r0, step, SITE, 2, eta=ETA, rerun_frozen=False)

    axes = {a["axis"] for a in exc.value.verification["mismatched"]}
    assert "n_updates" in axes
    assert "not evidence about feedback" in str(exc.value)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_compare_weights_is_computed_in_float64():
    """The headline ratio is two Frobenius norms; float32 leaves ~3e-4 in it.

    Same argument `plasticity.py` makes for `delta_frac`, and it bites harder
    here because the quantity being measured is a difference of two nearly equal
    matrices.

    The perturbation has to survive being added, which is the trap this test
    fell into once: `w0 + d` with `w0` at order 1 and `d` at 1e-7 is evaluated
    in float32, where most of `d` is lost to rounding, so `(w0 + d) - w0` is not
    `d` and the test measures the addition rather than the function. The fix is
    to compare against the difference that is actually representable -- computed
    in float64 from the same float32 tensors the function is handed -- and keep
    the tolerance tight, because the point is that `compare_weights` does not
    silently do the arithmetic in float32.
    """
    g = torch.Generator().manual_seed(3)
    w0 = torch.randn(64, 32, generator=g)
    perturbed = w0 + torch.randn(64, 32, generator=g) * 1e-7

    out = compare_weights(perturbed, w0, w0)
    representable = (perturbed.double() - w0.double()).norm().item()
    expected = representable / w0.double().norm().item()

    assert out["rel_fro_diff"] == pytest.approx(expected, rel=1e-12)
    assert out["bit_identical"] is False
    assert out["cos_delta"] != out["cos_delta"]  # NaN: the second delta is zero

    # A float32 pass would leave ~1e-4 relative error on a norm this size; the
    # assertion above is four orders tighter than that, so it can tell them
    # apart rather than merely agreeing with both.
    assert 0.0 < expected < 1e-6


def test_compare_weights_reports_identity_exactly():
    g = torch.Generator().manual_seed(4)
    w0 = torch.randn(16, 8, generator=g)
    out = compare_weights(w0, w0, w0)
    assert out["bit_identical"] is True
    assert out["fro_diff"] == 0.0
    assert out["rel_fro_diff"] == 0.0


def test_compare_states_matches_a_hand_computed_cosine():
    a = torch.tensor([[3.0, 0.0], [0.0, 0.0]])
    b = torch.tensor([[0.0, 4.0], [0.0, 0.0]])
    out = compare_states(a, b)
    assert out["cos"] == pytest.approx(0.0)
    assert out["l2_diff"] == pytest.approx(5.0)
    assert out["rel_l2_diff"] == pytest.approx(5.0 / 3.0)
    assert out["bit_identical"] is False


def test_activation_record_repr_states_the_footprint():
    rec = ActivationRecord(
        site=SITE, step_index=(0,), x=(torch.zeros(2, 4),), y=(torch.zeros(2, 3),),
        n_steps=1, store_dtype=torch.float32, weight_dtype=torch.float32,
        bytes_stored=1024 * 1024,
    )
    text = repr(rec)
    assert "MiB" in text and SITE in text and "samples=1" in text


def test_recorder_rejects_a_zero_length_run(tl_gpt2, tl_loop):
    r0, step = tl_loop
    with pytest.raises(ValueError, match="n_steps"):
        record_frozen_activations(tl_gpt2, r0, step, SITE, 0)


def test_the_rule_object_is_plasticitys_own(tl_gpt2, frozen_record):
    """No second copy of Oja lives here.

    The offline arm's whole claim is that it runs the identical rule; a
    reimplementation would be one more axis on which the arms could differ, and
    one the verifier could not see.
    """
    import inspect

    src = inspect.getsource(offline_control)
    assert "from plasticity import OjaPlasticity" in src
    for forbidden in ("def _hebb_term", "def _oja_decay", "class OjaPlasticity"):
        assert forbidden not in src, f"{forbidden!r}: the rule is reimplemented here"

    # And what comes back is that object's own log schema, not a private one.
    arm = replay_offline(tl_gpt2, frozen_record, eta=0.0)
    reference = OjaPlasticity(tl_gpt2, site=SITE, eta=0.0, mode="off").report()
    assert set(arm.report) == set(reference)
    assert arm.report["site"] == SITE
