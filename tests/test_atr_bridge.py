"""
Tests for `atr_bridge` -- the per-step seam in the parent project's ATR loop.

The test that matters is `test_step_reproduces_run_atr_loop_bit_exactly`.
Everything else here is hygiene. If the extracted step and
`atr_engine.run_atr_loop` ever disagree, then every plasticity number this repo
produces is measured against a loop the parent never ran, and no comparison with
the parent's recorded attractors -- the period-2 `Divine` cycle above all --
means anything. Bit-exact, not close: the finding EXP-001 turns on is
`cos(A, f(f(A))) = 1.000000`, and a tolerance is how you lose that.

The equivalence test needs the parent repo checked out. `ATR_PARENT_PATH`
overrides the location; absent, it skips -- and `ATR_REQUIRE_PARENT=1` turns
that skip into a failure, exactly as `conftest._unavailable` does for
`ATR_REQUIRE_MODEL`. Same reasoning: a skip is honest on a laptop that was
never going to have the parent, and is a lie in CI. This repo has already been
bitten once by a check that could not fail.

Run with:  .venv/bin/python -m pytest tests/test_atr_bridge.py -q
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch

from atr_bridge import (
    initial_state,
    load_state,
    make_atr_step,
    make_atr_step_from_state,
    renormalise,
)

PARENT_DEFAULT = "/workspace/lucier-gpt2-activ-tensor-reson-experiments"

# The `Divine` prompt: 9 tokens plus BOS = 10 positions, and the one the saved
# state below was produced from.
PROMPT = "The cat sat on the mat and then the"
LAYER_START = 0
LAYER_END = 11

# Five iterations of two loops is ten GPT-2 forwards, ~1s. The parent's own
# acceptance bar is 20; divergence from a bit-exact copy shows up at iteration
# 1 or never, and the suite is already slow enough.
N_ITER = 5

STATE_DIVINE = "experiments/gpt2_small/output_divine_motion/state_divine.pt"
DIVINE_INITIAL_NORM = 1468.4886474609375
DIVINE_SHAPE = (10, 768)


def _parent_path() -> str:
    return os.environ.get("ATR_PARENT_PATH", PARENT_DEFAULT)


def _parent_unavailable(reason: str):
    """Skip, or fail if the environment says the parent repo must be there.

    Mirrors `conftest._unavailable`. A green suite that skipped the one test
    proving the two engines agree is worse than a red one.
    """
    if os.environ.get("ATR_REQUIRE_PARENT"):
        pytest.fail(f"ATR_REQUIRE_PARENT is set but {reason}")
    pytest.skip(reason)


@pytest.fixture(scope="session")
def atr_engine():
    """The parent's `atr_engine` module, imported from a path, not installed.

    Loaded by file location on purpose: the parent is a sibling checkout, not a
    dependency, and pinning the import to `ATR_PARENT_PATH` keeps it obvious
    which copy of the loop the equivalence is against.
    """
    path = os.path.join(_parent_path(), "atr_engine.py")
    if not os.path.isfile(path):
        _parent_unavailable(f"parent ATR repo not found at {_parent_path()} (set ATR_PARENT_PATH)")
    spec = importlib.util.spec_from_file_location("_parent_atr_engine", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        _parent_unavailable(f"parent atr_engine is not importable: {exc}")
    return module


@pytest.fixture(scope="session")
def divine_state():
    """The parent's iteration-1000 `Divine` state -- a tensor sitting on the cycle."""
    path = os.path.join(_parent_path(), STATE_DIVINE)
    if not os.path.isfile(path):
        _parent_unavailable(f"saved state not found at {path} (set ATR_PARENT_PATH)")
    return load_state(path)


@pytest.fixture(scope="session")
def parent_snapshots(atr_engine, tl_gpt2):
    """`run_atr_loop`'s trajectory, every iteration 0..N_ITER, as ground truth."""
    return atr_engine.run_atr_loop(
        tl_gpt2,
        PROMPT,
        layer_start=LAYER_START,
        layer_end=LAYER_END,
        max_iter=N_ITER,
        schedule=list(range(N_ITER + 1)),
        verbose=False,
    )


# --------------------------------------------------------------------------
# The acceptance test
# --------------------------------------------------------------------------

def test_step_reproduces_run_atr_loop_bit_exactly(tl_gpt2, parent_snapshots):
    """Iterating `atr_step` must equal `run_atr_loop`, tensor for tensor, bit for bit.

    A failure here is not a tolerance to widen. It means the extraction differs
    from the parent's loop somewhere -- most likely the normalisation target
    (initial norm, not unit norm), the injection site, or normalising after
    injection instead of before -- and the bridge is simulating a different
    dynamical system than the one whose attractors this repo cites.
    """
    state = initial_state(tl_gpt2, PROMPT, layer_end=LAYER_END)
    assert torch.equal(state.tensor, parent_snapshots[0]["tensor"]), (
        "iteration 0 already differs: the read hook point or the prompt handling "
        "does not match the parent"
    )
    assert state.initial_norm == parent_snapshots[0]["tensor_norm"]

    step = make_atr_step(
        tl_gpt2, PROMPT, layer_start=LAYER_START, layer_end=LAYER_END
    )

    r = state.tensor
    deviations = []
    for i in range(1, N_ITER + 1):
        r = step(tl_gpt2, r)
        expected = parent_snapshots[i]["tensor"]
        deviations.append((r - expected).abs().max().item())
        assert torch.equal(r, expected), (
            f"iteration {i}: max|Δ| = {deviations[-1]:.3e} "
            f"(iterations 1..{i - 1} were exact)"
        )

    assert max(deviations) == 0.0


# --------------------------------------------------------------------------
# Resuming a saved trajectory
# --------------------------------------------------------------------------

def test_load_state_divine_has_the_documented_shape_and_norm(divine_state):
    """The checkpoint EXP-001 starts from, read rather than assumed."""
    assert tuple(divine_state.tensor.shape) == DIVINE_SHAPE
    assert divine_state.initial_norm == DIVINE_INITIAL_NORM
    assert divine_state.prompt == PROMPT
    assert divine_state.iteration == 1000
    assert divine_state.label == "Divine_Syntactic"


def test_resumed_step_preserves_shape_and_energy_shell(tl_gpt2, divine_state):
    """One step from iteration 1000 stays on the cycle's shell, not the tensor's own.

    `initial_norm` comes from the checkpoint (1468.49), while the resumed tensor
    itself has norm ~5098. A bridge that recomputed the norm from the resumed
    state would rescale the attractor to itself, which is a no-op that looks
    like a working loop and silently measures the wrong system.
    """
    step = make_atr_step_from_state(
        tl_gpt2, divine_state, layer_start=LAYER_START, layer_end=LAYER_END
    )
    assert step.initial_norm == DIVINE_INITIAL_NORM
    assert divine_state.tensor.norm().item() != pytest.approx(DIVINE_INITIAL_NORM)

    r1 = step(tl_gpt2, divine_state.tensor)
    assert tuple(r1.shape) == DIVINE_SHAPE
    assert torch.isfinite(r1).all()


def test_initial_state_from_prompt_reproduces_the_saved_initial_norm(tl_gpt2, divine_state):
    """A fresh iteration-0 pass on the `Divine` prompt gives the checkpoint's ‖x₀‖.

    Provenance check on the whole read path: same tokenisation (BOS prepended,
    10 positions), same read hook, same dtype as the run that produced the file.
    If this drifts, resumed runs and fresh runs are no longer comparable.
    """
    state = initial_state(tl_gpt2, divine_state.prompt, layer_end=LAYER_END)
    assert tuple(state.tensor.shape) == DIVINE_SHAPE
    assert state.initial_norm == divine_state.initial_norm


# --------------------------------------------------------------------------
# Loop invariants
# --------------------------------------------------------------------------

def test_injected_tensor_is_held_at_initial_norm(tl_gpt2):
    """What is re-injected each iteration always has norm ‖x₀‖ -- the room's energy.

    The iterate coming *out* of the model does not: it is free to grow (it does,
    by ~2x on the first step). Holding the input energy fixed is the whole
    reason the loop neither explodes nor decays, so if this stops holding, any
    convergence or divergence observed afterwards is an artefact of gain.
    """
    state = initial_state(tl_gpt2, PROMPT, layer_end=LAYER_END)
    step = make_atr_step(
        tl_gpt2, PROMPT, layer_start=LAYER_START, layer_end=LAYER_END
    )
    n = state.initial_norm

    r = state.tensor
    raw_norms = []
    for _ in range(N_ITER):
        assert renormalise(r, n).norm().item() == pytest.approx(n, rel=1e-6)
        r = step(tl_gpt2, r)
        raw_norms.append(r.norm().item())

    # The rescale is load-bearing, not a no-op on an already-normalised iterate.
    assert all(abs(x - n) > 1.0 for x in raw_norms), raw_norms


def test_step_is_deterministic(tl_gpt2):
    """Same model, same input, same output -- bit for bit.

    C0 compares two trajectories and calls any difference the plasticity layer's
    fault. That inference is only valid if the step itself is deterministic.
    """
    state = initial_state(tl_gpt2, PROMPT, layer_end=LAYER_END)
    step = make_atr_step(
        tl_gpt2, PROMPT, layer_start=LAYER_START, layer_end=LAYER_END
    )
    a = step(tl_gpt2, state.tensor)
    b = step(tl_gpt2, state.tensor)
    assert torch.equal(a, b)


def test_step_leaves_no_hooks_behind(tl_gpt2):
    """The model is handed back clean.

    The `tl_gpt2` fixture is session-scoped: an injection hook left installed
    would keep overwriting `hook_resid_pre` in every later test and every later
    experiment, with no error to point at.
    """
    step = make_atr_step(
        tl_gpt2, PROMPT, layer_start=LAYER_START, layer_end=LAYER_END
    )
    state = initial_state(tl_gpt2, PROMPT, layer_end=LAYER_END)
    step(tl_gpt2, state.tensor)

    left = {name: len(hp.fwd_hooks) for name, hp in tl_gpt2.hook_dict.items() if hp.fwd_hooks}
    assert left == {}


def test_step_leaves_no_hooks_behind_when_it_raises(tl_gpt2):
    """...including when the forward pass blows up mid-step.

    A wrong-length state is the realistic way in: resume a checkpoint against a
    prompt that tokenises to a different number of positions and the injection
    assignment fails. Without the `finally`, the model would be poisoned from
    that point on and the next test to fail would be an innocent one.
    """
    step = make_atr_step(
        tl_gpt2, PROMPT, layer_start=LAYER_START, layer_end=LAYER_END
    )
    wrong_length = torch.zeros(3, tl_gpt2.cfg.d_model)

    with pytest.raises(RuntimeError):
        step(tl_gpt2, wrong_length)

    left = {name: len(hp.fwd_hooks) for name, hp in tl_gpt2.hook_dict.items() if hp.fwd_hooks}
    assert left == {}


def test_step_does_not_mutate_its_input(tl_gpt2):
    """`r` goes in unchanged.

    The controls keep and compare earlier states (`states.append(r.clone())`);
    a step that rescaled its argument in place would rewrite the trajectory it
    is being compared against.
    """
    state = initial_state(tl_gpt2, PROMPT, layer_end=LAYER_END)
    r = state.tensor.clone()
    before = r.clone()
    step = make_atr_step(
        tl_gpt2, PROMPT, layer_start=LAYER_START, layer_end=LAYER_END
    )
    step(tl_gpt2, r)
    assert torch.equal(r, before)


def test_default_layer_end_is_the_last_block(tl_gpt2):
    """Omitting `layer_end` reads the top of the stack, not layer 0."""
    step = make_atr_step(tl_gpt2, PROMPT)
    assert step.hook_point_read == f"blocks.{tl_gpt2.cfg.n_layers - 1}.hook_resid_post"
    assert step.hook_point_write == "blocks.0.hook_resid_pre"


def test_missing_parent_repo_is_a_failure_when_required(monkeypatch):
    """`ATR_REQUIRE_PARENT` must convert the skip into a failure.

    The guard on the guard. If this stops working, the equivalence test can
    vanish from a CI run without turning it red -- which is exactly the
    false-green this suite exists to prevent.
    """
    monkeypatch.setenv("ATR_PARENT_PATH", "/nonexistent/atr/parent")
    monkeypatch.setenv("ATR_REQUIRE_PARENT", "1")
    # pytest's own outcomes derive from BaseException, not Exception, so that a
    # bare `except Exception` in test code cannot swallow them.
    with pytest.raises(BaseException) as exc:
        _parent_unavailable("parent ATR repo not found")
    assert exc.typename == "Failed"
    assert "ATR_REQUIRE_PARENT" in str(exc.value)

    monkeypatch.delenv("ATR_REQUIRE_PARENT")
    with pytest.raises(BaseException) as exc:
        _parent_unavailable("parent ATR repo not found")
    assert exc.typename == "Skipped"
