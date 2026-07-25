"""
Tests for `plasticity.py`.

The module's own header says it was "written but never executed against real
weights". These tests are the first execution. They are written in the spirit of
the controls table in README: each one names the thing that would be true of the
experiment if the test failed, not just the thing that would be true of the code.

The learning-rule tests reconstruct the expected update from activations captured
by an *independent* forward hook installed by the test. If the expectation came
out of `plasticity.py` the test would only prove the module is self-consistent.
"""

from __future__ import annotations

import contextlib
import copy
import math

import pytest
import torch

from conftest import Conv1D, LinearModel, ToyModel     # noqa: F401  (Conv1D re-exported for shape asserts)
from plasticity import OjaPlasticity, candidate_sites


# --------------------------------------------------------------------------
# Local helpers -- deliberately independent of the code under test
# --------------------------------------------------------------------------

@contextlib.contextmanager
def capture(module):
    """
    Record (x, y) for every forward pass through `module`, flattened the way the
    learning rules define them: x (N, n_in), y (N, n_out).
    """
    seen = []

    def _hook(_mod, inputs, output):
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).clone()
        y = output.detach().reshape(-1, output.shape[-1]).clone()
        seen.append((x, y))

    handle = module.register_forward_hook(_hook)
    try:
        yield seen
    finally:
        handle.remove()


def drive(model, r, atr_step, n=3):
    """n iterations of the caller's engine. Returns the final state."""
    for _ in range(n):
        r = atr_step(model, r)
    return r


def hebb_term(x, y):
    return (x.transpose(0, 1) @ y) / x.shape[0]


def oja_term(x, y, w):
    return hebb_term(x, y) - w @ ((y.transpose(0, 1) @ y) / y.shape[0])


def expected_update(seen, w=None):
    """
    Mean over hook firings of the per-firing rule. `w` None means raw Hebb.
    Summed in firing order to match the accumulator's float32 rounding.
    """
    total = None
    for x, y in seen:
        term = hebb_term(x, y) if w is None else oja_term(x, y, w)
        total = term if total is None else total + term
    return total / len(seen)


def resolve(model, path):
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0
    ).item()


REPORT_TYPES = {
    "site": str,
    "mode": str,
    "eta": float,
    "n_applied": int,
    "delta_norm": float,
    "delta_frac": float,
    "last_update_norm": float,
    "clipped": bool,
    "nonfinite": bool,
}


# --------------------------------------------------------------------------
# 1. Construction
# --------------------------------------------------------------------------

class TestConstruction:

    def test_invalid_mode_raises(self, toy_model, site):
        """A typo'd mode must not silently degrade to a different experiment."""
        with pytest.raises(ValueError, match="mode must be one of"):
            OjaPlasticity(toy_model, site=site, mode="oj")

    @pytest.mark.parametrize("mode", OjaPlasticity.VALID_MODES)
    def test_every_advertised_mode_constructs(self, toy_model, site, mode):
        """The four modes in the docstring are the four controls; all must exist."""
        assert OjaPlasticity(toy_model, site=site, mode=mode).mode == mode

    @pytest.mark.parametrize(
        "bad_site",
        ["transformer.h.1.ln_1",   # LayerNorm: weight is 1-D
         "transformer.h.1.mlp"],   # container: no .weight at all
    )
    def test_non_matrix_site_raises_typeerror(self, toy_model, bad_site):
        """
        Attaching to a 1-D or weightless module would make <x y^T> meaningless;
        failing loudly at construction is the only safe behaviour.
        """
        with pytest.raises(TypeError, match="no 2-D .weight"):
            OjaPlasticity(toy_model, site=bad_site)

    def test_dotted_path_resolves_through_modulelist(self, toy_model, site):
        """
        `transformer.h.1` indexes an nn.ModuleList. If numeric parts were treated
        as attributes the whole site-addressing scheme would be unusable.
        """
        p = OjaPlasticity(toy_model, site=site)
        assert p.module is toy_model.transformer.h[1].mlp.c_proj
        assert p.module.weight.shape == (64, 16)   # Conv1D: (n_in, n_out)

    def test_W0_is_an_independent_snapshot(self, toy_model, site):
        """
        Every measurement in report() is relative to W0. If W0 aliased the live
        parameter, delta_frac would read 0 no matter how far the weight drifted.
        """
        p = OjaPlasticity(toy_model, site=site)
        assert torch.equal(p.W0, p.module.weight)
        assert p.W0.data_ptr() != p.module.weight.data_ptr()
        assert p.W0_norm == pytest.approx(p.module.weight.norm().item())

        with torch.no_grad():
            p.module.weight.add_(1.0)
        assert not torch.equal(p.W0, p.module.weight)

    def test_initial_state_is_clean(self, toy_model, site):
        """A fresh instance must not claim to have applied or clipped anything."""
        p = OjaPlasticity(toy_model, site=site)
        assert p._acc is None
        assert p._n_batches == 0
        assert p.n_applied == 0
        assert not p.clipped
        assert not p.nonfinite
        assert torch.equal(p.delta, torch.zeros_like(p.W0))


# --------------------------------------------------------------------------
# 2. Hook lifecycle
# --------------------------------------------------------------------------

class TestHookLifecycle:

    def test_install_is_idempotent(self, toy_model, site):
        """
        A double-installed hook would double-count every batch, halving the
        effective eta relative to what report() claims.
        """
        p = OjaPlasticity(toy_model, site=site)
        before = len(p.module._forward_hooks)
        p.install()
        assert len(p.module._forward_hooks) == before + 1
        handle = p._handle
        p.install()
        assert len(p.module._forward_hooks) == before + 1
        assert p._handle is handle

    def test_remove_detaches(self, toy_model, site):
        """A leaked hook survives the experiment it belongs to and contaminates the next one."""
        p = OjaPlasticity(toy_model, site=site)
        before = len(p.module._forward_hooks)
        p.install()
        p.remove()
        assert len(p.module._forward_hooks) == before
        assert p._handle is None
        p.remove()   # second remove must not raise

    def test_context_manager_installs_and_removes(self, toy_model, site, r0, atr_step):
        """The documented usage is `with OjaPlasticity(...)`; it must be exception-safe."""
        p = OjaPlasticity(toy_model, site=site)
        baseline = len(p.module._forward_hooks)
        with p as ctx:
            assert ctx is p
            assert len(p.module._forward_hooks) == baseline + 1
            drive(toy_model, r0, atr_step, n=1)
            assert p._n_batches == 1
        assert len(p.module._forward_hooks) == baseline
        assert p._handle is None

    def test_context_manager_removes_on_exception(self, toy_model, site):
        with pytest.raises(RuntimeError):
            with OjaPlasticity(toy_model, site=site) as p:
                raise RuntimeError("boom")
        assert p._handle is None
        assert len(p.module._forward_hooks) == 0

    def test_no_accumulation_after_remove(self, toy_model, site, r0, atr_step):
        """C1's premise: once detached, the object is inert."""
        p = OjaPlasticity(toy_model, site=site).install()
        drive(toy_model, r0, atr_step, n=2)
        assert p._n_batches == 2
        p.remove()
        drive(toy_model, r0, atr_step, n=5)
        assert p._n_batches == 2


# --------------------------------------------------------------------------
# 3. Statistics collection
# --------------------------------------------------------------------------

class TestCollection:

    def test_accumulator_shape_and_count(self, toy_model, site, r0, atr_step):
        """
        The accumulator is the update-in-waiting: it must carry the weight's
        (n_in, n_out) shape and one contribution per forward pass.
        """
        p = OjaPlasticity(toy_model, site=site)
        assert p._acc is None
        with p:
            drive(toy_model, r0, atr_step, n=3)
        assert p._acc is not None
        assert p._acc.shape == p.module.weight.shape == (64, 16)
        assert p._n_batches == 3

    def test_collection_does_not_touch_the_weight(self, toy_model, site, r0, atr_step):
        """
        Control C0: watching must not perturb. If the weight moves before any
        apply(), no downstream trajectory difference is interpretable.
        """
        p = OjaPlasticity(toy_model, site=site, eta=1e-3)
        w_before = p.module.weight.detach().clone()
        with p:
            drive(toy_model, r0, atr_step, n=3)
        assert torch.equal(p.module.weight, w_before)

    def test_hook_does_not_disturb_the_trajectory(self, toy_model, site, r0, atr_step):
        """
        Control C0 proper, at eta=0: hooks plus a committed zero update must
        leave the iterated map bit-identical to the unhooked one.
        """
        clean = copy.deepcopy(toy_model)
        r_clean = drive(clean, r0, atr_step, n=4)

        p = OjaPlasticity(toy_model, site=site, eta=0.0, mode="oja")
        w_before = p.module.weight.detach().clone()
        with p:
            r = r0
            for _ in range(4):
                r = atr_step(toy_model, r)
                p.apply()
        assert torch.equal(r, r_clean)
        assert torch.equal(p.module.weight, w_before)
        assert p.report()["delta_norm"] == 0.0


# --------------------------------------------------------------------------
# 4. Learning-rule correctness
# --------------------------------------------------------------------------

class TestLearningRule:
    """
    The heart of it. If these fail, the repo is not running the rule its README
    argues for, and the Oja-vs-Hebb framing is unsupported.
    """

    ETA = 1.0   # large enough that delta is resolvable in float32, far below the ceiling

    def test_hebb_update_matches_closed_form(self, toy_model, site, r0, atr_step):
        """dW = mean over firings of <x y^T>. A mismatch means mode='hebb' is not raw Hebb."""
        p = OjaPlasticity(toy_model, site=site, eta=self.ETA, mode="hebb")
        with capture(p.module) as seen, p:
            drive(toy_model, r0, atr_step, n=3)

        expected = expected_update(seen)
        rep = p.apply()

        assert len(seen) == 3
        assert torch.allclose(p.delta, self.ETA * expected, rtol=1e-5, atol=1e-9)
        assert torch.allclose(p.module.weight, p.W0 + self.ETA * expected,
                              rtol=1e-6, atol=1e-9)
        assert not rep["clipped"]
        assert rep["n_applied"] == 1

    def test_oja_update_matches_closed_form(self, toy_model, site, r0, atr_step):
        """
        dW = <x y^T> - W <y y^T>. Losing the decay term silently turns the real
        experiment into control C3 (divergent raw Hebb).
        """
        p = OjaPlasticity(toy_model, site=site, eta=self.ETA, mode="oja")
        with capture(p.module) as seen, p:
            drive(toy_model, r0, atr_step, n=3)

        expected = expected_update(seen, w=p.W0)   # no delta yet, so W_eff == W0
        p.apply()

        assert torch.allclose(p.delta, self.ETA * expected, rtol=1e-5, atol=1e-9)
        assert torch.allclose(p.module.weight, p.W0 + self.ETA * expected,
                              rtol=1e-6, atol=1e-9)

    def test_oja_differs_from_hebb_by_exactly_the_decay_term(self, toy_model, site, r0, atr_step):
        """
        The decay term is the whole argument of README's "Why Oja rather than
        Hebb". It must be present and it must be W <y y^T>, nothing else.
        """
        model_h = copy.deepcopy(toy_model)
        p_h = OjaPlasticity(model_h, site=site, eta=self.ETA, mode="hebb")
        with capture(p_h.module) as seen_h, p_h:
            drive(model_h, r0, atr_step, n=3)
        p_h.apply()

        p_o = OjaPlasticity(toy_model, site=site, eta=self.ETA, mode="oja")
        with p_o:
            drive(toy_model, r0, atr_step, n=3)
        p_o.apply()

        decay = None
        for _x, y in seen_h:
            term = p_h.W0 @ ((y.transpose(0, 1) @ y) / y.shape[0])
            decay = term if decay is None else decay + term
        decay = decay / len(seen_h)

        assert torch.allclose(p_h.delta - p_o.delta, self.ETA * decay,
                              rtol=1e-4, atol=1e-9)
        assert decay.norm().item() > 0

    def test_oja_uses_the_effective_weight_after_a_first_apply(self, toy_model, site, r0, atr_step):
        """
        The decay term must track the live weight, not the frozen W0. Using W0
        forever makes the rule non-local in time and breaks Oja's fixed point.
        """
        p = OjaPlasticity(toy_model, site=site, eta=self.ETA, mode="oja")
        with capture(p.module) as seen, p:
            r = drive(toy_model, r0, atr_step, n=2)
            p.apply()
            w_eff = p.W0 + p.delta          # what the second round should decay against
            assert p.delta.norm().item() > 0
            seen.clear()
            delta_after_first = p.delta.clone()
            drive(toy_model, r, atr_step, n=2)

        expected = expected_update(seen, w=w_eff)
        p.apply()
        assert torch.allclose(p.delta - delta_after_first, self.ETA * expected,
                              rtol=1e-4, atol=1e-9)

    def test_batches_are_averaged_not_summed(self, toy_model, site, r0, atr_step):
        """
        apply() divides by _n_batches. If it summed, the effective learning rate
        would scale with cadence and every eta sweep would be mislabelled.
        """
        p = OjaPlasticity(toy_model, site=site, eta=self.ETA, mode="hebb")
        with capture(p.module) as seen, p:
            drive(toy_model, r0, atr_step, n=4)
        summed = expected_update(seen) * len(seen)
        p.apply()
        assert not torch.allclose(p.delta, self.ETA * summed, rtol=1e-3, atol=1e-9)
        assert torch.allclose(p.delta, self.ETA * summed / 4, rtol=1e-5, atol=1e-9)


# --------------------------------------------------------------------------
# 5. mode="off"
# --------------------------------------------------------------------------

class TestModeOff:

    def test_off_accumulates_but_never_writes(self, toy_model, site, r0, atr_step):
        """
        Control C0/C1's instrument: statistics with no effect. Any weight motion
        here means the "off" baseline is not a baseline.
        """
        p = OjaPlasticity(toy_model, site=site, eta=1.0, mode="off")
        w_before = p.module.weight.detach().clone()
        with p:
            drive(toy_model, r0, atr_step, n=3)
            assert p._acc is not None and p._n_batches == 3
            rep = p.apply()

        assert torch.equal(p.module.weight, w_before)   # bit-for-bit
        assert rep["delta_norm"] == 0.0
        assert rep["delta_frac"] == 0.0
        assert rep["n_applied"] == 0
        assert not rep["clipped"]

    def test_off_does_not_alter_the_trajectory(self, toy_model, site, r0, atr_step):
        clean = copy.deepcopy(toy_model)
        r_clean = drive(clean, r0, atr_step, n=4)
        with OjaPlasticity(toy_model, site=site, eta=1.0, mode="off") as p:
            r = r0
            for _ in range(4):
                r = atr_step(toy_model, r)
                p.apply()
        assert torch.equal(r, r_clean)


# --------------------------------------------------------------------------
# 6. mode="random"  (Control C2)
# --------------------------------------------------------------------------

def _amplified(model, site, gain=10.0):
    """
    Scale the target weight so the Oja decay term is a large fraction of the
    Hebb term. At the toy model's init scale (std 0.02) the decay contributes
    ~0.7% of the update norm, which is too small to tell the two rules apart by
    magnitude. Hebb ~ k, decay ~ k^3, so a gain of 10 makes them distinguishable.
    """
    with torch.no_grad():
        resolve(model, site).weight.mul_(gain)
    return model


class TestModeRandom:

    def test_random_is_reproducible_for_a_fixed_seed(self, toy_model, site, r0, atr_step):
        """
        C2 is a comparison. If the control is not reproducible, a difference
        between two runs cannot be attributed to the rule.
        """
        deltas = []
        for _ in range(2):
            model = copy.deepcopy(toy_model)
            p = OjaPlasticity(model, site=site, eta=1.0, mode="random", seed=0)
            with p:
                drive(model, r0, atr_step, n=3)
            p.apply()
            deltas.append(p.delta.clone())
        assert torch.equal(deltas[0], deltas[1])

    def test_random_differs_across_seeds(self, toy_model, site, r0, atr_step):
        """A seed that does not change the draw would make C2 a single sample."""
        out = []
        for seed in (0, 1):
            model = copy.deepcopy(toy_model)
            p = OjaPlasticity(model, site=site, eta=1.0, mode="random", seed=seed)
            with p:
                drive(model, r0, atr_step, n=3)
            p.apply()
            out.append(p.delta.clone())
        assert not torch.allclose(out[0], out[1])
        assert abs(cosine(out[0], out[1])) < 0.3

    def test_random_points_elsewhere_than_oja(self, toy_model, site, r0, atr_step):
        """
        The point of C2 is that only the *direction* differs. A random matrix
        aligned with the Oja update would test nothing.
        """
        model_r = copy.deepcopy(toy_model)
        p_r = OjaPlasticity(model_r, site=site, eta=1.0, mode="random", seed=0)
        with p_r:
            drive(model_r, r0, atr_step, n=3)
        p_r.apply()

        p_o = OjaPlasticity(toy_model, site=site, eta=1.0, mode="oja")
        with p_o:
            drive(toy_model, r0, atr_step, n=3)
        p_o.apply()

        assert abs(cosine(p_r.delta, p_o.delta)) < 0.2

    def test_random_is_norm_matched_to_oja(self, toy_model, site, r0, atr_step):
        """
        The class docstring defines "random" as norm-matched to what Oja would
        have applied. If it is matched to something else, C2 compares two updates
        of different magnitude and its verdict on "direction vs magnitude" is void.
        """
        model_o = _amplified(copy.deepcopy(toy_model), site)
        model_r = _amplified(copy.deepcopy(toy_model), site)

        p_o = OjaPlasticity(model_o, site=site, eta=1e-3, mode="oja")
        with p_o:
            drive(model_o, r0, atr_step, n=3)
        oja_norm = p_o.apply()["delta_norm"]

        p_r = OjaPlasticity(model_r, site=site, eta=1e-3, mode="random", seed=0)
        with p_r:
            drive(model_r, r0, atr_step, n=3)
        rand_norm = p_r.apply()["delta_norm"]

        assert rand_norm == pytest.approx(oja_norm, rel=1e-4)


# --------------------------------------------------------------------------
# 7. max_delta_frac ceiling
# --------------------------------------------------------------------------

class TestCeiling:

    def test_large_eta_is_clipped_at_the_ceiling(self, toy_model, site, r0, atr_step):
        """
        README: the ceiling "is the guard against silently destroying the model".
        Exceeding it without flagging is the failure mode that voids a whole run.
        """
        p = OjaPlasticity(toy_model, site=site, eta=1e3, mode="oja", max_delta_frac=0.05)
        with p:
            drive(toy_model, r0, atr_step, n=3)
        rep = p.apply()

        assert rep["clipped"] is True
        assert rep["delta_frac"] <= 0.05 + 1e-6
        assert rep["delta_frac"] == pytest.approx(0.05, rel=1e-5)
        assert torch.isfinite(p.module.weight).all()

    def test_tiny_eta_is_not_clipped(self, toy_model, site, r0, atr_step):
        """A false clipped flag would send the operator chasing a nonexistent regime."""
        p = OjaPlasticity(toy_model, site=site, eta=1e-9, mode="oja", max_delta_frac=0.05)
        with p:
            drive(toy_model, r0, atr_step, n=3)
        rep = p.apply()

        assert rep["clipped"] is False
        assert 0.0 < rep["delta_frac"] < 0.05

    def test_repeated_applies_accumulate_toward_the_ceiling(self, toy_model, site, r0, atr_step):
        """
        The ceiling is on *total* drift from W0, not on one step. If it were
        per-step, a long run could walk arbitrarily far while never flagging.
        """
        p = OjaPlasticity(toy_model, site=site, eta=1.0, mode="oja", max_delta_frac=0.02)
        fracs = []
        with p:
            r = r0
            for _ in range(6):
                r = drive(toy_model, r, atr_step, n=2)
                fracs.append(p.apply()["delta_frac"])

        assert all(f <= 0.02 + 1e-6 for f in fracs)
        assert fracs[0] < fracs[1] < fracs[2]
        # Non-decreasing to float32 precision; the ceiling, not the rule, ends it.
        assert not any(fracs[i] > fracs[i + 1] + 1e-7 for i in range(len(fracs) - 1))
        assert not any(f > 0.02 for f in fracs[:2])
        assert p.clipped is True
        assert fracs[-1] == pytest.approx(0.02, rel=1e-5)


# --------------------------------------------------------------------------
# 8. revert()
# --------------------------------------------------------------------------

class TestRevert:

    def test_revert_restores_bit_exactly(self, toy_model, site, r0, atr_step):
        """
        Control C1. Anything short of bit-exact means hidden state is
        accumulating and no A/B comparison in this repo is trustworthy.
        """
        p = OjaPlasticity(toy_model, site=site, eta=1.0, mode="oja")
        w_before = p.module.weight.detach().clone()
        with p:
            drive(toy_model, r0, atr_step, n=3)
            p.apply()
            assert not torch.equal(p.module.weight, w_before)
            drive(toy_model, r0, atr_step, n=1)   # leave something pending too
            p.revert()

        assert torch.equal(p.module.weight, w_before)
        assert torch.equal(p.delta, torch.zeros_like(p.W0))
        assert p._acc is None
        assert p._n_batches == 0

        rep = p.report()
        assert rep["delta_norm"] == 0.0
        assert rep["delta_frac"] == 0.0

    def test_revert_restores_the_trajectory(self, toy_model, site, r0, atr_step):
        """C1 as the README states it: the *trajectory* must come back, not just the weight."""
        clean = copy.deepcopy(toy_model)
        r_clean = drive(clean, r0, atr_step, n=3)

        p = OjaPlasticity(toy_model, site=site, eta=1.0, mode="oja")
        with p:
            drive(toy_model, r0, atr_step, n=3)
            p.apply()
        p.revert()
        assert torch.equal(drive(toy_model, r0, atr_step, n=3), r_clean)

    def test_accumulation_resumes_after_revert(self, toy_model, site, r0, atr_step):
        """revert() clears state; it must not disable the instrument."""
        p = OjaPlasticity(toy_model, site=site, eta=1.0, mode="oja")
        with p:
            drive(toy_model, r0, atr_step, n=2)
            p.apply()
            p.revert()
            drive(toy_model, r0, atr_step, n=2)
            assert p._n_batches == 2
            p.apply()
        assert p.delta.norm().item() > 0


# --------------------------------------------------------------------------
# 9. apply() edge cases
# --------------------------------------------------------------------------

class TestApplyEdgeCases:

    def test_apply_with_nothing_accumulated_is_a_safe_noop(self, toy_model, site):
        """
        The documented loop calls apply() on a cadence, which can land before any
        forward pass. Raising there would break the caller's engine.
        """
        p = OjaPlasticity(toy_model, site=site, eta=1.0, mode="oja")
        w_before = p.module.weight.detach().clone()
        rep = p.apply()
        assert isinstance(rep, dict)
        assert rep["n_applied"] == 0
        assert torch.equal(p.module.weight, w_before)

    def test_apply_twice_does_not_double_apply(self, toy_model, site, r0, atr_step):
        """
        Applying stale statistics twice would double the effective eta for that
        step, invisibly, exactly where the eta sweep is meant to be precise.
        """
        p = OjaPlasticity(toy_model, site=site, eta=1.0, mode="oja")
        with p:
            drive(toy_model, r0, atr_step, n=3)
        first = p.apply()
        w_after_first = p.module.weight.detach().clone()
        delta_after_first = p.delta.clone()

        second = p.apply()
        assert torch.equal(p.module.weight, w_after_first)
        assert torch.equal(p.delta, delta_after_first)
        assert second["n_applied"] == first["n_applied"] == 1

    def test_apply_consumes_the_accumulator(self, toy_model, site, r0, atr_step):
        p = OjaPlasticity(toy_model, site=site, eta=1.0, mode="oja")
        with p:
            drive(toy_model, r0, atr_step, n=2)
            assert p._acc is not None
            p.apply()
            assert p._acc is None
            assert p._n_batches == 0


# --------------------------------------------------------------------------
# 10. nonfinite
# --------------------------------------------------------------------------

class TestNonFinite:

    def test_nonfinite_activations_are_rejected_not_absorbed(self, toy_model, site, r0):
        """
        DESIGN's failure table treats `nonfinite` as a diagnostic. If a nan got
        into the accumulator instead of raising the flag, the weight would be
        destroyed and the report would still look healthy.
        """
        p = OjaPlasticity(toy_model, site=site, eta=1.0, mode="oja")
        bad = r0.clone()
        bad[0, 0, 0] = float("inf")
        w_before = p.module.weight.detach().clone()

        with p:
            with torch.no_grad():
                toy_model(bad)

        assert p.nonfinite is True
        assert p._acc is None            # the poisoned batch was dropped, not accumulated
        assert p._n_batches == 0
        assert torch.equal(p.module.weight, w_before)

        rep = p.apply()
        assert rep["nonfinite"] is True
        assert torch.equal(p.module.weight, w_before)
        assert torch.isfinite(p.module.weight).all()

    def test_nan_activations_are_rejected(self, toy_model, site, r0):
        p = OjaPlasticity(toy_model, site=site, eta=1.0, mode="oja")
        bad = r0.clone()
        bad[0, 1, 2] = float("nan")
        w_before = p.module.weight.detach().clone()
        with p:
            with torch.no_grad():
                toy_model(bad)
        assert p.nonfinite is True
        assert torch.equal(p.module.weight, w_before)

    def test_overflowing_step_does_not_corrupt_the_weight(self, toy_model, site, r0, atr_step):
        """
        eta far too high is the expected operator error (DESIGN failure table).
        The weight must survive it so the run can be diagnosed rather than lost.
        """
        p = OjaPlasticity(toy_model, site=site, eta=1e39, mode="oja")   # overflows float32
        w_before = p.module.weight.detach().clone()
        with p:
            drive(toy_model, r0, atr_step, n=2)
        rep = p.apply()

        assert rep["nonfinite"] is True
        assert rep["n_applied"] == 0
        assert torch.equal(p.module.weight, w_before)
        assert torch.isfinite(p.module.weight).all()
        assert torch.equal(p.delta, torch.zeros_like(p.W0))

    def test_overflowing_accumulated_delta_is_flagged_not_silently_zeroed(
        self, toy_model, site, r0, atr_step
    ):
        """
        The step itself stays finite, so the isfinite(step) guard above lets it
        through; it is the *accumulated* delta whose float32 norm overflows.

        Failure means the ceiling rescale computes ceiling/inf == 0, zeroes the
        delta, and reports delta_frac == 0.0 with nonfinite == False -- a run
        that blew up and reads as a run where nothing happened. c3_divergence_demo
        sets max_delta_frac=1e9 deliberately, so this is on a path the repo
        itself takes, and it was found only against real GPT-2 weights.
        """
        p = OjaPlasticity(toy_model, site=site, eta=1e22, mode="hebb",
                          max_delta_frac=1e9)
        w_before = p.module.weight.detach().clone()
        with p:
            drive(toy_model, r0, atr_step, n=1)

            # The regime this test exists for: entries representable, float32
            # sum-of-squares not. eta is tuned to sit exactly there.
            step = p.eta * (p._acc / p._n_batches)
            assert torch.isfinite(step).all(), "entries must stay finite"
            assert step.norm().isinf(), "float32 norm must overflow; lower eta"
            assert math.isfinite(step.double().norm().item())

            rep = p.apply()

        ceiling = p.max_delta_frac * p.W0_norm
        assert rep["delta_norm"] > 0.0, "delta was silently zeroed by ceiling/inf"
        assert rep["clipped"] is True
        assert rep["delta_norm"] == pytest.approx(ceiling, rel=1e-5)
        assert torch.isfinite(p.module.weight).all()
        p.revert()
        assert torch.equal(p.module.weight, w_before)

    def test_finite_run_does_not_raise_the_flag(self, toy_model, site, r0, atr_step):
        """A flag that fires on healthy runs is worse than no flag."""
        p = OjaPlasticity(toy_model, site=site, eta=1e-3, mode="oja")
        with p:
            drive(toy_model, r0, atr_step, n=5)
        assert p.apply()["nonfinite"] is False


# --------------------------------------------------------------------------
# 11. report() / __repr__
# --------------------------------------------------------------------------

class TestReport:

    def test_report_keys_and_types(self, toy_model, site, r0, atr_step):
        """
        DESIGN's measurement plan logs delta_norm, delta_frac, clipped and
        nonfinite every iteration. A missing or wrongly-typed key breaks the log
        schema for a run that may take days.
        """
        p = OjaPlasticity(toy_model, site=site, eta=1.0, mode="oja")
        with p:
            drive(toy_model, r0, atr_step, n=2)
        rep = p.apply()

        assert set(rep) == set(REPORT_TYPES)
        for key, typ in REPORT_TYPES.items():
            assert isinstance(rep[key], typ), f"{key}: {type(rep[key])}"
        assert rep["site"] == site
        assert rep["mode"] == "oja"
        assert rep["eta"] == 1.0

    def test_delta_frac_is_delta_norm_over_W0_norm(self, toy_model, site, r0, atr_step):
        """delta_frac is what the ceiling and the eta sweep are both read off."""
        p = OjaPlasticity(toy_model, site=site, eta=1.0, mode="oja")
        with p:
            drive(toy_model, r0, atr_step, n=3)
        rep = p.apply()
        assert rep["delta_frac"] == pytest.approx(rep["delta_norm"] / p.W0_norm, rel=1e-9)
        assert rep["last_update_norm"] == pytest.approx(rep["delta_norm"], rel=1e-5)

    def test_report_is_a_fresh_dict(self, toy_model, site):
        """Callers log it per iteration; a shared mutable dict would rewrite history."""
        p = OjaPlasticity(toy_model, site=site)
        a, b = p.report(), p.report()
        assert a == b and a is not b

    def test_repr_names_the_site(self, toy_model, site):
        """Logs identify a run by its repr; a repr without the site is unidentifiable."""
        text = repr(OjaPlasticity(toy_model, site=site, mode="hebb", eta=1e-5))
        assert isinstance(text, str)
        assert site in text
        assert "hebb" in text


# --------------------------------------------------------------------------
# 12. candidate_sites()
# --------------------------------------------------------------------------

class TestCandidateSites:

    def test_returns_the_conv1d_matrices_only(self, toy_model):
        """
        README tells the operator to pick a site from this list. A LayerNorm gain
        vector in the list is a TypeError waiting at construction time; a missing
        mlp.c_proj hides the recommended first target.
        """
        sites = candidate_sites(toy_model)
        expected = [
            f"transformer.h.{i}.{mod}"
            for i in (0, 1)
            for mod in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
        ]
        assert sites == expected

    def test_excludes_layernorm_and_non_prefixed_modules(self, toy_model):
        sites = candidate_sites(toy_model)
        assert not any("ln_" in s for s in sites)
        assert "transformer.ln_f" not in sites
        for name, mod in toy_model.named_modules():
            w = getattr(mod, "weight", None)
            if torch.is_tensor(w) and w.dim() == 1:
                assert name not in sites

    def test_prefix_filters(self, toy_model):
        assert all(s.startswith("transformer.h.0") for s in candidate_sites(
            toy_model, prefix="transformer.h.0"))
        assert candidate_sites(toy_model, prefix="transformer.h.0") == \
            candidate_sites(toy_model)[:4]
        assert candidate_sites(toy_model, prefix="nonexistent") == []

    def test_every_candidate_is_actually_constructible(self, toy_model):
        """The list is only useful if every entry survives OjaPlasticity's own check."""
        for s in candidate_sites(toy_model):
            p = OjaPlasticity(toy_model, site=s)
            assert p.W0.dim() == 2


# --------------------------------------------------------------------------
# 13. transposed=True  (nn.Linear)
# --------------------------------------------------------------------------
# The module header promises: "For nn.Linear, weight is (n_out, n_in), so set
# `transposed=True` and we handle it." conftest's LinearModel is deliberately
# non-square, so the two conventions are distinguishable by shape.

class TestTransposed:

    def test_construction_reads_the_linear_layout(self, linear_model):
        p = OjaPlasticity(linear_model, site="proj", transposed=True)
        assert p.W0.shape == (linear_model.d_hidden, linear_model.d_model)
        assert p.delta.shape == p.W0.shape

    @pytest.mark.parametrize("site_name", ["proj", "out"])
    def test_collection_alone_does_not_corrupt_the_weight(self, linear_model, r0, atr_step, site_name):
        """
        Whatever else is wrong with the transposed path, watching must be inert:
        the failure has to be confined to apply(), or C0 fails on nn.Linear too.
        """
        p = OjaPlasticity(linear_model, site=site_name, eta=1e-3,
                          mode="oja", transposed=True)
        w_before = p.module.weight.detach().clone()
        with p:
            drive(linear_model, r0, atr_step, n=3)

        assert p._n_batches == 3
        assert torch.equal(p.module.weight, w_before)
        assert p.module.weight.shape == w_before.shape
        # The rules are written in (n_in, n_out); the comment at plasticity.py:205
        # says collection happens in that convention and is flipped on apply.
        assert p._acc.shape == (p.W0.shape[1], p.W0.shape[0])

    @pytest.mark.parametrize("site_name", ["proj", "out"])
    def test_transposed_apply_keeps_a_valid_weight(self, linear_model, r0, atr_step, site_name):
        """
        The documented contract for transposed=True: the weight stays a valid
        (n_out, n_in) matrix, the model still runs, and revert() restores it.
        Failure means nn.Linear targets are unusable -- which is every non-GPT-2
        model this scaffold might be pointed at.
        """
        p = OjaPlasticity(linear_model, site=site_name, eta=1.0,
                          mode="oja", transposed=True)
        w_before = p.module.weight.detach().clone()

        with p:
            drive(linear_model, r0, atr_step, n=3)
            p.apply()

        assert p.module.weight.shape == w_before.shape
        assert torch.isfinite(p.module.weight).all()
        assert not torch.equal(p.module.weight, w_before)
        with torch.no_grad():
            assert linear_model(r0).shape == r0.shape

        p.revert()
        assert torch.equal(p.module.weight, w_before)

    def test_transposed_update_is_written_in_the_weight_layout(self, linear_model, r0, atr_step):
        """
        Stronger form: the applied delta must be the transpose of the (n_in, n_out)
        update the rules compute. If it is written unflipped, a square nn.Linear
        target learns a scrambled update with no error at all.
        """
        p = OjaPlasticity(linear_model, site="proj", eta=1.0,
                          mode="hebb", transposed=True)
        with capture(p.module) as seen, p:
            drive(linear_model, r0, atr_step, n=3)
        expected = expected_update(seen)          # (n_in, n_out)
        p.apply()
        assert torch.allclose(p.delta, expected.transpose(0, 1), rtol=1e-5, atol=1e-9)
