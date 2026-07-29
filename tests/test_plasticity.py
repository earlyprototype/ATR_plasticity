"""
Tests for `plasticity.py`, against real GPT-2 small.

The module's own header says it was "written but never executed against real
weights". These tests are that execution. They are written in the spirit of the
controls table in README: each one names the thing that would be true of the
*experiment* if the test failed, not just the thing that would be true of the
code.

The learning-rule tests reconstruct the expected update from activations
captured by an *independent* forward hook installed by the test. If the
expectation came out of `plasticity.py` the test would only prove the module is
self-consistent.

Weight hygiene: the model is session-scoped, so every mutating test reverts in a
`finally`. `conftest._target_weight_unchanged` is the backstop.
"""

from __future__ import annotations

import contextlib
import math

import pytest
import torch
import torch.nn as nn

from conftest import D_MLP, D_MODEL, N_LAYER, REPORT_TYPES, resolve
from plasticity import OjaPlasticity, candidate_sites, subspace_projector


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


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0
    ).item()


# --------------------------------------------------------------------------
# 1. The Conv1D convention everything downstream rests on
# --------------------------------------------------------------------------

class TestConv1DConvention:
    """
    `plasticity.py`'s rules are written for `y = x @ W` with W of shape
    `(n_in, n_out)`. That is HuggingFace's Conv1D and the transpose of
    `nn.Linear`.
    """

    def test_real_site_is_a_conv1d_with_n_in_by_n_out_weight(self, gpt2, site, hf_conv1d):
        """
        The premise of the whole repo. If GPT-2's c_proj were `(n_out, n_in)`,
        `transposed=False` would be wrong at every site and the Oja decay term
        `W <y y^T>` would not even be conformable.
        """
        mod = resolve(gpt2, site)
        assert isinstance(mod, hf_conv1d)
        assert mod.weight.shape == (D_MLP, D_MODEL)     # (n_in, n_out)
        assert mod.bias.shape == (D_MODEL,)
        assert mod.weight.dtype == torch.float32
        # Non-square, so the two conventions are distinguishable by shape.
        assert mod.weight.shape[0] != mod.weight.shape[1]

    def test_forward_is_x_at_w_plus_b(self, gpt2, site, r0, atr_step):
        """
        The hook reads `inputs[0]` as x and `output` as y and forms `<x y^T>`.
        That is only the outer product of pre- and post-synaptic activity if the
        module really computes `x @ W + b` in this layout. If it did not, every
        learning rule in the repo would be transposed against real weights and
        no assertion about shapes would notice. Checked on activations from a
        real forward pass, not synthetic ones.
        """
        mod = resolve(gpt2, site)
        seen = []
        handle = mod.register_forward_hook(
            lambda _m, i, o: seen.append((i[0].detach().clone(), o.detach().clone()))
        )
        try:
            atr_step(gpt2, r0)
        finally:
            handle.remove()

        assert len(seen) == 1
        x, y = seen[0]
        assert x.shape[-1] == D_MLP and y.shape[-1] == D_MODEL
        manual = x @ mod.weight + mod.bias
        assert torch.allclose(manual, y, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# 2. Construction
# --------------------------------------------------------------------------

class TestConstruction:

    def test_invalid_mode_raises(self, gpt2, site):
        """A typo'd mode must not silently degrade to a different experiment."""
        with pytest.raises(ValueError, match="mode must be one of"):
            OjaPlasticity(gpt2, site=site, mode="oj")

    @pytest.mark.parametrize("mode", OjaPlasticity.VALID_MODES)
    def test_every_advertised_mode_constructs(self, gpt2, site, mode):
        """
        The modes in the docstring are the experiment and its controls; all must
        exist. `anti_hebb` joined them for EXP-002 -- a mode advertised in the
        docstring and missing from VALID_MODES is a ValueError an hour into a
        sweep. Its own behaviour is tested in `test_antihebbian.py`.
        """
        assert OjaPlasticity(gpt2, site=site, mode=mode).mode == mode

    @pytest.mark.parametrize(
        "bad_site",
        ["transformer.h.6.ln_1",   # LayerNorm: weight is 1-D
         "transformer.h.6.mlp"],   # container: no .weight at all
    )
    def test_non_matrix_site_raises_typeerror(self, gpt2, bad_site):
        """
        Attaching to a 1-D or weightless module would make <x y^T> meaningless;
        failing loudly at construction is the only safe behaviour.
        """
        with pytest.raises(TypeError, match="no 2-D .weight"):
            OjaPlasticity(gpt2, site=bad_site)

    def test_dotted_path_resolves_through_modulelist(self, gpt2, site):
        """
        `transformer.h.6` indexes an `nn.ModuleList`. If the numeric part were
        treated as an attribute, the entire site-addressing scheme would be
        unusable on the model the experiment actually runs on.
        """
        p = OjaPlasticity(gpt2, site=site)
        assert p.module is gpt2.transformer.h[6].mlp.c_proj
        assert p.module.weight.shape == (D_MLP, D_MODEL)   # Conv1D: (n_in, n_out)

    def test_W0_is_an_independent_snapshot(self, gpt2, site):
        """
        Every measurement in report() is relative to W0. If W0 aliased the live
        parameter, delta_frac would read 0 no matter how far the weight drifted
        -- a sweep would report a model that never moved while wrecking it.
        """
        p = OjaPlasticity(gpt2, site=site)
        assert torch.equal(p.W0, p.module.weight)
        assert p.W0.data_ptr() != p.module.weight.data_ptr()
        assert p.W0_norm > 0
        assert p.W0_norm == pytest.approx(p.module.weight.double().norm().item())

        try:
            with torch.no_grad():
                p.module.weight.add_(1.0)
            assert not torch.equal(p.W0, p.module.weight)
        finally:
            with torch.no_grad():
                p.module.weight.copy_(p.W0)

    def test_initial_state_is_clean(self, gpt2, site):
        """A fresh instance must not claim to have applied or clipped anything."""
        p = OjaPlasticity(gpt2, site=site)
        assert p._acc is None
        assert p._n_batches == 0
        assert p.n_applied == 0
        assert not p.clipped
        assert not p.nonfinite
        assert torch.equal(p.delta, torch.zeros_like(p.W0))


# --------------------------------------------------------------------------
# 3. Hook lifecycle
# --------------------------------------------------------------------------

class TestHookLifecycle:

    def test_install_is_idempotent(self, gpt2, site):
        """
        A double-installed hook would double-count every batch, halving the
        effective eta relative to what report() claims.
        """
        p = OjaPlasticity(gpt2, site=site)
        before = len(p.module._forward_hooks)
        p.install()
        assert len(p.module._forward_hooks) == before + 1
        handle = p._handle
        p.install()
        assert len(p.module._forward_hooks) == before + 1
        assert p._handle is handle
        p.remove()

    def test_remove_detaches(self, gpt2, site):
        """A leaked hook survives the experiment it belongs to and contaminates the next one."""
        p = OjaPlasticity(gpt2, site=site)
        before = len(p.module._forward_hooks)
        p.install()
        p.remove()
        assert len(p.module._forward_hooks) == before
        assert p._handle is None
        p.remove()   # second remove must not raise

    def test_context_manager_installs_and_removes(self, gpt2, site, r0, atr_step):
        """The documented usage is `with OjaPlasticity(...)`; a leaked hook here
        means the documented usage is the one that contaminates the next run."""
        p = OjaPlasticity(gpt2, site=site)
        baseline = len(p.module._forward_hooks)
        with p as ctx:
            assert ctx is p
            assert len(p.module._forward_hooks) == baseline + 1
            drive(gpt2, r0, atr_step, n=1)
            assert p._n_batches == 1
        assert len(p.module._forward_hooks) == baseline
        assert p._handle is None

    def test_context_manager_removes_on_exception(self, gpt2, site):
        """An engine that raises mid-run must not leave the model instrumented."""
        with pytest.raises(RuntimeError):
            with OjaPlasticity(gpt2, site=site) as p:
                raise RuntimeError("boom")
        assert p._handle is None
        assert len(p.module._forward_hooks) == 0

    def test_no_accumulation_after_remove(self, gpt2, site, r0, atr_step):
        """C1's premise: once detached, the object is inert. If it is not, the
        post-revert baseline is still being driven by the instrument."""
        p = OjaPlasticity(gpt2, site=site).install()
        drive(gpt2, r0, atr_step, n=2)
        assert p._n_batches == 2
        p.remove()
        drive(gpt2, r0, atr_step, n=3)
        assert p._n_batches == 2


# --------------------------------------------------------------------------
# 4. Statistics collection
# --------------------------------------------------------------------------

class TestCollection:

    def test_accumulator_shape_and_count(self, gpt2, site, r0, atr_step):
        """
        The accumulator is the update-in-waiting: it must carry the weight's
        (n_in, n_out) shape and one contribution per forward pass.
        """
        p = OjaPlasticity(gpt2, site=site)
        assert p._acc is None
        with p:
            drive(gpt2, r0, atr_step, n=3)
        assert p._acc is not None
        assert p._acc.shape == p.module.weight.shape == (D_MLP, D_MODEL)
        assert p._n_batches == 3

    @pytest.mark.parametrize("mode", OjaPlasticity.VALID_MODES)
    def test_collection_does_not_touch_the_weight(self, gpt2, site, r0, atr_step, mode):
        """
        Control C0: watching must not perturb. If the weight moves before any
        apply(), no downstream trajectory difference is interpretable.

        Over every mode, because the rule branch runs inside the hook: a mode
        that wrote from the hook rather than from apply() would break C0 for
        that mode alone, and only that mode's runs would be confounded.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1e-3, mode=mode)
        w_before = p.module.weight.detach().clone()
        with p:
            drive(gpt2, r0, atr_step, n=3)
        assert torch.equal(p.module.weight, w_before)

    def test_hook_does_not_disturb_the_trajectory(self, gpt2, site, r0, atr_step):
        """
        Control C0 proper, at eta=0: hooks plus a committed zero update must
        leave the iterated map bit-identical to the unhooked one on 124M real
        parameters. Failure means every plasticity result is confounded with an
        instrumentation artefact.

        The baseline runs on the same model object rather than a copy: at eta=0
        with mode="oja" nothing is written, which the weight assertion below
        confirms, and deep-copying GPT-2 per test costs more than the test.
        """
        r_clean = drive(gpt2, r0, atr_step, n=4)

        p = OjaPlasticity(gpt2, site=site, eta=0.0, mode="oja")
        w_before = p.module.weight.detach().clone()
        try:
            with p:
                r = r0
                for _ in range(4):
                    r = atr_step(gpt2, r)
                    p.apply()
            assert torch.equal(r, r_clean)
            assert torch.equal(p.module.weight, w_before)
            assert p.report()["delta_norm"] == 0.0
        finally:
            p.revert()


# --------------------------------------------------------------------------
# 5. Learning-rule correctness
# --------------------------------------------------------------------------

class TestLearningRule:
    """
    The heart of it. If these fail, the repo is not running the rule its README
    argues for, and the Oja-vs-Hebb framing is unsupported.
    """

    # Large enough that delta is resolvable in float32 at GPT-2's weight scale,
    # small enough that neither rule reaches the 0.05 ceiling: at this site
    # ||W0||_F = 164.9, the mean Hebb term has norm ~331 and the mean Oja term
    # ~3.5e4, so the binding constraint is Oja (measured delta_frac 0.011).
    ETA = 5e-5

    def test_hebb_update_matches_closed_form(self, gpt2, site, r0, atr_step):
        """dW = mean over firings of <x y^T>. A mismatch means mode='hebb' is
        not raw Hebb, so control C3 is comparing Oja against something else."""
        p = OjaPlasticity(gpt2, site=site, eta=self.ETA, mode="hebb")
        try:
            with capture(p.module) as seen, p:
                drive(gpt2, r0, atr_step, n=3)

            expected = expected_update(seen)
            rep = p.apply()

            assert len(seen) == 3
            assert torch.allclose(p.delta, self.ETA * expected, rtol=1e-5, atol=1e-9)
            assert torch.allclose(p.module.weight, p.W0 + self.ETA * expected,
                                  rtol=1e-6, atol=1e-9)
            assert not rep["clipped"]
            assert rep["n_applied"] == 1
        finally:
            p.revert()

    def test_oja_update_matches_closed_form(self, gpt2, site, r0, atr_step):
        """
        dW = <x y^T> - W <y y^T>. Losing the decay term silently turns the real
        experiment into control C3 (divergent raw Hebb).
        """
        p = OjaPlasticity(gpt2, site=site, eta=self.ETA, mode="oja")
        try:
            with capture(p.module) as seen, p:
                drive(gpt2, r0, atr_step, n=3)

            expected = expected_update(seen, w=p.W0)   # no delta yet, so W_eff == W0
            rep = p.apply()

            assert not rep["clipped"]
            assert torch.allclose(p.delta, self.ETA * expected, rtol=1e-5, atol=1e-9)
            assert torch.allclose(p.module.weight, p.W0 + self.ETA * expected,
                                  rtol=1e-6, atol=1e-9)
        finally:
            p.revert()

    def test_oja_differs_from_hebb_by_exactly_the_decay_term(self, gpt2, site, r0, atr_step):
        """
        The decay term is the whole argument of README's "Why Oja rather than
        Hebb". It must be present and it must be W <y y^T>, nothing else.

        The two arms run sequentially on the same model with a revert between,
        so both start from the same W0 -- which is what makes their deltas
        comparable at all.
        """
        p_h = OjaPlasticity(gpt2, site=site, eta=self.ETA, mode="hebb")
        try:
            with capture(p_h.module) as seen_h, p_h:
                drive(gpt2, r0, atr_step, n=3)
            p_h.apply()
            delta_h = p_h.delta.clone()
        finally:
            p_h.revert()

        p_o = OjaPlasticity(gpt2, site=site, eta=self.ETA, mode="oja")
        try:
            with p_o:
                drive(gpt2, r0, atr_step, n=3)
            p_o.apply()
            delta_o = p_o.delta.clone()
        finally:
            p_o.revert()

        decay = None
        for _x, y in seen_h:
            term = p_h.W0 @ ((y.transpose(0, 1) @ y) / y.shape[0])
            decay = term if decay is None else decay + term
        decay = decay / len(seen_h)

        assert torch.allclose(delta_h - delta_o, self.ETA * decay,
                              rtol=1e-4, atol=1e-9)
        assert decay.norm().item() > 0

    def test_oja_uses_the_effective_weight_after_a_first_apply(self, gpt2, site, r0, atr_step):
        """
        The decay term must track the live weight, not the frozen W0. Using W0
        forever makes the rule non-local in time and breaks Oja's fixed point --
        which is the only reason the rule is bounded, and therefore the only
        reason C3 comes out the way README says it does.
        """
        p = OjaPlasticity(gpt2, site=site, eta=self.ETA, mode="oja")
        try:
            with capture(p.module) as seen, p:
                r = drive(gpt2, r0, atr_step, n=2)
                p.apply()
                w_eff = p.W0 + p.delta      # what the second round should decay against
                assert p.delta.norm().item() > 0
                seen.clear()
                delta_after_first = p.delta.clone()
                drive(gpt2, r, atr_step, n=2)

            expected = expected_update(seen, w=w_eff)
            rep = p.apply()
            assert not rep["clipped"]       # or the comparison is against a rescaled delta
            assert torch.allclose(p.delta - delta_after_first, self.ETA * expected,
                                  rtol=1e-4, atol=1e-9)
        finally:
            p.revert()

    def test_batches_are_averaged_not_summed(self, gpt2, site, r0, atr_step):
        """
        apply() divides by _n_batches. If it summed, the effective learning rate
        would scale with cadence and every eta sweep would be mislabelled.
        """
        p = OjaPlasticity(gpt2, site=site, eta=self.ETA, mode="hebb")
        try:
            with capture(p.module) as seen, p:
                drive(gpt2, r0, atr_step, n=4)
            summed = expected_update(seen) * len(seen)
            p.apply()
            assert not torch.allclose(p.delta, self.ETA * summed, rtol=1e-3, atol=1e-9)
            assert torch.allclose(p.delta, self.ETA * summed / 4, rtol=1e-5, atol=1e-9)
        finally:
            p.revert()

    def test_oja_update_is_larger_than_hebb_at_real_weight_scale(self, gpt2, site, r0, atr_step):
        """
        Recorded because it inverts the intuition a small-init toy network
        gives. At toy init scale (std 0.02) the Oja decay contributes under 1%
        of the update norm; on real GPT-2 weights `||W||_F` is ~165 at this site
        and the decay term dominates, so the Oja delta is ~100x the Hebb delta
        from the first apply onward. Anyone reading a delta_frac trace and
        expecting "Oja = Hebb minus a small correction" will misread the run,
        and any eta chosen for Hebb is far too large for Oja at the same site.

        `anti_hebb` is measured in the same sweep because the same trap catches
        the operator twice: it differs from Oja only in the sign of the
        *smaller* term, so at matched eta its update norm sits within a percent
        of Oja's (measured 36,496 against 36,339, with Hebb at 330). An eta
        chosen from either is right for the other -- and, less comfortably, a
        norm trace cannot tell you which of the two rules actually ran.
        """
        norms = {}
        for mode in ("hebb", "oja", "anti_hebb"):
            p = OjaPlasticity(gpt2, site=site, eta=1e-6, mode=mode)
            try:
                with p:
                    drive(gpt2, r0, atr_step, n=2)
                norms[mode] = p.apply()["delta_norm"]
            finally:
                p.revert()
        assert norms["oja"] > 10 * norms["hebb"]
        assert norms["anti_hebb"] > 10 * norms["hebb"]
        assert norms["anti_hebb"] == pytest.approx(norms["oja"], rel=0.05)


# --------------------------------------------------------------------------
# 6. mode="off"
# --------------------------------------------------------------------------

class TestModeOff:

    def test_off_accumulates_but_never_writes(self, gpt2, site, r0, atr_step):
        """
        Control C0/C1's instrument: statistics with no effect. Any weight motion
        here means the "off" baseline is not a baseline.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1.0, mode="off")
        w_before = p.module.weight.detach().clone()
        try:
            with p:
                drive(gpt2, r0, atr_step, n=3)
                assert p._acc is not None and p._n_batches == 3
                rep = p.apply()

            assert torch.equal(p.module.weight, w_before)   # bit-for-bit
            assert rep["delta_norm"] == 0.0
            assert rep["delta_frac"] == 0.0
            assert rep["n_applied"] == 0
            assert not rep["clipped"]
        finally:
            p.revert()

    def test_off_does_not_alter_the_trajectory(self, gpt2, site, r0, atr_step):
        """The weight not moving is necessary; the trajectory not moving is the
        claim C0 actually makes."""
        r_clean = drive(gpt2, r0, atr_step, n=4)
        p = OjaPlasticity(gpt2, site=site, eta=1.0, mode="off")
        try:
            with p:
                r = r0
                for _ in range(4):
                    r = atr_step(gpt2, r)
                    p.apply()
            assert torch.equal(r, r_clean)
        finally:
            p.revert()


# --------------------------------------------------------------------------
# 7. mode="random"  (Control C2)
# --------------------------------------------------------------------------

class TestModeRandom:

    ETA = 1e-6

    def _delta_after(self, gpt2, site, r0, atr_step, **kwargs):
        """One accumulate-and-apply cycle, weights handed back untouched."""
        p = OjaPlasticity(gpt2, site=site, eta=self.ETA, **kwargs)
        try:
            with p:
                drive(gpt2, r0, atr_step, n=3)
            rep = p.apply()
            return p.delta.clone(), rep
        finally:
            p.revert()

    def test_random_is_reproducible_for_a_fixed_seed(self, gpt2, site, r0, atr_step):
        """
        C2 is a comparison. If the control is not reproducible, a difference
        between two runs cannot be attributed to the rule.
        """
        a, _ = self._delta_after(gpt2, site, r0, atr_step, mode="random", seed=0)
        b, _ = self._delta_after(gpt2, site, r0, atr_step, mode="random", seed=0)
        assert torch.equal(a, b)

    def test_random_differs_across_seeds(self, gpt2, site, r0, atr_step):
        """A seed that does not change the draw would make C2 a single sample
        dressed up as a control."""
        a, _ = self._delta_after(gpt2, site, r0, atr_step, mode="random", seed=0)
        b, _ = self._delta_after(gpt2, site, r0, atr_step, mode="random", seed=1)
        assert not torch.allclose(a, b)
        assert abs(cosine(a, b)) < 0.3

    def test_random_points_elsewhere_than_oja(self, gpt2, site, r0, atr_step):
        """
        The point of C2 is that only the *direction* differs. A random matrix
        aligned with the Oja update would test nothing.
        """
        rand, _ = self._delta_after(gpt2, site, r0, atr_step, mode="random", seed=0)
        oja, _ = self._delta_after(gpt2, site, r0, atr_step, mode="oja")
        assert abs(cosine(rand, oja)) < 0.2

    def test_random_is_norm_matched_to_oja_on_real_weights(self, gpt2, site, r0, atr_step):
        """
        README defect 2: `mode="random"` was norm-matched to the raw Hebb term
        rather than to Oja, and the bias grew with weight and activation
        magnitude -- so real GPT-2 scale is exactly where it mattered and a toy
        is exactly where it hides. At this site the decay term dominates the
        Hebb term by ~100x, so a regression would mismatch the two arms by
        orders of magnitude, not percent.

        C2 is the control README calls decisive: if the arms differ in
        magnitude, the comparison of Oja against a random direction is void.
        """
        _, rep_o = self._delta_after(gpt2, site, r0, atr_step, mode="oja")
        _, rep_r = self._delta_after(gpt2, site, r0, atr_step, mode="random", seed=0)

        assert rep_o["delta_norm"] > 0
        assert rep_r["delta_norm"] == pytest.approx(rep_o["delta_norm"], rel=1e-5)


# --------------------------------------------------------------------------
# 8. max_delta_frac ceiling
# --------------------------------------------------------------------------

class TestCeiling:
    """
    Real weight norms are far from a toy's and the ceiling is a *fraction*, so
    the guard is checked at the scale it will actually run at.
    """

    def test_large_eta_is_clipped_at_the_ceiling(self, gpt2, site, r0, atr_step):
        """
        README: the ceiling "is the guard against silently destroying the
        model". An unenforced ceiling on a 124M-parameter model is a wrecked
        matrix and a report that still looks healthy.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1e3, mode="oja", max_delta_frac=0.05)
        try:
            with p:
                drive(gpt2, r0, atr_step, n=2)
            rep = p.apply()

            assert rep["clipped"] is True
            assert rep["nonfinite"] is False
            # Held to float32 precision, not exactly. Rescaling a 3072x768
            # float32 matrix by a float64 factor leaves the norm ~2.7e-8
            # relative above the ceiling, and correcting again is a no-op --
            # the correction factor rounds to 1.0 in float32.
            assert rep["delta_frac"] <= 0.05 + 1e-6
            assert rep["delta_frac"] == pytest.approx(0.05, rel=1e-4)
            assert torch.isfinite(p.module.weight).all()
        finally:
            p.revert()

    def test_tiny_eta_is_not_clipped(self, gpt2, site, r0, atr_step):
        """A ceiling that fires in the safe regime sends the operator chasing a
        regime that is not there, and caps an eta sweep before it begins."""
        p = OjaPlasticity(gpt2, site=site, eta=1e-9, mode="oja", max_delta_frac=0.05)
        try:
            with p:
                drive(gpt2, r0, atr_step, n=2)
            rep = p.apply()

            assert rep["clipped"] is False
            assert 0.0 < rep["delta_frac"] < 0.05
        finally:
            p.revert()

    def test_repeated_applies_accumulate_toward_the_ceiling(self, gpt2, site, r0, atr_step):
        """
        The ceiling is on *total* drift from W0, not on one step. If it were
        per-step, a long run could walk arbitrarily far while never flagging,
        and `max_delta_frac=0.05` would mean nothing at all.

        eta is chosen so the trace climbs for three applies and then binds:
        measured [0.0073, 0.0120, 0.0162, 0.02, 0.02, 0.02] at this site.
        """
        p = OjaPlasticity(gpt2, site=site, eta=2.9e-5, mode="oja", max_delta_frac=0.02)
        fracs = []
        try:
            with p:
                r = r0
                for _ in range(6):
                    r = drive(gpt2, r, atr_step, n=1)
                    fracs.append(p.apply()["delta_frac"])

            assert all(f <= 0.02 + 1e-6 for f in fracs)
            assert fracs[0] < fracs[1] < fracs[2]
            # Non-decreasing to float32 precision; the ceiling, not the rule, ends it.
            assert not any(fracs[i] > fracs[i + 1] + 1e-7 for i in range(len(fracs) - 1))
            assert not any(f > 0.02 for f in fracs[:2])
            assert p.clipped is True
            assert fracs[-1] == pytest.approx(0.02, rel=1e-5)
        finally:
            p.revert()


# --------------------------------------------------------------------------
# 9. revert()
# --------------------------------------------------------------------------

class TestRevert:

    def test_revert_clears_the_diagnostics_not_just_the_weights(
        self, gpt2, site, r0, atr_step
    ):
        """
        report() is the per-iteration log schema DESIGN.md specifies. If revert()
        restores the weights but leaves `clipped` set, an instance reused across
        a sweep reports a run as clipped that never clipped, and `n_applied`
        keeps counting across resets -- so the log says the ceiling bound at an
        eta where it did not, which is exactly the diagnostic an operator uses
        to decide whether a landscape change is real or an artefact of clipping.

        The controls dodge this by building a fresh instance per arm. A caller
        logging one instance does not, and the docs never told them to.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1e3, mode="oja", max_delta_frac=1e-4)
        try:
            with p:
                drive(gpt2, r0, atr_step, n=2)
                dirty = p.apply()

            # Precondition: this run really did trip the flags being tested.
            assert dirty["clipped"] is True
            assert dirty["n_applied"] == 1
            assert dirty["delta_norm"] > 0.0
            assert dirty["last_update_norm"] > 0.0
        finally:
            p.revert()

        clean = p.report()
        assert clean["clipped"] is False
        assert clean["nonfinite"] is False
        assert clean["n_applied"] == 0
        assert clean["delta_norm"] == 0.0
        assert clean["delta_frac"] == 0.0
        assert clean["last_update_norm"] == 0.0

    def test_a_reused_instance_reports_the_current_run_not_the_previous_one(
        self, gpt2, site, r0, atr_step
    ):
        """
        The same object, a clipping run then a safe one. Failure means the
        second run inherits the first's verdict -- the concrete way the stale
        diagnostics mislead, since a sweep down an eta ladder reusing one
        instance would report every eta as clipped once the largest one was.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1e3, mode="oja", max_delta_frac=1e-4)
        try:
            with p:
                drive(gpt2, r0, atr_step, n=2)
                assert p.apply()["clipped"] is True
            p.revert()

            p.eta = 1e-9                      # far below the ceiling
            with p:
                drive(gpt2, r0, atr_step, n=2)
                second = p.apply()

            assert second["clipped"] is False, "verdict inherited from the previous run"
            assert second["n_applied"] == 1, "n_applied counted across the reset"
        finally:
            p.revert()

    def test_revert_restores_bit_exactly(self, gpt2, site, r0, atr_step):
        """
        Control C1, on a 3072x768 matrix that took real compute to produce.
        Anything short of bit-exact means hidden state is accumulating and no
        A/B comparison in this repo is trustworthy -- and, with a session-scoped
        model, that every later test ran against a different GPT-2.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1e-4, mode="oja")
        w_before = p.module.weight.detach().clone()
        try:
            with p:
                drive(gpt2, r0, atr_step, n=3)
                p.apply()
                assert not torch.equal(p.module.weight, w_before)
                drive(gpt2, r0, atr_step, n=1)   # leave something pending too
        finally:
            p.revert()

        assert torch.equal(p.module.weight, w_before)
        assert torch.equal(p.delta, torch.zeros_like(p.W0))
        assert p._acc is None
        assert p._n_batches == 0

        rep = p.report()
        assert rep["delta_norm"] == 0.0
        assert rep["delta_frac"] == 0.0

    def test_revert_restores_the_trajectory(self, gpt2, site, r0, atr_step):
        """
        Restoring the weight is necessary but not sufficient: C1's claim is
        about the *trajectory*. If the iterated map does not come back, state is
        accumulating somewhere other than the weight.
        """
        r_clean = drive(gpt2, r0, atr_step, n=3)

        p = OjaPlasticity(gpt2, site=site, eta=1e-4, mode="oja")
        try:
            with p:
                drive(gpt2, r0, atr_step, n=3)
                p.apply()
            perturbed = drive(gpt2, r0, atr_step, n=3)
            assert not torch.equal(perturbed, r_clean)   # the run was non-vacuous
        finally:
            p.revert()

        assert torch.equal(drive(gpt2, r0, atr_step, n=3), r_clean)

    def test_accumulation_resumes_after_revert(self, gpt2, site, r0, atr_step):
        """revert() clears state; if it also disabled the instrument, a sweep
        would silently stop learning after its first reset."""
        p = OjaPlasticity(gpt2, site=site, eta=1e-4, mode="oja")
        try:
            with p:
                drive(gpt2, r0, atr_step, n=2)
                p.apply()
                p.revert()
                drive(gpt2, r0, atr_step, n=2)
                assert p._n_batches == 2
                p.apply()
            assert p.delta.norm().item() > 0
        finally:
            p.revert()


# --------------------------------------------------------------------------
# 10. apply() edge cases and the report contract
# --------------------------------------------------------------------------

class TestApplyEdgeCases:

    def test_apply_with_nothing_accumulated_is_a_safe_noop(self, gpt2, site):
        """
        The documented loop calls apply() on a cadence, which can land before
        any forward pass. Raising there would break the caller's engine.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1.0, mode="oja")
        w_before = p.module.weight.detach().clone()
        rep = p.apply()
        assert isinstance(rep, dict)
        assert rep["n_applied"] == 0
        assert torch.equal(p.module.weight, w_before)

    def test_apply_twice_does_not_double_apply(self, gpt2, site, r0, atr_step):
        """
        Applying stale statistics twice would double the effective eta for that
        step, invisibly, exactly where the eta sweep is meant to be precise.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1e-5, mode="oja")
        try:
            with p:
                drive(gpt2, r0, atr_step, n=3)
            first = p.apply()
            w_after_first = p.module.weight.detach().clone()
            delta_after_first = p.delta.clone()

            second = p.apply()
            assert torch.equal(p.module.weight, w_after_first)
            assert torch.equal(p.delta, delta_after_first)
            assert second["n_applied"] == first["n_applied"] == 1
        finally:
            p.revert()

    def test_apply_consumes_the_accumulator(self, gpt2, site, r0, atr_step):
        """Statistics left in the accumulator would be re-applied on the next
        cadence tick, so one batch would contribute to two updates."""
        p = OjaPlasticity(gpt2, site=site, eta=1e-5, mode="oja")
        try:
            with p:
                drive(gpt2, r0, atr_step, n=2)
                assert p._acc is not None
                p.apply()
                assert p._acc is None
                assert p._n_batches == 0
        finally:
            p.revert()

    def test_apply_moves_the_weight_and_report_describes_the_move(
        self, gpt2, site, r0, atr_step
    ):
        """
        The read/write half of the scaffold, end to end. `apply()` must actually
        move the matrix, `report()` must describe the move in the schema the run
        log depends on, and `revert()` must put back the original bits -- not
        something within float tolerance of them.
        """
        w_before = resolve(gpt2, site).weight.detach().clone()

        p = OjaPlasticity(gpt2, site=site, eta=1e-5, mode="oja")
        try:
            with p:
                drive(gpt2, r0, atr_step, n=3)
                rep = p.apply()

            assert set(rep) == set(REPORT_TYPES)
            for key, typ in REPORT_TYPES.items():
                assert isinstance(rep[key], typ), f"{key}: {type(rep[key])}"
            assert rep["site"] == site
            assert rep["mode"] == "oja"
            assert rep["n_applied"] == 1
            assert rep["nonfinite"] is False
            assert rep["clipped"] is False
            assert 0.0 < rep["delta_frac"] < 0.05
            assert math.isfinite(rep["delta_norm"]) and rep["delta_norm"] > 0
            assert rep["delta_frac"] == pytest.approx(
                rep["delta_norm"] / p.W0_norm, rel=1e-9
            )

            w_after = resolve(gpt2, site).weight
            assert not torch.equal(w_after, w_before)
            assert torch.isfinite(w_after).all()
        finally:
            p.revert()

        assert torch.equal(resolve(gpt2, site).weight, w_before)
        assert torch.equal(p.delta, torch.zeros_like(p.W0))
        assert p.report()["delta_norm"] == 0.0


# --------------------------------------------------------------------------
# 11. nonfinite
# --------------------------------------------------------------------------

class TestNonFinite:

    def test_nonfinite_activations_are_rejected_not_absorbed(self, gpt2, site, r0, atr_step):
        """
        DESIGN's failure table treats `nonfinite` as a diagnostic. If a nan got
        into the accumulator instead of raising the flag, the weight would be
        destroyed and the report would still look healthy.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1.0, mode="oja")
        bad = r0.clone()
        bad[0, 0, 0] = float("inf")
        w_before = p.module.weight.detach().clone()

        try:
            with p:
                atr_step(gpt2, bad)

            assert p.nonfinite is True
            assert p._acc is None        # the poisoned batch was dropped, not accumulated
            assert p._n_batches == 0
            assert torch.equal(p.module.weight, w_before)

            rep = p.apply()
            assert rep["nonfinite"] is True
            assert torch.equal(p.module.weight, w_before)
            assert torch.isfinite(p.module.weight).all()
        finally:
            p.revert()

    def test_nan_activations_are_rejected(self, gpt2, site, r0, atr_step):
        """Same guard, the other non-finite value: a nan that is absorbed
        propagates to the weight and every later iteration reads garbage."""
        p = OjaPlasticity(gpt2, site=site, eta=1.0, mode="oja")
        bad = r0.clone()
        bad[0, 1, 2] = float("nan")
        w_before = p.module.weight.detach().clone()
        try:
            with p:
                atr_step(gpt2, bad)
            assert p.nonfinite is True
            assert torch.equal(p.module.weight, w_before)
        finally:
            p.revert()

    def test_overflowing_step_does_not_corrupt_the_weight(self, gpt2, site, r0, atr_step):
        """
        eta far too high is the expected operator error (DESIGN failure table).
        The weight must survive it so the run can be diagnosed rather than lost
        -- on a session-scoped model, "lost" means every later result too.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1e39, mode="oja")   # overflows float32
        w_before = p.module.weight.detach().clone()
        try:
            with p:
                drive(gpt2, r0, atr_step, n=2)
            rep = p.apply()

            assert rep["nonfinite"] is True
            assert rep["n_applied"] == 0
            assert torch.equal(p.module.weight, w_before)
            assert torch.isfinite(p.module.weight).all()
            assert torch.equal(p.delta, torch.zeros_like(p.W0))
        finally:
            p.revert()

    def test_overflowing_accumulated_delta_is_flagged_not_silently_zeroed(
        self, gpt2, site, r0, atr_step
    ):
        """
        The step itself stays finite, so the isfinite(step) guard above lets it
        through; it is the *accumulated* delta whose float32 norm overflows.

        Failure means the ceiling rescale computes ceiling/inf == 0, zeroes the
        delta, and reports delta_frac == 0.0 with nonfinite == False -- a run
        that blew up and reads as a run where nothing happened. c3_divergence_demo
        sets max_delta_frac=1e9 deliberately, so this is on a path the repo
        itself takes.

        eta re-derived on this 3072x768 matrix rather than carried over from a
        smaller one: the mean Hebb term here has ||.||_F = 331 and max entry
        26.5, so the float32 sum-of-squares overflows above eta ~ 5.6e16
        (measured: 5e16 finite, 1e17 inf) while entries stay representable up to
        eta ~ 1.3e37. 1e22 sits in the middle of that window. The asserts below
        pin the regime, so if the window ever moves the test says so instead of
        quietly testing the ordinary clipping path.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1e22, mode="hebb",
                          max_delta_frac=1e9)
        w_before = p.module.weight.detach().clone()
        try:
            with p:
                drive(gpt2, r0, atr_step, n=1)

                # The regime this test exists for: entries representable, float32
                # sum-of-squares not. eta is tuned to sit exactly there.
                step = p.eta * (p._acc / p._n_batches)
                assert torch.isfinite(step).all(), "entries must stay finite; lower eta"
                assert step.norm().isinf(), "float32 norm must overflow; raise eta"
                assert math.isfinite(step.double().norm().item())

                rep = p.apply()

            ceiling = p.max_delta_frac * p.W0_norm
            assert rep["delta_norm"] > 0.0, "delta was silently zeroed by ceiling/inf"
            assert rep["clipped"] is True
            assert rep["delta_norm"] == pytest.approx(ceiling, rel=1e-5)
            assert torch.isfinite(p.module.weight).all()
        finally:
            p.revert()
        assert torch.equal(p.module.weight, w_before)

    def test_finite_run_does_not_raise_the_flag(self, gpt2, site, r0, atr_step):
        """A flag that fires on healthy runs is worse than no flag: the operator
        learns to ignore it, and the one real blow-up goes unnoticed."""
        p = OjaPlasticity(gpt2, site=site, eta=1e-6, mode="oja")
        try:
            with p:
                drive(gpt2, r0, atr_step, n=4)
            assert p.apply()["nonfinite"] is False
        finally:
            p.revert()


# --------------------------------------------------------------------------
# 12. report() / __repr__
# --------------------------------------------------------------------------

class TestReport:

    def test_report_keys_and_types(self, gpt2, site, r0, atr_step):
        """
        DESIGN's measurement plan logs delta_norm, delta_frac, clipped and
        nonfinite every iteration. A missing or wrongly-typed key breaks the log
        schema for a run that may take days.
        """
        p = OjaPlasticity(gpt2, site=site, eta=1e-5, mode="oja")
        try:
            with p:
                drive(gpt2, r0, atr_step, n=2)
            rep = p.apply()

            assert set(rep) == set(REPORT_TYPES)
            for key, typ in REPORT_TYPES.items():
                assert isinstance(rep[key], typ), f"{key}: {type(rep[key])}"
            assert rep["site"] == site
            assert rep["mode"] == "oja"
            assert rep["eta"] == 1e-5
        finally:
            p.revert()

    def test_delta_frac_is_delta_norm_over_W0_norm(self, gpt2, site, r0, atr_step):
        """delta_frac is what the ceiling and the eta sweep are both read off.
        If it is not that ratio, both are calibrated against a fiction."""
        p = OjaPlasticity(gpt2, site=site, eta=1e-5, mode="oja")
        try:
            with p:
                drive(gpt2, r0, atr_step, n=3)
            rep = p.apply()
            assert rep["delta_frac"] == pytest.approx(rep["delta_norm"] / p.W0_norm, rel=1e-9)
            assert rep["last_update_norm"] == pytest.approx(rep["delta_norm"], rel=1e-5)
        finally:
            p.revert()

    def test_report_is_a_fresh_dict(self, gpt2, site):
        """Callers log it per iteration; a shared mutable dict would rewrite
        history, so the trace on disk would show the last state at every step."""
        p = OjaPlasticity(gpt2, site=site)
        a, b = p.report(), p.report()
        assert a == b and a is not b

    def test_repr_names_the_site(self, gpt2, site):
        """Logs identify a run by its repr; a repr without the site cannot be
        matched to the sweep entry that produced it."""
        text = repr(OjaPlasticity(gpt2, site=site, mode="hebb", eta=1e-5))
        assert isinstance(text, str)
        assert site in text
        assert "hebb" in text


# --------------------------------------------------------------------------
# 13. candidate_sites()
# --------------------------------------------------------------------------

class TestCandidateSites:

    def test_returns_forty_eight_sites_on_gpt2_small(self, gpt2, site):
        """
        12 layers x {attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj}. A count
        other than 48 means the enumerator is either missing targets the
        operator is told to choose from, or offering ones that are not
        plasticity targets at all.
        """
        sites = candidate_sites(gpt2)
        assert len(sites) == 48
        assert len(set(sites)) == 48
        assert site in sites
        expected = [
            f"transformer.h.{i}.{m}"
            for i in range(N_LAYER)
            for m in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
        ]
        assert sites == expected

    def test_every_site_resolves_to_a_two_d_weight(self, gpt2):
        """
        README sends the operator to this list to pick a site. An entry that
        does not survive `OjaPlasticity`'s own 2-D check is a TypeError waiting
        at construction time, hours into a sweep.
        """
        for s in candidate_sites(gpt2):
            mod = resolve(gpt2, s)
            assert mod.weight.dim() == 2
            assert OjaPlasticity(gpt2, site=s).W0.shape == mod.weight.shape

    def test_layernorms_embeddings_and_one_d_parameters_are_excluded(self, gpt2):
        """
        A LayerNorm gain is 1-D: `<x y^T>` against it is not a matrix and the
        rule is meaningless. The embeddings are 2-D and would pass a naive
        filter, but they sit outside the block stack and are not what "one
        weight matrix under the loop" means.
        """
        sites = candidate_sites(gpt2)
        assert not any(".ln_" in s for s in sites)
        assert "transformer.ln_f" not in sites
        assert "transformer.wte" not in sites and "transformer.wpe" not in sites
        assert not any(s == "lm_head" for s in sites)
        for name, mod in gpt2.named_modules():
            w = getattr(mod, "weight", None)
            if torch.is_tensor(w) and w.dim() != 2:
                assert name not in sites
        # Biases are parameters, not modules, so they cannot appear at all.
        assert not any(s.endswith(".bias") for s in sites)

    def test_prefix_filters(self, gpt2):
        """The prefix is how a layer sweep is scoped. If it does not filter, a
        sweep meant for one layer silently runs the whole stack."""
        first_layer = candidate_sites(gpt2, prefix="transformer.h.0")
        assert all(s.startswith("transformer.h.0") for s in first_layer)
        assert first_layer == candidate_sites(gpt2)[:4]
        assert candidate_sites(gpt2, prefix="nonexistent") == []

    def test_c_attn_is_offered_but_documented_as_the_one_to_avoid(self, gpt2):
        """
        `attn.c_attn` packs Q, K and V into one (768, 2304) matrix, so a
        Hebbian update there is three experiments at once. It is a legitimate
        2-D site and is listed; the warning has to live in the documentation,
        and if that warning is ever deleted the operator loses the only signal
        that the default-looking choice is the wrong one.
        """
        sites = candidate_sites(gpt2)
        c_attn = [s for s in sites if s.endswith("attn.c_attn")]
        assert len(c_attn) == N_LAYER
        assert resolve(gpt2, c_attn[0]).weight.shape == (D_MODEL, 3 * D_MODEL)
        doc = candidate_sites.__doc__ or ""
        assert "attn.c_attn" in doc and "Avoid" in doc


# --------------------------------------------------------------------------
# 14. transposed=True  (nn.Linear)
# --------------------------------------------------------------------------
# GPT-2 is Conv1D at every site, so there is NO real-model target for the
# `transposed=True` branch. The module below exists solely to exercise that code
# path; it is a code-path fixture, not a model, and nothing about GPT-2 or about
# the experiment should be inferred from it. Deleting it would drop the only
# coverage of a branch `plasticity.py` advertises in its header ("For nn.Linear,
# weight is (n_out, n_in), so set `transposed=True` and we handle it") and that
# README records as defect 1 -- it was a bare `pass` and never transposed at all.
#
# Both layers are deliberately NON-SQUARE (768 <-> 3072, GPT-2's own widths). A
# square weight makes the two conventions indistinguishable by shape, so a
# transpose bug survives every assertion that only checks shapes and silently
# applies the update in the wrong orientation. Non-square is the only honest
# test of `transposed=True`.

class LinearProbe(nn.Module):
    """An `nn.Linear` target for the `transposed=True` path. Not a model."""

    def __init__(self, d_model: int = D_MODEL, d_hidden: int = D_MLP):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.proj = nn.Linear(d_model, d_hidden, bias=False)   # weight (3072, 768)
        self.out = nn.Linear(d_hidden, d_model, bias=False)    # weight (768, 3072)

    def forward(self, x):
        return self.out(torch.tanh(self.proj(x)))


@pytest.fixture
def linear_probe() -> LinearProbe:
    torch.manual_seed(1234)
    m = LinearProbe()
    m.eval()
    m.requires_grad_(False)
    return m


@pytest.fixture
def linear_step():
    """`atr_step` for the probe: it is not a transformer, so it has no
    `.transformer` attribute for the GPT-2 step to call."""

    def _step(model, r):
        with torch.no_grad():
            out = model(r)
        return out / (out.norm() + 1e-12)

    return _step


class TestTransposed:

    def test_construction_reads_the_linear_layout(self, linear_probe):
        """If W0 came back in the rules' (n_in, n_out) convention instead of the
        weight's own, delta would not even be addable to the parameter."""
        p = OjaPlasticity(linear_probe, site="proj", transposed=True)
        assert p.W0.shape == (linear_probe.d_hidden, linear_probe.d_model)
        assert p.delta.shape == p.W0.shape

    @pytest.mark.parametrize("site_name", ["proj", "out"])
    def test_collection_alone_does_not_corrupt_the_weight(
        self, linear_probe, r0, linear_step, site_name
    ):
        """
        Whatever else is wrong with the transposed path, watching must be inert:
        the failure has to be confined to apply(), or C0 fails on nn.Linear too
        and the scaffold cannot be pointed at any non-GPT-2 model.
        """
        p = OjaPlasticity(linear_probe, site=site_name, eta=1e-3,
                          mode="oja", transposed=True)
        w_before = p.module.weight.detach().clone()
        with p:
            drive(linear_probe, r0, linear_step, n=3)

        assert p._n_batches == 3
        assert torch.equal(p.module.weight, w_before)
        assert p.module.weight.shape == w_before.shape
        # The rules are written in (n_in, n_out); plasticity.py's comment says
        # collection happens in that convention and is flipped on apply.
        assert p._acc.shape == (p.W0.shape[1], p.W0.shape[0])

    @pytest.mark.parametrize("site_name", ["proj", "out"])
    def test_transposed_apply_keeps_a_valid_weight(
        self, linear_probe, r0, linear_step, site_name
    ):
        """
        The documented contract for transposed=True: the weight stays a valid
        (n_out, n_in) matrix, the model still runs, and revert() restores it.
        Failure means nn.Linear targets are unusable -- which is every
        non-GPT-2 model this scaffold might be pointed at.
        """
        p = OjaPlasticity(linear_probe, site=site_name, eta=1e-4,
                          mode="oja", transposed=True)
        w_before = p.module.weight.detach().clone()

        with p:
            drive(linear_probe, r0, linear_step, n=3)
            p.apply()

        assert p.module.weight.shape == w_before.shape
        assert torch.isfinite(p.module.weight).all()
        assert not torch.equal(p.module.weight, w_before)
        with torch.no_grad():
            assert linear_probe(r0).shape == r0.shape

        p.revert()
        assert torch.equal(p.module.weight, w_before)

    def test_transposed_update_is_written_in_the_weight_layout(
        self, linear_probe, r0, linear_step
    ):
        """
        Stronger form: the applied delta must be the transpose of the
        (n_in, n_out) update the rules compute. If it is written unflipped, a
        square nn.Linear target learns a scrambled update with no error at all
        -- README defect 1, in the shape it took before it was fixed.
        """
        p = OjaPlasticity(linear_probe, site="proj", eta=1.0,
                          mode="hebb", transposed=True)
        with capture(p.module) as seen, p:
            drive(linear_probe, r0, linear_step, n=3)
        expected = expected_update(seen)          # (n_in, n_out)
        p.apply()
        assert torch.allclose(p.delta, expected.transpose(0, 1), rtol=1e-5, atol=1e-9)


# --------------------------------------------------------------------------
# 15. Subspace projection
# --------------------------------------------------------------------------
# Issue #25's third knob: restrict the drift to a chosen subspace of the output
# space, so a direction can be picked and everything else left alone. The update
# is multiplied by an orthogonal projector before the ceiling, which is one line
# of arithmetic and two ways to be silently wrong -- a matrix that is not a
# projection, and a "random" control projected after its norm was matched rather
# than before.

class TestSubspaceProjection:

    ETA = 1e-6

    @staticmethod
    def directions(gpt2, k=2) -> torch.Tensor:
        """
        Real directions in residual-stream space: k unembedding rows.

        Issue #25 names exactly these -- an unembedding row, a mean of
        embeddings, a J-space basis vector -- so the projector is built from one
        rather than from a random matrix that would make the test easier and
        the claim weaker.
        """
        return gpt2.lm_head.weight[:k].detach().clone()

    def _delta_in(self, gpt2, site, r0, atr_step, **kwargs):
        """One accumulate-and-apply cycle; the weight is handed back untouched."""
        p = OjaPlasticity(gpt2, site=site, eta=self.ETA, **kwargs)
        try:
            with p:
                drive(gpt2, r0, atr_step, n=2)
            rep = p.apply()
            return p.delta.clone(), rep
        finally:
            p.revert()

    def test_the_projected_update_lies_in_the_subspace(self, gpt2, site, r0, atr_step):
        """
        The claim, to numerical tolerance: after projection, the part of the
        update outside the chosen subspace is gone.

        The unprojected arm is measured alongside because without it the test
        passes for a projector that is secretly the identity, or for an update
        that happened to live in the subspace already -- and at 2 directions out
        of 768 the second is not a possibility anyone should assume away.
        """
        basis = self.directions(gpt2, k=2)
        P = subspace_projector(basis)

        plain, _ = self._delta_in(gpt2, site, r0, atr_step, mode="oja")
        projected, rep = self._delta_in(gpt2, site, r0, atr_step, mode="oja", project=P)

        def outside(d):
            return ((d - d @ P).double().norm() / d.double().norm()).item()

        assert projected.norm().item() > 0
        assert rep["nonfinite"] is False
        assert outside(projected) < 1e-5, "the update left the subspace"
        # ...and it was a real restriction: the same rule unprojected is almost
        # entirely outside it (measured ~1.0 -- two directions out of 768).
        assert outside(plain) > 0.9
        assert projected.double().norm().item() < plain.double().norm().item()

    def test_a_one_dimensional_projection_leaves_rank_one_drift(
        self, gpt2, site, r0, atr_step
    ):
        """
        The strongest form of the claim, available only for k=1: every row of
        the update must be a multiple of the chosen direction, so the delta is
        the outer product of its own coefficients with that direction.

        This is what "aimable drift" means concretely -- and it is the check
        that a projector applied on the wrong side would fail, since `P @ upd`
        is conformable too at this site if the widths happen to match.
        """
        v = self.directions(gpt2, k=1)                      # (1, 768)
        P = subspace_projector(v)
        unit = (v / v.norm()).flatten()

        delta, _ = self._delta_in(gpt2, site, r0, atr_step, mode="oja", project=P)

        coeffs = delta @ unit                               # (n_in,)
        reconstructed = torch.outer(coeffs, unit)
        assert delta.norm().item() > 0
        assert torch.allclose(delta, reconstructed, rtol=1e-4, atol=1e-9)
        assert torch.linalg.matrix_rank(delta.double(), rtol=1e-6).item() == 1

    def test_the_random_control_is_projected_before_it_is_norm_matched(
        self, gpt2, site, r0, atr_step
    ):
        """
        C2 must stay a comparison of directions *within* the subspace. If the
        noise were matched to the unprojected update and projected afterwards,
        the random arm would carry ~sqrt(k/n_out) of the Oja arm's norm -- a
        factor of 20 at k=2, d_model=768 -- and C2 would be comparing
        magnitudes again, which is the exact confound it exists to remove
        (README defect 2, in a new place).
        """
        P = subspace_projector(self.directions(gpt2, k=2))

        oja, rep_o = self._delta_in(gpt2, site, r0, atr_step, mode="oja", project=P)
        rand, rep_r = self._delta_in(gpt2, site, r0, atr_step, mode="random",
                                     project=P, seed=0)

        assert rep_o["delta_norm"] > 0
        assert rep_r["delta_norm"] == pytest.approx(rep_o["delta_norm"], rel=1e-5)
        outside = ((rand - rand @ P).double().norm() / rand.double().norm()).item()
        assert outside < 1e-5, "the random arm drifted outside the subspace"
        assert abs(cosine(rand, oja)) < 0.5

    def test_no_projector_is_the_previous_behaviour_exactly(
        self, gpt2, site, r0, atr_step
    ):
        """
        The default path must be untouched, bit-for-bit. A new optional argument
        that changes the answer when it is not passed would invalidate every
        number already recorded in this repo.
        """
        a, _ = self._delta_in(gpt2, site, r0, atr_step, mode="oja")
        b, _ = self._delta_in(gpt2, site, r0, atr_step, mode="oja", project=None)
        assert torch.equal(a, b)

    @pytest.mark.parametrize(
        "bad, match",
        [(torch.eye(D_MODEL) * 2.0, "not idempotent"),      # scaling, not projecting
         (torch.eye(D_MLP), "project must be"),             # the input axis, not output
         (torch.ones(D_MODEL, D_MODEL), "not idempotent")],
    )
    def test_a_matrix_that_is_not_a_projection_is_refused(self, gpt2, site, bad, match):
        """
        `2I` is the dangerous one: it is symmetric, square, correctly shaped,
        and doubles every update. Nothing downstream would notice -- the run
        would simply be at twice the eta the log records.
        """
        with pytest.raises(ValueError, match=match):
            OjaPlasticity(gpt2, site=site, project=bad)

    def test_subspace_projector_builds_a_projection(self, gpt2):
        """
        The helper is what callers will actually use, so its output has to
        satisfy the constructor's check by construction rather than by luck.
        """
        basis = self.directions(gpt2, k=3)
        P = subspace_projector(basis)

        assert P.shape == (D_MODEL, D_MODEL)
        assert P.dtype == basis.dtype
        assert torch.allclose(P, P.transpose(0, 1), atol=1e-6)      # symmetric
        assert torch.allclose(P @ P, P, atol=1e-5)                  # idempotent
        assert torch.linalg.matrix_rank(P.double(), rtol=1e-6).item() == 3
        # Every basis direction survives projection; the projector spans them.
        for row in basis:
            assert torch.allclose(row @ P, row, rtol=1e-4, atol=1e-4)

    def test_subspace_projector_rejects_a_degenerate_basis(self, gpt2):
        """
        A dependent basis makes the span smaller than the caller asked for, and
        the run would confine drift to fewer directions than the log claims.
        """
        v = self.directions(gpt2, k=1)
        with pytest.raises(ValueError, match="linearly dependent"):
            subspace_projector(torch.cat([v, 2.0 * v], dim=0))
        with pytest.raises(ValueError, match="basis must be"):
            subspace_projector(v.flatten())
